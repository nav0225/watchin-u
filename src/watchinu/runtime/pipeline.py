"""Wires frames -> light tier -> visit state machine -> heavy tier -> decisions.

The pipeline is source-agnostic: the live RTSP reader and recorded-footage replay feed it
identically, so every behaviour can be reproduced offline from a clip.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from watchinu.config import AppConfig, BayConfig
from watchinu.domain import BBox, Decision, Frame, Layout, Stage
from watchinu.models.base import ModelSet
from watchinu.runtime.aggregator import SlotObservations, TemplateSpec, VisitAggregator
from watchinu.runtime.scheduler import HeavyScheduler
from watchinu.runtime.shared import SharedFrames
from watchinu.runtime.sinks import EventSink
from watchinu.runtime.slots import observe_slots, out_of_view_slots, slot_image_points
from watchinu.runtime.visit_machine import EventKind, VisitEvent, VisitState, VisitStateMachine
from watchinu.vision.drift import ViewDriftChecker
from watchinu.vision.geometry import (
    BedHomography,
    blank_regions,
    point_in_polygon,
    polygon_mask,
    polygon_rect,
    to_pixels,
    union_coverage,
)
from watchinu.vision.overlay import draw_boxes, draw_slots, encode_jpeg
from watchinu.vision.quality import assess, change_score, thumbnail

log = logging.getLogger(__name__)


@dataclass(slots=True)
class _Geometry:
    """Bay geometry in crop-local pixel coordinates for one frame size."""

    size: tuple[int, int]
    rect: tuple[int, int, int, int]
    poly_local: np.ndarray
    mask_local: np.ndarray  # 255 inside the bay polygon (and therefore inside the frame)
    homography: BedHomography

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Bay crop with everything outside the polygon blacked out (as in training)."""
        x0, y0, x1, y1 = self.rect
        out = image[y0:y1, x0:x1].copy()
        out[self.mask_local == 0] = 0
        return out


@dataclass(slots=True)
class _Evidence:
    score: float
    jpeg: bytes


class BayProcessor:
    def __init__(
        self,
        cfg: AppConfig,
        bay: BayConfig,
        models: ModelSet,
        sink: EventSink,
        id_factory: Callable[[], str] | None = None,
        shared: SharedFrames | None = None,
    ) -> None:
        self.cfg = cfg
        self.bay = bay
        self.shared = shared or SharedFrames(models.detector, cfg.schedule.lookback_s)
        self.stage_model = models.stage_for(bay.id)
        self.templates = {
            Layout(layout): (name, template)
            for layout, (name, template) in cfg.templates_for(bay).items()
        }
        self.models = models
        self.sink = sink
        self.machine = (
            VisitStateMachine(cfg.visit, id_factory) if id_factory else VisitStateMachine(cfg.visit)
        )
        self.scheduler = HeavyScheduler(cfg.schedule, cfg.visit.provisional_after_s)
        self.aggregator: VisitAggregator | None = None
        self._geom: _Geometry | None = None
        self._last_tick: float | None = None
        self._prev_thumb: np.ndarray | None = None
        self._heavy_ts: set[float] = set()
        self._best: _Evidence | None = None
        self._final_sent = False

    # -- lifecycle ------------------------------------------------------------------
    def restore(self, snapshot: dict) -> None:
        """Resume after a restart. In-memory evidence is lost, so the visit is flagged."""
        self.machine.restore(snapshot)
        if self.machine.visit_id is not None:
            self._new_aggregator()
            assert self.aggregator is not None
            self.aggregator.flag("restarted_mid_visit")

    def _template_specs(self) -> dict[Layout, TemplateSpec]:
        specs = {}
        for layout, (name, template) in self.templates.items():
            oov: frozenset = frozenset()
            if self._geom is not None:
                points = slot_image_points(self._geom.homography, template)
                oov = frozenset(out_of_view_slots(points, self._geom.mask_local))
            specs[layout] = TemplateSpec(name, template, oov)
        return specs

    def _new_aggregator(self) -> None:
        self.aggregator = VisitAggregator(
            self.cfg.decision, self._template_specs(), self.models.versions()
        )
        self.scheduler.reset()
        self._heavy_ts.clear()
        self._best = None
        self._prev_thumb = None
        self._final_sent = False

    # -- per frame ------------------------------------------------------------------
    def process(self, ts: float, image: np.ndarray) -> None:
        """``image`` is the full frame with OSD regions already blanked."""
        interval = (
            self.cfg.schedule.idle_interval_s
            if self.machine.state is VisitState.IDLE
            else self.cfg.schedule.open_interval_s
        )
        if self._last_tick is not None and ts - self._last_tick < interval - 1e-6:
            return
        self._last_tick = ts

        geom = self._geometry(image)
        crop = geom.crop(image)
        stage, conf = self._stage(crop)

        for event in self.machine.step(ts, stage, conf):
            self._handle(event)

        if self.machine.state is VisitState.OPEN and self.aggregator is not None:
            if conf >= self.cfg.visit.stage_min_confidence:
                self.aggregator.add_stage(ts, stage)
            # Only trucks carrying wrapped wire rods ever wake the heavy tier.
            if stage is Stage.WIRE_ROD:
                self._watch(ts, image, crop, geom)

    def _watch(self, ts: float, image: np.ndarray, crop: np.ndarray, geom: _Geometry) -> None:
        thumb = thumbnail(crop)
        change = change_score(self._prev_thumb, thumb)
        self._prev_thumb = thumb
        self.shared.remember(ts, image)

        opened_at = self.machine.opened_at
        assert opened_at is not None and self.aggregator is not None
        if not self.aggregator.product_confirmed():
            return
        if self.scheduler.should_run(ts, opened_at, change):
            self.scheduler.mark_run(ts)
            self._heavy(ts, image, geom)

    def _bay_detections(
        self, ts: float, image: np.ndarray, geom: _Geometry
    ) -> tuple[list[BBox], list[BBox]]:
        """This bay's coils (centre inside the lane polygon) and the workers over it,
        from the shared full-frame detections, in crop-local coordinates."""
        x0, y0, x1, y1 = geom.rect
        coils: list[BBox] = []
        persons: list[BBox] = []
        for det in self.shared.detections(ts, image):
            local = det.box.shifted(-x0, -y0)
            if det.label == self.cfg.models.coil_label:
                cx, cy = local.center
                if det.confidence >= self.cfg.decision.coil_min_confidence and point_in_polygon(
                    cx, cy, geom.poly_local
                ):
                    coils.append(local)
            elif det.label == self.cfg.models.person_label and local.intersection(
                BBox(0, 0, x1 - x0, y1 - y0)
            ):
                persons.append(local)
        return coils, persons

    def _heavy(self, ts: float, image: np.ndarray, geom: _Geometry) -> None:
        if self.aggregator is None or ts in self._heavy_ts:
            return
        crop = geom.crop(image)
        quality = assess(crop, geom.mask_local)
        if not quality.passes(self.cfg.quality):
            return
        self._heavy_ts.add(ts)

        coils, persons = self._bay_detections(ts, image, geom)
        occlusion = union_coverage(persons, geom.mask_local)

        layout_probs = None
        if self.models.layout is not None and occlusion <= self.cfg.quality.max_occlusion_fraction:
            layout_probs = self.models.layout.classify(crop)

        slots: SlotObservations = {}
        if self.models.detector is not None:
            for layout, (_, template) in self.templates.items():
                slots[layout] = observe_slots(
                    coils, persons, geom.homography, template, geom.mask_local
                )
        self.aggregator.add_observation(ts, layout_probs, len(coils), slots)

        score = len(coils) - 10.0 * occlusion
        if self._best is None or score >= self._best.score:
            annotated = crop.copy()
            draw_boxes(annotated, coils, "coil")
            draw_boxes(annotated, persons, "person")
            shown = self.aggregator.leading_layout()
            if shown not in self.templates:
                shown = next(iter(self.templates))
            draw_slots(
                annotated,
                slot_image_points(geom.homography, self.templates[shown][1]),
                slots.get(shown, (set(), set()))[0],
            )
            self._best = _Evidence(score, encode_jpeg(annotated))

    # -- events ---------------------------------------------------------------------
    def _handle(self, event: VisitEvent) -> None:
        cam, bay = self.cfg.camera.id, self.bay.id
        if event.kind is EventKind.OPENED:
            self._new_aggregator()
            self.sink.visit_opened(cam, bay, event.visit_id, event.ts, dict(event.data))
            if event.data.get("arrived_covered"):
                assert self.aggregator is not None
                self.aggregator.flag("arrived_covered")
        elif event.kind is EventKind.PROVISIONAL_DUE:
            self._emit(event.visit_id, "provisional", event.ts, before_ts=None)
        elif event.kind is EventKind.COVER_DETECTED:
            self._analyse_lookback(before_ts=event.ts)
            self._emit(event.visit_id, "final", event.ts, before_ts=event.ts)
        elif event.kind is EventKind.COVER_REMOVED:
            self._final_sent = False  # a new final supersedes the old one at the next cover
        elif event.kind is EventKind.CLOSED:
            reason = str(event.data.get("reason", ""))
            if not self._final_sent and self.aggregator is not None:
                if reason != "departed":
                    self.aggregator.flag(reason)
                self._emit(event.visit_id, "final", event.ts, before_ts=event.ts)
            self.sink.visit_closed(cam, bay, event.visit_id, event.ts, reason)
            self.aggregator = None
        self.sink.save_state(cam, bay, self.machine.snapshot())

    def _analyse_lookback(self, before_ts: float) -> None:
        """Run the heavy tier on buffered frames from just before the tarp went on."""
        if self._geom is None:
            return
        for ts in self.shared.history(before_ts):
            if ts in self._heavy_ts:
                continue
            image = self.shared.image_at(ts)
            if image is not None:
                self._heavy(ts, image, self._geom)

    def _emit(self, visit_id: str, kind: str, ts: float, before_ts: float | None) -> None:
        if self.aggregator is None:
            return
        decision: Decision = self.aggregator.decide(kind, ts, before_ts=before_ts)
        if kind == "final":
            self._final_sent = True
        evidence = self._best.jpeg if self._best is not None else None
        self.sink.decision(self.cfg.camera.id, self.bay.id, visit_id, decision, evidence)

    # -- helpers --------------------------------------------------------------------
    def _geometry(self, image: np.ndarray) -> _Geometry:
        h, w = image.shape[:2]
        if self._geom is None or self._geom.size != (w, h):
            poly = to_pixels(self.bay.polygon, w, h)
            rect = polygon_rect(poly, w, h)
            offset = np.array(rect[:2], dtype=np.float32)
            poly_local = poly - offset
            mask_local = polygon_mask(poly_local, (rect[3] - rect[1], rect[2] - rect[0]))
            quad_local = to_pixels(self.bay.bed_quad, w, h) - offset
            self._geom = _Geometry((w, h), rect, poly_local, mask_local, BedHomography(quad_local))
            if self.aggregator is not None:  # e.g. restored before the first frame
                for layout, spec in self._template_specs().items():
                    self.aggregator.boards[layout].out_of_view = set(spec.out_of_view)
        return self._geom

    def _stage(self, crop: np.ndarray) -> tuple[Stage, float]:
        if self.stage_model is None:
            return Stage.UNKNOWN, 0.0
        probs = self.stage_model.classify(crop)
        label, p = max(probs.items(), key=lambda kv: kv[1])
        try:
            return Stage(label), p
        except ValueError:
            return Stage.UNKNOWN, 0.0


class CameraProcessor:
    """Fans each frame out to every bay, behind the OSD mask and the view-drift guard."""

    def __init__(
        self,
        cfg: AppConfig,
        models: ModelSet,
        sink: EventSink,
        drift_checker: ViewDriftChecker | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.sink = sink
        self.drift_checker = drift_checker
        self.shared = SharedFrames(models.detector, cfg.schedule.lookback_s)
        self.bays = [
            BayProcessor(cfg, bay, models, sink, id_factory, self.shared) for bay in cfg.bays
        ]
        self.suspended = False
        self._last_drift_check: float | None = None

    def process(self, frame: Frame) -> None:
        if self.drift_checker is not None and (
            self._last_drift_check is None
            or frame.ts - self._last_drift_check >= self.cfg.schedule.drift_check_interval_s
        ):
            self._last_drift_check = frame.ts
            result = self.drift_checker.check(frame.image)
            if not result.ok and not self.suspended:
                self.suspended = True
                self.sink.alert(
                    self.cfg.camera.id,
                    "view_drift",
                    f"camera view changed ({result.reason}, shift {result.shift_px:.0f}px); "
                    "decisions suspended until the view is restored or recalibrated",
                    frame.ts,
                )
            elif result.ok and self.suspended:
                self.suspended = False
                self.sink.alert(self.cfg.camera.id, "view_restored", "view matches", frame.ts)
        if self.suspended:
            return
        image = blank_regions(frame.image, self.cfg.camera.osd_masks)
        for bay in self.bays:
            bay.process(frame.ts, image)
