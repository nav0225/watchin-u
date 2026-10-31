"""Wires frames -> light tier -> visit state machine -> heavy tier -> decisions.

The pipeline is source-agnostic: the live RTSP reader and recorded-footage replay feed it
identically, so every behaviour can be reproduced offline from a clip.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from watchinu.config import AppConfig, BayConfig
from watchinu.domain import BBox, Decision, Frame, Stage
from watchinu.models.base import ModelSet
from watchinu.runtime.aggregator import VisitAggregator
from watchinu.runtime.scheduler import HeavyScheduler
from watchinu.runtime.sinks import EventSink
from watchinu.runtime.slots import observe_slots, slot_centres_uv
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
from watchinu.vision.overlay import decode_jpeg, draw_boxes, draw_slots, encode_jpeg
from watchinu.vision.quality import assess, change_score, thumbnail

log = logging.getLogger(__name__)


@dataclass(slots=True)
class _Geometry:
    """Bay geometry in crop-local pixel coordinates for one frame size."""

    size: tuple[int, int]
    rect: tuple[int, int, int, int]
    poly_local: np.ndarray
    mask_local: np.ndarray
    homography: BedHomography

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Bay crop with everything outside the polygon blacked out (as in training)."""
        x0, y0, x1, y1 = self.rect
        out = image[y0:y1, x0:x1].copy()
        out[self.mask_local == 0] = 0
        return out


@dataclass(slots=True)
class _Buffered:
    ts: float
    jpeg: bytes


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
    ) -> None:
        self.cfg = cfg
        self.bay = bay
        self.template = cfg.template_for(bay)
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
        self._lookback: deque[_Buffered] = deque()
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

    def _new_aggregator(self) -> None:
        self.aggregator = VisitAggregator(
            self.cfg.decision, self.template, self.bay.slot_template, self.models.versions()
        )
        self.scheduler.reset()
        self._lookback.clear()
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

        if self.machine.state is VisitState.OPEN and stage is Stage.OPEN:
            self._watch(ts, crop, geom)

    def _watch(self, ts: float, crop: np.ndarray, geom: _Geometry) -> None:
        thumb = thumbnail(crop)
        change = change_score(self._prev_thumb, thumb)
        self._prev_thumb = thumb

        self._lookback.append(_Buffered(ts, encode_jpeg(crop, 95)))
        while self._lookback and ts - self._lookback[0].ts > self.cfg.schedule.lookback_s:
            self._lookback.popleft()

        opened_at = self.machine.opened_at
        assert opened_at is not None
        if self.scheduler.should_run(ts, opened_at, change):
            self.scheduler.mark_run(ts)
            self._heavy(ts, crop, geom)

    def _heavy(self, ts: float, crop: np.ndarray, geom: _Geometry) -> None:
        if self.aggregator is None or ts in self._heavy_ts:
            return
        quality = assess(crop, geom.mask_local)
        if not quality.passes(self.cfg.quality):
            return
        self._heavy_ts.add(ts)

        coils: list[BBox] = []
        persons: list[BBox] = []
        if self.models.detector is not None:
            for det in self.models.detector.detect(crop):
                cx, cy = det.box.center
                if det.label == self.cfg.models.coil_label:
                    if det.confidence >= self.cfg.decision.coil_min_confidence and (
                        point_in_polygon(cx, cy, geom.poly_local)
                    ):
                        coils.append(det.box)
                elif det.label == self.cfg.models.person_label:
                    persons.append(det.box)
        occlusion = union_coverage(persons, geom.mask_local)

        layout_probs = None
        if self.models.layout is not None and occlusion <= self.cfg.quality.max_occlusion_fraction:
            layout_probs = self.models.layout.classify(crop)

        if self.models.detector is not None:
            occupied, visible = observe_slots(coils, persons, geom.homography, self.template)
        else:
            occupied, visible = set(), set()
        self.aggregator.add_observation(ts, layout_probs, len(coils), occupied, visible)

        score = len(coils) - 10.0 * occlusion
        if self._best is None or score >= self._best.score:
            annotated = crop.copy()
            draw_boxes(annotated, coils, "coil")
            draw_boxes(annotated, persons, "person")
            centres = slot_centres_uv(self.template)
            pts = geom.homography.to_image(np.array(list(centres.values()), dtype=np.float32))
            draw_slots(
                annotated,
                {s: (float(x), float(y)) for s, (x, y) in zip(centres, pts, strict=True)},
                occupied,
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
        for item in list(self._lookback):
            if item.ts < before_ts and item.ts not in self._heavy_ts:
                self._heavy(item.ts, decode_jpeg(item.jpeg), self._geom)

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
        return self._geom

    def _stage(self, crop: np.ndarray) -> tuple[Stage, float]:
        if self.models.stage is None:
            return Stage.UNKNOWN, 0.0
        probs = self.models.stage.classify(crop)
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
        self.bays = [BayProcessor(cfg, bay, models, sink, id_factory) for bay in cfg.bays]
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
