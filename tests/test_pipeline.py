"""End-to-end: synthetic frames -> light tier -> state machine -> heavy tier -> decisions."""

from __future__ import annotations

import numpy as np

from watchinu.domain import BBox, Detection, Frame, Layout
from watchinu.models.base import ModelSet
from watchinu.models.fake import FunctionClassifier, FunctionDetector
from watchinu.runtime.pipeline import CameraProcessor
from watchinu.runtime.sinks import MemorySink
from watchinu.runtime.slots import slot_centres_uv
from watchinu.vision.drift import DriftResult
from watchinu.vision.geometry import BedHomography, to_pixels

from .helpers import make_config, textured

CODES = {"no_truck": 40, "open": 100, "covering": 160, "covered": 210}
W, H = 320, 180


def stage_from_image(image: np.ndarray) -> dict[str, float]:
    mean = float(image.mean())
    label = min(CODES, key=lambda k: abs(CODES[k] - mean))
    return {label: 0.95, **{k: 0.05 / 3 for k in CODES if k != label}}


def make_models(cfg, n_coils: int = 10, layout: str = "linear") -> ModelSet:
    bay = cfg.bays[0]
    homography = BedHomography(to_pixels(bay.bed_quad, W, H))
    centres = slot_centres_uv(cfg.template_for(bay))
    pts = homography.to_image(np.array(list(centres.values()), np.float32))[:n_coils]

    def detect(image: np.ndarray) -> list[Detection]:
        if abs(float(image.mean()) - CODES["open"]) > 20:
            return []
        return [Detection("coil", 0.9, BBox(x - 6, y - 6, x + 6, y + 6)) for x, y in pts]

    return ModelSet(
        stage=FunctionClassifier(stage_from_image, "stage"),
        layout=FunctionClassifier(lambda img: {layout: 0.9, "criss_cross": 0.1}, "layout"),
        detector=FunctionDetector(detect, "det"),
    )


def run(cfg, models, timeline, step: float = 5.0, processor=None, sink=None):
    sink = sink or MemorySink()
    camera = processor or CameraProcessor(cfg, models, sink, id_factory=lambda: "visit-1")
    t = 1_700_000_000.0
    for stage, seconds in timeline:
        img = textured(CODES[stage], (H, W))
        end = t + seconds
        while t < end:
            camera.process(Frame(ts=t, image=img))
            t += step
    return sink, camera


NORMAL = [("no_truck", 60), ("open", 900), ("covering", 120), ("covered", 300), ("no_truck", 300)]


def test_full_visit_produces_provisional_and_final_decisions():
    cfg = make_config()
    sink, _ = run(cfg, make_models(cfg), NORMAL)
    kinds = [r["type"] for r in sink.records]
    assert kinds == ["opened", "decision", "decision", "closed"]

    provisional, final = (r["decision"] for r in sink.of_type("decision"))
    assert provisional.kind == "provisional"
    assert final.kind == "final"
    assert final.layout is Layout.LINEAR
    assert final.counts.observed == 10
    assert final.counts.inferred_total == 10
    assert not final.needs_review, final.reasons
    assert sink.of_type("decision")[1]["has_evidence"]
    assert sink.of_type("closed")[0]["reason"] == "departed"
    assert ("cam01", "left") in sink.states


def test_partially_visible_load_reports_observed_and_inferred_separately():
    cfg = make_config()
    sink, _ = run(cfg, make_models(cfg, n_coils=8), NORMAL)
    final = sink.of_type("decision")[-1]["decision"]
    # The far row is clearly visible and empty here, so it must NOT be inferred as full.
    assert final.counts.observed == 8
    assert final.counts.empty_slots == 2
    assert final.counts.unseen_slots == 0
    assert final.counts.inferred_total == 8


def test_heavy_model_sleeps_for_most_of_the_visit():
    cfg = make_config()
    models = make_models(cfg)
    calls = {"stage": 0, "detector": 0}
    stage_fn, det = models.stage, models.detector

    class Counting:
        name, version = "c", "1"

        def __init__(self, inner, key):
            self.inner, self.key = inner, key

        def classify(self, img):
            calls[self.key] += 1
            return self.inner.classify(img)

        def detect(self, img):
            calls[self.key] += 1
            return self.inner.detect(img)

    models.stage, models.detector = Counting(stage_fn, "stage"), Counting(det, "detector")
    # A 45-minute visit, one frame every 2 s from the camera.
    run(
        cfg,
        models,
        [("no_truck", 60), ("open", 2700), ("covered", 300), ("no_truck", 300)],
        step=2.0,
    )
    frames = (60 + 2700 + 300 + 300) // 2
    assert calls["stage"] < frames / 4  # light tier is duty-cycled too
    # provisional burst (~13) + checkpoints (~4) + look-back before the tarp (<=18)
    assert calls["detector"] <= 40, calls


def test_truck_arriving_covered_needs_review():
    cfg = make_config()
    sink, _ = run(cfg, make_models(cfg), [("no_truck", 30), ("covered", 300), ("no_truck", 300)])
    final = sink.of_type("decision")[0]["decision"]
    assert final.layout is Layout.UNKNOWN
    assert final.needs_review
    assert "arrived_covered" in final.reasons


def test_restart_mid_visit_resumes_and_flags():
    cfg = make_config()
    models = make_models(cfg)
    sink, _ = run(cfg, models, [("no_truck", 30), ("open", 300)])
    snapshot = sink.states[("cam01", "left")]

    restarted = CameraProcessor(cfg, models, sink, id_factory=lambda: "visit-2")
    restarted.bays[0].restore(snapshot)
    run(
        cfg,
        models,
        [("open", 400), ("covered", 300), ("no_truck", 300)],
        processor=restarted,
        sink=sink,
    )
    final = sink.of_type("decision")[-1]["decision"]
    assert "restarted_mid_visit" in final.reasons
    assert sink.of_type("closed")[0]["visit_id"] == "visit-1"


class _DriftStub:
    def __init__(self) -> None:
        self.ok = True

    def check(self, image):
        return DriftResult(
            self.ok,
            1.0 if self.ok else 0.0,
            0.0 if self.ok else 99.0,
            "" if self.ok else "view_shifted",
        )


def test_view_drift_suspends_processing_and_alerts():
    cfg = make_config(schedule={"drift_check_interval_s": 0})
    drift = _DriftStub()
    drift.ok = False
    sink = MemorySink()
    camera = CameraProcessor(cfg, make_models(cfg), sink, drift_checker=drift)
    run(cfg, None, NORMAL, processor=camera, sink=sink)
    assert [r["kind"] for r in sink.of_type("alert")] == ["view_drift"]
    assert not sink.of_type("opened")
