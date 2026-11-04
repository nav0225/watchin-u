"""End-to-end: synthetic frames -> light tier -> state machine -> heavy tier -> decisions."""

from __future__ import annotations

import numpy as np

from watchinu.domain import BBox, Detection, Frame, Layout, Product
from watchinu.models.base import ModelSet
from watchinu.models.fake import FunctionClassifier, FunctionDetector
from watchinu.runtime.pipeline import CameraProcessor
from watchinu.runtime.sinks import MemorySink
from watchinu.runtime.slots import slot_image_points
from watchinu.vision.drift import DriftResult
from watchinu.vision.geometry import BedHomography, to_pixels

from .helpers import BASE_CONFIG, make_config, textured

# Synthetic frames encode the stage in their mean brightness.
CODES = {"no_truck": 30, "wire_rod": 80, "other_load": 130, "covering": 175, "covered": 220}
W, H = 320, 180


def stage_from_image(image: np.ndarray) -> dict[str, float]:
    mean = float(image.mean())
    label = min(CODES, key=lambda k: abs(CODES[k] - mean))
    return {label: 0.95, **{k: 0.05 / 4 for k in CODES if k != label}}


class Counting:
    """Wraps a fake model and records the timestamp-free call count."""

    def __init__(self, inner):
        self.inner, self.calls = inner, 0
        self.name, self.version = inner.name, inner.version

    def classify(self, img):
        self.calls += 1
        return self.inner.classify(img)

    def detect(self, img):
        self.calls += 1
        return self.inner.detect(img)


def make_models(cfg, n_coils: int = 10, layout: str = "linear", extra=()) -> ModelSet:
    bay = cfg.bays[0]
    template = cfg.templates_for(bay)[layout][1]
    points = slot_image_points(BedHomography(to_pixels(bay.bed_quad, W, H)), template)
    pts = [p for p in points.values() if 0 <= p[1] < H][:n_coils] + list(extra)

    def detect(image: np.ndarray) -> list[Detection]:
        if abs(float(image.mean()) - CODES["wire_rod"]) > 20:
            return []
        return [
            Detection("coil", 0.9, BBox(x - 6, y - 6, x + 6, min(y + 6, H - 1))) for x, y in pts
        ]

    return ModelSet(
        stage=FunctionClassifier(stage_from_image, "stage"),
        layout=FunctionClassifier(lambda img: {layout: 0.9, "mixed": 0.1}, "layout"),
        detector=Counting(FunctionDetector(detect, "det")),
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


NORMAL = [
    ("no_truck", 60),
    ("wire_rod", 900),
    ("covering", 120),
    ("covered", 300),
    ("no_truck", 300),
]


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
    models.stage = Counting(models.stage)
    # A 45-minute visit, one frame every 2 s from the camera.
    run(
        cfg,
        models,
        [("no_truck", 60), ("wire_rod", 2700), ("covered", 300), ("no_truck", 300)],
        step=2.0,
    )
    frames = (60 + 2700 + 300 + 300) // 2
    assert models.stage.calls < frames / 4  # light tier is duty-cycled too
    # provisional burst (~10) + checkpoints (~4) + look-back before the tarp (<=18)
    assert models.detector.calls <= 40, models.detector.calls


def test_ingot_truck_is_recorded_but_never_wakes_the_heavy_model():
    cfg = make_config()
    models = make_models(cfg)
    sink, _ = run(
        cfg, models, [("no_truck", 60), ("other_load", 900), ("covered", 300), ("no_truck", 300)]
    )
    assert models.detector.calls == 0
    assert [r["type"] for r in sink.records] == ["opened", "decision", "decision", "closed"]
    final = sink.of_type("decision")[-1]["decision"]
    assert final.product is Product.OTHER
    assert final.reasons == ["not_wire_rod_load"]
    assert not final.needs_review


def test_heavy_model_waits_for_wire_rods_to_be_confirmed():
    cfg = make_config(decision={"min_wire_rod_votes": 3})
    models = make_models(cfg)
    # 40 s of truck: the visit opens at 30 s but only 1 wire-rod vote is in -> no heavy run.
    run(cfg, models, [("no_truck", 30), ("wire_rod", 40), ("no_truck", 300)])
    assert models.detector.calls == 0
    models = make_models(cfg)
    run(cfg, models, [("no_truck", 30), ("wire_rod", 90), ("no_truck", 300)])
    assert models.detector.calls > 0


def test_criss_cross_load_with_last_coil_cut_off_by_the_frame():
    # Bed extends below the image: the near row of the 2x7 criss-cross grid is out of view.
    bay = {**BASE_CONFIG["bays"][0], "bed_quad": [[0.1, 1.1], [0.9, 1.1], [0.9, 0.05], [0.1, 0.05]]}
    cfg = make_config(bays=[bay])
    edge_coil = [(W * 0.3, H - 4)]  # half a coil peeking in at the bottom edge -> slot (0, 0)
    models = make_models(cfg, n_coils=12, layout="criss_cross", extra=edge_coil)
    sink, _ = run(
        cfg, models, [("no_truck", 60), ("wire_rod", 900), ("covered", 300), ("no_truck", 300)]
    )
    final = sink.of_type("decision")[-1]["decision"]
    assert final.layout is Layout.CRISS_CROSS
    assert final.counts.template == "cc_2x7"
    assert final.counts.observed == 13
    assert final.counts.out_of_view_slots == 1
    assert final.counts.inferred_total == 14
    assert "load_extends_beyond_camera_view" in final.reasons
    assert not final.needs_review, final.reasons


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
    sink, _ = run(cfg, models, [("no_truck", 30), ("wire_rod", 300)])
    snapshot = sink.states[("cam01", "left")]

    restarted = CameraProcessor(cfg, models, sink, id_factory=lambda: "visit-2")
    restarted.bays[0].restore(snapshot)
    run(
        cfg,
        models,
        [("wire_rod", 400), ("covered", 300), ("no_truck", 300)],
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


def _two_lane_config():
    left = {
        "id": "left",
        "polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]],
        "bed_quad": [[0.05, 0.95], [0.45, 0.95], [0.45, 0.05], [0.05, 0.05]],
        "slot_templates": {"linear": "std_2x5"},
    }
    right = {
        "id": "right",
        "polygon": [[0.5, 0], [1, 0], [1, 1], [0.5, 1]],
        "bed_quad": [[0.55, 0.95], [0.95, 0.95], [0.95, 0.05], [0.55, 0.05]],
        "slot_templates": {"linear": "std_2x5"},
    }
    return make_config(bays=[left, right])


def test_two_lanes_share_one_detector_pass_and_count_only_their_own_coils():
    cfg = _two_lane_config()
    # Left lane: wire rods with 10 coils; right lane: ingot truck (other_load).
    left_img = textured(CODES["wire_rod"], (H, W // 2))
    right_img = textured(CODES["other_load"], (H, W // 2))
    frame_img = np.hstack([left_img, right_img])
    bay = cfg.bays[0]
    pts = slot_image_points(
        BedHomography(to_pixels(bay.bed_quad, W, H)), cfg.templates_for(bay)["linear"][1]
    )
    # Coils in BOTH halves; the right-half ones must never reach the left lane's count.
    boxes = [BBox(x - 5, y - 5, x + 5, y + 5) for x, y in pts.values()]
    boxes += [b.shifted(W / 2, 0) for b in boxes]
    detector = Counting(FunctionDetector(lambda img: [Detection("coil", 0.9, b) for b in boxes]))
    models = ModelSet(
        stage=FunctionClassifier(stage_from_image),
        layout=FunctionClassifier(lambda img: {"linear": 0.9, "mixed": 0.1}),
        detector=detector,
    )
    sink = MemorySink()
    ids = iter(f"v{i}" for i in range(10))
    camera = CameraProcessor(cfg, models, sink, id_factory=lambda: next(ids))
    t = 1_700_000_000.0
    for img, seconds in (
        (textured(CODES["no_truck"], (H, W)), 60),
        (frame_img, 900),
        (textured(CODES["covered"], (H, W)), 300),
        (textured(CODES["no_truck"], (H, W)), 300),
    ):
        for _ in range(int(seconds / 5)):
            camera.process(Frame(ts=t, image=img))
            t += 5
    finals = {
        r["bay"]: r["decision"] for r in sink.of_type("decision") if r["decision"].kind == "final"
    }
    assert finals["left"].product is Product.WIRE_ROD
    assert finals["left"].counts.observed == 10  # not 20
    assert finals["right"].product is Product.OTHER  # ingot lane: logged, not inspected
    # one detector pass per heavy timestamp, shared by both lanes
    assert detector.calls == camera.shared.detector_calls
    assert detector.calls <= 40
