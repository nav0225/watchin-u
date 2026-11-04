from __future__ import annotations

import numpy as np

from watchinu.config import DecisionConfig, ScheduleConfig, SlotTemplate
from watchinu.domain import BBox, Layout, Product, Stage
from watchinu.runtime.aggregator import TemplateSpec, VisitAggregator
from watchinu.runtime.scheduler import HeavyScheduler
from watchinu.runtime.slots import (
    SlotBoard,
    SlotStatus,
    observe_slots,
    out_of_view_slots,
    slot_centres_uv,
    slot_image_points,
    slot_of,
)
from watchinu.vision.geometry import BedHomography, polygon_mask

T = SlotTemplate(columns=2, rows=5)
CC = SlotTemplate(columns=2, rows=7)
# Bed fills x 0..200, y 0..500; far end at the top of the image.
H = BedHomography(np.array([[0, 500], [200, 500], [200, 0], [0, 0]], np.float32))


def coil_at(slot: tuple[int, int], half: float = 20, template: SlotTemplate = T) -> BBox:
    ((x, y),) = H.to_image(np.array([slot_centres_uv(template)[slot]], np.float32))
    return BBox(x - half, y - half, x + half, y + half)


def test_slot_of_bins_and_rejects_off_bed_points():
    assert slot_of(0.1, 0.05, T) == (0, 0)
    assert slot_of(0.9, 0.95, T) == (1, 4)
    assert slot_of(1.05, 0.5, T) == (1, 2)  # within margin -> clamped
    assert slot_of(2.0, 0.5, T) is None


def test_observe_slots_occupied_visible_and_occluded():
    coils = [coil_at((0, 0)), coil_at((1, 0))]
    worker = coil_at((0, 3), half=40)  # a worker standing over slot (0, 3)
    occupied, visible = observe_slots(coils, [worker], H, T)
    assert occupied == {(0, 0), (1, 0)}
    assert (0, 3) not in visible
    assert (1, 3) in visible and (1, 3) not in occupied


def test_slots_outside_the_camera_view_are_never_seen_empty():
    # The camera only sees y >= 120: the far rows (small v... far end at top) are cut off.
    view = polygon_mask(
        np.array([[0, 120], [199, 120], [199, 499], [0, 499]], np.float32), (500, 200)
    )
    points = slot_image_points(H, T)
    oov = out_of_view_slots(points, view)
    assert oov == {(0, 4), (1, 4)}  # far row centres at y = 50
    _, visible = observe_slots([], [], H, T, view)
    assert not (visible & oov)
    board = SlotBoard(T, history=3, out_of_view=oov)
    for _ in range(3):
        occupied = {(c, r) for c in range(2) for r in range(4)}
        board.update(occupied, observe_slots([coil_at(s) for s in occupied], [], H, T, view)[1])
    summary = board.summary(infer_unseen=True, template_name="std_2x5")
    assert (summary.observed, summary.out_of_view_slots, summary.inferred_total) == (8, 2, 10)
    assert "outside camera view" in summary.basis
    # A coil detected at the very edge still counts as observed for its slot.
    board.update({(0, 4)}, {(0, 4)})
    assert board.status((0, 4)) is SlotStatus.OCCUPIED


def test_slot_board_majority_and_summary():
    board = SlotBoard(T, history=3)
    all_visible = set(slot_centres_uv(T))
    for _ in range(2):
        board.update({(0, 0), (1, 0)}, all_visible - {(0, 4), (1, 4)})
    board.update({(0, 0)}, all_visible - {(0, 4), (1, 4)})  # one missed detection
    assert board.status((1, 0)) is SlotStatus.OCCUPIED  # 2 of 3 says occupied
    assert board.status((0, 4)) is SlotStatus.UNSEEN
    summary = board.summary(infer_unseen=True, template_name="std_2x5")
    assert summary.observed == 2
    assert summary.unseen_slots == 2
    assert summary.empty_slots == 6
    assert summary.inferred_total == 4
    assert "assumed occupied" in summary.basis
    assert board.summary(infer_unseen=False, template_name="std_2x5").inferred_total is None


SPECS = {
    Layout.LINEAR: TemplateSpec("std_2x5", T),
    Layout.CRISS_CROSS: TemplateSpec("cc_2x7", CC, frozenset({(1, 0)})),
}


def _agg(**kw) -> VisitAggregator:
    agg = VisitAggregator(DecisionConfig(**kw), SPECS, {"layout": "x@1"})
    for t in range(5):
        agg.add_stage(t, Stage.WIRE_ROD)
    return agg


def test_layout_vote_and_review_flags():
    agg = _agg(min_layout_votes=3)
    for t in range(5):
        agg.add_observation(t, {"linear": 0.9, "criss_cross": 0.1}, 4, {})
    d = agg.decide("provisional", 10)
    assert d.layout is Layout.LINEAR
    assert d.product is Product.WIRE_ROD
    assert d.layout_agreement == 1.0
    assert not d.needs_review
    assert d.model_versions == {"layout": "x@1"}

    few = _agg(min_layout_votes=3)
    few.add_observation(0, {"linear": 0.9}, 1, {})
    d = few.decide("provisional", 1)
    assert d.needs_review and "too_few_layout_votes" in d.reasons


def test_not_visible_frames_carry_no_vote():
    agg = _agg(min_layout_votes=1)
    agg.add_observation(0, {"not_visible": 0.95, "linear": 0.05}, 0, {})
    assert agg.decide("provisional", 1).layout is Layout.UNKNOWN


def test_final_uses_recent_window_before_cover():
    agg = _agg(min_layout_votes=3, final_window_s=100)
    for t in range(0, 50, 10):  # early frames look criss-cross
        agg.add_observation(t, {"criss_cross": 0.9, "linear": 0.1}, 2, {})
    for t in range(500, 560, 10):  # last minutes before the tarp: linear
        agg.add_observation(t, {"linear": 0.9, "criss_cross": 0.1}, 8, {})
    agg.add_observation(700, {"criss_cross": 1.0}, 0, {})  # after cover start
    d = agg.decide("final", 700, before_ts=600)
    assert d.layout is Layout.LINEAR
    assert d.n_layout_votes == 6


def test_criss_cross_counts_use_their_own_template_and_report_cut_off_coil():
    agg = _agg(min_layout_votes=1)
    all_cc = set(slot_centres_uv(CC))
    occupied = all_cc - {(1, 0)}  # 13 seen; (1, 0) is outside the camera view
    agg.add_observation(0, {"criss_cross": 1.0}, 13, {Layout.CRISS_CROSS: (occupied, occupied)})
    d = agg.decide("final", 10)
    assert d.layout is Layout.CRISS_CROSS
    assert d.counts.template == "cc_2x7"
    assert (d.counts.observed, d.counts.out_of_view_slots, d.counts.inferred_total) == (13, 1, 14)
    assert "load_extends_beyond_camera_view" in d.reasons
    assert not d.needs_review  # a known camera limitation is information, not an alarm


def test_mixed_layout_falls_back_to_robust_count():
    agg = _agg(min_layout_votes=1)
    for t, n in enumerate([9, 14, 11, 12]):  # 14 is a one-frame false positive
        agg.add_observation(t, {"mixed": 1.0}, n, {})
    d = agg.decide("final", 10)
    assert d.layout is Layout.MIXED
    assert d.counts.observed == 12
    assert d.counts.inferred_total is None


def test_product_gate_and_other_loads():
    cfg = DecisionConfig(min_wire_rod_votes=3)
    agg = VisitAggregator(cfg, SPECS)
    agg.add_stage(0, Stage.WIRE_ROD)
    agg.add_stage(10, Stage.COVERED)  # tarp frames are not product votes
    assert not agg.product_confirmed()
    agg.add_stage(20, Stage.WIRE_ROD)
    agg.add_stage(30, Stage.WIRE_ROD)
    assert agg.product_confirmed()

    ingots = VisitAggregator(cfg, SPECS)
    for t in range(10):
        ingots.add_stage(t, Stage.OTHER_LOAD)
    d = ingots.decide("final", 100)
    assert d.product is Product.OTHER
    assert d.reasons == ["not_wire_rod_load"]
    assert not d.needs_review

    unsure = VisitAggregator(cfg, SPECS)
    for t in range(10):
        unsure.add_stage(t, Stage.WIRE_ROD if t % 2 else Stage.OTHER_LOAD)
    assert "product_uncertain" in unsure.decide("final", 100).reasons

    never_seen = VisitAggregator(cfg, SPECS)
    d = never_seen.decide("final", 100)
    assert d.product is Product.UNKNOWN and d.needs_review


def test_scheduler_provisional_spike_and_checkpoint():
    s = HeavyScheduler(
        ScheduleConfig(
            provisional_interval_s=10,
            checkpoint_interval_s=600,
            change_spike=0.1,
            change_settle=0.03,
        ),
        provisional_after_s=120,
    )
    runs = []
    for t in range(0, 130, 5):
        if s.should_run(t, 0, 0.0):
            s.mark_run(t)
            runs.append(t)
    assert runs == list(range(0, 130, 10))
    assert not s.should_run(200, 0, 0.2)  # burst in progress: wait
    assert not s.should_run(210, 0, 0.06)
    assert s.should_run(220, 0, 0.01)  # settled -> run once
    s.mark_run(220)
    assert not s.should_run(500, 0, 0.0)
    assert s.should_run(820, 0, 0.0)  # checkpoint
