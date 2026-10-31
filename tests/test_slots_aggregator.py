from __future__ import annotations

import numpy as np

from watchinu.config import DecisionConfig, ScheduleConfig, SlotTemplate
from watchinu.domain import BBox, Layout
from watchinu.runtime.aggregator import VisitAggregator
from watchinu.runtime.scheduler import HeavyScheduler
from watchinu.runtime.slots import SlotBoard, SlotStatus, observe_slots, slot_centres_uv, slot_of
from watchinu.vision.geometry import BedHomography

T = SlotTemplate(columns=2, rows=5)
# Bed fills x 0..200, y 0..500; far end at the top of the image.
H = BedHomography(np.array([[0, 500], [200, 500], [200, 0], [0, 0]], np.float32))


def coil_at(slot: tuple[int, int], half: float = 20) -> BBox:
    ((x, y),) = H.to_image(np.array([slot_centres_uv(T)[slot]], np.float32))
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


def _agg(**kw) -> VisitAggregator:
    return VisitAggregator(DecisionConfig(**kw), T, "std_2x5", {"layout": "x@1"})


def test_layout_vote_and_review_flags():
    agg = _agg(min_layout_votes=3)
    for t in range(5):
        agg.add_observation(t, {"linear": 0.9, "criss_cross": 0.1}, 4, set(), set())
    d = agg.decide("provisional", 10)
    assert d.layout is Layout.LINEAR
    assert d.layout_agreement == 1.0
    assert not d.needs_review
    assert d.model_versions == {"layout": "x@1"}

    few = _agg(min_layout_votes=3)
    few.add_observation(0, {"linear": 0.9}, 1, set(), set())
    d = few.decide("provisional", 1)
    assert d.needs_review and "too_few_layout_votes" in d.reasons


def test_not_visible_frames_carry_no_vote():
    agg = _agg(min_layout_votes=1)
    agg.add_observation(0, {"not_visible": 0.95, "linear": 0.05}, 0, set(), set())
    assert agg.decide("provisional", 1).layout is Layout.UNKNOWN


def test_final_uses_recent_window_before_cover():
    agg = _agg(min_layout_votes=3, final_window_s=100)
    for t in range(0, 50, 10):  # early frames look criss-cross
        agg.add_observation(t, {"criss_cross": 0.9, "linear": 0.1}, 2, set(), set())
    for t in range(500, 560, 10):  # last minutes before the tarp: linear
        agg.add_observation(t, {"linear": 0.9, "criss_cross": 0.1}, 8, set(), set())
    agg.add_observation(700, {"criss_cross": 1.0}, 0, set(), set())  # after cover start
    d = agg.decide("final", 700, before_ts=600)
    assert d.layout is Layout.LINEAR
    assert d.n_layout_votes == 6


def test_non_template_layout_uses_robust_count():
    agg = _agg(min_layout_votes=1)
    for t, n in enumerate([9, 14, 11, 12]):  # 14 is a one-frame false positive
        agg.add_observation(t, {"criss_cross": 1.0}, n, set(), set())
    d = agg.decide("final", 10)
    assert d.layout is Layout.CRISS_CROSS
    assert d.counts.observed == 12
    assert d.counts.inferred_total is None


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
