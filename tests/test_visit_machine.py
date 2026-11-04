from __future__ import annotations

import itertools

from watchinu.config import VisitConfig
from watchinu.domain import Stage
from watchinu.runtime.visit_machine import EventKind, VisitState, VisitStateMachine

CFG = VisitConfig(
    open_after_s=30,
    close_after_s=120,
    covered_after_s=60,
    reopen_after_s=120,
    provisional_after_s=120,
    max_visit_s=3 * 3600,
    max_gap_s=60,
    stage_min_confidence=0.6,
)


def machine() -> VisitStateMachine:
    counter = itertools.count(1)
    return VisitStateMachine(CFG, id_factory=lambda: f"v{next(counter)}")


def feed(m, t0: float, seconds: float, stage: Stage, step: float = 10.0, conf: float = 1.0):
    events = []
    t = t0
    while t < t0 + seconds:
        events += m.step(t, stage, conf)
        t += step
    return t, events


def kinds(events):
    return [e.kind for e in events]


def test_normal_visit_lifecycle():
    m = machine()
    t, ev = feed(m, 0, 60, Stage.NO_TRUCK)
    assert ev == []
    t_open = t
    t, ev1 = feed(m, t, 600, Stage.WIRE_ROD)
    t_cover = t
    t, ev2 = feed(m, t, 30, Stage.COVERING)
    t, ev3 = feed(m, t, 300, Stage.COVERED)
    t_leave = t
    t, ev4 = feed(m, t, 200, Stage.NO_TRUCK)

    assert kinds(ev1) == [EventKind.OPENED, EventKind.PROVISIONAL_DUE]
    assert ev1[0].ts == t_open  # visit start is backdated to the first evidence
    assert ev1[1].ts >= t_open + CFG.provisional_after_s
    assert kinds(ev2 + ev3) == [EventKind.COVER_DETECTED]
    assert (ev2 + ev3)[0].ts == t_cover
    assert kinds(ev4) == [EventKind.CLOSED]
    assert ev4[0].data["reason"] == "departed"
    assert ev4[0].ts == t_leave
    assert m.state is VisitState.IDLE


def test_brief_truck_sighting_does_not_open_a_visit():
    m = machine()
    t, _ = feed(m, 0, 20, Stage.WIRE_ROD)  # 20 s < open_after_s
    _, ev = feed(m, t, 100, Stage.NO_TRUCK)
    assert ev == []


def test_low_confidence_frames_are_neutral():
    m = machine()
    t, ev = feed(m, 0, 20, Stage.WIRE_ROD)
    t, ev2 = feed(m, t, 20, Stage.NO_TRUCK, conf=0.3)  # ignored, does not break the run
    t, ev3 = feed(m, t, 20, Stage.WIRE_ROD)
    assert EventKind.OPENED in kinds(ev + ev2 + ev3)


def test_camera_outage_never_looks_like_departure():
    m = machine()
    t, _ = feed(m, 0, 300, Stage.WIRE_ROD)
    # 15-minute outage (no frames), then a few unknown frames and the bed again.
    t += 900
    t, ev = feed(m, t, 60, Stage.UNKNOWN)
    t, ev2 = feed(m, t, 60, Stage.WIRE_ROD)
    assert EventKind.CLOSED not in kinds(ev + ev2)
    assert m.state is VisitState.OPEN


def test_truck_arriving_already_covered():
    m = machine()
    _, ev = feed(m, 0, 60, Stage.COVERED)
    assert kinds(ev) == [EventKind.OPENED, EventKind.COVER_DETECTED]
    assert ev[0].data["arrived_covered"] is True


def test_tarp_removed_and_reapplied_gives_second_cover_event():
    m = machine()
    t, _ = feed(m, 0, 300, Stage.WIRE_ROD)
    t, ev1 = feed(m, t, 120, Stage.COVERED)
    t, ev2 = feed(m, t, 200, Stage.WIRE_ROD)
    t, ev3 = feed(m, t, 120, Stage.COVERED)
    assert kinds(ev1) == [EventKind.COVER_DETECTED]
    assert kinds(ev2) == [EventKind.COVER_REMOVED]
    assert kinds(ev3) == [EventKind.COVER_DETECTED]


def test_departure_without_tarp_is_flagged():
    m = machine()
    t, _ = feed(m, 0, 300, Stage.WIRE_ROD)
    _, ev = feed(m, t, 200, Stage.NO_TRUCK)
    assert kinds(ev) == [EventKind.CLOSED]
    assert ev[0].data["reason"] == "departed_uncovered"


def test_timeout_then_requires_absence_before_next_visit():
    m = machine()
    _, ev = feed(m, 0, CFG.max_visit_s + 600, Stage.WIRE_ROD)
    assert kinds(ev).count(EventKind.OPENED) == 1
    assert ev[-1].kind is EventKind.CLOSED and ev[-1].data["reason"] == "timeout"


def test_snapshot_restore_resumes_identically():
    a, b = machine(), machine()
    t, _ = feed(a, 0, 200, Stage.WIRE_ROD)
    b.restore(a.snapshot())
    _, ev_a = feed(a, t, 100, Stage.COVERED)
    _, ev_b = feed(b, t, 100, Stage.COVERED)
    assert [(e.kind, e.ts, e.visit_id) for e in ev_a] == [(e.kind, e.ts, e.visit_id) for e in ev_b]


def test_other_load_trucks_open_and_close_visits_too():
    """Ingot trucks are visits as well (audit trail); the product gate is downstream."""
    m = machine()
    t, ev1 = feed(m, 0, 300, Stage.OTHER_LOAD)
    t, ev2 = feed(m, t, 120, Stage.COVERED)
    _, ev3 = feed(m, t, 200, Stage.NO_TRUCK)
    assert kinds(ev1) == [EventKind.OPENED, EventKind.PROVISIONAL_DUE]
    assert kinds(ev2) == [EventKind.COVER_DETECTED]
    assert kinds(ev3) == [EventKind.CLOSED]
