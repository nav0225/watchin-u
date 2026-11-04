"""Per-bay visit state machine, driven only by frame timestamps (never the wall clock).

    IDLE --truck present >= open_after_s--> OPEN --tarp >= covered_after_s--> COVERED
      ^                                      |  ^                               |
      |                                      |  +---- open bed >= reopen_after_s+
      +------ no truck >= close_after_s -----+----------------------------------+

Every transition needs the condition to *hold* for a minimum duration (debouncing), and
low-confidence or unknown frames are neutral: they neither confirm nor break a run.
A gap with no evidence longer than ``max_gap_s`` (camera outage) breaks every run, so an
outage can never be mistaken for "the truck left".
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from watchinu.config import VisitConfig
from watchinu.domain import Stage


class VisitState(StrEnum):
    IDLE = "idle"
    OPEN = "open"
    COVERED = "covered"


class EventKind(StrEnum):
    OPENED = "opened"
    PROVISIONAL_DUE = "provisional_due"
    COVER_DETECTED = "cover_detected"
    COVER_REMOVED = "cover_removed"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class VisitEvent:
    kind: EventKind
    ts: float
    visit_id: str
    data: dict[str, Any] = field(default_factory=dict)


class _Hold:
    """A contiguous run of evidence that a condition is true."""

    __slots__ = ("last", "since")

    def __init__(self) -> None:
        self.since: float | None = None
        self.last: float | None = None

    def observe(self, ts: float, value: bool | None, max_gap: float) -> None:
        if self.last is not None and ts - self.last > max_gap:
            self.since = self.last = None
        if value is None:
            return
        if value:
            if self.since is None:
                self.since = ts
            self.last = ts
        else:
            self.since = self.last = None

    def held_for(self) -> float:
        if self.since is None or self.last is None:
            return 0.0
        return self.last - self.since

    def reset(self) -> None:
        self.since = self.last = None


def _new_id() -> str:
    return uuid.uuid4().hex


class VisitStateMachine:
    def __init__(self, cfg: VisitConfig, id_factory: Callable[[], str] = _new_id) -> None:
        self.cfg = cfg
        self._id_factory = id_factory
        self.state = VisitState.IDLE
        self.visit_id: str | None = None
        self.opened_at: float | None = None
        self.cover_started_at: float | None = None
        self._provisional_sent = False
        self._require_absence = False
        self._present = _Hold()
        self._tarp = _Hold()
        self._open_bed = _Hold()
        self._absent = _Hold()

    # -- public ---------------------------------------------------------------------
    def step(self, ts: float, stage: Stage, confidence: float = 1.0) -> list[VisitEvent]:
        if confidence < self.cfg.stage_min_confidence:
            stage = Stage.UNKNOWN
        known = stage is not Stage.UNKNOWN
        gap = self.cfg.max_gap_s
        self._present.observe(ts, stage.truck_present if known else None, gap)
        self._tarp.observe(ts, stage.tarp_on if known else None, gap)
        self._open_bed.observe(ts, stage.bed_open if known else None, gap)
        self._absent.observe(ts, stage is Stage.NO_TRUCK if known else None, gap)

        events: list[VisitEvent] = []
        if self.state is VisitState.IDLE:
            self._step_idle(events)
        elif self.state is VisitState.OPEN:
            self._step_open(ts, events)
        else:
            self._step_covered(events)

        if (
            self.visit_id is not None
            and self.opened_at is not None
            and ts - self.opened_at >= self.cfg.max_visit_s
        ):
            self._close(ts, "timeout", events)
            self._require_absence = True
        return events

    def snapshot(self) -> dict[str, Any]:
        """Serialisable state, persisted so a restart can resume an open visit."""
        holds = {name: [h.since, h.last] for name, h in self._holds().items()}
        return {
            "state": self.state.value,
            "visit_id": self.visit_id,
            "opened_at": self.opened_at,
            "cover_started_at": self.cover_started_at,
            "provisional_sent": self._provisional_sent,
            "require_absence": self._require_absence,
            "holds": holds,
        }

    def restore(self, snap: dict[str, Any]) -> None:
        self.state = VisitState(snap["state"])
        self.visit_id = snap["visit_id"]
        self.opened_at = snap["opened_at"]
        self.cover_started_at = snap["cover_started_at"]
        self._provisional_sent = snap["provisional_sent"]
        self._require_absence = snap["require_absence"]
        for name, hold in self._holds().items():
            hold.since, hold.last = snap["holds"][name]

    # -- transitions ----------------------------------------------------------------
    def _holds(self) -> dict[str, _Hold]:
        return {
            "present": self._present,
            "tarp": self._tarp,
            "open_bed": self._open_bed,
            "absent": self._absent,
        }

    def _step_idle(self, events: list[VisitEvent]) -> None:
        if self._require_absence:
            if self._absent.held_for() >= self.cfg.close_after_s:
                self._require_absence = False
            return
        if self._present.held_for() < self.cfg.open_after_s:
            return
        assert self._present.since is not None
        self.visit_id = self._id_factory()
        self.opened_at = self._present.since
        self._provisional_sent = False
        arrived_covered = self._tarp.held_for() >= self.cfg.open_after_s
        events.append(
            VisitEvent(
                EventKind.OPENED,
                self.opened_at,
                self.visit_id,
                {"arrived_covered": arrived_covered},
            )
        )
        if arrived_covered:
            # Truck came in already tarped (re-entry, or loaded elsewhere): no evidence.
            self.state = VisitState.COVERED
            self.cover_started_at = self.opened_at
            events.append(
                VisitEvent(
                    EventKind.COVER_DETECTED,
                    self.opened_at,
                    self.visit_id,
                    {"arrived_covered": True},
                )
            )
        else:
            self.state = VisitState.OPEN

    def _step_open(self, ts: float, events: list[VisitEvent]) -> None:
        assert self.visit_id is not None and self.opened_at is not None
        if not self._provisional_sent and ts - self.opened_at >= self.cfg.provisional_after_s:
            self._provisional_sent = True
            events.append(VisitEvent(EventKind.PROVISIONAL_DUE, ts, self.visit_id))
        if self._tarp.held_for() >= self.cfg.covered_after_s:
            assert self._tarp.since is not None
            self.state = VisitState.COVERED
            self.cover_started_at = self._tarp.since
            events.append(VisitEvent(EventKind.COVER_DETECTED, self._tarp.since, self.visit_id))
        elif self._absent.held_for() >= self.cfg.close_after_s:
            assert self._absent.since is not None
            self._close(self._absent.since, "departed_uncovered", events)

    def _step_covered(self, events: list[VisitEvent]) -> None:
        assert self.visit_id is not None
        if self._absent.held_for() >= self.cfg.close_after_s:
            assert self._absent.since is not None
            self._close(self._absent.since, "departed", events)
        elif self._open_bed.held_for() >= self.cfg.reopen_after_s:
            # Tarp pulled back (e.g. to fix the load): back to watching the bed.
            assert self._open_bed.since is not None
            self.state = VisitState.OPEN
            self.cover_started_at = None
            events.append(VisitEvent(EventKind.COVER_REMOVED, self._open_bed.since, self.visit_id))

    def _close(self, ts: float, reason: str, events: list[VisitEvent]) -> None:
        assert self.visit_id is not None
        events.append(VisitEvent(EventKind.CLOSED, ts, self.visit_id, {"reason": reason}))
        self.state = VisitState.IDLE
        self.visit_id = None
        self.opened_at = None
        self.cover_started_at = None
        self._provisional_sent = False
        for hold in self._holds().values():
            hold.reset()
