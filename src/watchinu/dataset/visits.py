"""Visit log: per-truck ground truth written once per visit (cheap, high value).

A row per truck visit with a few timestamps gives, for free:
* the visit each frame belongs to (leakage-safe splits);
* a stage label for *every* frame (no_truck / open / covering / covered);
* a layout label for every open-bed frame after loading starts;
* ground truth for the visit-level evaluation (layout, final coil count).

CSV columns (timestamps ISO-8601; without an offset they are read in the plant timezone):
    visit_id, camera_id, bay_id, arrived, loading_start, tarp_start, tarp_done, departed,
    layout, coil_count, truck_ref, notes
``loading_start``, ``tarp_start`` and ``tarp_done`` may be blank if not observed.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from watchinu.dataset.manifest import FrameRecord
from watchinu.domain import Layout, Stage

VISIT_LOG_COLUMNS = [
    "visit_id",
    "camera_id",
    "bay_id",
    "arrived",
    "loading_start",
    "tarp_start",
    "tarp_done",
    "departed",
    "layout",
    "coil_count",
    "truck_ref",
    "notes",
]

NOT_VISIBLE = "not_visible"


class VisitLabel(BaseModel):
    visit_id: str
    camera_id: str
    bay_id: str
    arrived: datetime
    loading_start: datetime | None = None
    tarp_start: datetime | None = None
    tarp_done: datetime | None = None
    departed: datetime
    layout: Layout | None = None
    coil_count: int | None = None
    truck_ref: str | None = None
    notes: str | None = None

    def contains(self, t: datetime) -> bool:
        return self.arrived <= t <= self.departed

    def stage_at(self, t: datetime) -> Stage:
        if not self.contains(t):
            return Stage.NO_TRUCK
        if self.tarp_done is not None and t >= self.tarp_done:
            return Stage.COVERED
        if self.tarp_start is not None and t >= self.tarp_start:
            return Stage.COVERING
        return Stage.OPEN

    def transitions(self) -> list[datetime]:
        return [
            t
            for t in (self.arrived, self.tarp_start, self.tarp_done, self.departed)
            if t is not None
        ]


def _parse_time(text: str, tz: ZoneInfo) -> datetime | None:
    text = text.strip()
    if not text:
        return None
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=tz)


def load_visit_log(path: str | Path, default_tz: str = "Asia/Kolkata") -> list[VisitLabel]:
    tz = ZoneInfo(default_tz)
    out: list[VisitLabel] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            missing = [
                c
                for c in ("visit_id", "camera_id", "bay_id", "arrived", "departed")
                if not (row.get(c) or "").strip()
            ]
            if missing:
                raise ValueError(f"{path}:{i}: missing {', '.join(missing)}")
            out.append(
                VisitLabel(
                    visit_id=row["visit_id"].strip(),
                    camera_id=row["camera_id"].strip(),
                    bay_id=row["bay_id"].strip(),
                    arrived=_parse_time(row["arrived"], tz),
                    loading_start=_parse_time(row.get("loading_start") or "", tz),
                    tarp_start=_parse_time(row.get("tarp_start") or "", tz),
                    tarp_done=_parse_time(row.get("tarp_done") or "", tz),
                    departed=_parse_time(row["departed"], tz),
                    layout=(row.get("layout") or "").strip() or None,
                    coil_count=int(row["coil_count"])
                    if (row.get("coil_count") or "").strip()
                    else None,
                    truck_ref=(row.get("truck_ref") or "").strip() or None,
                    notes=(row.get("notes") or "").strip() or None,
                )
            )
    ids = [v.visit_id for v in out]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate visit_id")
    for v in out:
        if v.departed < v.arrived:
            raise ValueError(f"{path}: visit {v.visit_id} departs before it arrives")
    return out


def assign_visits(records: list[FrameRecord], visits: list[VisitLabel]) -> int:
    """Set ``visit_id`` on frames captured during a visit (same camera). Returns count."""
    n = 0
    for rec in records:
        rec.visit_id = None
        if rec.captured_at is None:
            continue
        for v in visits:
            if v.camera_id == rec.camera_id and v.contains(rec.captured_at):
                rec.visit_id = v.visit_id
                n += 1
                break
    return n


def visit_groups(visits: list[VisitLabel]) -> dict[str, str]:
    """visit_id -> group id. Visits overlapping in time on one camera share a group,
    because their frames show both trucks and must land in the same split."""
    parent = {v.visit_id: v.visit_id for v in visits}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ordered = sorted(visits, key=lambda v: (v.camera_id, v.arrived))
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            if b.camera_id != a.camera_id or b.arrived > a.departed:
                continue
            parent[find(b.visit_id)] = find(a.visit_id)
    return {v.visit_id: find(v.visit_id) for v in visits}


def frame_stage(
    t: datetime, camera_id: str, visits: list[VisitLabel], margin: timedelta
) -> Stage | None:
    """Stage label at time ``t``; None when within ``margin`` of a transition (ambiguous)."""
    for v in visits:
        if v.camera_id != camera_id:
            continue
        if any(abs(t - tr) < margin for tr in v.transitions()):
            return None
        if v.contains(t):
            return v.stage_at(t)
    return Stage.NO_TRUCK


def frame_layout_label(
    t: datetime, camera_id: str, visits: list[VisitLabel], margin: timedelta
) -> str | None:
    """Layout class for the layout classifier, ``not_visible`` when no load is visible,
    None when unusable (no truck, unknown layout, or near a transition)."""
    stage = frame_stage(t, camera_id, visits, margin)
    if stage is None or stage is Stage.NO_TRUCK:
        return None
    visit = next(v for v in visits if v.camera_id == camera_id and v.contains(t))
    if stage is not Stage.OPEN:
        return NOT_VISIBLE
    loading = visit.loading_start or visit.arrived
    if t < loading + margin:
        return NOT_VISIBLE if t < loading - margin else None
    if visit.layout is None or visit.layout is Layout.UNKNOWN:
        return None
    return visit.layout.value


def write_visit_log_template(path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(VISIT_LOG_COLUMNS)
