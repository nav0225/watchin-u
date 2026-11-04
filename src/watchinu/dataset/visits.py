"""Visit log: per-truck ground truth written once per visit (cheap, high value).

A row per truck visit with a few timestamps gives, for free:
* the visit each frame belongs to (leakage-safe splits);
* a stage label for *every* frame (no_truck / wire_rod / other_load / covering / covered);
* a layout label for every open-bed frame after loading starts;
* ground truth for the visit-level evaluation (product, layout, final coil count).

CSV columns (timestamps ISO-8601; without an offset they are read in the plant timezone):
    visit_id, camera_id, bay_id, arrived, loading_start, tarp_start, tarp_done, departed,
    product, layout, coil_count, truck_ref, notes
``loading_start``, ``tarp_start`` and ``tarp_done`` may be blank if not observed.
``product`` is ``wire_rod`` or what else the truck carries (``ingot``, ``empty``, ...).
A blank product excludes that visit's open-bed frames from stage training.
``window`` = ``clip`` marks a short recording: ``arrived``/``departed`` are then just the
recording bounds (the truck was there before and after), not real arrival/departure.
``stage`` (optional) states what was visible throughout the row's window when the real
transition times were not observed (e.g. a truck already tarped when recording started);
it overrides the stage derived from the tarp columns. Rows with the same ``truck_ref``
are one physical truck seen in several recordings and always share a split.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from watchinu.dataset.manifest import FrameRecord
from watchinu.domain import Layout, Product, Stage

VISIT_LOG_COLUMNS = [
    "visit_id",
    "camera_id",
    "bay_id",
    "arrived",
    "loading_start",
    "tarp_start",
    "tarp_done",
    "departed",
    "product",
    "layout",
    "coil_count",
    "truck_ref",
    "notes",
    "window",
    "stage",
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
    product: Product | None = None
    product_label: str | None = None  # as written in the log, e.g. "ingot"
    layout: Layout | None = None
    coil_count: int | None = None
    truck_ref: str | None = None
    notes: str | None = None
    clip: bool = False  # arrived/departed are recording bounds, not truck transitions
    stage: Stage | None = None  # observed stage for the whole window (overrides tarp times)

    def contains(self, t: datetime) -> bool:
        return self.arrived <= t <= self.departed

    def stage_at(self, t: datetime) -> Stage | None:
        """None when the stage cannot be labelled (open bed of unknown product)."""
        if not self.contains(t):
            return Stage.NO_TRUCK
        if self.stage is not None:
            return self.stage
        if self.tarp_done is not None and t >= self.tarp_done:
            return Stage.COVERED
        if self.tarp_start is not None and t >= self.tarp_start:
            return Stage.COVERING
        if self.product is Product.WIRE_ROD:
            # Before the first coil is placed the bed is empty: not (yet) a wire-rod frame.
            if self.loading_start is not None and t < self.loading_start:
                return Stage.OTHER_LOAD
            return Stage.WIRE_ROD
        if self.product is Product.OTHER:
            return Stage.OTHER_LOAD
        return None

    def transitions(self) -> list[datetime]:
        """Times where the stage really changes (ambiguous frames nearby are skipped)."""
        bounds = () if self.clip else (self.arrived, self.departed)
        events = (self.loading_start, self.tarp_start, self.tarp_done)
        return [t for t in (*bounds, *events) if t is not None]


def parse_product(text: str | None) -> Product | None:
    norm = (text or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not norm:
        return None
    return Product.WIRE_ROD if norm in ("wire_rod", "wire_rods", "wirerod") else Product.OTHER


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
                    product=parse_product(row.get("product")),
                    product_label=(row.get("product") or "").strip() or None,
                    layout=(row.get("layout") or "").strip() or None,
                    coil_count=int(row["coil_count"])
                    if (row.get("coil_count") or "").strip()
                    else None,
                    truck_ref=(row.get("truck_ref") or "").strip() or None,
                    notes=(row.get("notes") or "").strip() or None,
                    clip=(row.get("window") or "").strip().lower() == "clip",
                    stage=(row.get("stage") or "").strip() or None,
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
    """visit_id -> group id for leakage-safe splits. Rows share a group when they overlap
    in time on one camera (their frames show both trucks) or have the same ``truck_ref``
    (one truck seen in several recordings)."""
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
    by_truck: dict[str, str] = {}
    for v in visits:
        if v.truck_ref:
            first = by_truck.setdefault(v.truck_ref, v.visit_id)
            parent[find(v.visit_id)] = find(first)
    return {v.visit_id: find(v.visit_id) for v in visits}


def _in_scope(v: VisitLabel, camera_id: str, bay_id: str | None) -> bool:
    return v.camera_id == camera_id and (bay_id is None or v.bay_id == bay_id)


def frame_stage(
    t: datetime,
    camera_id: str,
    visits: list[VisitLabel],
    margin: timedelta,
    bay_id: str | None = None,
) -> Stage | None:
    """Stage label at time ``t``; None when within ``margin`` of a transition (ambiguous).

    With ``bay_id`` only that lane's visits count: an ingot truck in one lane and a
    wire-rod truck in the other at the same moment give two different labels.
    """
    for v in visits:
        if not _in_scope(v, camera_id, bay_id):
            continue
        if any(abs(t - tr) < margin for tr in v.transitions()):
            return None
        if v.contains(t):
            return v.stage_at(t)
    return Stage.NO_TRUCK


def frame_layout_label(
    t: datetime,
    camera_id: str,
    visits: list[VisitLabel],
    margin: timedelta,
    bay_id: str | None = None,
) -> str | None:
    """Layout class for the layout classifier, ``not_visible`` when no load is visible,
    None when unusable (no truck, unknown layout, or near a transition)."""
    stage = frame_stage(t, camera_id, visits, margin, bay_id)
    if stage is None or stage is Stage.NO_TRUCK:
        return None
    visit = next(v for v in visits if _in_scope(v, camera_id, bay_id) and v.contains(t))
    if stage is not Stage.WIRE_ROD:
        # Tarp on, empty bed or ingots: no wire-rod arrangement to classify.
        return NOT_VISIBLE
    # A clip without loading_start was recorded with the load already on the bed.
    loading = visit.loading_start or (None if visit.clip else visit.arrived)
    if loading is not None and t < loading + margin:
        return NOT_VISIBLE if t < loading - margin else None
    if visit.layout is None or visit.layout is Layout.UNKNOWN:
        return None
    return visit.layout.value


def write_visit_log_template(path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(VISIT_LOG_COLUMNS)


def visit_log_from_clips(
    records: list[FrameRecord], bay_id: str, path: str | Path, tz: str = "Asia/Kolkata"
) -> int:
    """Pre-fill one visit-log row per clip (short recordings, one truck each).

    Time columns come from the frames, so the log always matches the manifest; product,
    layout and coil_count are left for a person to fill in. Returns the number of rows.
    """
    zone = ZoneInfo(tz)
    spans: dict[str, tuple[datetime, datetime, str]] = {}
    for rec in records:
        if rec.captured_at is None:
            continue
        lo, hi, cam = spans.get(rec.video, (rec.captured_at, rec.captured_at, rec.camera_id))
        spans[rec.video] = (min(lo, rec.captured_at), max(hi, rec.captured_at), cam)
    pad = timedelta(seconds=1)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(VISIT_LOG_COLUMNS)
        for video, (lo, hi, cam) in sorted(spans.items(), key=lambda kv: kv[1][0]):
            name = Path(video.removeprefix("timelapse:")).stem
            w.writerow(
                [
                    name,
                    cam,
                    bay_id,
                    (lo - pad).astimezone(zone).isoformat(timespec="seconds"),
                    "",
                    "",
                    "",
                    (hi + pad).astimezone(zone).isoformat(timespec="seconds"),
                    "",  # product: wire_rod / ingot / empty / ...
                    "",  # layout: linear / criss_cross / mixed
                    "",  # coil_count (from the loading slip)
                    "",
                    f"clip {Path(video).name}",
                    "clip",
                    "",  # stage: fill if it is constant over the clip (e.g. covered)
                ]
            )
    return len(spans)
