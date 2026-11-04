"""Label audit: coils per lane per recording, against the visit log.

Run it on pre-labels to judge how much correcting they need, and on corrected CVAT
labels before training: a wire-rod lane with far fewer coils than its ``coil_count``, an
ingot lane with coils, or a covered truck with coils all point at labelling mistakes
(or at a visit-log error), clip by clip.
"""

from __future__ import annotations

import statistics
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from watchinu.config import AppConfig
from watchinu.dataset.manifest import FrameRecord
from watchinu.dataset.visits import VisitLabel
from watchinu.domain import BBox
from watchinu.vision.geometry import point_in_polygon, to_pixels

YoloLabels = dict[str, list[tuple[int, float, float, float, float]]]


def read_yolo_labels_any(path: str | Path) -> YoloLabels:
    """Stem -> boxes, from a folder of ``.txt`` files or a CVAT "YOLO 1.1" zip."""
    p = Path(path)
    texts: dict[str, str] = {}
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as zf:
            for name in zf.namelist():
                if name.endswith(".txt") and "obj_train_data/" in name:
                    texts[Path(name).stem] = zf.read(name).decode("utf-8")
    else:
        texts = {f.stem: f.read_text(encoding="utf-8") for f in p.rglob("*.txt")}
    out: YoloLabels = {}
    for stem, text in texts.items():
        rows = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 5:
                cx, cy, w, h = (float(v) for v in parts[1:])
                rows.append((int(parts[0]), cx, cy, w, h))
        out[stem] = rows
    return out


@dataclass(frozen=True, slots=True)
class LaneClipStats:
    clip: str
    lane: str
    truck: str | None
    product: str | None
    stage: str | None
    truth_count: int | None
    frames: int
    coils_median: float
    coils_min: int
    coils_max: int

    @property
    def flag(self) -> str:
        """A short hint where labels and the visit log disagree."""
        visible = self.stage in (None, "wire_rod")
        if self.product is None and self.coils_max > 0:
            return "coils in an empty lane"
        if self.product not in (None, "wire_rod") and self.coils_max > 0:
            return "coils on a non-wire-rod truck"
        if self.product == "wire_rod" and not visible and self.coils_max > 2:
            return "coils on a covered truck"
        if (
            self.product == "wire_rod"
            and visible
            and self.truth_count
            and self.coils_max > self.truth_count + 1
        ):
            return "more coils than coil_count (duplicates?)"
        return ""


def audit_labels(
    records: list[FrameRecord],
    labels: YoloLabels,
    visits: list[VisitLabel],
    cfg: AppConfig,
    coil_class: int = 0,
) -> list[LaneClipStats]:
    per: dict[tuple[str, str], list[int]] = defaultdict(list)
    first_t: dict[tuple[str, str], datetime] = {}
    for rec in records:
        stem = Path(rec.image).stem
        if stem not in labels or rec.captured_at is None:
            continue
        coils = []
        for cls, cx, cy, w, h in labels[stem]:
            if cls == coil_class:
                coils.append(
                    BBox(
                        (cx - w / 2) * rec.width,
                        (cy - h / 2) * rec.height,
                        (cx + w / 2) * rec.width,
                        (cy + h / 2) * rec.height,
                    )
                )
        clip = Path(rec.video).stem
        for bay in cfg.bays:
            poly = to_pixels(bay.polygon, rec.width, rec.height)
            n = sum(point_in_polygon(*b.center, poly) for b in coils)
            per[(clip, bay.id)].append(n)
            first_t.setdefault((clip, bay.id), rec.captured_at)

    out: list[LaneClipStats] = []
    for (clip, lane), counts in sorted(per.items()):
        t = first_t[(clip, lane)] + timedelta(seconds=1)
        truth = next((v for v in visits if v.bay_id == lane and v.contains(t)), None)
        out.append(
            LaneClipStats(
                clip=clip,
                lane=lane,
                truck=truth.truck_ref if truth else None,
                product=truth.product.value if truth and truth.product else None,
                stage=truth.stage.value if truth and truth.stage else None,
                truth_count=truth.coil_count if truth else None,
                frames=len(counts),
                coils_median=float(statistics.median(counts)),
                coils_min=min(counts),
                coils_max=max(counts),
            )
        )
    return out


def to_markdown(stats: list[LaneClipStats]) -> str:
    lines = [
        "| clip | lane | truck | product | stage | coil_count | frames "
        "| coils/frame median [min-max] | check |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in stats:
        lines.append(
            f"| {s.clip} | {s.lane} | {s.truck or ''} | {s.product or 'empty'} | {s.stage or ''} | "
            f"{s.truth_count if s.truth_count is not None else ''} | {s.frames} | "
            f"{s.coils_median:g} [{s.coils_min}-{s.coils_max}] | {s.flag} |"
        )
    return "\n".join(lines) + "\n"
