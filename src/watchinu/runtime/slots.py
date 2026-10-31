"""Slot-based verification of a linear (template) load.

Coil box centres are projected onto the bed plane (unit square) and binned into the
template grid, e.g. 2 columns x 5 rows. Each slot then accumulates evidence over time:

* OCCUPIED - a coil was seen in it (majority of the recent observations);
* EMPTY    - the slot was visible and repeatedly had no coil;
* UNSEEN   - never visible (hidden by workers, by nearer coils, or at the far end).

Observed and inferred counts are reported separately; an UNSEEN slot is only counted
as occupied when inference is explicitly enabled, and the basis is recorded.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from enum import StrEnum

import numpy as np

from watchinu.config import SlotTemplate
from watchinu.domain import BBox, CountSummary
from watchinu.vision.geometry import BedHomography

Slot = tuple[int, int]  # (column, row); row 0 = near end


class SlotStatus(StrEnum):
    OCCUPIED = "occupied"
    EMPTY = "empty"
    UNSEEN = "unseen"


def all_slots(template: SlotTemplate) -> list[Slot]:
    return [(c, r) for r in range(template.rows) for c in range(template.columns)]


def slot_of(u: float, v: float, template: SlotTemplate, margin: float = 0.15) -> Slot | None:
    """Grid cell for a bed-plane point, or None if it is clearly off the bed."""
    if not (-margin <= u <= 1 + margin and -margin <= v <= 1 + margin):
        return None
    col = min(template.columns - 1, max(0, int(u * template.columns)))
    row = min(template.rows - 1, max(0, int(v * template.rows)))
    return col, row


def slot_centres_uv(template: SlotTemplate) -> dict[Slot, tuple[float, float]]:
    return {
        (c, r): ((c + 0.5) / template.columns, (r + 0.5) / template.rows)
        for c, r in all_slots(template)
    }


def observe_slots(
    coils: list[BBox],
    persons: list[BBox],
    homography: BedHomography,
    template: SlotTemplate,
) -> tuple[set[Slot], set[Slot]]:
    """Return (occupied, visible) slots for one frame.

    A slot without a coil counts as *visible* only if its image-space centre is not
    covered by a worker or by a coil that belongs to a different (nearer) slot.
    """
    assigned: list[tuple[BBox, Slot]] = []
    if coils:
        uv = homography.to_bed(np.array([b.center for b in coils], dtype=np.float32))
        for box, (u, v) in zip(coils, uv, strict=True):
            slot = slot_of(float(u), float(v), template)
            if slot is not None:
                assigned.append((box, slot))
    occupied = {slot for _, slot in assigned}

    centres = slot_centres_uv(template)
    order = list(centres)
    img_pts = homography.to_image(np.array([centres[s] for s in order], dtype=np.float32))
    visible = set(occupied)
    for slot, (x, y) in zip(order, img_pts, strict=True):
        if slot in occupied:
            continue
        if any(p.contains(float(x), float(y)) for p in persons):
            continue
        if any(b.contains(float(x), float(y)) for b, s in assigned if s != slot):
            continue
        visible.add(slot)
    return occupied, visible


class SlotBoard:
    def __init__(self, template: SlotTemplate, history: int = 3) -> None:
        self.template = template
        self._obs: dict[Slot, deque[bool]] = {s: deque(maxlen=history) for s in all_slots(template)}

    def update(self, occupied: Iterable[Slot], visible: Iterable[Slot]) -> None:
        occ = set(occupied)
        for slot in set(visible) | occ:
            if slot in self._obs:
                self._obs[slot].append(slot in occ)

    def status(self, slot: Slot) -> SlotStatus:
        obs = self._obs[slot]
        if not obs:
            return SlotStatus.UNSEEN
        hits = sum(obs)
        misses = len(obs) - hits
        if hits == misses:  # tie: trust the latest observation
            return SlotStatus.OCCUPIED if obs[-1] else SlotStatus.EMPTY
        return SlotStatus.OCCUPIED if hits > misses else SlotStatus.EMPTY

    def statuses(self) -> dict[Slot, SlotStatus]:
        return {s: self.status(s) for s in self._obs}

    def summary(self, infer_unseen: bool, template_name: str) -> CountSummary:
        st = self.statuses().values()
        observed = sum(1 for s in st if s is SlotStatus.OCCUPIED)
        empty = sum(1 for s in st if s is SlotStatus.EMPTY)
        unseen = sum(1 for s in st if s is SlotStatus.UNSEEN)
        if infer_unseen and observed > 0:
            inferred = observed + unseen
            basis = (
                f"{observed} observed + {unseen} unseen slot(s) assumed occupied "
                f"(template {template_name}, {empty} seen empty)"
            )
        else:
            inferred = None
            basis = f"{observed} observed, {unseen} unseen, {empty} empty (no inference)"
        return CountSummary(observed, empty, unseen, inferred, basis)
