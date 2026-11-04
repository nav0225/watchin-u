"""Slot-based verification of a templated load (linear 2x5, criss-cross 2x7, ...).

Coil box centres are projected onto the bed plane (unit square) and binned into the
template grid. Each slot then accumulates evidence over time:

* OCCUPIED    - a coil was seen in it (majority of the recent observations);
* EMPTY       - the slot was visible and repeatedly had no coil;
* UNSEEN      - inside the camera view but never visible (workers, nearer coils);
* OUT_OF_VIEW - its centre lies outside the camera view (load cut off by the image edge).

Observed and inferred counts are reported separately; UNSEEN / OUT_OF_VIEW slots are only
counted as occupied when inference is explicitly enabled, and the basis is recorded.
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
    OUT_OF_VIEW = "out_of_view"


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


def slot_image_points(
    homography: BedHomography, template: SlotTemplate
) -> dict[Slot, tuple[float, float]]:
    """Slot centres in image coordinates."""
    centres = slot_centres_uv(template)
    pts = homography.to_image(np.array(list(centres.values()), dtype=np.float32))
    return {s: (float(x), float(y)) for s, (x, y) in zip(centres, pts, strict=True)}


def _in_view(x: float, y: float, view_mask: np.ndarray | None) -> bool:
    if view_mask is None:
        return True
    h, w = view_mask.shape[:2]
    xi, yi = int(x), int(y)
    return 0 <= xi < w and 0 <= yi < h and bool(view_mask[yi, xi])


def out_of_view_slots(
    points: dict[Slot, tuple[float, float]], view_mask: np.ndarray | None
) -> set[Slot]:
    return {s for s, (x, y) in points.items() if not _in_view(x, y, view_mask)}


def observe_slots(
    coils: list[BBox],
    persons: list[BBox],
    homography: BedHomography,
    template: SlotTemplate,
    view_mask: np.ndarray | None = None,
) -> tuple[set[Slot], set[Slot]]:
    """Return (occupied, visible) slots for one frame.

    A slot without a coil counts as *visible* (i.e. seen empty) only if its image-space
    centre is inside the camera view / bay polygon and not covered by a worker or by a
    coil that belongs to a different (nearer) slot.
    """
    assigned: list[tuple[BBox, Slot]] = []
    if coils:
        uv = homography.to_bed(np.array([b.center for b in coils], dtype=np.float32))
        for box, (u, v) in zip(coils, uv, strict=True):
            slot = slot_of(float(u), float(v), template)
            if slot is not None:
                assigned.append((box, slot))
    occupied = {slot for _, slot in assigned}

    visible = set(occupied)
    for slot, (x, y) in slot_image_points(homography, template).items():
        if slot in occupied or not _in_view(x, y, view_mask):
            continue
        if any(p.contains(x, y) for p in persons):
            continue
        if any(b.contains(x, y) for b, s in assigned if s != slot):
            continue
        visible.add(slot)
    return occupied, visible


class SlotBoard:
    def __init__(
        self,
        template: SlotTemplate,
        history: int = 3,
        out_of_view: set[Slot] | None = None,
    ) -> None:
        self.template = template
        self.out_of_view = set(out_of_view or ())
        self._obs: dict[Slot, deque[bool]] = {s: deque(maxlen=history) for s in all_slots(template)}

    def update(self, occupied: Iterable[Slot], visible: Iterable[Slot]) -> None:
        occ = set(occupied)
        for slot in set(visible) | occ:
            if slot in self._obs:
                self._obs[slot].append(slot in occ)

    def status(self, slot: Slot) -> SlotStatus:
        obs = self._obs[slot]
        if not obs:
            return SlotStatus.OUT_OF_VIEW if slot in self.out_of_view else SlotStatus.UNSEEN
        hits = sum(obs)
        misses = len(obs) - hits
        if hits == misses:  # tie: trust the latest observation
            return SlotStatus.OCCUPIED if obs[-1] else SlotStatus.EMPTY
        return SlotStatus.OCCUPIED if hits > misses else SlotStatus.EMPTY

    def statuses(self) -> dict[Slot, SlotStatus]:
        return {s: self.status(s) for s in self._obs}

    def summary(self, infer_unseen: bool, template_name: str) -> CountSummary:
        st = list(self.statuses().values())
        observed = st.count(SlotStatus.OCCUPIED)
        empty = st.count(SlotStatus.EMPTY)
        unseen = st.count(SlotStatus.UNSEEN)
        oov = st.count(SlotStatus.OUT_OF_VIEW)
        if infer_unseen and observed > 0:
            inferred: int | None = observed + unseen + oov
            basis = (
                f"{observed} observed + {unseen} hidden + {oov} outside camera view, "
                f"assumed occupied (template {template_name}, {empty} seen empty)"
            )
        else:
            inferred = None
            basis = (
                f"{observed} observed, {unseen} hidden, {oov} outside camera view, "
                f"{empty} empty (no inference)"
            )
        return CountSummary(observed, empty, unseen, inferred, basis, oov, template_name)
