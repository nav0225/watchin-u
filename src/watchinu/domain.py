"""Core domain types shared by the runtime, dataset tooling and storage layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class Stage(StrEnum):
    """Coarse visit stage, predicted per frame by the light (Tier-1) classifier.

    Loading and lacing are deliberately one class (``OPEN``): both show an open bed and
    are hard to tell apart from a single frame. Activity is measured separately with the
    change score.
    """

    NO_TRUCK = "no_truck"
    OPEN = "open"
    COVERING = "covering"
    COVERED = "covered"
    UNKNOWN = "unknown"

    @property
    def truck_present(self) -> bool:
        return self in (Stage.OPEN, Stage.COVERING, Stage.COVERED)

    @property
    def tarp_on(self) -> bool:
        return self in (Stage.COVERING, Stage.COVERED)


class Layout(StrEnum):
    """Coil arrangement on the truck bed."""

    LINEAR = "linear"
    CRISS_CROSS = "criss_cross"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned box in pixel coordinates (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def shifted(self, dx: float, dy: float) -> BBox:
        return BBox(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def intersection(self, other: BBox) -> float:
        w = min(self.x2, other.x2) - max(self.x1, other.x1)
        h = min(self.y2, other.y2) - max(self.y1, other.y1)
        return max(0.0, w) * max(0.0, h)

    def iou(self, other: BBox) -> float:
        inter = self.intersection(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    box: BBox


@dataclass(slots=True)
class Frame:
    """A decoded frame plus timing. ``ts`` is wall-clock epoch seconds (UTC)."""

    ts: float
    image: np.ndarray
    index: int = 0
    pts_s: float = 0.0
    stream: str = "main"
    source: str = ""


@dataclass(slots=True)
class CountSummary:
    """Coil count for one visit. Observed and inferred counts are kept apart on purpose."""

    observed: int
    empty_slots: int
    unseen_slots: int
    inferred_total: int | None
    basis: str


@dataclass(slots=True)
class Decision:
    """A provisional or final arrangement decision for one visit."""

    kind: str  # "provisional" | "final"
    ts: float
    layout: Layout
    layout_confidence: float
    layout_agreement: float
    n_layout_votes: int
    counts: CountSummary
    needs_review: bool
    reasons: list[str] = field(default_factory=list)
    model_versions: dict[str, str] = field(default_factory=dict)
