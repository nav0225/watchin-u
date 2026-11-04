"""Deterministic stand-in models for tests and pipeline dry runs."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import numpy as np

from watchinu.dataset.visits import VisitLabel, frame_stage
from watchinu.domain import Detection, Stage


class FunctionClassifier:
    """Classifier whose output is a pure function of the image (e.g. a colour code)."""

    def __init__(self, fn: Callable[[np.ndarray], dict[str, float]], name: str = "fake-cls"):
        self._fn = fn
        self.name = name
        self.version = "test"

    def classify(self, image: np.ndarray) -> dict[str, float]:
        return self._fn(image)


class FunctionDetector:
    def __init__(self, fn: Callable[[np.ndarray], list[Detection]], name: str = "fake-det"):
        self._fn = fn
        self.name = name
        self.version = "test"

    def detect(self, image: np.ndarray) -> list[Detection]:
        return self._fn(image)


class VisitLogStageOracle:
    """Stage "classifier" that reads the answer from the visit log.

    Replaying with the oracle isolates the heavy tier: any remaining error is in the
    detector / layout classifier, not in visit detection. Call :meth:`set_time` with
    each frame's timestamp before processing it.
    """

    name = "visit-log-oracle"
    version = "oracle"

    def __init__(self, visits: list[VisitLabel], camera_id: str, bay_id: str | None = None) -> None:
        self._visits = visits
        self._camera_id = camera_id
        self._bay_id = bay_id
        self._now: datetime | None = None

    def set_time(self, ts: float) -> None:
        self._now = datetime.fromtimestamp(ts, tz=UTC)

    def classify(self, image: np.ndarray) -> dict[str, float]:
        if self._now is None:
            return {Stage.UNKNOWN.value: 1.0}
        stage = frame_stage(self._now, self._camera_id, self._visits, timedelta(0), self._bay_id)
        return {(stage or Stage.UNKNOWN).value: 1.0}
