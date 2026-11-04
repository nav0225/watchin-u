"""Per-camera state shared by all bays: one detector pass and one look-back per frame.

Both lanes are visible in the same frame, so running the detector separately on each bay
crop would double the heavy compute and cut coils that straddle the lane boundary in
half. Instead the detector runs at most once per frame timestamp, on the full frame, and
every bay keeps the detections whose centre lies in its own polygon.
"""

from __future__ import annotations

from collections import OrderedDict, deque

import numpy as np

from watchinu.domain import Detection
from watchinu.models.base import Detector
from watchinu.vision.overlay import decode_jpeg, encode_jpeg


class SharedFrames:
    def __init__(
        self,
        detector: Detector | None,
        lookback_s: float,
        cache_size: int = 64,
        jpeg_quality: int = 90,
    ) -> None:
        self.detector = detector
        self.lookback_s = lookback_s
        self.jpeg_quality = jpeg_quality
        self._cache_size = cache_size
        self._detections: OrderedDict[float, list[Detection]] = OrderedDict()
        self._history: deque[tuple[float, bytes]] = deque()
        self.detector_calls = 0

    # -- look-back ------------------------------------------------------------------
    def remember(self, ts: float, image: np.ndarray) -> None:
        """Keep a (compressed) full frame for the look-back; idempotent per timestamp."""
        if self._history and self._history[-1][0] >= ts:
            return
        self._history.append((ts, encode_jpeg(image, self.jpeg_quality)))
        while self._history and ts - self._history[0][0] > self.lookback_s:
            self._history.popleft()

    def history(self, before_ts: float) -> list[float]:
        return [t for t, _ in self._history if t < before_ts]

    def image_at(self, ts: float) -> np.ndarray | None:
        for t, jpeg in self._history:
            if t == ts:
                return decode_jpeg(jpeg)
        return None

    # -- detections -----------------------------------------------------------------
    def detections(self, ts: float, image: np.ndarray) -> list[Detection]:
        """Full-frame detections for this timestamp, computed once for all bays."""
        if self.detector is None:
            return []
        cached = self._detections.get(ts)
        if cached is not None:
            self._detections.move_to_end(ts)
            return cached
        result = self.detector.detect(image)
        self.detector_calls += 1
        self._detections[ts] = result
        while len(self._detections) > self._cache_size:
            self._detections.popitem(last=False)
        return result
