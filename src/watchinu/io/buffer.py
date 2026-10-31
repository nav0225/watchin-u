"""Single-slot "latest frame wins" buffer between the camera reader and the consumer.

A slow consumer never builds a queue: older frames are overwritten and counted as drops,
so processing latency stays bounded no matter how long inference takes.
"""

from __future__ import annotations

import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class LatestBuffer(Generic[T]):
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: T | None = None
        self._seq = 0
        self._dropped = 0
        self._closed = False

    def put(self, item: T) -> None:
        with self._cond:
            if self._item is not None:
                self._dropped += 1
            self._item = item
            self._seq += 1
            self._cond.notify_all()

    def get_newer(self, last_seq: int, timeout: float | None = None) -> tuple[int, T] | None:
        """Wait for an item newer than ``last_seq``. Returns None on timeout or close."""
        with self._cond:
            if not self._cond.wait_for(
                lambda: self._closed or (self._seq > last_seq and self._item is not None),
                timeout=timeout,
            ):
                return None
            if self._item is None or self._seq <= last_seq:
                return None
            item = self._item
            self._item = None
            return self._seq, item

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def dropped(self) -> int:
        with self._cond:
            return self._dropped
