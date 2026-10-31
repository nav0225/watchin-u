"""Live RTSP ingest from the Hikvision camera with reconnect and stall detection."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from urllib.parse import quote

import cv2

from watchinu.domain import Frame
from watchinu.io.buffer import LatestBuffer

log = logging.getLogger(__name__)

# Force RTSP over TCP: UDP drops packets on a busy plant network and smears frames.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def build_rtsp_url(host: str, port: int, path: str, user: str, password: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return f"rtsp://{quote(user, safe='')}:{quote(password, safe='')}@{host}:{port}{path}"


_CRED_RE = re.compile(r"(rtsp://[^:/@]+:)[^@]*(@)")


def redact(url: str) -> str:
    """Hide the password so URLs can be logged safely."""
    return _CRED_RE.sub(r"\1***\2", url)


@dataclass(slots=True)
class StreamStats:
    frames: int = 0
    reconnects: int = 0
    last_frame_wall: float = 0.0


class RtspFrameSource:
    """Background reader thread -> :class:`LatestBuffer`. Iterate to consume frames.

    Reconnects with exponential backoff (capped) when the stream errors or stalls.
    """

    def __init__(
        self,
        url: str,
        stream: str = "main",
        stall_timeout_s: float = 10.0,
        max_backoff_s: float = 60.0,
    ) -> None:
        self._url = url
        self._stream = stream
        self._stall_timeout_s = stall_timeout_s
        self._max_backoff_s = max_backoff_s
        self._buffer: LatestBuffer[Frame] = LatestBuffer()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stats = StreamStats()

    def __enter__(self) -> RtspFrameSource:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"rtsp-{self._stream}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._buffer.close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        backoff = 1.0
        index = 0
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                log.warning("cannot open %s; retry in %.0fs", redact(self._url), backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self._max_backoff_s)
                continue
            log.info("connected to %s", redact(self._url))
            connected_at = time.monotonic()
            last_ok = time.monotonic()
            while not self._stop.is_set():
                ok, image = cap.read()
                now = time.monotonic()
                if not ok or image is None:
                    if now - last_ok > self._stall_timeout_s:
                        log.warning("stream stalled for %.0fs; reconnecting", now - last_ok)
                        break
                    time.sleep(0.05)
                    continue
                last_ok = now
                wall = time.time()
                self.stats.frames += 1
                self.stats.last_frame_wall = wall
                self._buffer.put(Frame(ts=wall, image=image, index=index, stream=self._stream))
                index += 1
            cap.release()
            if self._stop.is_set():
                break
            self.stats.reconnects += 1
            # A connection that lived a while resets the backoff.
            backoff = (
                1.0
                if time.monotonic() - connected_at > 60
                else min(backoff * 2, self._max_backoff_s)
            )
            self._stop.wait(backoff)

    def frames(self, timeout: float = 5.0) -> Iterator[Frame | None]:
        """Yield the newest frame each time one is available; ``None`` on timeout."""
        last = 0
        while not self._stop.is_set():
            got = self._buffer.get_newer(last, timeout=timeout)
            if got is None:
                yield None
                continue
            last, frame = got
            yield frame

    @property
    def dropped(self) -> int:
        return self._buffer.dropped
