"""Replay recorded footage as a deterministic frame source (tests, offline eval, dataset)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2

from watchinu.domain import Frame

VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".ts", ".mov"}

_STAMP_RE = re.compile(r"(\d{8})T(\d{6})")


def parse_start_time(path: str | Path, tz: str = "Asia/Kolkata") -> datetime | None:
    """Start time from a ``..._YYYYMMDDTHHMMSS...`` filename (as written by ``capture``)."""
    match = _STAMP_RE.search(Path(path).name)
    if not match:
        return None
    naive = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
    return naive.replace(tzinfo=ZoneInfo(tz))


def list_videos(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_SUFFIXES)


class VideoFileSource:
    """Yields one frame every ``every_s`` seconds of video time.

    Skipped frames are only ``grab()``-ed (decoded, not converted), which is the cheapest
    exact way to step through H.264. ``ts`` = start time + presentation time, so replayed
    footage drives the visit state machine exactly like the live camera.
    """

    def __init__(
        self,
        path: str | Path,
        every_s: float = 1.0,
        start_time: datetime | None = None,
        stream: str = "main",
    ) -> None:
        self.path = Path(path)
        self.every_s = every_s
        self.start_time = start_time or parse_start_time(self.path)
        self.stream = stream

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise OSError(f"cannot open video {self.path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        base = self.start_time.timestamp() if self.start_time else 0.0
        next_t = 0.0
        index = -1
        try:
            while cap.grab():
                index += 1
                pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                # Some containers report no timestamps; fall back to index / fps.
                pts_s = index / fps if (pts_ms <= 0 < index and fps > 0) else pts_ms / 1000.0
                if pts_s + 1e-6 < next_t:
                    continue
                ok, image = cap.retrieve()
                if not ok or image is None:
                    continue
                next_t = pts_s + self.every_s
                yield Frame(
                    ts=base + pts_s,
                    image=image,
                    index=index,
                    pts_s=pts_s,
                    stream=self.stream,
                    source=str(self.path),
                )
        finally:
            cap.release()


class ImageSequenceSource:
    """Replays time-lapse JPEGs (``..._YYYYMMDDTHHMMSS.jpg``) in time order."""

    def __init__(self, paths: list[Path], tz: str = "Asia/Kolkata", stream: str = "main") -> None:
        timed = [(parse_start_time(p, tz), p) for p in paths]
        missing = [p.name for t, p in timed if t is None]
        if missing:
            raise ValueError(f"no timestamp in image name(s): {', '.join(missing[:5])}")
        self._items = sorted((t, p) for t, p in timed if t is not None)
        self.stream = stream

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Frame]:
        if not self._items:
            return
        t0 = self._items[0][0]
        for i, (t, path) in enumerate(self._items):
            image = cv2.imread(str(path))
            if image is None:
                continue
            yield Frame(
                ts=t.timestamp(),
                image=image,
                index=i,
                pts_s=(t - t0).total_seconds(),
                stream=self.stream,
                source=str(path),
            )
