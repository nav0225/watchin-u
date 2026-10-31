"""Sample frames from recorded footage into an image folder plus manifest records."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import cv2

from watchinu.dataset.manifest import FrameRecord
from watchinu.io.video import VideoFileSource
from watchinu.vision.hashing import dhash, to_hex
from watchinu.vision.quality import assess

log = logging.getLogger(__name__)


def extract_video(
    video: str | Path,
    out_dir: str | Path,
    every_s: float,
    camera_id: str,
    stream: str = "main",
    start_time: datetime | None = None,
    jpeg_quality: int = 95,
) -> list[FrameRecord]:
    """Write one JPEG every ``every_s`` seconds of footage; return their manifest records.

    Frames are saved full-size and un-cropped. Cropping to the bay happens at export
    time, with exactly the transform the runtime applies.
    """
    video = Path(video)
    out_dir = Path(out_dir)
    img_dir = out_dir / "images" / video.stem
    img_dir.mkdir(parents=True, exist_ok=True)
    source = VideoFileSource(video, every_s=every_s, start_time=start_time, stream=stream)
    records: list[FrameRecord] = []
    for frame in source:
        frame_id = f"{video.stem}_{round(frame.pts_s * 1000):09d}"
        path = img_dir / f"{frame_id}.jpg"
        if not cv2.imwrite(str(path), frame.image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
            raise OSError(f"failed to write {path}")
        q = assess(frame.image)
        h, w = frame.image.shape[:2]
        records.append(
            FrameRecord(
                frame_id=frame_id,
                image=str(path.relative_to(out_dir)),
                video=str(video),
                camera_id=camera_id,
                stream=stream,
                pts_s=round(frame.pts_s, 3),
                frame_index=frame.index,
                captured_at=(
                    datetime.fromtimestamp(frame.ts, tz=UTC) if source.start_time else None
                ),
                width=w,
                height=h,
                blur_var=round(q.blur_var, 2),
                brightness=round(q.brightness, 2),
                overexposed_fraction=round(q.overexposed_fraction, 4),
                dhash=to_hex(dhash(frame.image)),
            )
        )
    log.info("%s: %d frames", video.name, len(records))
    return records
