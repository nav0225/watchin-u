"""Sample frames from recorded footage into an image folder plus manifest records."""

from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

import cv2

from watchinu.dataset.manifest import FrameRecord
from watchinu.io.video import VideoFileSource, parse_start_time
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


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_images(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p] if p.suffix.lower() in IMAGE_SUFFIXES else []
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES)


def index_images(
    images: list[Path],
    out_dir: str | Path,
    camera_id: str,
    stream: str = "main",
    tz: str = "Asia/Kolkata",
) -> list[FrameRecord]:
    """Add time-lapse JPEGs (``..._YYYYMMDDTHHMMSS.jpg``) to the dataset.

    Files are copied byte-for-byte (no re-encoding). Each day's sequence is grouped under
    one ``video`` key with ``pts_s`` = seconds since local midnight, so de-duplication
    and visit assignment work exactly as for extracted video frames.
    """
    out_dir = Path(out_dir)
    records: list[FrameRecord] = []
    for i, path in enumerate(sorted(images)):
        taken = parse_start_time(path, tz)
        image = cv2.imread(str(path))
        if image is None:
            log.warning("skip unreadable image %s", path)
            continue
        day = taken.strftime("%Y%m%d") if taken else "undated"
        group = f"{camera_id}_{stream}_{day}"
        dst = out_dir / "images" / group / path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        midnight = taken.replace(hour=0, minute=0, second=0, microsecond=0) if taken else None
        q = assess(image)
        h, w = image.shape[:2]
        records.append(
            FrameRecord(
                frame_id=path.stem,
                image=str(dst.relative_to(out_dir)),
                video=f"timelapse:{group}",
                camera_id=camera_id,
                stream=stream,
                pts_s=(taken - midnight).total_seconds() if taken and midnight else float(i),
                frame_index=i,
                captured_at=taken.astimezone(UTC) if taken else None,
                width=w,
                height=h,
                blur_var=round(q.blur_var, 2),
                brightness=round(q.brightness, 2),
                overexposed_fraction=round(q.overexposed_fraction, 4),
                dhash=to_hex(dhash(image)),
                source_image=str(path),
            )
        )
    log.info("indexed %d time-lapse images", len(records))
    return records
