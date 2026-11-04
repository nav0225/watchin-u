"""Contact sheets: a few thumbnails per clip on one image, for fast human review.

Useful to fill the visit log (what does each clip show?) and to share a whole recording
session as a handful of images instead of gigabytes of video.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from watchinu.dataset.manifest import FrameRecord


def build_contact_sheets(
    records: list[FrameRecord],
    images_root: str | Path,
    out_dir: str | Path,
    per_clip: int = 4,
    clips_per_sheet: int = 8,
    thumb_width: int = 480,
) -> list[Path]:
    images_root, out_dir = Path(images_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_clip: dict[str, list[FrameRecord]] = defaultdict(list)
    for rec in records:
        by_clip[rec.video].append(rec)
    clips = sorted(by_clip, key=lambda v: (by_clip[v][0].captured_at is None, v))
    thumb_h = thumb_width * 9 // 16
    label_h = 36
    sheets: list[Path] = []
    for start in range(0, len(clips), clips_per_sheet):
        rows = []
        for video in clips[start : start + clips_per_sheet]:
            recs = sorted(by_clip[video], key=lambda r: r.pts_s)
            picks = [
                recs[round(i * (len(recs) - 1) / max(1, per_clip - 1))] for i in range(per_clip)
            ]
            tiles = []
            for rec in picks:
                img = cv2.imread(str(images_root / rec.image))
                tile = (
                    cv2.resize(img, (thumb_width, thumb_h), interpolation=cv2.INTER_AREA)
                    if img is not None
                    else np.zeros((thumb_h, thumb_width, 3), np.uint8)
                )
                cv2.putText(
                    tile,
                    f"t={rec.pts_s:.0f}s",
                    (8, thumb_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                tiles.append(tile)
            label = np.full((label_h, thumb_width * per_clip, 3), 32, np.uint8)
            when = recs[0].captured_at.isoformat(timespec="seconds") if recs[0].captured_at else ""
            text = f"{Path(video.removeprefix('timelapse:')).name}   {when}   frames={len(recs)}"
            cv2.putText(label, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            rows.append(np.vstack([label, np.hstack(tiles)]))
        sheet = np.vstack(rows)
        path = out_dir / f"contact_sheet_{start // clips_per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        sheets.append(path)
    return sheets
