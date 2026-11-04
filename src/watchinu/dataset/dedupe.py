"""Near-duplicate removal for a fixed camera.

A parked truck yields hundreds of almost identical frames. Labelling them wastes annotator
time and over-weights static scenes, but a perceptual hash of the whole frame is the wrong
tool here: on a fixed camera it barely moves while workers lace or a tarp is pulled (on
the 2026-11-04 footage a 64-bit dHash collapsed 365 of 393 frames, including clips with
constant activity).

Policy, per video in time order, compared with the last *kept* frame:
* keep if the scene changed (``region_change`` >= ``min_change``) and at least
  ``min_gap_s`` passed - captures activity without keeping every second of it;
* keep anyway after ``max_gap_s`` - even a static clip contributes evenly spaced frames.

OSD overlays (the ticking clock) are blanked first. Frames are only *marked*
(``duplicate_of``), never deleted, so the decision stays reversible.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from watchinu.dataset.manifest import FrameRecord
from watchinu.vision.geometry import blank_regions
from watchinu.vision.quality import region_change, thumbnail

PointSeq = Sequence[tuple[float, float]]


def _thumb(path: Path, masks: Sequence[PointSeq]) -> np.ndarray | None:
    # 1/4-scale JPEG decode is ~10x cheaper than a full decode and plenty for a thumbnail.
    image = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if image is None:
        return None
    return thumbnail(blank_regions(image, masks))


def mark_duplicates(
    records: list[FrameRecord],
    images_root: str | Path,
    min_change: float = 0.03,
    min_gap_s: float = 3.0,
    max_gap_s: float = 10.0,
    osd_masks: Sequence[PointSeq] = (),
) -> int:
    """Set ``duplicate_of`` on redundant frames; returns how many were marked."""
    root = Path(images_root)
    by_video: dict[str, list[FrameRecord]] = defaultdict(list)
    for rec in records:
        by_video[rec.video].append(rec)
    n_dup = 0
    for recs in by_video.values():
        recs.sort(key=lambda r: r.pts_s)
        kept: tuple[FrameRecord, np.ndarray] | None = None
        for rec in recs:
            thumb = _thumb(root / rec.image, osd_masks)
            if thumb is None:  # unreadable: never silently drop it as a "duplicate"
                rec.duplicate_of = None
                continue
            if kept is not None:
                gap = rec.pts_s - kept[0].pts_s
                changed = region_change(kept[1], thumb) >= min_change
                if gap < max_gap_s and not (changed and gap >= min_gap_s):
                    rec.duplicate_of = kept[0].frame_id
                    n_dup += 1
                    continue
            rec.duplicate_of = None
            kept = (rec, thumb)
    return n_dup
