"""Near-duplicate removal: a parked truck produces hundreds of nearly identical frames.

Labelling those wastes annotator time and over-weights static scenes, so frames whose
hash is within ``max_distance`` bits of a recently kept frame from the same video are
marked ``duplicate_of`` (never deleted - the decision stays reversible).
"""

from __future__ import annotations

from collections import defaultdict, deque

from watchinu.dataset.manifest import FrameRecord
from watchinu.vision.hashing import from_hex, hamming


def mark_duplicates(records: list[FrameRecord], max_distance: int = 6, window: int = 30) -> int:
    by_video: dict[str, list[FrameRecord]] = defaultdict(list)
    for rec in records:
        by_video[rec.video].append(rec)
    n_dup = 0
    for recs in by_video.values():
        recs.sort(key=lambda r: r.pts_s)
        kept: deque[tuple[str, int]] = deque(maxlen=window)
        for rec in recs:
            h = from_hex(rec.dhash)
            match = next((fid for fid, kh in kept if hamming(h, kh) <= max_distance), None)
            if match is None:
                rec.duplicate_of = None
                kept.append((rec.frame_id, h))
            else:
                rec.duplicate_of = match
                n_dup += 1
    return n_dup
