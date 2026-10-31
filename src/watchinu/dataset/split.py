"""Leakage-safe, deterministic, stratified train/val/test splits.

The unit of splitting is a *visit group* (overlapping visits merged), never a frame:
frames from one visit are near-duplicates, and a frame-level split would let the test
set leak into training and inflate every metric. Frames outside any visit are grouped by
day. Visits listed in a frozen test file always stay in test across re-splits, so the
gold set never drifts as data grows.
"""

from __future__ import annotations

import random
from collections import defaultdict

from watchinu.dataset.manifest import FrameRecord
from watchinu.dataset.visits import VisitLabel, visit_groups

SPLITS = ("train", "val", "test")


def _group_key(rec: FrameRecord, groups: dict[str, str]) -> str:
    if rec.visit_id is not None:
        return f"visit:{groups.get(rec.visit_id, rec.visit_id)}"
    if rec.day is not None:
        return f"day:{rec.camera_id}:{rec.day}"
    return f"video:{rec.video}"


def assign_splits(
    records: list[FrameRecord],
    visits: list[VisitLabel],
    val_frac: float = 0.15,
    test_frac: float = 0.2,
    seed: int = 42,
    frozen_test: set[str] | None = None,
) -> dict[str, str]:
    """Assign ``split`` on every record. Returns group key -> split."""
    if not 0 <= val_frac < 1 or not 0 <= test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("invalid split fractions")
    groups = visit_groups(visits)
    layout_of = {v.visit_id: (v.layout.value if v.layout else "unknown") for v in visits}
    frozen = frozen_test or set()

    stratum_of: dict[str, str] = {}
    forced_test: set[str] = set()
    for rec in records:
        key = _group_key(rec, groups)
        if rec.visit_id is None:
            stratum_of.setdefault(key, "negative")
        else:
            stratum_of.setdefault(key, layout_of.get(rec.visit_id, "unknown"))
            if rec.visit_id in frozen:
                forced_test.add(key)

    by_stratum: dict[str, list[str]] = defaultdict(list)
    for key, stratum in stratum_of.items():
        if key not in forced_test:
            by_stratum[stratum].append(key)

    assignment = dict.fromkeys(forced_test, "test")
    rng = random.Random(seed)
    for stratum in sorted(by_stratum):
        keys = sorted(by_stratum[stratum])
        rng.shuffle(keys)
        n = len(keys)
        n_test = round(n * test_frac)
        n_val = round(n * val_frac)
        if n >= 3:  # every stratum with enough groups is represented in val and test
            n_test, n_val = max(1, n_test), max(1, n_val)
        for i, key in enumerate(keys):
            assignment[key] = "test" if i < n_test else "val" if i < n_test + n_val else "train"

    for rec in records:
        rec.split = assignment[_group_key(rec, groups)]
    return assignment


def split_summary(records: list[FrameRecord]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {s: {"frames": 0, "visits": 0} for s in SPLITS}
    visits: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        if rec.split in out:
            out[rec.split]["frames"] += 1
            if rec.visit_id:
                visits[rec.split].add(rec.visit_id)
    for s in SPLITS:
        out[s]["visits"] = len(visits[s])
    return out
