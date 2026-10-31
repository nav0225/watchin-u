"""Frame manifest: one JSON line per extracted frame, the backbone of the dataset.

Every frame keeps its provenance (source video, presentation time, wall-clock time),
its quality metrics and hash, and later its visit and split. Images are referenced by
paths relative to the manifest's directory, so a dataset folder can be moved as a unit.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class FrameRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: str
    image: str
    video: str
    camera_id: str
    stream: str
    pts_s: float
    frame_index: int
    captured_at: datetime | None
    width: int
    height: int
    blur_var: float
    brightness: float
    overexposed_fraction: float
    dhash: str
    duplicate_of: str | None = None
    visit_id: str | None = None
    split: str | None = None

    @property
    def day(self) -> str | None:
        return self.captured_at.date().isoformat() if self.captured_at else None


def read_manifest(path: str | Path) -> list[FrameRecord]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        return [FrameRecord.model_validate_json(line) for line in fh if line.strip()]


def write_manifest(path: str | Path, records: Iterable[FrameRecord]) -> None:
    """Atomic write: a crash mid-write never leaves a truncated manifest behind."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(rec.model_dump_json() + "\n")
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
