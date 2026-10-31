"""Thin wrapper over Ultralytics training that records provenance with every run.

Each run directory gets ``provenance.json``: git commit, dataset card (with the manifest
digest), arguments and library version — so any deployed weight file can be traced
back to the exact code and data that produced it.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def dataset_card_for(data: str | Path) -> dict[str, Any] | None:
    p = Path(data)
    card = (p.parent if p.is_file() else p) / "dataset_card.json"
    return json.loads(card.read_text()) if card.exists() else None


def train(
    data: str | Path,
    model: str,
    project: str | Path = "runs",
    name: str | None = None,
    epochs: int = 100,
    imgsz: int = 1280,
    batch: int = 8,
    device: str = "0",
    extra: dict[str, Any] | None = None,
) -> Path:
    try:
        from ultralytics import YOLO
        from ultralytics import __version__ as ultralytics_version
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("ultralytics is not installed: pip install -e '.[ml]'") from exc

    args: dict[str, Any] = {
        "data": str(data),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": str(project),
        "name": name,
        "seed": 42,
        "deterministic": True,
        **(extra or {}),
    }
    yolo = YOLO(model)
    yolo.train(**args)
    run_dir = Path(yolo.trainer.save_dir)
    provenance = {
        "created_at": datetime.now(UTC).isoformat(),
        "base_model": model,
        "args": args,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "dataset_card": dataset_card_for(data),
        "ultralytics_version": ultralytics_version,
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    return run_dir
