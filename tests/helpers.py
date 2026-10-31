from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from watchinu.config import AppConfig

BASE_CONFIG = {
    "camera": {
        "id": "cam01",
        "host": "10.0.0.1",
        "credentials_env": {"user": "TEST_CAM_USER", "password": "TEST_CAM_PASSWORD"},
    },
    "slot_templates": {"std_2x5": {"columns": 2, "rows": 5}},
    "bays": [
        {
            "id": "left",
            "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            # near-left, near-right, far-right, far-left (far end at the top of the image)
            "bed_quad": [[0.1, 0.95], [0.9, 0.95], [0.9, 0.05], [0.1, 0.05]],
            "slot_template": "std_2x5",
        }
    ],
}


def make_config(**overrides: object) -> AppConfig:
    raw = {**BASE_CONFIG, **overrides}
    return AppConfig.model_validate(raw)


def write_video(path: Path, frames: list[np.ndarray], fps: float = 10.0) -> Path:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    assert writer.isOpened()
    for f in frames:
        writer.write(f)
    writer.release()
    return path


def textured(value: int, shape: tuple[int, int] = (180, 320), amp: int = 30) -> np.ndarray:
    """Checkerboard around ``value`` so the mean is ``value`` and the frame is sharp."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((yy // 8) + (xx // 8)) % 2) * 2 - 1
    img = np.clip(value + amp * checker, 0, 255).astype(np.uint8)
    return cv2.merge([img, img, img])
