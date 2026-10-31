"""Cheap per-frame quality metrics and the change score used for duty-cycling."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from watchinu.config import QualityConfig

THUMB_SIZE = (96, 54)  # 16:9, matches both camera streams


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


@dataclass(frozen=True, slots=True)
class FrameQuality:
    blur_var: float
    brightness: float
    overexposed_fraction: float

    def problems(self, cfg: QualityConfig) -> list[str]:
        out: list[str] = []
        if self.blur_var < cfg.min_blur_var:
            out.append("blurred")
        if self.brightness < cfg.min_brightness:
            out.append("too_dark")
        if self.overexposed_fraction > cfg.max_overexposed_fraction:
            out.append("overexposed")
        return out

    def passes(self, cfg: QualityConfig) -> bool:
        return not self.problems(cfg)


def assess(image: np.ndarray, mask: np.ndarray | None = None) -> FrameQuality:
    """Blur (variance of Laplacian), mean brightness and blown-highlight fraction.

    ``mask`` (uint8, 255 = keep) restricts the statistics to e.g. the bay polygon, so
    blacked-out pixels outside the bay do not drag the numbers down.
    """
    gray = to_gray(image)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    if mask is not None and mask.any():
        sel = mask > 0
        # Erode so the artificial polygon edge does not count as sharp detail.
        inner = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
        lap_vals = lap[inner] if inner.any() else lap[sel]
        vals = gray[sel]
    else:
        lap_vals = lap.ravel()
        vals = gray.ravel()
    return FrameQuality(
        blur_var=float(lap_vals.var()) if lap_vals.size else 0.0,
        brightness=float(vals.mean()) if vals.size else 0.0,
        overexposed_fraction=float((vals >= 250).mean()) if vals.size else 0.0,
    )


def thumbnail(image: np.ndarray) -> np.ndarray:
    """Small blurred grayscale thumbnail used for change detection."""
    small = cv2.resize(to_gray(image), THUMB_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (3, 3), 0)


def change_score(prev_thumb: np.ndarray | None, cur_thumb: np.ndarray) -> float:
    """Normalised mean absolute difference in 0..1 (0 = identical)."""
    if prev_thumb is None:
        return 0.0
    diff = cv2.absdiff(prev_thumb, cur_thumb)
    return float(diff.mean()) / 255.0
