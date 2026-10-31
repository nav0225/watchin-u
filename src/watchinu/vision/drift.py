"""View-drift guard: detects that the camera view no longer matches the calibration.

PTZ control is locked, but a power cycle, firmware update or someone re-enabling PTZ can
still move the view. Every polygon in the config would then be wrong, so the pipeline
suspends decisions and raises an alert instead of producing confident garbage.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from watchinu.vision.quality import to_gray


@dataclass(frozen=True, slots=True)
class DriftResult:
    ok: bool
    inlier_ratio: float
    shift_px: float
    reason: str = ""


class ViewDriftChecker:
    def __init__(
        self,
        reference: np.ndarray,
        static_mask: np.ndarray | None = None,
        min_inlier_ratio: float = 0.3,
        max_shift_px: float = 25.0,
        n_features: int = 1500,
    ) -> None:
        self._orb = cv2.ORB.create(nfeatures=n_features)
        self._mask = static_mask
        self._min_inlier_ratio = min_inlier_ratio
        self._max_shift_px = max_shift_px
        self._ref_kp, self._ref_desc = self._orb.detectAndCompute(to_gray(reference), static_mask)
        if self._ref_desc is None or len(self._ref_kp) < 20:
            raise ValueError("reference image has too few features in the static region")
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def check(self, image: np.ndarray) -> DriftResult:
        kp, desc = self._orb.detectAndCompute(to_gray(image), self._mask)
        if desc is None or len(kp) < 10:
            return DriftResult(False, 0.0, float("inf"), "no_features")
        pairs = self._matcher.knnMatch(self._ref_desc, desc, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < 10:
            return DriftResult(False, 0.0, float("inf"), "too_few_matches")
        src = np.array([self._ref_kp[m.queryIdx].pt for m in good], dtype=np.float32)
        dst = np.array([kp[m.trainIdx].pt for m in good], dtype=np.float32)
        matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
        if matrix is None or inliers is None:
            return DriftResult(False, 0.0, float("inf"), "no_transform")
        ratio = float(inliers.sum()) / len(good)
        shift = float(np.hypot(matrix[0, 2], matrix[1, 2]))
        if ratio < self._min_inlier_ratio:
            return DriftResult(False, ratio, shift, "low_inlier_ratio")
        if shift > self._max_shift_px:
            return DriftResult(False, ratio, shift, "view_shifted")
        return DriftResult(True, ratio, shift)
