from __future__ import annotations

import cv2
import numpy as np
import pytest

from watchinu.domain import BBox
from watchinu.vision.drift import ViewDriftChecker
from watchinu.vision.geometry import (
    BedHomography,
    blank_regions,
    crop_to_polygon,
    polygon_mask,
    to_pixels,
    union_coverage,
)
from watchinu.vision.hashing import dhash, hamming
from watchinu.vision.quality import assess, change_score, thumbnail

from .helpers import textured


def test_to_pixels_scales_normalised_points():
    pts = to_pixels([(0.5, 0.25), (1.0, 1.0)], 200, 100)
    np.testing.assert_allclose(pts, [[100, 25], [200, 100]])


def test_crop_masks_outside_polygon_and_reports_offset():
    image = np.full((100, 200, 3), 255, np.uint8)
    tri = np.array([[50, 10], [150, 10], [150, 90]], np.float32)
    crop, (ox, oy) = crop_to_polygon(image, tri)
    assert (ox, oy) == (50, 10)
    assert crop.shape[:2] == (81, 101)
    assert crop[70, 5].sum() == 0  # bottom-left of the bounding box is outside the triangle
    assert crop[5, 95].sum() == 255 * 3


def test_blank_regions_does_not_modify_input():
    image = np.full((10, 10, 3), 200, np.uint8)
    out = blank_regions(image, [[(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]])
    assert image.min() == 200
    assert out[1, 1].sum() == 0 and out[8, 8].sum() == 600


def test_bed_homography_round_trip():
    quad = np.array([[100, 900], [900, 950], [700, 100], [300, 120]], np.float32)
    h = BedHomography(quad)
    np.testing.assert_allclose(h.to_bed(quad), [[0, 0], [1, 0], [1, 1], [0, 1]], atol=1e-4)
    uv = np.array([[0.25, 0.3], [0.75, 0.9]], np.float32)
    np.testing.assert_allclose(h.to_bed(h.to_image(uv)), uv, atol=1e-4)


def test_union_coverage_counts_overlap_once():
    mask = polygon_mask(np.array([[0, 0], [99, 0], [99, 99], [0, 99]], np.float32), (100, 100))
    boxes = [BBox(0, 0, 50, 100), BBox(25, 0, 50, 100)]
    assert union_coverage(boxes, mask, scale=1.0) == pytest.approx(0.5, abs=0.02)


def test_quality_metrics():
    sharp = textured(120)
    blurred = cv2.GaussianBlur(sharp, (31, 31), 10)
    assert assess(sharp).blur_var > 10 * assess(blurred).blur_var
    assert assess(textured(120)).brightness == pytest.approx(120, abs=1)
    white = np.full((50, 50, 3), 255, np.uint8)
    assert assess(white).overexposed_fraction == 1.0


def test_change_score():
    a, b = thumbnail(textured(100)), thumbnail(textured(180))
    assert change_score(None, a) == 0.0
    assert change_score(a, a) == 0.0
    assert change_score(a, b) > 0.2


def test_dhash_is_robust_to_noise_but_not_to_change():
    rng = np.random.default_rng(0)
    base = cv2.resize(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8), (320, 240))
    noisy = np.clip(base.astype(int) + rng.integers(-4, 5, base.shape), 0, 255).astype(np.uint8)
    other = cv2.resize(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8), (320, 240))
    assert hamming(dhash(base), dhash(noisy)) <= 6
    assert hamming(dhash(base), dhash(other)) > 15


def _scene(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((360, 640, 3), np.uint8)
    for _ in range(150):  # random rectangles = plenty of ORB corners
        x, y = rng.integers(0, 600), rng.integers(0, 330)
        cv2.rectangle(
            img,
            (int(x), int(y)),
            (int(x) + 30, int(y) + 20),
            tuple(int(c) for c in rng.integers(0, 255, 3)),
            -1,
        )
    return img


def _shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))


def test_drift_checker_accepts_small_jitter_and_flags_moves():
    ref = _scene()
    checker = ViewDriftChecker(ref, max_shift_px=25)
    assert checker.check(_shift(ref, 3, -2)).ok
    moved = checker.check(_shift(ref, 60, 30))
    assert not moved.ok
    assert moved.reason == "view_shifted"
    assert not checker.check(_scene(seed=7)).ok  # a different view entirely
