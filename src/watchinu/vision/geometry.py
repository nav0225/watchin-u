"""Polygon, masking and bed-plane homography helpers."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from watchinu.domain import BBox

PointSeq = Sequence[tuple[float, float]]


def to_pixels(points: PointSeq, width: int, height: int) -> np.ndarray:
    """Normalised (0..1) points -> float32 pixel array of shape (N, 2)."""
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return arr * np.array([width, height], dtype=np.float32)


def polygon_mask(poly_px: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """uint8 mask (255 inside) of ``shape`` = (height, width)."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(poly_px).astype(np.int32)], 255)
    return mask


def point_in_polygon(x: float, y: float, poly_px: np.ndarray) -> bool:
    contour = poly_px.astype(np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


def blank_regions(image: np.ndarray, polys_norm: Sequence[PointSeq], value: int = 0) -> np.ndarray:
    """Return a copy of ``image`` with each normalised polygon filled with ``value``."""
    if not polys_norm:
        return image
    out = image.copy()
    h, w = image.shape[:2]
    for poly in polys_norm:
        cv2.fillPoly(out, [np.round(to_pixels(poly, w, h)).astype(np.int32)], value)
    return out


def polygon_rect(poly_px: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    """Bounding rectangle (x0, y0, x1, y1) of the polygon, clipped to the frame."""
    x, y, bw, bh = cv2.boundingRect(np.round(poly_px).astype(np.int32))
    return max(0, x), max(0, y), min(width, x + bw), min(height, y + bh)


def crop_to_polygon(
    image: np.ndarray, poly_px: np.ndarray, mask_outside: bool = True
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop to the polygon's bounding rectangle; optionally black out pixels outside it.

    Returns the crop and its (x, y) offset in the source image. Masking outside the bay
    keeps the neighbouring truck out of the model input — training exports apply exactly
    the same transform so train and inference inputs match.
    """
    h, w = image.shape[:2]
    x0, y0, x1, y1 = polygon_rect(poly_px, w, h)
    crop = image[y0:y1, x0:x1].copy()
    if mask_outside:
        local = poly_px - np.array([x0, y0], dtype=np.float32)
        mask = polygon_mask(local, crop.shape[:2])
        crop[mask == 0] = 0
    return crop, (x0, y0)


class BedHomography:
    """Maps image points onto the bed plane, normalised to the unit square.

    ``u`` runs across the bed (0 = left wall, 1 = right wall) and ``v`` along it
    (0 = near end, 1 = far end). The quad is traced along the tops of the side walls,
    which sit at roughly coil mid-height, so coil box centres project sensibly.
    """

    _UNIT = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    def __init__(self, quad_px: np.ndarray) -> None:
        quad = np.asarray(quad_px, dtype=np.float32).reshape(4, 2)
        self._h = cv2.getPerspectiveTransform(quad, self._UNIT)
        self._h_inv = cv2.getPerspectiveTransform(self._UNIT, quad)

    def to_bed(self, points_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return cv2.perspectiveTransform(pts, self._h).reshape(-1, 2)

    def to_image(self, points_uv: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_uv, dtype=np.float32).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return cv2.perspectiveTransform(pts, self._h_inv).reshape(-1, 2)


def union_coverage(boxes: Sequence[BBox], region_mask: np.ndarray, scale: float = 0.25) -> float:
    """Fraction of ``region_mask`` (255 = region) covered by the union of ``boxes``.

    Rasterised at reduced ``scale`` — exact enough for an occlusion gate, and cheap.
    """
    if not boxes:
        return 0.0
    small = cv2.resize(
        region_mask,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )
    region = small > 0
    total = int(region.sum())
    if total == 0:
        return 0.0
    cover = np.zeros_like(region)
    for b in boxes:
        x1, y1 = int(b.x1 * scale), int(b.y1 * scale)
        x2, y2 = int(np.ceil(b.x2 * scale)), int(np.ceil(b.y2 * scale))
        cover[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)] = True
    return float((cover & region).sum()) / total
