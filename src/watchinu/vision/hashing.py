"""Perceptual difference hash for near-duplicate frame removal."""

from __future__ import annotations

import cv2
import numpy as np

from watchinu.vision.quality import to_gray


def dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """64-bit difference hash (for ``hash_size`` 8): robust to compression and small noise."""
    small = cv2.resize(to_gray(image), (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def to_hex(value: int, hash_size: int = 8) -> str:
    return f"{value:0{hash_size * hash_size // 4}x}"


def from_hex(text: str) -> int:
    return int(text, 16)
