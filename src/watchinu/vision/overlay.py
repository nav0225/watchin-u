"""Debug/evidence overlays: every decision is saved with what the model actually saw."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from watchinu.domain import BBox

_COLOURS = {
    "coil": (0, 200, 0),
    "person": (0, 140, 255),
    "polygon": (255, 200, 0),
    "slot_occupied": (0, 200, 0),
    "slot_other": (0, 0, 220),
}


def draw_polygon(image: np.ndarray, poly_px: np.ndarray, colour: tuple[int, int, int]) -> None:
    pts = np.round(poly_px).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], isClosed=True, color=colour, thickness=2)


def draw_boxes(
    image: np.ndarray, boxes: Sequence[BBox], label: str, colour: tuple[int, int, int] | None = None
) -> None:
    c = colour or _COLOURS.get(label, (255, 255, 255))
    for b in boxes:
        cv2.rectangle(image, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), c, 2)


def draw_slots(
    image: np.ndarray, centres_px: dict[tuple[int, int], tuple[float, float]], occupied: set
) -> None:
    for slot, (x, y) in centres_px.items():
        key = "slot_occupied" if slot in occupied else "slot_other"
        cv2.circle(image, (int(x), int(y)), 8, _COLOURS[key], -1)
        cv2.putText(
            image,
            f"{slot[0]},{slot[1]}",
            (int(x) + 10, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            _COLOURS[key],
            2,
        )


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def decode_jpeg(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("JPEG decoding failed")
    return image
