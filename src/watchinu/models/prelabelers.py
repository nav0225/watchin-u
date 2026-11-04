"""Bootstrap pre-labellers: get from zero labels to CVAT corrections as fast as possible.

There is no public coil detector, and open-vocabulary text prompts need a text encoder
that is not always installable. What works on this footage (checked 2026-11-04):

* a prompt-free open-vocabulary segmenter (YOLOE ``*-pf``) proposes instance masks for
  almost every coil - under arbitrary names ("sleeping bag", "laundry", ...);
* BALCO wire-rod coils have an unmistakable wrap: white film with saturated blue bands.
  Ingots (grey metal), tarps (yellow / black), the shed and the trucks fail that test.

So coils = proposals whose mask passes a measured colour test, minus group boxes and
duplicates. Workers come from a COCO model, which knows ``person`` well. These are
*pre-labels* for human correction, not ground truth; once ~200 frames are corrected the
fine-tuned detector replaces this.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from watchinu.domain import BBox, Detection
from watchinu.models.base import Detector


@dataclass(frozen=True, slots=True)
class WrapColourFilter:
    """Colour test for the BALCO wrap (OpenCV HSV: hue 0-179). Tuned on 2026-11-04."""

    blue_hue: tuple[int, int] = (95, 130)
    blue_min_saturation: int = 90
    blue_min_value: int = 60
    white_max_saturation: int = 55
    white_min_value: int = 150
    min_blue_fraction: float = 0.10
    min_wrap_fraction: float = 0.55  # blue + white
    min_area_fraction: float = 0.002
    max_area_fraction: float = 0.15
    nms_iou: float = 0.5
    max_contained: int = 1  # a proposal containing 2+ accepted coils is a group box


DEFAULT_WRAP_FILTER = WrapColourFilter()


def wrap_fractions(hsv: np.ndarray, mask: np.ndarray, f: WrapColourFilter) -> tuple[float, float]:
    """(blue, white) pixel fractions inside ``mask`` of an HSV image."""
    pix = hsv[mask]
    if pix.size == 0:
        return 0.0, 0.0
    h, s, v = pix[:, 0], pix[:, 1], pix[:, 2]
    blue = (h >= f.blue_hue[0]) & (h <= f.blue_hue[1])
    blue &= (s >= f.blue_min_saturation) & (v >= f.blue_min_value)
    white = (s <= f.white_max_saturation) & (v >= f.white_min_value)
    return float(blue.mean()), float(white.mean())


def _contains(outer: BBox, inner: BBox, tol: float = 5.0) -> bool:
    return (
        outer.x1 <= inner.x1 + tol
        and outer.y1 <= inner.y1 + tol
        and outer.x2 >= inner.x2 - tol
        and outer.y2 >= inner.y2 - tol
    )


def select_coils(
    image: np.ndarray,
    boxes: Sequence[BBox],
    confidences: Sequence[float],
    masks: Sequence[np.ndarray],
    f: WrapColourFilter = DEFAULT_WRAP_FILTER,
    label: str = "coil",
) -> list[Detection]:
    """Turn class-agnostic instance proposals into coil detections."""
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    accepted: list[tuple[float, BBox]] = []
    for box, conf, mask in zip(boxes, confidences, masks, strict=True):
        area = box.area / (w * h)
        if not f.min_area_fraction <= area <= f.max_area_fraction:
            continue
        blue, white = wrap_fractions(hsv, mask.astype(bool), f)
        if blue >= f.min_blue_fraction and blue + white >= f.min_wrap_fraction:
            accepted.append((float(conf), box))
    # Group boxes: one proposal spanning several coils.
    accepted = [
        (c, b)
        for c, b in accepted
        if sum(_contains(b, o) for _, o in accepted if o is not b) <= f.max_contained
    ]
    accepted.sort(key=lambda cb: -cb[0])
    kept: list[tuple[float, BBox]] = []
    for c, b in accepted:
        if all(b.iou(k) < f.nms_iou for _, k in kept):
            kept.append((c, b))
    return [Detection(label, c, b) for c, b in kept]


class WrapColourCoilProposer:
    """Detector: YOLOE prompt-free instance proposals + the wrap colour test."""

    STATS_SCALE = 4

    def __init__(
        self,
        weights: str = "yoloe-11s-seg-pf.pt",
        imgsz: int = 1280,
        conf: float = 0.05,
        device: str = "cpu",
        colour: WrapColourFilter = DEFAULT_WRAP_FILTER,
    ) -> None:
        from watchinu.models.ultralytics_backend import load_yolo

        self._model = load_yolo(weights)
        self._imgsz, self._conf, self._device, self._colour = imgsz, conf, device, colour
        self.name = f"wrap-colour+{Path(weights).stem}"
        self.version = "bootstrap"

    def detect(self, image: np.ndarray) -> list[Detection]:
        # Colour statistics do not need full-resolution masks: rasterise each proposal's
        # polygon at 1/STATS_SCALE on a downscaled image (retina masks for ~100 proposals
        # on a 2560x1440 frame dominate the run time otherwise).
        res = self._model.predict(
            image, imgsz=self._imgsz, conf=self._conf, device=self._device, verbose=False
        )[0]
        if res.boxes is None or res.masks is None:
            return []
        k = self.STATS_SCALE
        h, w = image.shape[:2]
        small = cv2.resize(image, (w // k, h // k), interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        boxes, masks = [], []
        for xyxy, polys in zip(res.boxes.xyxy.cpu().numpy(), res.masks.xy, strict=True):
            mask = np.zeros((sh, sw), np.uint8)
            if len(polys):
                cv2.fillPoly(mask, [np.round(np.asarray(polys) / k).astype(np.int32)], 1)
            masks.append(mask.astype(bool))
            boxes.append(BBox(*(float(v) / k for v in xyxy)))
        confs = res.boxes.conf.cpu().numpy().tolist()
        found = select_coils(small, boxes, confs, masks, self._colour)
        return [
            Detection(
                d.label, d.confidence, BBox(d.box.x1 * k, d.box.y1 * k, d.box.x2 * k, d.box.y2 * k)
            )
            for d in found
        ]


class LabelFilter:
    """Keeps (and renames) selected labels of another detector, e.g. COCO ``person``."""

    def __init__(self, inner: Detector, keep: dict[str, str]) -> None:
        self._inner, self._keep = inner, keep
        self.name, self.version = inner.name, inner.version

    def detect(self, image: np.ndarray) -> list[Detection]:
        return [
            Detection(self._keep[d.label], d.confidence, d.box)
            for d in self._inner.detect(image)
            if d.label in self._keep
        ]


class CompositeDetector:
    def __init__(self, *detectors: Detector) -> None:
        self._detectors = detectors
        self.name = "+".join(d.name for d in detectors)
        self.version = "+".join(d.version for d in detectors)

    def detect(self, image: np.ndarray) -> list[Detection]:
        return [det for d in self._detectors for det in d.detect(image)]
