"""Ultralytics YOLO backend (detector + classifiers). Imported lazily: install ``.[ml]``."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from watchinu.domain import BBox, Detection


def _file_version(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        return p.name  # hub model name such as "yolo26s.pt"
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _load(weights: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("ultralytics is not installed: pip install -e '.[ml]'") from exc
    return YOLO(weights)


class UltralyticsDetector:
    def __init__(
        self,
        weights: str,
        imgsz: int = 1280,
        conf: float = 0.25,
        iou: float = 0.6,
        device: str = "cpu",
        prompts: list[str] | None = None,
    ) -> None:
        self._model = _load(weights)
        if prompts:
            # Open-vocabulary models (YOLO-World / YOLOE) for zero-shot pre-labelling.
            self._model.set_classes(prompts)
        self._imgsz = imgsz
        self._conf = conf
        self._iou = iou
        self._device = device
        self.name = Path(weights).stem
        self.version = _file_version(weights)

    def detect(self, image: np.ndarray) -> list[Detection]:
        result = self._model.predict(
            image,
            imgsz=self._imgsz,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=False,
        )[0]
        names = result.names
        out: list[Detection] = []
        boxes = result.boxes
        if boxes is None:
            return out
        for xyxy, conf, cls in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
            strict=True,
        ):
            out.append(
                Detection(
                    label=str(names[int(cls)]),
                    confidence=float(conf),
                    box=BBox(*(float(v) for v in xyxy)),
                )
            )
        return out


class UltralyticsClassifier:
    def __init__(self, weights: str, imgsz: int = 224, device: str = "cpu") -> None:
        self._model = _load(weights)
        self._imgsz = imgsz
        self._device = device
        self.name = Path(weights).stem
        self.version = _file_version(weights)

    def classify(self, image: np.ndarray) -> dict[str, float]:
        result = self._model.predict(image, imgsz=self._imgsz, device=self._device, verbose=False)[
            0
        ]
        probs = result.probs.data.cpu().numpy()
        return {str(result.names[i]): float(p) for i, p in enumerate(probs)}
