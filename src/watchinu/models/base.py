"""Model interfaces. The runtime depends only on these, so backends are swappable
(Ultralytics YOLO today, RF-DETR / ONNX Runtime / TensorRT later)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from watchinu.domain import Detection


class Detector(Protocol):
    name: str
    version: str

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detections in ``image`` pixel coordinates."""
        ...


class Classifier(Protocol):
    name: str
    version: str

    def classify(self, image: np.ndarray) -> dict[str, float]:
        """Class name -> probability (sums to ~1)."""
        ...


@dataclass(slots=True)
class ModelSet:
    stage: Classifier | None = None
    layout: Classifier | None = None
    detector: Detector | None = None
    extra_versions: dict[str, str] = field(default_factory=dict)
    # Optional per-bay stage models (e.g. the visit-log oracle, which answers per lane).
    stage_by_bay: dict[str, Classifier] = field(default_factory=dict)

    def stage_for(self, bay_id: str) -> Classifier | None:
        return self.stage_by_bay.get(bay_id, self.stage)

    def versions(self) -> dict[str, str]:
        out = dict(self.extra_versions)
        for role, model in (("stage", self.stage), ("layout", self.layout)):
            if model is not None:
                out[role] = f"{model.name}@{model.version}"
        if self.detector is not None:
            out["detector"] = f"{self.detector.name}@{self.detector.version}"
        return out
