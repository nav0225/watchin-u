"""Model-assisted labelling: run any detector over frames and package its predictions as
a CVAT "YOLO 1.1" import, so annotators correct boxes instead of drawing them.

Bootstrap with a zero-shot open-vocabulary model (YOLO-World / YOLOE prompts such as
"wrapped wire coil"), then switch to our own fine-tuned v0 once ~200 frames are labelled.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from pathlib import Path

import cv2

from watchinu.models.base import Detector


def to_yolo_line(cls: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    x1, x2 = max(0.0, x1), min(float(w), x2)
    y1, y2 = max(0.0, y1), min(float(h), y2)
    return (
        f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
        f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
    )


def prelabel_to_cvat_zip(
    detector: Detector,
    image_paths: Iterable[Path],
    class_names: list[str],
    out_zip: str | Path,
    label_map: dict[str, str] | None = None,
    min_confidence: float = 0.25,
) -> int:
    """Write a CVAT-importable zip. ``label_map`` maps detector labels (e.g. prompts) to
    our class names. Returns the number of boxes written."""
    mapping = label_map or {}
    n_boxes = 0
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    names_in_zip: list[str] = []
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in image_paths:
            image = cv2.imread(str(path))
            if image is None:
                raise OSError(f"cannot read {path}")
            h, w = image.shape[:2]
            lines = []
            for det in detector.detect(image):
                name = mapping.get(det.label, det.label)
                if det.confidence < min_confidence or name not in class_names:
                    continue
                b = det.box
                lines.append(to_yolo_line(class_names.index(name), b.x1, b.y1, b.x2, b.y2, w, h))
            zf.writestr(
                f"obj_train_data/{path.stem}.txt", "\n".join(lines) + ("\n" if lines else "")
            )
            names_in_zip.append(f"data/obj_train_data/{path.name}")
            n_boxes += len(lines)
        zf.writestr("obj.names", "\n".join(class_names) + "\n")
        zf.writestr(
            "obj.data",
            f"classes = {len(class_names)}\nnames = data/obj.names\ntrain = data/train.txt\n",
        )
        zf.writestr("train.txt", "\n".join(names_in_zip) + "\n")
    return n_boxes
