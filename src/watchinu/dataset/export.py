"""Build Ultralytics-ready training datasets from the manifest.

Detection (``build_detection_dataset``): CVAT "YOLO 1.1" labels on full frames are
cropped to the bay with exactly the transform the runtime uses (bounding rectangle +
black outside the polygon), so the model trains on what it will see in production.

Classification (``build_classification_dataset``): stage and layout labels are derived
from the visit log, so no per-frame tagging is needed. Output is the ``split/class/``
folder layout that ``yolo classify train`` expects.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

import cv2
import yaml

from watchinu.dataset.manifest import FrameRecord
from watchinu.dataset.visits import VisitLabel, frame_layout_label, frame_stage
from watchinu.vision.geometry import blank_regions, crop_to_polygon, to_pixels

YoloBox = tuple[int, float, float, float, float]  # class, cx, cy, w, h (normalised)


def read_yolo_labels(path: Path) -> list[YoloBox]:
    out: list[YoloBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{path}: expected 5 fields, got {len(parts)}: {line!r}")
        out.append((int(parts[0]), *(float(p) for p in parts[1:])))  # type: ignore[arg-type]
    return out


def crop_yolo_labels(
    labels: list[YoloBox],
    img_w: int,
    img_h: int,
    rect: tuple[int, int, int, int],
    min_visible: float = 0.5,
) -> list[YoloBox]:
    """Re-express full-frame labels in crop coordinates.

    Boxes are clipped to the crop; boxes with less than ``min_visible`` of their area
    inside the crop are dropped (a sliver of a coil is not a training target).
    """
    x0, y0, x1, y1 = rect
    cw, ch = x1 - x0, y1 - y0
    out: list[YoloBox] = []
    for cls, cx, cy, w, h in labels:
        bx1, by1 = (cx - w / 2) * img_w, (cy - h / 2) * img_h
        bx2, by2 = (cx + w / 2) * img_w, (cy + h / 2) * img_h
        area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        if area <= 0:
            continue
        nx1, ny1 = max(bx1, x0), max(by1, y0)
        nx2, ny2 = min(bx2, x1), min(by2, y1)
        inter = max(0.0, nx2 - nx1) * max(0.0, ny2 - ny1)
        if inter / area < min_visible:
            continue
        out.append(
            (
                cls,
                ((nx1 + nx2) / 2 - x0) / cw,
                ((ny1 + ny2) / 2 - y0) / ch,
                (nx2 - nx1) / cw,
                (ny2 - ny1) / ch,
            )
        )
    return out


def _manifest_digest(records: Sequence[FrameRecord]) -> str:
    h = hashlib.sha256()
    for rec in sorted(records, key=lambda r: r.frame_id):
        h.update(f"{rec.frame_id}|{rec.split}|{rec.visit_id}\n".encode())
    return h.hexdigest()[:16]


def _prepare(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)


def build_detection_dataset(
    records: list[FrameRecord],
    images_root: str | Path,
    labels_dir: str | Path,
    out_dir: str | Path,
    class_names: list[str],
    crop_polygon: Sequence[tuple[float, float]] | None = None,
    include_duplicates: bool = False,
    osd_masks: Sequence[Sequence[tuple[float, float]]] = (),
) -> dict[str, object]:
    """Only frames with a label file are exported; an empty file is a verified negative.

    The runtime detector sees full frames with the OSD overlays blanked, so that is the
    default here too (``crop_polygon`` is only for experiments)."""
    images_root, labels_dir, out_dir = Path(images_root), Path(labels_dir), Path(out_dir)
    _prepare(out_dir)
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    used: list[FrameRecord] = []
    for rec in records:
        if rec.split is None or (rec.duplicate_of and not include_duplicates):
            continue
        label_path = labels_dir / f"{Path(rec.image).stem}.txt"
        if not label_path.exists():
            continue
        image = cv2.imread(str(images_root / rec.image))
        if image is None:
            raise OSError(f"cannot read {images_root / rec.image}")
        image = blank_regions(image, osd_masks)
        labels = read_yolo_labels(label_path)
        h, w = image.shape[:2]
        if crop_polygon is not None:
            image, (ox, oy) = crop_to_polygon(image, to_pixels(crop_polygon, w, h))
            ch, cw = image.shape[:2]
            labels = crop_yolo_labels(labels, w, h, (ox, oy, ox + cw, oy + ch))
        img_out = out_dir / "images" / rec.split / f"{rec.frame_id}.jpg"
        lbl_out = out_dir / "labels" / rec.split / f"{rec.frame_id}.txt"
        img_out.parent.mkdir(parents=True, exist_ok=True)
        lbl_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_out), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lbl_out.write_text(
            "".join(f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n" for c, x, y, bw, bh in labels),
            encoding="utf-8",
        )
        stats[rec.split]["images"] += 1
        for c, *_ in labels:
            stats[rec.split][class_names[c] if c < len(class_names) else f"class_{c}"] += 1
        used.append(rec)

    # No "path": Ultralytics then resolves splits relative to this file, so the dataset
    # folder stays valid when copied from a laptop to Google Drive / Colab / the server.
    data_yaml = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(class_names)),
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    card = {
        "task": "detect",
        "classes": class_names,
        "cropped_to_bay": crop_polygon is not None,
        "manifest_digest": _manifest_digest(used),
        "splits": {k: dict(v) for k, v in stats.items()},
    }
    (out_dir / "dataset_card.json").write_text(json.dumps(card, indent=2))
    return card


BayCrop = tuple[str | None, Sequence[tuple[float, float]] | None]  # (bay id, polygon)


def build_classification_dataset(
    records: list[FrameRecord],
    visits: list[VisitLabel],
    images_root: str | Path,
    out_dir: str | Path,
    task: str,
    bays: Sequence[BayCrop] = ((None, None),),
    boundary_margin_s: float = 60.0,
    include_duplicates: bool = False,
    osd_masks: Sequence[Sequence[tuple[float, float]]] = (),
) -> dict[str, object]:
    """``task`` is ``stage`` or ``layout``; one sample per frame **per bay**.

    Each bay's sample is that bay's crop (outside the polygon blacked out, as at runtime)
    labelled from *that lane's* visits. ``bays=((None, None),)`` means the whole frame
    with lane-agnostic labels. Ambiguous frames near transitions are skipped.
    """
    if task not in ("stage", "layout"):
        raise ValueError("task must be 'stage' or 'layout'")
    images_root, out_dir = Path(images_root), Path(out_dir)
    _prepare(out_dir)
    margin = timedelta(seconds=boundary_margin_s)
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    used: list[FrameRecord] = []
    for rec in records:
        if rec.split is None or rec.captured_at is None:
            continue
        if rec.duplicate_of and not include_duplicates:
            continue
        labels: list[tuple[str | None, Sequence[tuple[float, float]] | None, str]] = []
        for bay_id, polygon in bays:
            if task == "stage":
                stage = frame_stage(rec.captured_at, rec.camera_id, visits, margin, bay_id)
                label = stage.value if stage is not None else None
            else:
                label = frame_layout_label(rec.captured_at, rec.camera_id, visits, margin, bay_id)
            if label is not None:
                labels.append((bay_id, polygon, label))
        if not labels:
            continue
        image = cv2.imread(str(images_root / rec.image))
        if image is None:
            raise OSError(f"cannot read {images_root / rec.image}")
        image = blank_regions(image, osd_masks)  # exactly what the runtime classifiers see
        h, w = image.shape[:2]
        for bay_id, polygon, label in labels:
            sample = (
                image if polygon is None else crop_to_polygon(image, to_pixels(polygon, w, h))[0]
            )
            name = f"{rec.frame_id}__{bay_id}.jpg" if bay_id else f"{rec.frame_id}.jpg"
            dst = out_dir / rec.split / label / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), sample, [cv2.IMWRITE_JPEG_QUALITY, 95])
            stats[rec.split][label] += 1
        used.append(rec)
    card = {
        "task": f"classify-{task}",
        "bays": [b for b, _ in bays],
        "cropped_to_bay": any(p is not None for _, p in bays),
        "boundary_margin_s": boundary_margin_s,
        "manifest_digest": _manifest_digest(used),
        "splits": {k: dict(v) for k, v in stats.items()},
    }
    (out_dir / "dataset_card.json").write_text(json.dumps(card, indent=2))
    return card
