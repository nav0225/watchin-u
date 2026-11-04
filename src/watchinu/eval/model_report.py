"""Model-level evaluation on a held-out split, as a compact report to paste back.

Detection uses Ultralytics validation (mAP, precision, recall per class). Classification
is evaluated here image by image, so the report has a full confusion matrix and
per-class recall - the numbers that matter for the stage gate (does an ingot truck ever
get called ``wire_rod``?) and for the layout decision.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def classification_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]
) -> dict[str, Any]:
    """Accuracy, per-class precision/recall/support and the confusion matrix."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred differ in length")
    pairs = Counter(zip(y_true, y_pred, strict=True))
    per_class: dict[str, dict[str, float | int | None]] = {}
    for c in classes:
        tp = pairs[(c, c)]
        support = sum(n for (t, _), n in pairs.items() if t == c)
        predicted = sum(n for (_, p), n in pairs.items() if p == c)
        per_class[c] = {
            "support": support,
            "precision": round(tp / predicted, 4) if predicted else None,
            "recall": round(tp / support, 4) if support else None,
        }
    correct = sum(n for (t, p), n in pairs.items() if t == p)
    return {
        "n": len(y_true),
        "accuracy": round(correct / len(y_true), 4) if y_true else None,
        "per_class": per_class,
        "confusion": {t: {p: pairs[(t, p)] for p in classes} for t in classes},
    }


def _run_summary(weights: Path) -> dict[str, Any]:
    """Best epoch from ``results.csv`` and provenance, when weights sit in a run folder."""
    run_dir = weights.parent.parent
    out: dict[str, Any] = {}
    results = run_dir / "results.csv"
    if results.exists():
        with results.open(newline="") as fh:
            rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(fh)]
        if rows:
            key = next(
                (k for k in ("metrics/mAP50-95(B)", "metrics/accuracy_top1") if k in rows[0]),
                None,
            )
            best = max(rows, key=lambda r: float(r[key])) if key else rows[-1]
            out["epochs_run"] = len(rows)
            out["best_epoch"] = int(float(best.get("epoch", len(rows))))
            if key:
                out["best_" + key.split("/")[-1]] = round(float(best[key]), 4)
    provenance = run_dir / "provenance.json"
    if provenance.exists():
        prov = json.loads(provenance.read_text())
        card = prov.get("dataset_card") or {}
        out["git_commit"] = (prov.get("git_commit") or "")[:12]
        out["base_model"] = prov.get("base_model")
        out["dataset_digest"] = card.get("manifest_digest")
        out["dataset_splits"] = card.get("splits")
    return out


def evaluate_detector(
    weights: str | Path,
    data_yaml: str | Path,
    split: str = "test",
    imgsz: int = 1280,
    device: str = "cpu",
) -> dict[str, Any]:
    from ultralytics import YOLO

    weights = Path(weights)
    metrics = YOLO(str(weights)).val(
        data=str(data_yaml), split=split, imgsz=imgsz, device=device, plots=False, verbose=False
    )
    box = metrics.box
    names = metrics.names
    per_class = {}
    for i, c in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        per_class[names[int(c)]] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "mAP50": round(float(ap50), 4),
            "mAP50-95": round(float(ap), 4),
        }
    return {
        "task": "detect",
        "weights": str(weights),
        "split": split,
        "generated_at": datetime.now(UTC).isoformat(),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "per_class": per_class,
        "run": _run_summary(weights),
    }


def evaluate_classifier(
    weights: str | Path,
    data_dir: str | Path,
    split: str = "test",
    imgsz: int = 224,
    device: str = "cpu",
) -> dict[str, Any]:
    from watchinu.models.ultralytics_backend import UltralyticsClassifier

    weights, split_dir = Path(weights), Path(data_dir) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"no {split} split at {split_dir}")
    clf = UltralyticsClassifier(str(weights), imgsz=imgsz, device=device)
    import cv2

    y_true: list[str] = []
    y_pred: list[str] = []
    for class_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            probs = clf.classify(image)
            y_true.append(class_dir.name)
            y_pred.append(max(probs, key=lambda k: probs[k]))
    classes = sorted(set(y_true) | set(y_pred))
    return {
        "task": "classify",
        "weights": str(weights),
        "split": split,
        "generated_at": datetime.now(UTC).isoformat(),
        **classification_metrics(y_true, y_pred, classes),
        "run": _run_summary(weights),
    }


def to_markdown(report: dict[str, Any]) -> str:
    title = Path(report["weights"]).parent.parent.name or Path(report["weights"]).stem
    lines = [f"## {title} ({report['task']}, split={report['split']})", ""]
    run = report.get("run") or {}
    if run:
        lines.append(
            "run: " + ", ".join(f"{k}={v}" for k, v in run.items() if k != "dataset_splits")
        )
        if run.get("dataset_splits"):
            lines.append(f"dataset: {json.dumps(run['dataset_splits'])}")
        lines.append("")
    if report["task"] == "detect":
        lines += [
            "| class | precision | recall | mAP50 | mAP50-95 |",
            "|---|---|---|---|---|",
            f"| **all** | {report['precision']} | {report['recall']} | {report['mAP50']} | "
            f"{report['mAP50-95']} |",
        ]
        for name, m in report["per_class"].items():
            lines.append(
                f"| {name} | {m['precision']} | {m['recall']} | {m['mAP50']} | {m['mAP50-95']} |"
            )
    else:
        classes = list(report["confusion"])
        lines += [f"accuracy: **{report['accuracy']}** on {report['n']} images", ""]
        lines += ["| class | support | precision | recall |", "|---|---|---|---|"]
        for c in classes:
            m = report["per_class"][c]
            lines.append(f"| {c} | {m['support']} | {m['precision']} | {m['recall']} |")
        lines += ["", "confusion (rows = truth, columns = predicted):", ""]
        lines.append("| truth \\ pred | " + " | ".join(classes) + " |")
        lines.append("|---" * (len(classes) + 1) + "|")
        for t in classes:
            row = " | ".join(str(report["confusion"][t][p]) for p in classes)
            lines.append(f"| {t} | {row} |")
    return "\n".join(lines) + "\n"
