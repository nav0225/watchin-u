"""Integration with the real Ultralytics library (skipped unless ``.[ml]`` is installed).

Proves that our exported datasets are accepted by Ultralytics training, that the
training wrapper records provenance, and that the backends parse real model output.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("ultralytics")

from watchinu.dataset.export import build_classification_dataset, build_detection_dataset
from watchinu.dataset.manifest import FrameRecord
from watchinu.dataset.visits import VisitLabel
from watchinu.domain import Product
from watchinu.eval.model_report import evaluate_classifier, evaluate_detector, to_markdown
from watchinu.models.ultralytics_backend import UltralyticsClassifier, UltralyticsDetector
from watchinu.training import train

T0 = datetime.fromisoformat("2026-10-31T10:00:00+05:30")


def _frame(i: int, root: Path, with_coils: bool) -> tuple[FrameRecord, list[str]]:
    rng = np.random.default_rng(i)
    img = np.full((360, 640, 3), 60, np.uint8)
    labels = []
    if with_coils:
        for k in range(3):
            cx, cy = 120 + 180 * k + int(rng.integers(-20, 20)), 180 + int(rng.integers(-40, 40))
            cv2.ellipse(img, (cx, cy), (60, 45), 0, 0, 360, (235, 235, 235), -1)
            cv2.ellipse(img, (cx, cy), (60, 45), 0, 0, 360, (200, 90, 30), 6)
            labels.append(f"0 {cx / 640:.6f} {cy / 360:.6f} {130 / 640:.6f} {100 / 360:.6f}")
    path = root / "images" / f"f{i}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    rec = FrameRecord(
        frame_id=f"f{i}",
        image=f"images/f{i}.jpg",
        video="v.mkv",
        camera_id="cam01",
        stream="main",
        pts_s=i,
        frame_index=i,
        captured_at=T0 + timedelta(minutes=i),
        width=640,
        height=360,
        blur_var=100,
        brightness=60,
        overexposed_fraction=0,
        dhash="0" * 16,
        split="train" if i % 4 else "val",
    )
    return rec, labels


def test_detector_finetune_and_backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # weights download here
    root = tmp_path / "frames"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    records = []
    for i in range(16):
        rec, labels = _frame(i, root, with_coils=i % 5 != 0)
        (labels_dir / f"f{i}.txt").write_text("\n".join(labels) + ("\n" if labels else ""))
        records.append(rec)
    build_detection_dataset(records, root, labels_dir, tmp_path / "ds", ["coil", "person"])

    run_dir = train(
        tmp_path / "ds" / "data.yaml",
        "yolo26n.pt",
        project=tmp_path / "runs",
        name="smoke",
        epochs=1,
        imgsz=320,
        batch=4,
        device="cpu",
        extra={"workers": 0, "plots": False},
    )
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert provenance["dataset_card"]["task"] == "detect"
    assert provenance["base_model"] == "yolo26n.pt"

    det = UltralyticsDetector(str(run_dir / "weights" / "best.pt"), imgsz=320, conf=0.0)
    img = cv2.imread(str(root / "images" / "f1.jpg"))
    out = det.detect(img)
    assert all(d.label in {"coil", "person"} for d in out)
    assert all(0 <= d.box.x1 <= d.box.x2 <= 640 for d in out)
    assert len(det.version) == 12  # sha256 prefix of the weights file

    report = evaluate_detector(
        run_dir / "weights" / "best.pt", tmp_path / "ds" / "data.yaml", split="val", imgsz=320
    )
    assert set(report["per_class"]) <= {"coil", "person"}
    assert report["run"]["base_model"] == "yolo26n.pt"
    assert "| **all** |" in to_markdown(report)


def test_classifier_dataset_trains_and_backend_parses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "frames"
    visits = [
        VisitLabel(
            visit_id="v1",
            camera_id="cam01",
            bay_id="left",
            arrived=T0 + timedelta(minutes=4),
            departed=T0 + timedelta(minutes=12),
            product=Product.WIRE_ROD,
        )
    ]
    records = [_frame(i, root, with_coils=4 <= i <= 12)[0] for i in range(16)]
    build_classification_dataset(
        records, visits, root, tmp_path / "cls", "stage", boundary_margin_s=0
    )
    run_dir = train(
        tmp_path / "cls",
        "yolo26n-cls.pt",
        project=tmp_path / "runs",
        name="cls",
        epochs=1,
        imgsz=64,
        batch=4,
        device="cpu",
        extra={"workers": 0, "plots": False},
    )
    clf = UltralyticsClassifier(str(run_dir / "weights" / "best.pt"), imgsz=64)
    probs = clf.classify(cv2.imread(str(root / "images" / "f5.jpg")))
    assert set(probs) == {"no_truck", "wire_rod"}
    assert abs(sum(probs.values()) - 1.0) < 1e-3

    report = evaluate_classifier(
        run_dir / "weights" / "best.pt", tmp_path / "cls", split="train", imgsz=64
    )
    assert report["n"] > 0 and report["accuracy"] is not None
    assert "confusion (rows = truth" in to_markdown(report)


def test_wrap_colour_proposer_runs_on_the_real_model(tmp_path, monkeypatch):
    """Exercises the real YOLOE mask polygons -> quarter-scale colour test path."""
    from watchinu.models.prelabelers import WrapColourCoilProposer

    weights = Path(__file__).resolve().parents[1] / "yoloe-11s-seg-pf.pt"
    monkeypatch.chdir(tmp_path)  # downloads here if the repo has no local copy
    proposer = WrapColourCoilProposer(
        weights=str(weights) if weights.exists() else "yoloe-11s-seg-pf.pt"
    )
    img = np.full((720, 1280, 3), 70, np.uint8)
    for k in range(3):  # three wrap-coloured "coils" (white film, blue band)
        x = 200 + 300 * k
        cv2.ellipse(img, (x, 360), (110, 90), 0, 0, 360, (230, 230, 230), -1)
        cv2.ellipse(img, (x, 400), (110, 50), 0, 0, 180, (200, 60, 20), -1)
    out = proposer.detect(img)
    assert all(d.label == "coil" for d in out)
    assert all(0 <= d.box.x1 < d.box.x2 <= 1280 and 0 <= d.box.y1 < d.box.y2 <= 720 for d in out)
