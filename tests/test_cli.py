"""Smoke test of the real command sequence on a synthetic recording."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from watchinu.cli import main

from .helpers import BASE_CONFIG, textured, write_video

REPO = Path(__file__).resolve().parents[1]


def test_dataset_and_replay_commands_end_to_end(tmp_path):
    # 2 fps "recording": 1 min empty, 5 min open bed, 2 min covered, 3 min empty.
    frames = []
    for value, minutes in ((40, 1), (100, 5), (210, 2), (40, 3)):
        frames += [textured(value)] * (minutes * 60 * 2)
    video = write_video(tmp_path / "cam01_main_20261031T100000.avi", frames, fps=2)

    visit_log = tmp_path / "visits.csv"
    visit_log.write_text(
        "visit_id,camera_id,bay_id,arrived,loading_start,tarp_start,tarp_done,departed,"
        "layout,coil_count,truck_ref,notes\n"
        "v1,cam01,left,2026-10-31T10:01:00,2026-10-31T10:02:00,2026-10-31T10:06:00,"
        "2026-10-31T10:06:30,2026-10-31T10:08:00,linear,10,,\n"
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(BASE_CONFIG))
    frames_dir = tmp_path / "frames"
    manifest = frames_dir / "manifest.jsonl"

    assert (
        main(["extract", "--input", str(video), "--out", str(frames_dir), "--every-s", "10"]) == 0
    )
    assert main(["extract", "--input", str(video), "--out", str(frames_dir)]) == 0  # resumable
    assert sum(1 for _ in manifest.open()) == 66
    assert main(["dedupe", "--manifest", str(manifest)]) == 0
    assert main(["split", "--manifest", str(manifest), "--visit-log", str(visit_log)]) == 0
    assert (
        main(
            [
                "export-cls",
                "--manifest",
                str(manifest),
                "--visit-log",
                str(visit_log),
                "--task",
                "stage",
                "--out",
                str(tmp_path / "stage"),
                "--margin-s",
                "0",
            ]
        )
        == 0
    )
    card = json.loads((tmp_path / "stage" / "dataset_card.json").read_text())
    assert card["task"] == "classify-stage"

    out = tmp_path / "replay"
    assert (
        main(
            [
                "replay",
                "--config",
                str(cfg_path),
                "--video",
                str(video),
                "--out",
                str(out),
                "--every-s",
                "5",
                "--stage-oracle",
                str(visit_log),
            ]
        )
        == 0
    )
    events = [json.loads(line) for line in (out / "events.jsonl").open()]
    assert [e["type"] for e in events] == ["opened", "decision", "decision", "closed"]

    report_path = tmp_path / "report.md"
    assert (
        main(
            [
                "eval",
                "--predictions",
                str(out / "events.jsonl"),
                "--visit-log",
                str(visit_log),
                "--out",
                str(report_path),
            ]
        )
        == 0
    )
    assert "| visit_recall | 1.0 |" in report_path.read_text()


def test_render_config_draws_on_frame(tmp_path):
    frame = tmp_path / "frame.jpg"
    cv2.imwrite(str(frame), np.zeros((360, 640, 3), np.uint8))
    out = tmp_path / "check.png"
    assert (
        main(
            [
                "render-config",
                "--config",
                str(REPO / "configs" / "camera01.yaml"),
                "--image",
                str(frame),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert cv2.imread(str(out)).sum() > 0
