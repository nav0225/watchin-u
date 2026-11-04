"""Smoke test of the real command sequence on a synthetic recording."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

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
        "product,layout,coil_count,truck_ref,notes\n"
        "v1,cam01,left,2026-10-31T10:01:00,,2026-10-31T10:06:00,"
        "2026-10-31T10:06:30,2026-10-31T10:08:00,wire_rod,linear,10,,\n"
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


def test_timelapse_images_index_and_replay_a_full_visit(tmp_path):
    """A 40-minute visit recorded as one JPEG every 10 s, end to end."""
    tl = tmp_path / "timelapse"
    tl.mkdir()
    start = datetime(2026, 11, 4, 10, 0, 0)
    for i in range(6 * 50):  # 50 minutes
        t = start + timedelta(seconds=10 * i)
        minute = i / 6
        value = 40 if minute < 2 or minute >= 46 else (210 if minute >= 42 else 100)
        cv2.imwrite(str(tl / f"cam01_main_{t:%Y%m%dT%H%M%S}.jpg"), textured(value))
    visit_log = tmp_path / "visits.csv"
    visit_log.write_text(
        "visit_id,camera_id,bay_id,arrived,loading_start,tarp_start,tarp_done,departed,"
        "product,layout,coil_count,truck_ref,notes\n"
        "t1,cam01,left,2026-11-04T10:02:00,,2026-11-04T10:42:00,2026-11-04T10:43:00,"
        "2026-11-04T10:46:00,wire_rod,criss_cross,14,,\n"
    )
    frames_dir = tmp_path / "frames"
    assert main(["extract", "--input", str(tl), "--out", str(frames_dir)]) == 0
    assert main(["extract", "--input", str(tl), "--out", str(frames_dir)]) == 0  # resumable
    records = [json.loads(line) for line in (frames_dir / "manifest.jsonl").open()]
    assert len(records) == 300
    assert records[0]["captured_at"].startswith("2026-11-04T04:30:00")  # 10:00 IST in UTC
    assert main(["dedupe", "--manifest", str(frames_dir / "manifest.jsonl")]) == 0

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(BASE_CONFIG))
    out = tmp_path / "replay"
    assert (
        main(
            [
                "replay",
                "--config",
                str(cfg_path),
                "--video",
                str(tl),
                "--out",
                str(out),
                "--stage-oracle",
                str(visit_log),
            ]
        )
        == 0
    )
    events = [json.loads(line) for line in (out / "events.jsonl").open()]
    assert [e["type"] for e in events] == ["opened", "decision", "decision", "closed"]
    assert events[2]["decision"]["product"] == "wire_rod"


def test_short_clips_without_timestamps_get_a_prefilled_visit_log(tmp_path):
    """Phone/NVR clips named e.g. VID_001.avi: file time -> start, one log row per clip."""
    clips = tmp_path / "clips"
    clips.mkdir()
    for i, value in enumerate((100, 130)):
        video = write_video(clips / f"VID_00{i}.avi", [textured(value)] * 20, fps=2)  # 10 s
        stamp = datetime(2026, 11, 4, 9 + i, 0, tzinfo=ZoneInfo("Asia/Kolkata")).timestamp()
        os.utime(video, (stamp, stamp))
    frames_dir = tmp_path / "frames"
    assert (
        main(
            [
                "extract",
                "--input",
                str(clips),
                "--out",
                str(frames_dir),
                "--every-s",
                "1",
                "--start-time-from-mtime",
            ]
        )
        == 0
    )
    manifest = frames_dir / "manifest.jsonl"
    log_path = tmp_path / "visit_log.csv"
    assert main(["visit-log-from-clips", "--manifest", str(manifest), "--out", str(log_path)]) == 0
    rows = list(csv.DictReader(log_path.open()))
    assert [r["visit_id"] for r in rows] == ["VID_000", "VID_001"]
    assert rows[0]["arrived"] == "2026-11-04T08:59:59+05:30"
    assert rows[0]["product"] == "" and rows[0]["bay_id"] == "left"

    sheets = tmp_path / "sheets"
    assert main(["contact-sheet", "--manifest", str(manifest), "--out", str(sheets)]) == 0
    sheet = cv2.imread(str(sheets / "contact_sheet_01.jpg"))
    assert sheet.shape[1] == 480 * 4 and sheet.shape[0] > 2 * 270
