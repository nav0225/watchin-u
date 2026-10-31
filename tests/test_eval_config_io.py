from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from watchinu.config import load_config
from watchinu.dataset.capture import build_capture_cmd
from watchinu.dataset.visits import load_visit_log
from watchinu.domain import CountSummary, Decision, Layout
from watchinu.eval.visit_metrics import evaluate, load_predictions, to_markdown
from watchinu.io.buffer import LatestBuffer
from watchinu.io.rtsp import build_rtsp_url, redact
from watchinu.runtime.sinks import JsonlSink

from .helpers import BASE_CONFIG, make_config

REPO = Path(__file__).resolve().parents[1]


def test_shipped_camera_config_is_valid():
    cfg = load_config(REPO / "configs" / "camera01.yaml")
    assert cfg.camera.streams["main"].endswith("/101")
    assert cfg.template_for(cfg.bay("left")).n_slots == 10


def test_config_rejects_unknown_template_and_bad_coordinates():
    bad_bay = {**BASE_CONFIG["bays"][0], "slot_template": "nope"}
    with pytest.raises(ValidationError):
        make_config(bays=[bad_bay])
    bad_poly = {**BASE_CONFIG["bays"][0], "polygon": [[0, 0], [2.0, 0], [1, 1]]}
    with pytest.raises(ValidationError):
        make_config(bays=[bad_poly])


def test_credentials_come_from_environment_only(monkeypatch):
    cfg = make_config()
    monkeypatch.delenv("TEST_CAM_USER", raising=False)
    with pytest.raises(RuntimeError, match="TEST_CAM_USER"):
        cfg.camera.credentials()
    monkeypatch.setenv("TEST_CAM_USER", "analytics")
    monkeypatch.setenv("TEST_CAM_PASSWORD", "p@ss:word")
    assert cfg.camera.credentials() == ("analytics", "p@ss:word")


def test_rtsp_url_escapes_and_redacts_credentials():
    url = build_rtsp_url("10.110.4.8", 554, "Streaming/Channels/101", "analytics", "p@ss:w/rd")
    assert url == "rtsp://analytics:p%40ss%3Aw%2Frd@10.110.4.8:554/Streaming/Channels/101"
    assert redact(url) == "rtsp://analytics:***@10.110.4.8:554/Streaming/Channels/101"
    assert "p%40ss" not in redact(url)


def test_capture_command_is_lossless_and_segmented():
    cmd = build_capture_cmd("rtsp://u:p@h/x", "/data/raw", "cam01", "main", segment_s=300)
    joined = " ".join(cmd)
    assert "-c copy" in joined and "-rtsp_transport tcp" in joined
    assert "-segment_format matroska" in joined
    assert cmd[-1].endswith("cam01_main_%Y%m%dT%H%M%S.mkv")


def test_latest_buffer_drops_stale_frames():
    buf: LatestBuffer[int] = LatestBuffer()
    for i in range(5):
        buf.put(i)
    seq, item = buf.get_newer(0, timeout=0.1)
    assert item == 4 and buf.dropped == 4
    assert buf.get_newer(seq, timeout=0.05) is None
    t = threading.Timer(0.05, lambda: buf.put(9))
    t.start()
    assert buf.get_newer(seq, timeout=1.0) == (seq + 1, 9)


VISITS = """visit_id,camera_id,bay_id,arrived,loading_start,tarp_start,tarp_done,departed,layout,coil_count,truck_ref,notes
g1,cam01,left,2026-10-31T10:00:00,,,,2026-10-31T10:45:00,linear,10,,
g2,cam01,left,2026-10-31T12:00:00,,,,2026-10-31T12:45:00,criss_cross,14,,
g3,cam01,left,2026-10-31T15:00:00,,,,2026-10-31T15:45:00,linear,10,,
"""


def _decision(layout: Layout, observed: int, inferred: int | None, review=False) -> Decision:
    return Decision(
        "final", 0.0, layout, 0.9, 1.0, 6, CountSummary(observed, 0, 0, inferred, "b"), review
    )


def test_visit_level_evaluation(tmp_path):
    (tmp_path / "v.csv").write_text(VISITS)
    gts = load_visit_log(tmp_path / "v.csv")
    sink = JsonlSink(tmp_path / "events.jsonl")
    ts = lambda s: datetime.fromisoformat(f"2026-10-31T{s}:00+05:30").timestamp()  # noqa: E731
    # g1: correct; g2: layout wrong, count right; g3: missed; plus one spurious visit.
    sink.visit_opened("cam01", "left", "p1", ts("10:01"), {})
    sink.decision("cam01", "left", "p1", _decision(Layout.LINEAR, 8, 10), None)
    sink.visit_opened("cam01", "left", "p2", ts("12:02"), {})
    sink.decision("cam01", "left", "p2", _decision(Layout.LINEAR, 14, None, True), None)
    sink.visit_opened("cam01", "left", "p3", ts("18:00"), {})

    report = evaluate(load_predictions(tmp_path / "events.jsonl"), gts)
    assert report["visits_matched"] == 2
    assert report["visits_missed"] == 1
    assert report["visits_spurious"] == 1
    assert report["layout_accuracy"] == 0.5
    assert report["layout_confusion"] == {"criss_cross -> linear": 1, "linear -> linear": 1}
    assert report["count_exact_rate"] == 1.0
    assert report["observed_le_true_rate"] == 1.0
    assert report["needs_review_rate"] == 0.5
    assert "| visit_recall |" in to_markdown(report)
