from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from watchinu.config import load_config
from watchinu.dataset.capture import build_capture_cmd, build_timelapse_cmd
from watchinu.dataset.visits import load_visit_log
from watchinu.domain import CountSummary, Decision, Layout, Product
from watchinu.eval.model_report import classification_metrics
from watchinu.eval.visit_metrics import evaluate, load_predictions, to_markdown
from watchinu.io.buffer import LatestBuffer
from watchinu.io.rtsp import build_rtsp_url, redact
from watchinu.runtime.sinks import JsonlSink

from .helpers import BASE_CONFIG, make_config

REPO = Path(__file__).resolve().parents[1]


def test_shipped_camera_config_is_valid():
    cfg = load_config(REPO / "configs" / "camera01.yaml")
    assert cfg.camera.streams["main"].endswith("/101")
    templates = cfg.templates_for(cfg.bay("left"))
    assert templates["linear"][1].n_slots == 10
    assert templates["criss_cross"][1].n_slots == 14


def test_config_rejects_unknown_template_and_bad_coordinates():
    bad_bay = {**BASE_CONFIG["bays"][0], "slot_templates": {"linear": "nope"}}
    with pytest.raises(ValidationError):
        make_config(bays=[bad_bay])
    bad_poly = {**BASE_CONFIG["bays"][0], "polygon": [[0, 0], [2.0, 0], [1, 1]]}
    with pytest.raises(ValidationError):
        make_config(bays=[bad_poly])
    bad_layout = {**BASE_CONFIG["bays"][0], "slot_templates": {"zigzag": "std_2x5"}}
    with pytest.raises(ValidationError):
        make_config(bays=[bad_layout])


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


def test_timelapse_command_writes_timestamped_jpegs():
    cmd = build_timelapse_cmd("rtsp://u:p@h/x", "/data/tl", "cam01", "main", every_s=10)
    joined = " ".join(cmd)
    assert "-vf fps=1/10" in joined and "-f image2 -strftime 1" in joined
    assert cmd[-1].endswith("cam01_main_%Y%m%dT%H%M%S.jpg")
    with pytest.raises(ValueError):
        build_timelapse_cmd("rtsp://u:p@h/x", "/data/tl", "cam01", "main", every_s=1)


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


VISITS = """visit_id,camera_id,bay_id,arrived,loading_start,tarp_start,tarp_done,departed,product,layout,coil_count,truck_ref,notes
g1,cam01,left,2026-10-31T10:00:00,,,,2026-10-31T10:45:00,wire_rod,linear,10,,
g2,cam01,left,2026-10-31T12:00:00,,,,2026-10-31T12:45:00,wire_rod,criss_cross,14,,
g3,cam01,left,2026-10-31T15:00:00,,,,2026-10-31T15:45:00,wire_rod,linear,10,,
g4,cam01,left,2026-10-31T16:30:00,,,,2026-10-31T17:15:00,ingot,,,,
"""


def _decision(
    layout: Layout, observed: int, inferred: int | None, review=False, product=Product.WIRE_ROD
) -> Decision:
    counts = CountSummary(observed, 0, 0, inferred, "b")
    return Decision("final", 0.0, layout, 0.9, 1.0, 6, counts, review, product)


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
    # g4: ingot truck correctly kept out of the wire-rod checks.
    sink.visit_opened("cam01", "left", "p4", ts("16:31"), {})
    sink.decision(
        "cam01", "left", "p4", _decision(Layout.UNKNOWN, 0, None, product=Product.OTHER), None
    )

    report = evaluate(load_predictions(tmp_path / "events.jsonl"), gts)
    assert report["visits_matched"] == 3
    assert report["visits_missed"] == 1
    assert report["product_accuracy"] == 1.0
    assert report["product_confusion"] == {"other -> other": 1, "wire_rod -> wire_rod": 2}
    assert report["visits_spurious"] == 1
    assert report["layout_accuracy"] == 0.5
    assert report["layout_confusion"] == {"criss_cross -> linear": 1, "linear -> linear": 1}
    assert report["count_exact_rate"] == 1.0
    assert report["observed_le_true_rate"] == 1.0
    assert report["needs_review_rate"] == round(1 / 3, 4)
    assert "| visit_recall |" in to_markdown(report)


def test_classification_metrics_confusion_and_recall():
    truth = ["wire_rod"] * 4 + ["other_load"] * 3 + ["covered"] * 2
    pred = [
        "wire_rod",
        "wire_rod",
        "wire_rod",
        "other_load",
        "other_load",
        "other_load",
        "wire_rod",
        "covered",
        "covered",
    ]
    m = classification_metrics(truth, pred, ["covered", "other_load", "wire_rod"])
    assert m["accuracy"] == round(7 / 9, 4)
    assert m["confusion"]["other_load"]["wire_rod"] == 1  # an ingot truck called wire rods
    assert m["per_class"]["wire_rod"] == {"support": 4, "precision": 0.75, "recall": 0.75}
    assert m["per_class"]["covered"]["recall"] == 1.0


def test_env_file_loader_never_overrides_real_environment(tmp_path, monkeypatch):
    from watchinu.cli import load_env_file

    env = tmp_path / ".env"
    env.write_text(
        '# comment\nexport A_KEY="quoted value"\nB_KEY=plain\nC_KEY=from-file\n\nbad line\n'
    )
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.delenv("B_KEY", raising=False)
    monkeypatch.setenv("C_KEY", "from-env")
    assert load_env_file(env) == 2
    import os

    assert (os.environ["A_KEY"], os.environ["B_KEY"], os.environ["C_KEY"]) == (
        "quoted value",
        "plain",
        "from-env",
    )
