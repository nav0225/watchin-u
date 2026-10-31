from __future__ import annotations

import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from watchinu.dataset.dedupe import mark_duplicates
from watchinu.dataset.export import (
    build_classification_dataset,
    build_detection_dataset,
    crop_yolo_labels,
)
from watchinu.dataset.extract import extract_video
from watchinu.dataset.manifest import read_manifest, write_manifest
from watchinu.dataset.prelabel import prelabel_to_cvat_zip
from watchinu.dataset.split import assign_splits, split_summary
from watchinu.dataset.visits import (
    NOT_VISIBLE,
    assign_visits,
    frame_layout_label,
    frame_stage,
    load_visit_log,
    visit_groups,
)
from watchinu.domain import BBox, Detection, Stage
from watchinu.io.video import VideoFileSource, parse_start_time
from watchinu.models.fake import FunctionDetector

from .helpers import textured, write_video

IST = "+05:30"


def test_parse_start_time_from_capture_filename():
    t = parse_start_time("cam01_main_20261031T141500.mkv")
    assert t is not None and t.isoformat() == "2026-10-31T14:15:00+05:30"
    assert parse_start_time("random.mp4") is None


def test_video_source_samples_by_time(tmp_path):
    frames = [textured(40 + 5 * i) for i in range(30)]  # 3 s at 10 fps
    video = write_video(tmp_path / "cam01_main_20261031T100000.avi", frames, fps=10)
    got = list(VideoFileSource(video, every_s=1.0))
    assert [round(f.pts_s, 1) for f in got] == [0.0, 1.0, 2.0]
    start = datetime(2026, 10, 31, 4, 30, tzinfo=UTC)  # 10:00 IST
    assert got[1].ts == pytest.approx(start.timestamp() + 1.0, abs=0.05)


def _scene(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (18, 32, 3), dtype=np.uint8)
    return cv2.resize(small, (320, 180), interpolation=cv2.INTER_NEAREST)


def _extract(tmp_path: Path):
    # Two structurally different scenes, 2 s each (dHash ignores pure brightness changes).
    frames = [_scene(1) for _ in range(20)] + [_scene(2) for _ in range(20)]
    video = write_video(tmp_path / "cam01_main_20261031T100000.avi", frames, fps=10)
    out = tmp_path / "frames"
    records = extract_video(video, out, every_s=0.5, camera_id="cam01")
    write_manifest(out / "manifest.jsonl", records)
    return out, records


def test_extract_writes_images_and_manifest(tmp_path):
    out, records = _extract(tmp_path)
    assert len(records) == 8
    assert all((out / r.image).exists() for r in records)
    assert records[0].captured_at is not None
    assert read_manifest(out / "manifest.jsonl") == records


def test_dedupe_keeps_one_frame_per_static_scene(tmp_path):
    _, records = _extract(tmp_path)
    n = mark_duplicates(records)
    kept = [r for r in records if r.duplicate_of is None]
    assert n == len(records) - len(kept)
    assert len(kept) == 2  # one per scene


VISIT_CSV = """visit_id,camera_id,bay_id,arrived,loading_start,tarp_start,tarp_done,departed,layout,coil_count,truck_ref,notes
v1,cam01,left,2026-10-31T10:00:00,2026-10-31T10:05:00,2026-10-31T10:40:00,2026-10-31T10:50:00,2026-10-31T11:00:00,linear,10,CG04AB1234,
v2,cam01,right,2026-10-31T10:30:00,,,,2026-10-31T11:30:00,criss_cross,14,,overlaps v1
v3,cam01,left,2026-10-31T13:00:00,,,,2026-10-31T13:45:00,linear,10,,
v4,cam01,left,2026-11-01T09:00:00,,,,2026-11-01T09:45:00,criss_cross,,,
"""


def visits(tmp_path):
    path = tmp_path / "visits.csv"
    path.write_text(VISIT_CSV)
    return load_visit_log(path)


def test_visit_log_parsing_and_groups(tmp_path):
    vs = visits(tmp_path)
    assert vs[0].arrived.isoformat() == f"2026-10-31T10:00:00{IST}"
    assert vs[0].coil_count == 10 and vs[3].coil_count is None
    groups = visit_groups(vs)
    assert groups["v1"] == groups["v2"]  # overlapping on one camera -> same split group
    assert groups["v3"] != groups["v1"]


def test_stage_and_layout_labels_from_visit_log(tmp_path):
    vs = visits(tmp_path)
    m = timedelta(seconds=60)
    at = lambda hhmm: datetime.fromisoformat(f"2026-10-31T{hhmm}:00{IST}")  # noqa: E731
    assert frame_stage(at("09:00"), "cam01", vs, m) is Stage.NO_TRUCK
    assert frame_stage(at("10:00"), "cam01", vs, m) is None  # on a transition
    assert frame_stage(at("10:20"), "cam01", vs, m) is Stage.OPEN
    assert frame_stage(at("10:45"), "cam01", vs, m) is Stage.COVERING
    assert frame_stage(at("10:55"), "cam01", vs, m) is Stage.COVERED
    assert frame_layout_label(at("10:02"), "cam01", vs, m) == NOT_VISIBLE  # before loading
    assert frame_layout_label(at("10:20"), "cam01", vs, m) == "linear"
    assert frame_layout_label(at("10:55"), "cam01", vs, m) == NOT_VISIBLE  # tarped
    assert frame_layout_label(at("09:00"), "cam01", vs, m) is None


def _records_for(vs, tmp_path):
    from watchinu.dataset.manifest import FrameRecord

    recs = []
    times = [f"2026-10-31T{h}:00{IST}" for h in ("08:00", "10:10", "10:35", "13:10", "13:20")]
    times.append(f"2026-11-01T09:10:00{IST}")
    for i, t in enumerate(times):
        recs.append(
            FrameRecord(
                frame_id=f"f{i}",
                image=f"images/f{i}.jpg",
                video="v.mkv",
                camera_id="cam01",
                stream="main",
                pts_s=i,
                frame_index=i,
                captured_at=datetime.fromisoformat(t),
                width=320,
                height=180,
                blur_var=100,
                brightness=100,
                overexposed_fraction=0,
                dhash="0" * 16,
            )
        )
    return recs


def test_splits_are_grouped_deterministic_and_respect_frozen_test(tmp_path):
    vs = visits(tmp_path)
    recs = _records_for(vs, tmp_path)
    assert assign_visits(recs, vs) == 5
    assert recs[0].visit_id is None and recs[2].visit_id == "v1"
    a = assign_splits(recs, vs, val_frac=0.3, test_frac=0.3, seed=1, frozen_test={"v3"})
    split_of = {r.frame_id: r.split for r in recs}
    assert split_of["f3"] == split_of["f4"] == "test"  # frozen, and same visit together
    assert split_of["f1"] == split_of["f2"]  # v1/v2 overlap -> same group
    recs2 = _records_for(vs, tmp_path)
    assign_visits(recs2, vs)
    assert assign_splits(recs2, vs, val_frac=0.3, test_frac=0.3, seed=1, frozen_test={"v3"}) == a
    assert sum(v["frames"] for v in split_summary(recs).values()) == 6


def test_crop_yolo_labels_clips_and_drops():
    labels = [
        (0, 0.5, 0.5, 0.2, 0.2),  # fully inside the crop
        (0, 0.05, 0.5, 0.1, 0.1),  # entirely left of the crop -> dropped
        (1, 0.25, 0.5, 0.1, 0.2),  # half inside -> kept and clipped (min_visible 0.5)
    ]
    out = crop_yolo_labels(labels, 1000, 1000, (250, 0, 1000, 1000))
    assert len(out) == 2
    cls, cx, _, w, _ = out[0]
    assert (cls, round(cx, 4), round(w, 4)) == (0, round(250 / 750, 4), round(200 / 750, 4))
    assert out[1][0] == 1 and out[1][3] == pytest.approx(50 / 750)


def test_build_detection_dataset(tmp_path):
    out, records = _extract(tmp_path)
    for i, r in enumerate(records):
        r.split = "train" if i % 2 else "val"
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / f"{Path(records[0].image).stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (labels / f"{Path(records[1].image).stem}.txt").write_text("")  # verified negative
    card = build_detection_dataset(
        records,
        out,
        labels,
        tmp_path / "ds",
        ["coil", "person"],
        crop_polygon=[(0.25, 0), (1, 0), (1, 1), (0.25, 1)],
    )
    assert card["splits"] == {"val": {"images": 1, "coil": 1}, "train": {"images": 1}}
    img = cv2.imread(str(next((tmp_path / "ds/images/val").iterdir())))
    assert img.shape[1] == 240  # cropped to the bay
    assert (tmp_path / "ds/data.yaml").exists()


def test_build_classification_dataset(tmp_path):
    out, records = _extract(tmp_path)  # 10:00:00-10:00:04 IST, i.e. at v1 arrival
    vs = visits(tmp_path)
    for r in records:
        r.split = "train"
    card = build_classification_dataset(
        records, vs, out, tmp_path / "cls", "stage", boundary_margin_s=0
    )
    assert card["splits"]["train"] == {"open": 8}
    assert len(list((tmp_path / "cls/train/open").iterdir())) == 8
    # Near-transition frames are excluded with a margin (all 8 are within 60 s of arrival).
    card = build_classification_dataset(records, vs, out, tmp_path / "cls2", "stage")
    assert card["splits"] == {}
    # Duplicates are excluded by default.
    mark_duplicates(records)
    card = build_classification_dataset(
        records, vs, out, tmp_path / "cls3", "stage", boundary_margin_s=0
    )
    assert card["splits"]["train"] == {"open": 2}


def test_prelabel_writes_cvat_yolo_bundle(tmp_path):
    out, records = _extract(tmp_path)
    images = [out / r.image for r in records[:2]]
    det = FunctionDetector(
        lambda img: [
            Detection("wrapped wire coil", 0.8, BBox(32, 18, 96, 54)),
            Detection("wrapped wire coil", 0.05, BBox(0, 0, 10, 10)),  # below threshold
            Detection("truck", 0.9, BBox(0, 0, 320, 180)),  # not one of our classes
        ]
    )
    n = prelabel_to_cvat_zip(
        det,
        images,
        ["coil", "person"],
        tmp_path / "pre.zip",
        {"wrapped wire coil": "coil"},
        min_confidence=0.25,
    )
    assert n == 2
    with zipfile.ZipFile(tmp_path / "pre.zip") as zf:
        names = set(zf.namelist())
        assert {"obj.names", "obj.data", "train.txt"} <= names
        label = zf.read(f"obj_train_data/{images[0].stem}.txt").decode()
    assert label == "0 0.200000 0.200000 0.200000 0.200000\n"
