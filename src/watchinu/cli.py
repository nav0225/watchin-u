"""``watchinu`` command line: dataset tooling, training, replay, live run and evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from watchinu.config import AppConfig
from watchinu.models.base import Classifier, ModelSet
from watchinu.vision.drift import ViewDriftChecker

log = logging.getLogger("watchinu")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_time(text: str | None, tz: str) -> datetime | None:
    if not text:
        return None
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=ZoneInfo(tz))


# -- dataset ------------------------------------------------------------------------
def cmd_capture(a: argparse.Namespace) -> int:
    from watchinu.config import load_config
    from watchinu.dataset.capture import build_capture_cmd, build_timelapse_cmd, run_capture
    from watchinu.io.rtsp import build_rtsp_url

    cfg = load_config(a.config)
    user, password = cfg.camera.credentials()
    url = build_rtsp_url(
        cfg.camera.host, cfg.camera.rtsp_port, cfg.camera.streams[a.stream], user, password
    )
    out = Path(a.out) / cfg.camera.id / a.stream
    if a.mode == "timelapse":
        out = out / "timelapse"
        cmd = build_timelapse_cmd(url, out, cfg.camera.id, a.stream, a.every_s)
    else:
        cmd = build_capture_cmd(url, out, cfg.camera.id, a.stream, a.segment_s)
    run_capture(cmd, out)
    return 0


def cmd_extract(a: argparse.Namespace) -> int:
    from watchinu.dataset.extract import extract_video, index_images, list_images
    from watchinu.dataset.manifest import read_manifest, write_manifest
    from watchinu.io.video import list_videos, parse_start_time

    out = Path(a.out)
    manifest_path = out / "manifest.jsonl"
    records = read_manifest(manifest_path)
    done_videos = {r.video for r in records}
    done_images = {r.source_image for r in records if r.source_image}
    start = _parse_time(a.start_time, a.timezone)
    videos = list_videos(a.input)
    if start is not None and len(videos) > 1:
        log.error("--start-time only makes sense for a single video")
        return 2
    for video in videos:
        if str(video) in done_videos:
            log.info("skip %s (already extracted)", video.name)
            continue
        video_start = start
        if video_start is None and a.start_time_from_mtime and parse_start_time(video) is None:
            # Short clips without a time in their name: use the file time (kept by zips).
            video_start = datetime.fromtimestamp(video.stat().st_mtime, ZoneInfo(a.timezone))
        records += extract_video(video, out, a.every_s, a.camera_id, a.stream, video_start)
        write_manifest(manifest_path, records)  # checkpoint after every video
    images = [p for p in list_images(a.input) if str(p) not in done_images]
    images = [p for p in images if out.resolve() not in p.resolve().parents]
    if images:
        records += index_images(images, out, a.camera_id, a.stream, a.timezone)
        write_manifest(manifest_path, records)
    no_time = sum(1 for r in records if r.captured_at is None)
    if no_time:
        log.warning(
            "%d frames have no capture time: pass --start-time or use capture-style "
            "file names (..._YYYYMMDDTHHMMSS.mkv / .jpg)",
            no_time,
        )
    log.info("manifest: %s (%d frames)", manifest_path, len(records))
    return 0


def cmd_dedupe(a: argparse.Namespace) -> int:
    from watchinu.config import load_config
    from watchinu.dataset.dedupe import mark_duplicates
    from watchinu.dataset.manifest import read_manifest, write_manifest

    records = read_manifest(a.manifest)
    masks = load_config(a.config).camera.osd_masks if a.config else []
    n = mark_duplicates(
        records, Path(a.manifest).parent, a.min_change, a.min_gap_s, a.max_gap_s, masks
    )
    write_manifest(a.manifest, records)
    log.info("%d of %d frames marked as near-duplicates", n, len(records))
    return 0


def cmd_visit_log_template(a: argparse.Namespace) -> int:
    from watchinu.dataset.visits import write_visit_log_template

    write_visit_log_template(a.out)
    log.info("wrote %s", a.out)
    return 0


def cmd_visit_log_from_clips(a: argparse.Namespace) -> int:
    from watchinu.dataset.manifest import read_manifest
    from watchinu.dataset.visits import visit_log_from_clips

    n = visit_log_from_clips(read_manifest(a.manifest), a.bay, a.out, a.timezone)
    log.info("wrote %s with %d clip rows: fill in product, layout and coil_count", a.out, n)
    return 0


def cmd_contact_sheet(a: argparse.Namespace) -> int:
    from watchinu.dataset.contact_sheet import build_contact_sheets
    from watchinu.dataset.manifest import read_manifest

    records = read_manifest(a.manifest)
    sheets = build_contact_sheets(records, Path(a.manifest).parent, a.out, a.per_clip)
    for sheet in sheets:
        print(sheet)
    return 0


def cmd_split(a: argparse.Namespace) -> int:
    from watchinu.dataset.manifest import read_manifest, write_manifest
    from watchinu.dataset.split import assign_splits, split_summary
    from watchinu.dataset.visits import assign_visits, load_visit_log

    records = read_manifest(a.manifest)
    visits = load_visit_log(a.visit_log, a.timezone)
    n = assign_visits(records, visits)
    frozen = set(Path(a.frozen_test).read_text().split()) if a.frozen_test else None
    if a.group_by == "clip":
        log.warning(
            "--group-by clip: one truck can appear in several splits; val/test scores are "
            "optimistic (bootstrap only, never report them)"
        )
    assign_splits(records, visits, a.val, a.test, a.seed, frozen, a.group_by)
    write_manifest(a.manifest, records)
    log.info("%d frames inside visits", n)
    print(json.dumps(split_summary(records), indent=2))
    return 0


def cmd_export_detect(a: argparse.Namespace) -> int:
    from watchinu.config import load_config
    from watchinu.dataset.export import build_detection_dataset
    from watchinu.dataset.manifest import read_manifest

    records = read_manifest(a.manifest)
    cfg = load_config(a.config) if a.config else None
    poly = cfg.bay(a.bay).polygon if cfg and a.bay else None
    masks = cfg.camera.osd_masks if cfg else []
    card = build_detection_dataset(
        records,
        Path(a.manifest).parent,
        a.labels,
        a.out,
        a.classes.split(","),
        poly,
        osd_masks=masks,
    )
    print(json.dumps(card, indent=2))
    return 0


def cmd_export_cls(a: argparse.Namespace) -> int:
    from watchinu.config import load_config
    from watchinu.dataset.export import BayCrop, build_classification_dataset
    from watchinu.dataset.manifest import read_manifest
    from watchinu.dataset.visits import load_visit_log

    records = read_manifest(a.manifest)
    visits = load_visit_log(a.visit_log, a.timezone)
    bays: list[BayCrop] = [(None, None)]
    masks = []
    if a.config:
        cfg = load_config(a.config)
        chosen = cfg.bays if a.bay in (None, "all") else [cfg.bay(a.bay)]
        bays = [(b.id, b.polygon) for b in chosen]
        masks = cfg.camera.osd_masks
    card = build_classification_dataset(
        records,
        visits,
        Path(a.manifest).parent,
        a.out,
        a.task,
        bays,
        a.margin_s,
        osd_masks=masks,
    )
    print(json.dumps(card, indent=2))
    return 0


def cmd_prelabel(a: argparse.Namespace) -> int:
    from watchinu.dataset.prelabel import prelabel_to_cvat_zip, write_flat_images_zip
    from watchinu.models.base import Detector
    from watchinu.models.ultralytics_backend import UltralyticsDetector

    label_map: dict[str, str] = {}
    detector: Detector
    if a.method == "wrap-colour":
        from watchinu.models.prelabelers import (
            CompositeDetector,
            LabelFilter,
            WrapColourCoilProposer,
        )

        persons = UltralyticsDetector(a.person_weights, imgsz=a.imgsz, conf=0.3, device=a.device)
        detector = CompositeDetector(
            WrapColourCoilProposer(imgsz=a.imgsz, device=a.device),
            LabelFilter(persons, {"person": "person"}),
        )
    else:
        if not a.weights:
            log.error("--weights is required for --method weights")
            return 2
        label_map = dict(p.split("=", 1) for p in a.prompts.split(",")) if a.prompts else {}
        detector = UltralyticsDetector(
            a.weights,
            imgsz=a.imgsz,
            conf=a.min_conf,
            device=a.device,
            prompts=list(label_map) or None,
        )
    if a.manifest:
        from watchinu.dataset.manifest import read_manifest

        root = Path(a.manifest).parent
        records = read_manifest(a.manifest)
        images = [root / r.image for r in records if r.duplicate_of is None]
    elif a.images:
        images = sorted(p for p in Path(a.images).rglob("*.jpg"))
    else:
        log.error("pass --manifest (non-duplicate frames only) or --images")
        return 2
    n = prelabel_to_cvat_zip(detector, images, a.classes.split(","), a.out, label_map, a.min_conf)
    log.info("%d boxes on %d images -> %s", n, len(images), a.out)
    if a.images_zip:
        write_flat_images_zip(images, a.images_zip)
        log.info("images for the CVAT task -> %s", a.images_zip)
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    import yaml

    from watchinu.training import train

    extra = yaml.safe_load(Path(a.hyp).read_text()) if a.hyp else None
    run_dir = train(a.data, a.model, a.project, a.name, a.epochs, a.imgsz, a.batch, a.device, extra)
    log.info("run saved to %s", run_dir)
    return 0


# -- runtime ------------------------------------------------------------------------
def _models(cfg: AppConfig, stage_override: Classifier | None = None) -> ModelSet:
    from watchinu.models.ultralytics_backend import UltralyticsClassifier, UltralyticsDetector

    m = cfg.models
    stage: Classifier | None = stage_override
    if stage is None and m.stage_classifier:
        stage = UltralyticsClassifier(m.stage_classifier, m.stage_imgsz, m.device)
    return ModelSet(
        stage=stage,
        layout=UltralyticsClassifier(m.layout_classifier, m.layout_imgsz, m.device)
        if m.layout_classifier
        else None,
        detector=UltralyticsDetector(m.detector, m.detector_imgsz, device=m.device)
        if m.detector
        else None,
    )


def _drift_checker(cfg: AppConfig) -> ViewDriftChecker | None:
    import cv2

    from watchinu.vision.geometry import polygon_mask, to_pixels

    ref_path = cfg.camera.reference_image
    if not ref_path:
        log.warning("no reference_image configured: view-drift guard disabled")
        return None
    ref = cv2.imread(ref_path)
    if ref is None:
        raise OSError(f"cannot read reference image {ref_path}")
    h, w = ref.shape[:2]
    mask = (
        polygon_mask(to_pixels(cfg.camera.static_region, w, h), (h, w))
        if cfg.camera.static_region
        else None
    )
    return ViewDriftChecker(ref, mask, cfg.drift.min_inlier_ratio, cfg.drift.max_shift_px)


def cmd_replay(a: argparse.Namespace) -> int:
    from watchinu.config import load_config
    from watchinu.dataset.extract import list_images
    from watchinu.dataset.visits import load_visit_log
    from watchinu.domain import Frame
    from watchinu.io.video import ImageSequenceSource, VideoFileSource, list_videos
    from watchinu.models.fake import VisitLogStageOracle
    from watchinu.runtime.pipeline import CameraProcessor
    from watchinu.runtime.sinks import JsonlSink

    cfg = load_config(a.config)
    oracles: list[VisitLogStageOracle] = []
    models = _models(cfg)
    if a.stage_oracle:
        visits = load_visit_log(a.stage_oracle, cfg.camera.timezone)
        for bay in cfg.bays:  # ground-truth stages, answered per lane
            oracle = VisitLogStageOracle(visits, cfg.camera.id, bay.id)
            models.stage_by_bay[bay.id] = oracle
            oracles.append(oracle)
    if models.stage is None and not models.stage_by_bay:
        log.error(
            "no stage classifier configured (models.stage_classifier) and no "
            "--stage-oracle: visits can never open"
        )
        return 2
    out = Path(a.out)
    sink = JsonlSink(out / "events.jsonl", out / "evidence")
    camera = CameraProcessor(cfg, models, sink, _drift_checker(cfg))
    start = _parse_time(a.start_time, cfg.camera.timezone)
    sources: list[Iterable[Frame]] = []
    for video in list_videos(a.video):
        video_source = VideoFileSource(video, every_s=a.every_s, start_time=start)
        if video_source.start_time is None:
            log.error("%s: unknown start time; pass --start-time", video.name)
            return 2
        sources.append(video_source)
    images = list_images(a.video)
    if images:
        sources.append(ImageSequenceSource(images, cfg.camera.timezone))
    if not sources:
        log.error("no videos or time-lapse images under %s", a.video)
        return 2
    for source in sources:
        for frame in source:
            for oracle in oracles:
                oracle.set_time(frame.ts)
            camera.process(frame)
    log.info("events written to %s", out / "events.jsonl")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    from watchinu.config import load_config
    from watchinu.io.rtsp import RtspFrameSource, build_rtsp_url
    from watchinu.runtime.pipeline import CameraProcessor
    from watchinu.runtime.sinks import JsonlSink, MultiSink
    from watchinu.store.db import make_engine, make_session_factory
    from watchinu.store.repository import DbSink, Repository

    cfg = load_config(a.config)
    repo = Repository(make_session_factory(make_engine(cfg.storage.database_url())))
    sink = MultiSink(DbSink(repo, cfg.storage.evidence_dir), JsonlSink(Path(a.log)))
    camera = CameraProcessor(cfg, _models(cfg), sink, _drift_checker(cfg))
    for bay in camera.bays:
        snap = repo.load_state(cfg.camera.id, bay.bay.id)
        if snap:
            bay.restore(snap)
            log.info("bay %s: resumed state %s", bay.bay.id, snap.get("state"))
    user, password = cfg.camera.credentials()
    url = build_rtsp_url(
        cfg.camera.host, cfg.camera.rtsp_port, cfg.camera.streams["main"], user, password
    )
    with RtspFrameSource(url, "main") as source:
        for frame in source.frames():
            if frame is None:
                log.warning(
                    "no frames from camera (reconnects so far: %d)", source.stats.reconnects
                )
                continue
            camera.process(frame)
    return 0


def cmd_eval(a: argparse.Namespace) -> int:
    from watchinu.dataset.visits import load_visit_log
    from watchinu.eval.visit_metrics import evaluate, load_predictions, to_markdown

    report = evaluate(
        load_predictions(a.predictions), load_visit_log(a.visit_log, a.timezone), a.tolerance_s
    )
    md = to_markdown(report)
    if a.out:
        Path(a.out).write_text(md)
    print(md)
    return 0


def cmd_evaluate_model(a: argparse.Namespace) -> int:
    from watchinu.eval.model_report import evaluate_classifier, evaluate_detector, to_markdown

    if a.task == "detect":
        report = evaluate_detector(a.weights, a.data, a.split, a.imgsz, a.device)
    else:
        report = evaluate_classifier(a.weights, a.data, a.split, a.imgsz, a.device)
    md = to_markdown(report)
    out = Path(a.out) if a.out else Path(a.weights).parent.parent / f"report_{a.split}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    out.with_suffix(".md").write_text(md)
    print(md)
    log.info("report saved to %s.{json,md}", out)
    return 0


def cmd_render_config(a: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    from watchinu.config import load_config
    from watchinu.runtime.slots import slot_image_points
    from watchinu.vision.geometry import BedHomography, to_pixels
    from watchinu.vision.overlay import draw_polygon, draw_slots

    cfg = load_config(a.config)
    image = cv2.imread(a.image)
    if image is None:
        log.error("cannot read %s", a.image)
        return 2
    h, w = image.shape[:2]
    shade = image.copy()
    for poly in cfg.camera.osd_masks:
        cv2.fillPoly(shade, [np.round(to_pixels(poly, w, h)).astype(np.int32)], (0, 0, 255))
    image = cv2.addWeighted(shade, 0.4, image, 0.6, 0)
    if cfg.camera.static_region:
        draw_polygon(image, to_pixels(cfg.camera.static_region, w, h), (255, 0, 0))
    for bay in cfg.bays:
        draw_polygon(image, to_pixels(bay.polygon, w, h), (0, 255, 255))
        quad = to_pixels(bay.bed_quad, w, h)
        draw_polygon(image, quad, (255, 255, 0))
        for name, (x, y) in zip(("NL", "NR", "FR", "FL"), quad, strict=True):
            cv2.putText(
                image,
                name,
                (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                2,
            )
        templates = cfg.templates_for(bay)
        if a.layout not in templates:
            log.error("bay %s has no slot template for layout %s", bay.id, a.layout)
            return 2
        draw_slots(image, slot_image_points(BedHomography(quad), templates[a.layout][1]), set())
    cv2.imwrite(a.out, image)
    log.info("wrote %s", a.out)
    return 0


def cmd_db_check(a: argparse.Namespace) -> int:
    from watchinu.config import load_config
    from watchinu.store.health import check_database

    url = load_config(a.config).storage.database_url()
    checks = check_database(url, REPO_ROOT / "alembic.ini")
    for c in checks:
        print(f"[{'OK' if c.ok else 'FAIL'}] {c.name}: {c.detail}")
    return 0 if all(c.ok for c in checks) else 1


def load_env_file(path: Path) -> int:
    """Minimal ``.env`` reader (KEY=VALUE lines); never overrides the real environment."""
    n = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            n += 1
    return n


def cmd_db_upgrade(a: argparse.Namespace) -> int:
    from alembic import command
    from alembic.config import Config

    command.upgrade(Config(str(REPO_ROOT / "alembic.ini")), "head")
    return 0


# -- parser -------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="watchinu", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--env-file", default=".env", help="KEY=VALUE file loaded if present")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(
        name: str, fn: Callable[[argparse.Namespace], int], help_: str
    ) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(fn=fn)
        return sp

    def add_tz(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--timezone", default="Asia/Kolkata", help="timezone for naive timestamps")

    s = add("capture", cmd_capture, "record the camera losslessly into time-stamped segments")
    s.add_argument("--config", required=True)
    s.add_argument("--stream", choices=["main", "sub"], default="main")
    s.add_argument("--out", default="data/raw")
    s.add_argument("--segment-s", type=int, default=300)
    s.add_argument(
        "--mode",
        choices=["video", "timelapse"],
        default="video",
        help="video: lossless segments; timelapse: one JPEG every --every-s "
        "(whole visits at ~1/6-1/12 of the video storage)",
    )
    s.add_argument("--every-s", type=float, default=10.0, help="time-lapse interval (s)")

    s = add("extract", cmd_extract, "sample frames from footage into a manifest")
    s.add_argument(
        "--input", required=True, help="video file, or folder of videos / time-lapse JPEGs"
    )
    s.add_argument("--out", default="data/frames")
    s.add_argument("--every-s", type=float, default=5.0)
    s.add_argument("--camera-id", default="cam01")
    s.add_argument("--stream", default="main")
    s.add_argument("--start-time", help="wall-clock start of a single video (ISO-8601)")
    s.add_argument(
        "--start-time-from-mtime",
        action="store_true",
        help="for clips with no time in their name: use the file modification time",
    )
    add_tz(s)

    s = add("dedupe", cmd_dedupe, "mark near-duplicate frames in a manifest")
    s.add_argument("--manifest", required=True)
    s.add_argument("--config", help="camera config: blank its OSD overlays before comparing")
    s.add_argument("--min-change", type=float, default=0.03, help="per-region change to keep")
    s.add_argument("--min-gap-s", type=float, default=3.0)
    s.add_argument("--max-gap-s", type=float, default=10.0, help="keep at least this often")

    s = add("visit-log-template", cmd_visit_log_template, "write an empty visit-log CSV")
    s.add_argument("--out", default="visit_log.csv")

    s = add("visit-log-from-clips", cmd_visit_log_from_clips, "one visit-log row per clip")
    s.add_argument("--manifest", required=True)
    s.add_argument("--bay", default="left")
    s.add_argument("--out", default="visit_log.csv")
    add_tz(s)

    s = add("contact-sheet", cmd_contact_sheet, "thumbnail grids per clip for quick review")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", default="contact_sheets")
    s.add_argument("--per-clip", type=int, default=4)

    s = add("split", cmd_split, "assign visits and leakage-safe train/val/test splits")
    s.add_argument("--manifest", required=True)
    s.add_argument("--visit-log", required=True)
    s.add_argument("--val", type=float, default=0.15)
    s.add_argument("--test", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--frozen-test", help="file with visit ids that must stay in test")
    s.add_argument(
        "--group-by",
        choices=["visit", "clip"],
        default="visit",
        help="visit: leakage-safe (default); clip: bootstrap when a session is one group",
    )
    add_tz(s)

    s = add("export-detect", cmd_export_detect, "build a YOLO detection dataset")
    s.add_argument("--manifest", required=True)
    s.add_argument("--labels", required=True, help="folder of YOLO .txt labels (CVAT export)")
    s.add_argument("--out", required=True)
    s.add_argument("--classes", default="coil,person")
    s.add_argument("--config", help="camera config: blank its OSD overlays (recommended)")
    s.add_argument("--bay", help="experimental: crop to one bay (runtime uses full frames)")

    s = add("export-cls", cmd_export_cls, "build a stage/layout classification dataset")
    s.add_argument("--manifest", required=True)
    s.add_argument("--visit-log", required=True)
    s.add_argument("--task", choices=["stage", "layout"], required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--config", help="crop per bay with lane-specific labels (recommended)")
    s.add_argument("--bay", default="all", help="one bay id, or 'all' (default)")
    s.add_argument("--margin-s", type=float, default=60.0)
    add_tz(s)

    s = add("prelabel", cmd_prelabel, "pre-label frames for CVAT with a detector")
    s.add_argument(
        "--method",
        choices=["wrap-colour", "weights"],
        default="weights",
        help="wrap-colour: zero-label bootstrap (YOLOE proposals + BALCO wrap colour test); "
        "weights: a trained detector (our v0+) or an open-vocabulary model with --prompts",
    )
    s.add_argument("--weights", help="detector weights for --method weights")
    s.add_argument("--person-weights", default="yolo26s.pt", help="COCO model for workers")
    s.add_argument("--manifest", help="use the manifest's non-duplicate frames (recommended)")
    s.add_argument("--images", help="or: a folder of .jpg frames")
    s.add_argument("--out", required=True, help="output .zip for CVAT 'YOLO 1.1' import")
    s.add_argument("--images-zip", help="also write a flat zip of the images for the CVAT task")
    s.add_argument("--classes", default="coil,person")
    s.add_argument("--prompts", help='open-vocab prompts: "wrapped wire coil=coil,person=person"')
    s.add_argument("--imgsz", type=int, default=1280)
    s.add_argument("--min-conf", type=float, default=0.15)
    s.add_argument("--device", default="cpu")

    s = add("train", cmd_train, "fine-tune a pretrained model (records provenance)")
    s.add_argument("--data", required=True, help="data.yaml (detect) or folder (classify)")
    s.add_argument("--model", required=True, help="e.g. yolo26s.pt or yolo26s-cls.pt")
    s.add_argument("--project", default="runs")
    s.add_argument("--name")
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--imgsz", type=int, default=1280)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--device", default="0")
    s.add_argument("--hyp", help="YAML of extra Ultralytics args (configs/train/*.yaml)")

    s = add("replay", cmd_replay, "run the full pipeline on recorded footage")
    s.add_argument("--config", required=True)
    s.add_argument(
        "--video", required=True, help="video file, or folder of videos / time-lapse JPEGs"
    )
    s.add_argument("--out", required=True)
    s.add_argument("--every-s", type=float, default=1.0)
    s.add_argument("--start-time")
    s.add_argument("--stage-oracle", help="visit log: use ground-truth stages")

    s = add("run", cmd_run, "run live on the camera (database + JSONL log)")
    s.add_argument("--config", required=True)
    s.add_argument("--log", default="data/events.jsonl")

    s = add("eval", cmd_eval, "visit-level evaluation of replay output vs the visit log")
    s.add_argument("--predictions", required=True, help="events.jsonl from replay/run")
    s.add_argument("--visit-log", required=True)
    s.add_argument("--tolerance-s", type=float, default=300.0)
    s.add_argument("--out")
    add_tz(s)

    s = add("evaluate-model", cmd_evaluate_model, "score trained weights on a held-out split")
    s.add_argument("--task", choices=["detect", "classify"], required=True)
    s.add_argument("--weights", required=True)
    s.add_argument("--data", required=True, help="data.yaml (detect) or dataset folder (classify)")
    s.add_argument("--split", default="test")
    s.add_argument("--imgsz", type=int, default=1280)
    s.add_argument("--device", default="cpu")
    s.add_argument("--out", help="report path without suffix (default: next to the run)")

    s = add("render-config", cmd_render_config, "draw configured geometry on a frame")
    s.add_argument("--config", required=True)
    s.add_argument("--image", required=True)
    s.add_argument("--out", default="config_check.png")
    s.add_argument("--layout", default="linear", help="which slot template to draw")

    add("db-upgrade", cmd_db_upgrade, "apply database migrations")
    s = add("db-check", cmd_db_check, "smoke-test the database (connect, schema, round trip)")
    s.add_argument("--config", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env_file = Path(args.env_file)
    if env_file.is_file():
        load_env_file(env_file)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
