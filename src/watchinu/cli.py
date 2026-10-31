"""``watchinu`` command line: dataset tooling, training, replay, live run and evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
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
    from watchinu.dataset.capture import run_capture
    from watchinu.io.rtsp import build_rtsp_url

    cfg = load_config(a.config)
    user, password = cfg.camera.credentials()
    url = build_rtsp_url(
        cfg.camera.host, cfg.camera.rtsp_port, cfg.camera.streams[a.stream], user, password
    )
    run_capture(url, Path(a.out) / cfg.camera.id / a.stream, cfg.camera.id, a.stream, a.segment_s)
    return 0


def cmd_extract(a: argparse.Namespace) -> int:
    from watchinu.dataset.extract import extract_video
    from watchinu.dataset.manifest import read_manifest, write_manifest
    from watchinu.io.video import list_videos

    out = Path(a.out)
    manifest_path = out / "manifest.jsonl"
    records = read_manifest(manifest_path)
    done = {r.video for r in records}
    start = _parse_time(a.start_time, a.timezone)
    videos = list_videos(a.input)
    if start is not None and len(videos) > 1:
        log.error("--start-time only makes sense for a single video")
        return 2
    for video in videos:
        if str(video) in done:
            log.info("skip %s (already extracted)", video.name)
            continue
        records += extract_video(video, out, a.every_s, a.camera_id, a.stream, start)
        write_manifest(manifest_path, records)  # checkpoint after every video
    no_time = sum(1 for r in records if r.captured_at is None)
    if no_time:
        log.warning(
            "%d frames have no capture time: pass --start-time or use "
            "capture-style file names (..._YYYYMMDDTHHMMSS.mkv)",
            no_time,
        )
    log.info("manifest: %s (%d frames)", manifest_path, len(records))
    return 0


def cmd_dedupe(a: argparse.Namespace) -> int:
    from watchinu.dataset.dedupe import mark_duplicates
    from watchinu.dataset.manifest import read_manifest, write_manifest

    records = read_manifest(a.manifest)
    n = mark_duplicates(records, a.max_distance, a.window)
    write_manifest(a.manifest, records)
    log.info("%d of %d frames marked as near-duplicates", n, len(records))
    return 0


def cmd_visit_log_template(a: argparse.Namespace) -> int:
    from watchinu.dataset.visits import write_visit_log_template

    write_visit_log_template(a.out)
    log.info("wrote %s", a.out)
    return 0


def cmd_split(a: argparse.Namespace) -> int:
    from watchinu.dataset.manifest import read_manifest, write_manifest
    from watchinu.dataset.split import assign_splits, split_summary
    from watchinu.dataset.visits import assign_visits, load_visit_log

    records = read_manifest(a.manifest)
    visits = load_visit_log(a.visit_log, a.timezone)
    n = assign_visits(records, visits)
    frozen = set(Path(a.frozen_test).read_text().split()) if a.frozen_test else None
    assign_splits(records, visits, a.val, a.test, a.seed, frozen)
    write_manifest(a.manifest, records)
    log.info("%d frames inside visits", n)
    print(json.dumps(split_summary(records), indent=2))
    return 0


def cmd_export_detect(a: argparse.Namespace) -> int:
    from watchinu.config import load_config
    from watchinu.dataset.export import build_detection_dataset
    from watchinu.dataset.manifest import read_manifest

    records = read_manifest(a.manifest)
    poly = load_config(a.config).bay(a.bay).polygon if a.bay else None
    card = build_detection_dataset(
        records, Path(a.manifest).parent, a.labels, a.out, a.classes.split(","), poly
    )
    print(json.dumps(card, indent=2))
    return 0


def cmd_export_cls(a: argparse.Namespace) -> int:
    from watchinu.config import load_config
    from watchinu.dataset.export import build_classification_dataset
    from watchinu.dataset.manifest import read_manifest
    from watchinu.dataset.visits import load_visit_log

    records = read_manifest(a.manifest)
    visits = load_visit_log(a.visit_log, a.timezone)
    poly = load_config(a.config).bay(a.bay).polygon if a.bay else None
    card = build_classification_dataset(
        records, visits, Path(a.manifest).parent, a.out, a.task, poly, a.margin_s
    )
    print(json.dumps(card, indent=2))
    return 0


def cmd_prelabel(a: argparse.Namespace) -> int:
    from watchinu.dataset.prelabel import prelabel_to_cvat_zip
    from watchinu.models.ultralytics_backend import UltralyticsDetector

    label_map = dict(p.split("=", 1) for p in a.prompts.split(",")) if a.prompts else {}
    detector = UltralyticsDetector(
        a.weights,
        imgsz=a.imgsz,
        conf=a.min_conf,
        device=a.device,
        prompts=list(label_map) or None,
    )
    images = sorted(p for p in Path(a.images).rglob("*.jpg"))
    n = prelabel_to_cvat_zip(detector, images, a.classes.split(","), a.out, label_map, a.min_conf)
    log.info("%d boxes on %d images -> %s", n, len(images), a.out)
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
    from watchinu.dataset.visits import load_visit_log
    from watchinu.io.video import VideoFileSource, list_videos
    from watchinu.models.fake import VisitLogStageOracle
    from watchinu.runtime.pipeline import CameraProcessor
    from watchinu.runtime.sinks import JsonlSink

    cfg = load_config(a.config)
    oracle = None
    if a.stage_oracle:
        oracle = VisitLogStageOracle(
            load_visit_log(a.stage_oracle, cfg.camera.timezone), cfg.camera.id
        )
    models = _models(cfg, oracle)
    if models.stage is None:
        log.error(
            "no stage classifier configured (models.stage_classifier) and no "
            "--stage-oracle: visits can never open"
        )
        return 2
    out = Path(a.out)
    sink = JsonlSink(out / "events.jsonl", out / "evidence")
    camera = CameraProcessor(cfg, models, sink, _drift_checker(cfg))
    start = _parse_time(a.start_time, cfg.camera.timezone)
    for video in list_videos(a.video):
        source = VideoFileSource(video, every_s=a.every_s, start_time=start)
        if source.start_time is None:
            log.error("%s: unknown start time; pass --start-time", video.name)
            return 2
        for frame in source:
            if oracle is not None:
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


def cmd_render_config(a: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    from watchinu.config import load_config
    from watchinu.runtime.slots import slot_centres_uv
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
        centres = slot_centres_uv(cfg.template_for(bay))
        pts = BedHomography(quad).to_image(np.array(list(centres.values()), dtype=np.float32))
        draw_slots(
            image, {s: (float(x), float(y)) for s, (x, y) in zip(centres, pts, strict=True)}, set()
        )
    cv2.imwrite(a.out, image)
    log.info("wrote %s", a.out)
    return 0


def cmd_db_upgrade(a: argparse.Namespace) -> int:
    from alembic import command
    from alembic.config import Config

    command.upgrade(Config(str(REPO_ROOT / "alembic.ini")), "head")
    return 0


# -- parser -------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="watchinu", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
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

    s = add("extract", cmd_extract, "sample frames from footage into a manifest")
    s.add_argument("--input", required=True, help="video file or folder")
    s.add_argument("--out", default="data/frames")
    s.add_argument("--every-s", type=float, default=5.0)
    s.add_argument("--camera-id", default="cam01")
    s.add_argument("--stream", default="main")
    s.add_argument("--start-time", help="wall-clock start of a single video (ISO-8601)")
    add_tz(s)

    s = add("dedupe", cmd_dedupe, "mark near-duplicate frames in a manifest")
    s.add_argument("--manifest", required=True)
    s.add_argument("--max-distance", type=int, default=6)
    s.add_argument("--window", type=int, default=30)

    s = add("visit-log-template", cmd_visit_log_template, "write an empty visit-log CSV")
    s.add_argument("--out", default="visit_log.csv")

    s = add("split", cmd_split, "assign visits and leakage-safe train/val/test splits")
    s.add_argument("--manifest", required=True)
    s.add_argument("--visit-log", required=True)
    s.add_argument("--val", type=float, default=0.15)
    s.add_argument("--test", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--frozen-test", help="file with visit ids that must stay in test")
    add_tz(s)

    s = add("export-detect", cmd_export_detect, "build a YOLO detection dataset")
    s.add_argument("--manifest", required=True)
    s.add_argument("--labels", required=True, help="folder of YOLO .txt labels (CVAT export)")
    s.add_argument("--out", required=True)
    s.add_argument("--classes", default="coil,person")
    s.add_argument("--config")
    s.add_argument("--bay", help="crop to this bay (needs --config)")

    s = add("export-cls", cmd_export_cls, "build a stage/layout classification dataset")
    s.add_argument("--manifest", required=True)
    s.add_argument("--visit-log", required=True)
    s.add_argument("--task", choices=["stage", "layout"], required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--config")
    s.add_argument("--bay")
    s.add_argument("--margin-s", type=float, default=60.0)
    add_tz(s)

    s = add("prelabel", cmd_prelabel, "pre-label frames for CVAT with a detector")
    s.add_argument("--weights", required=True, help="e.g. yolov8s-worldv2.pt or our v0")
    s.add_argument("--images", required=True)
    s.add_argument("--out", required=True, help="output .zip for CVAT 'YOLO 1.1' import")
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
    s.add_argument("--video", required=True, help="file or folder")
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

    s = add("render-config", cmd_render_config, "draw configured geometry on a frame")
    s.add_argument("--config", required=True)
    s.add_argument("--image", required=True)
    s.add_argument("--out", default="config_check.png")

    add("db-upgrade", cmd_db_upgrade, "apply database migrations")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
