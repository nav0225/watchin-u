"""RTSP recorders (ffmpeg, auto-restart): lossless video segments or a JPEG time-lapse.

Video (``build_capture_cmd``) keeps every frame; use it for short clips and NVR-like
evidence. Time-lapse (``build_timelapse_cmd``) keeps one high-quality JPEG every few
seconds: a full 40-minute visit at 10 s is 240 frames. Measured on the 2026-11-04 footage
(2560x1440, ``-q:v 2``): ~1.0 MB/frame, i.e. ~0.36 GB/h at 10 s versus ~2.5 GB/h of video
(~7x less) - and it is exactly what the stage/layout models and the trigger logic consume
(the runtime samples every 5-10 s).

* ``-c copy``: no re-encoding, so recorded data is bit-identical to what the live
  pipeline will decode (train/inference parity) and costs almost no CPU.
* Matroska segments survive an abrupt stop (an MP4 cut mid-write loses its index).
* Segment names carry the local start time (``cam01_main_20261031T141500.mkv``), which
  replay and frame extraction use as the time base. Set ``TZ`` to the plant timezone.
* Note: ffmpeg takes the RTSP URL (with password) on its command line, visible to other
  local users via ``ps``. Run the recorder on a dedicated, access-controlled machine.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from watchinu.io.rtsp import redact

log = logging.getLogger(__name__)


def build_capture_cmd(
    url: str,
    out_dir: str | Path,
    camera_id: str,
    stream: str,
    segment_s: int = 300,
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    pattern = str(Path(out_dir) / f"{camera_id}_{stream}_%Y%m%dT%H%M%S.mkv")
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-rtsp_transport",
        "tcp",
        "-timeout",
        "10000000",  # 10 s socket timeout -> ffmpeg exits and we reconnect
        "-i",
        url,
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-f",
        "segment",
        "-segment_time",
        str(segment_s),
        "-segment_atclocktime",
        "1",
        "-segment_format",
        "matroska",
        "-reset_timestamps",
        "1",
        "-strftime",
        "1",
        pattern,
    ]


def build_timelapse_cmd(
    url: str,
    out_dir: str | Path,
    camera_id: str,
    stream: str,
    every_s: float = 10.0,
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    """One JPEG (quality ~95) every ``every_s`` seconds, named by local wall-clock time."""
    if every_s < 2:
        raise ValueError("time-lapse interval must be >= 2 s (file names have 1 s resolution)")
    pattern = str(Path(out_dir) / f"{camera_id}_{stream}_%Y%m%dT%H%M%S.jpg")
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-rtsp_transport",
        "tcp",
        "-timeout",
        "10000000",
        "-i",
        url,
        "-map",
        "0:v:0",
        "-vf",
        f"fps=1/{every_s:g}",
        "-q:v",
        "2",
        "-f",
        "image2",
        "-strftime",
        "1",
        pattern,
    ]


def run_capture(cmd: list[str], out_dir: str | Path, max_backoff_s: float = 60.0) -> None:
    """Run a recorder command until interrupted, restarting it with exponential backoff."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    safe = [redact(c) for c in cmd]
    backoff = 1.0
    while True:
        started = time.monotonic()
        log.info("starting recorder: %s", " ".join(safe))
        proc = subprocess.Popen(cmd)
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            log.info("stopping recorder")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            return
        ran = time.monotonic() - started
        backoff = 1.0 if ran > 60 else min(backoff * 2, max_backoff_s)
        log.warning(
            "recorder exited with code %s after %.0fs; restart in %.0fs", code, ran, backoff
        )
        time.sleep(backoff)
