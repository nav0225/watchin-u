"""Lossless RTSP recorder: ffmpeg stream-copy into time-stamped segments, auto-restart.

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


def run_capture(
    url: str,
    out_dir: str | Path,
    camera_id: str,
    stream: str,
    segment_s: int = 300,
    max_backoff_s: float = 60.0,
    ffmpeg: str = "ffmpeg",
) -> None:
    """Record until interrupted, restarting ffmpeg with exponential backoff on failure."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = build_capture_cmd(url, out_dir, camera_id, stream, segment_s, ffmpeg)
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
