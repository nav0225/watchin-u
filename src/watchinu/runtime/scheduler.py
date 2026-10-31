"""Decides when the heavy tier (detector + layout classifier) runs during a visit.

The heavy model sleeps most of the ~45-minute visit. It wakes:
* densely during the first ``provisional_after_s`` (provisional decision at ~2 min);
* once after each loading burst settles (change score spikes, then calms down);
* on a slow periodic checkpoint;
* on cover detection, over the look-back buffer (handled by the pipeline).
"""

from __future__ import annotations

from watchinu.config import ScheduleConfig


class HeavyScheduler:
    def __init__(self, cfg: ScheduleConfig, provisional_after_s: float) -> None:
        self.cfg = cfg
        self.provisional_after_s = provisional_after_s
        self.reset()

    def reset(self) -> None:
        self.last_run: float | None = None
        self._spiked = False

    def should_run(self, ts: float, opened_at: float, change: float) -> bool:
        since_last = float("inf") if self.last_run is None else ts - self.last_run
        if ts - opened_at <= self.provisional_after_s:
            return since_last >= self.cfg.provisional_interval_s
        if change >= self.cfg.change_spike:
            self._spiked = True
            return False
        if self._spiked and change <= self.cfg.change_settle:
            self._spiked = False
            return True
        return since_last >= self.cfg.checkpoint_interval_s

    def mark_run(self, ts: float) -> None:
        self.last_run = ts
