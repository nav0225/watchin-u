"""Turns a visit's per-frame observations into provisional and final decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from watchinu.config import DecisionConfig, SlotTemplate
from watchinu.domain import CountSummary, Decision, Layout
from watchinu.runtime.slots import Slot, SlotBoard

LAYOUT_CLASSES: tuple[Layout, ...] = (Layout.LINEAR, Layout.CRISS_CROSS, Layout.MIXED)


@dataclass(frozen=True, slots=True)
class LayoutVote:
    ts: float
    probs: dict[str, float]


class VisitAggregator:
    def __init__(
        self,
        cfg: DecisionConfig,
        template: SlotTemplate,
        template_name: str,
        model_versions: dict[str, str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.template_name = template_name
        self.board = SlotBoard(template, history=cfg.slot_history)
        self.model_versions = dict(model_versions or {})
        self._votes: list[LayoutVote] = []
        self._counts: list[tuple[float, int]] = []
        self.flags: list[str] = []

    @property
    def n_observations(self) -> int:
        return len(self._counts)

    def add_observation(
        self,
        ts: float,
        layout_probs: dict[str, float] | None,
        coil_count: int,
        occupied: set[Slot],
        visible: set[Slot],
    ) -> None:
        if layout_probs:
            self._votes.append(LayoutVote(ts, dict(layout_probs)))
        self._counts.append((ts, coil_count))
        self.board.update(occupied, visible)

    def flag(self, reason: str) -> None:
        if reason not in self.flags:
            self.flags.append(reason)

    def decide(self, kind: str, ts: float, before_ts: float | None = None) -> Decision:
        votes = [v for v in self._votes if before_ts is None or v.ts < before_ts]
        if kind == "final" and votes:
            # The final arrangement is what the bed looked like just before the tarp:
            # weigh only the recent window if it has enough votes.
            cutoff = votes[-1].ts - self.cfg.final_window_s
            recent = [v for v in votes if v.ts >= cutoff]
            if len(recent) >= self.cfg.min_layout_votes:
                votes = recent
        layout, confidence, agreement, n_votes = self._layout(votes)

        reasons = list(self.flags)
        if n_votes < self.cfg.min_layout_votes:
            reasons.append("too_few_layout_votes")
        if n_votes and agreement < self.cfg.min_layout_agreement:
            reasons.append("layout_votes_disagree")
        if n_votes and confidence < self.cfg.min_layout_confidence:
            reasons.append("low_layout_confidence")

        counts = self._counts_summary(layout, before_ts)
        return Decision(
            kind=kind,
            ts=ts,
            layout=layout,
            layout_confidence=confidence,
            layout_agreement=agreement,
            n_layout_votes=n_votes,
            counts=counts,
            needs_review=bool(reasons),
            reasons=reasons,
            model_versions=dict(self.model_versions),
        )

    # -- internals ------------------------------------------------------------------
    @staticmethod
    def _layout(votes: list[LayoutVote]) -> tuple[Layout, float, float, int]:
        names = [c.value for c in LAYOUT_CLASSES]
        rows = []
        for v in votes:
            vec = np.array([max(0.0, v.probs.get(n, 0.0)) for n in names], dtype=np.float64)
            # Frames where the classifier mostly says "not_visible" carry no vote.
            if vec.sum() < 0.5:
                continue
            rows.append(vec / vec.sum())
        if not rows:
            return Layout.UNKNOWN, 0.0, 0.0, 0
        mat = np.stack(rows)
        mean = mat.mean(axis=0)
        winner = int(np.argmax(mean))
        agreement = float((mat.argmax(axis=1) == winner).mean())
        return LAYOUT_CLASSES[winner], float(mean[winner]), agreement, len(rows)

    def _counts_summary(self, layout: Layout, before_ts: float | None) -> CountSummary:
        if layout is Layout.LINEAR:
            return self.board.summary(self.cfg.infer_unseen_slots, self.template_name)
        counts = sorted(
            (c for t, c in self._counts if before_ts is None or t < before_ts), reverse=True
        )
        # Second-highest per-frame count: robust to a single frame's false positive.
        robust = counts[1] if len(counts) > 1 else (counts[0] if counts else 0)
        return CountSummary(
            observed=robust,
            empty_slots=0,
            unseen_slots=0,
            inferred_total=None,
            basis=f"layout {layout.value}: no slot template; robust max per-frame count",
        )
