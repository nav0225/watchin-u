"""Turns a visit's per-frame observations into provisional and final decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from watchinu.config import DecisionConfig, SlotTemplate
from watchinu.domain import CountSummary, Decision, Layout, Product, Stage
from watchinu.runtime.slots import Slot, SlotBoard

LAYOUT_CLASSES: tuple[Layout, ...] = (Layout.LINEAR, Layout.CRISS_CROSS, Layout.MIXED)

SlotObservations = dict[Layout, tuple[set[Slot], set[Slot]]]  # layout -> (occupied, visible)


@dataclass(frozen=True, slots=True)
class LayoutVote:
    ts: float
    probs: dict[str, float]


@dataclass(frozen=True, slots=True)
class TemplateSpec:
    name: str
    template: SlotTemplate
    out_of_view: frozenset[Slot] = frozenset()


class VisitAggregator:
    def __init__(
        self,
        cfg: DecisionConfig,
        templates: dict[Layout, TemplateSpec],
        model_versions: dict[str, str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.templates = dict(templates)
        self.boards = {
            layout: SlotBoard(spec.template, cfg.slot_history, set(spec.out_of_view))
            for layout, spec in templates.items()
        }
        self.model_versions = dict(model_versions or {})
        self._votes: list[LayoutVote] = []
        self._counts: list[tuple[float, int]] = []
        self._stages: list[tuple[float, Stage]] = []
        self.flags: list[str] = []

    # -- inputs ---------------------------------------------------------------------
    def add_stage(self, ts: float, stage: Stage) -> None:
        """Light-tier vote on what is on the bed (only open-bed stages count)."""
        if stage.bed_open:
            self._stages.append((ts, stage))

    def add_observation(
        self,
        ts: float,
        layout_probs: dict[str, float] | None,
        coil_count: int,
        slots: SlotObservations,
    ) -> None:
        if layout_probs:
            self._votes.append(LayoutVote(ts, dict(layout_probs)))
        self._counts.append((ts, coil_count))
        for layout, (occupied, visible) in slots.items():
            if layout in self.boards:
                self.boards[layout].update(occupied, visible)

    def flag(self, reason: str) -> None:
        if reason not in self.flags:
            self.flags.append(reason)

    # -- queries --------------------------------------------------------------------
    @property
    def n_observations(self) -> int:
        return len(self._counts)

    @property
    def wire_rod_votes(self) -> int:
        return sum(1 for _, s in self._stages if s is Stage.WIRE_ROD)

    def product_confirmed(self) -> bool:
        """The light tier has seen wrapped wire rods often enough to wake the heavy tier."""
        return self.wire_rod_votes >= self.cfg.min_wire_rod_votes

    def leading_layout(self) -> Layout:
        return self._layout(self._votes)[0]

    # -- decisions ------------------------------------------------------------------
    def decide(self, kind: str, ts: float, before_ts: float | None = None) -> Decision:
        product, product_reasons = self._product(before_ts)
        if product is Product.OTHER:
            # Ingots / other products: recorded for the audit trail, nothing to verify.
            return Decision(
                kind=kind,
                ts=ts,
                layout=Layout.UNKNOWN,
                layout_confidence=0.0,
                layout_agreement=0.0,
                n_layout_votes=0,
                counts=CountSummary(0, 0, 0, None, "not a wire-rod load"),
                needs_review=bool(product_reasons),
                product=product,
                reasons=["not_wire_rod_load", *product_reasons],
                model_versions=dict(self.model_versions),
            )

        votes = [v for v in self._votes if before_ts is None or v.ts < before_ts]
        if kind == "final" and votes:
            # The final arrangement is what the bed looked like just before the tarp:
            # weigh only the recent window if it has enough votes.
            cutoff = votes[-1].ts - self.cfg.final_window_s
            recent = [v for v in votes if v.ts >= cutoff]
            if len(recent) >= self.cfg.min_layout_votes:
                votes = recent
        layout, confidence, agreement, n_votes = self._layout(votes)

        reasons = list(self.flags) + product_reasons
        if n_votes < self.cfg.min_layout_votes:
            reasons.append("too_few_layout_votes")
        if n_votes and agreement < self.cfg.min_layout_agreement:
            reasons.append("layout_votes_disagree")
        if n_votes and confidence < self.cfg.min_layout_confidence:
            reasons.append("low_layout_confidence")

        counts = self._counts_summary(layout, before_ts)
        if counts.out_of_view_slots:
            reasons.append("load_extends_beyond_camera_view")
        return Decision(
            kind=kind,
            ts=ts,
            layout=layout,
            layout_confidence=confidence,
            layout_agreement=agreement,
            n_layout_votes=n_votes,
            counts=counts,
            needs_review=any(r != "load_extends_beyond_camera_view" for r in reasons),
            product=product,
            reasons=reasons,
            model_versions=dict(self.model_versions),
        )

    # -- internals ------------------------------------------------------------------
    def _product(self, before_ts: float | None) -> tuple[Product, list[str]]:
        stages = [s for t, s in self._stages if before_ts is None or t < before_ts]
        if not stages:
            return Product.UNKNOWN, ["product_not_seen"]
        share = sum(1 for s in stages if s is Stage.WIRE_ROD) / len(stages)
        product = Product.WIRE_ROD if share >= 0.5 else Product.OTHER
        reasons = (
            ["product_uncertain"] if abs(share - 0.5) < self.cfg.product_uncertain_band else []
        )
        return product, reasons

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
        if layout in self.boards:
            return self.boards[layout].summary(
                self.cfg.infer_unseen_slots, self.templates[layout].name
            )
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
