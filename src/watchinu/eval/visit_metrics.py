"""Visit-level evaluation against the visit log — the metric the business cares about.

Detector mAP (from ``yolo val``) says how well boxes are drawn; this says whether each
truck got the right arrangement and count, and how many trucks were missed entirely.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from watchinu.dataset.visits import VisitLabel


@dataclass(slots=True)
class PredictedVisit:
    visit_id: str
    camera_id: str
    bay_id: str
    opened_ts: float
    closed_ts: float | None = None
    layout: str = "unknown"
    observed: int | None = None
    inferred_total: int | None = None
    needs_review: bool = True
    reasons: list[str] = field(default_factory=list)


def load_predictions(jsonl_path: str | Path) -> list[PredictedVisit]:
    """Read the replay/runtime JSONL event log; the last final decision per visit wins."""
    visits: dict[str, PredictedVisit] = {}
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            vid = rec.get("visit_id")
            if rec["type"] == "opened":
                visits[vid] = PredictedVisit(vid, rec["camera"], rec["bay"], rec["ts"])
            elif rec["type"] == "decision" and vid in visits:
                d = rec["decision"]
                if d["kind"] != "final":
                    continue
                v = visits[vid]
                v.layout = d["layout"]
                v.observed = d["counts"]["observed"]
                v.inferred_total = d["counts"]["inferred_total"]
                v.needs_review = d["needs_review"]
                v.reasons = list(d["reasons"])
            elif rec["type"] == "closed" and vid in visits:
                visits[vid].closed_ts = rec["ts"]
    return list(visits.values())


def match_visits(
    preds: list[PredictedVisit], gts: list[VisitLabel], tolerance_s: float = 300.0
) -> tuple[list[tuple[VisitLabel, PredictedVisit | None]], list[PredictedVisit]]:
    """Greedy one-to-one match on camera, bay and opening time."""
    used: set[str] = set()
    pairs: list[tuple[VisitLabel, PredictedVisit | None]] = []
    for gt in sorted(gts, key=lambda g: g.arrived):
        lo = gt.arrived.timestamp() - tolerance_s
        hi = gt.departed.timestamp()
        cands = [
            p
            for p in preds
            if p.visit_id not in used
            and p.camera_id == gt.camera_id
            and p.bay_id == gt.bay_id
            and lo <= p.opened_ts <= hi
        ]
        best = min(cands, key=lambda p: abs(p.opened_ts - gt.arrived.timestamp()), default=None)
        if best is not None:
            used.add(best.visit_id)
        pairs.append((gt, best))
    spurious = [p for p in preds if p.visit_id not in used]
    return pairs, spurious


def evaluate(
    preds: list[PredictedVisit], gts: list[VisitLabel], tolerance_s: float = 300.0
) -> dict[str, Any]:
    pairs, spurious = match_visits(preds, gts, tolerance_s)
    matched = [(g, p) for g, p in pairs if p is not None]
    layout_pairs = [(g, p) for g, p in matched if g.layout is not None]
    confusion: Counter[str] = Counter(
        f"{g.layout.value if g.layout else '?'} -> {p.layout}" for g, p in layout_pairs
    )
    layout_correct = sum(1 for g, p in layout_pairs if g.layout and g.layout.value == p.layout)

    count_pairs = [(g, p) for g, p in matched if g.coil_count is not None]
    errors: list[int] = []
    exact = observed_le_true = 0
    for g, p in count_pairs:
        assert g.coil_count is not None
        predicted = p.inferred_total if p.inferred_total is not None else p.observed
        if predicted is None:
            continue
        errors.append(abs(predicted - g.coil_count))
        exact += int(predicted == g.coil_count)
        if p.observed is not None and p.observed <= g.coil_count:
            observed_le_true += 1

    def rate(n: int, d: int) -> float | None:
        return round(n / d, 4) if d else None

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "visits_ground_truth": len(gts),
        "visits_matched": len(matched),
        "visits_missed": len(gts) - len(matched),
        "visits_spurious": len(spurious),
        "visit_recall": rate(len(matched), len(gts)),
        "layout_evaluated": len(layout_pairs),
        "layout_accuracy": rate(layout_correct, len(layout_pairs)),
        "layout_unknown_rate": rate(
            sum(1 for _, p in matched if p.layout == "unknown"), len(matched)
        ),
        "layout_confusion": dict(sorted(confusion.items())),
        "count_evaluated": len(errors),
        "count_exact_rate": rate(exact, len(errors)),
        "count_mae": round(sum(errors) / len(errors), 3) if errors else None,
        "observed_le_true_rate": rate(observed_le_true, len(errors)),
        "needs_review_rate": rate(sum(1 for _, p in matched if p.needs_review), len(matched)),
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = ["# Visit-level evaluation", "", "| Metric | Value |", "|---|---|"]
    for key, value in report.items():
        if key == "layout_confusion":
            continue
        lines.append(f"| {key} | {value} |")
    lines += ["", "## Layout confusion (ground truth -> predicted)", ""]
    lines += [f"- {k}: {v}" for k, v in report.get("layout_confusion", {}).items()] or ["- none"]
    return "\n".join(lines) + "\n"
