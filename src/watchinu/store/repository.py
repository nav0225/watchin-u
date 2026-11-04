"""Data access for the runtime. Every public method is one transaction."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from watchinu.domain import Decision
from watchinu.store.db import (
    Alert,
    DecisionRow,
    Evidence,
    MachineState,
    OutboxRow,
    Visit,
    from_epoch,
    utcnow,
)


class Repository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sf = session_factory

    def open_visit(
        self, visit_id: str, camera_id: str, bay_id: str, ts: float, arrived_covered: bool
    ) -> None:
        with self._sf.begin() as s:
            if s.get(Visit, visit_id) is not None:
                return  # idempotent: replaying the same event is harmless
            s.add(
                Visit(
                    id=visit_id,
                    camera_id=camera_id,
                    bay_id=bay_id,
                    opened_at=from_epoch(ts),
                    arrived_covered=arrived_covered,
                )
            )

    def close_visit(self, visit_id: str, ts: float, reason: str) -> None:
        with self._sf.begin() as s:
            visit = s.get(Visit, visit_id)
            if visit is None or visit.closed_at is not None:
                return
            visit.closed_at = from_epoch(ts)
            visit.close_reason = reason
            s.add(
                OutboxRow(
                    topic="visit.closed",
                    payload={"visit_id": visit_id, "closed_at": ts, "reason": reason},
                )
            )

    def add_decision(
        self,
        visit_id: str,
        decision: Decision,
        evidence_uri: str | None = None,
        evidence_sha256: str | None = None,
    ) -> int:
        """Append a decision, superseding the visit's previous one, plus its outbox row."""
        with self._sf.begin() as s:
            previous = s.scalar(
                select(DecisionRow.id)
                .where(DecisionRow.visit_id == visit_id)
                .order_by(DecisionRow.id.desc())
                .limit(1)
            )
            c = decision.counts
            row = DecisionRow(
                visit_id=visit_id,
                kind=decision.kind,
                product=decision.product.value,
                decided_at=from_epoch(decision.ts),
                layout=decision.layout.value,
                layout_confidence=decision.layout_confidence,
                layout_agreement=decision.layout_agreement,
                n_layout_votes=decision.n_layout_votes,
                observed_count=c.observed,
                empty_slots=c.empty_slots,
                unseen_slots=c.unseen_slots,
                out_of_view_slots=c.out_of_view_slots,
                count_template=c.template,
                inferred_total=c.inferred_total,
                inference_basis=c.basis,
                needs_review=decision.needs_review,
                reasons=list(decision.reasons),
                model_versions=dict(decision.model_versions),
                supersedes_id=previous,
            )
            s.add(row)
            s.flush()
            if evidence_uri is not None and evidence_sha256 is not None:
                s.add(
                    Evidence(
                        visit_id=visit_id,
                        decision_id=row.id,
                        kind=decision.kind,
                        uri=evidence_uri,
                        sha256=evidence_sha256,
                    )
                )
            s.add(
                OutboxRow(
                    topic=f"decision.{decision.kind}",
                    payload={
                        "visit_id": visit_id,
                        "decision_id": row.id,
                        "product": decision.product.value,
                        "layout": decision.layout.value,
                        "observed_count": c.observed,
                        "inferred_total": c.inferred_total,
                        "needs_review": decision.needs_review,
                        "reasons": list(decision.reasons),
                    },
                )
            )
            return row.id

    def latest_decision(self, visit_id: str) -> DecisionRow | None:
        with self._sf() as s:
            return s.scalar(
                select(DecisionRow)
                .where(DecisionRow.visit_id == visit_id)
                .order_by(DecisionRow.id.desc())
                .limit(1)
            )

    def add_alert(self, camera_id: str, kind: str, message: str, ts: float) -> None:
        with self._sf.begin() as s:
            s.add(Alert(camera_id=camera_id, kind=kind, message=message, raised_at=from_epoch(ts)))
            s.add(
                OutboxRow(
                    topic=f"alert.{kind}",
                    payload={"camera_id": camera_id, "message": message, "ts": ts},
                )
            )

    def save_state(self, camera_id: str, bay_id: str, snapshot: dict[str, Any]) -> None:
        with self._sf.begin() as s:
            row = s.get(MachineState, (camera_id, bay_id))
            if row is None:
                s.add(MachineState(camera_id=camera_id, bay_id=bay_id, snapshot=snapshot))
            else:
                row.snapshot = snapshot
                row.updated_at = utcnow()

    def load_state(self, camera_id: str, bay_id: str) -> dict[str, Any] | None:
        with self._sf() as s:
            row = s.get(MachineState, (camera_id, bay_id))
            return dict(row.snapshot) if row is not None else None

    def pending_outbox(self, limit: int = 100) -> list[OutboxRow]:
        with self._sf() as s:
            return list(
                s.scalars(
                    select(OutboxRow)
                    .where(OutboxRow.sent_at.is_(None))
                    .order_by(OutboxRow.id)
                    .limit(limit)
                )
            )

    def mark_sent(self, outbox_id: int, error: str | None = None) -> None:
        with self._sf.begin() as s:
            row = s.get(OutboxRow, outbox_id)
            if row is None:
                return
            row.attempts += 1
            if error is None:
                row.sent_at = utcnow()
            else:
                row.last_error = error


class DbSink:
    """:class:`~watchinu.runtime.sinks.EventSink` backed by the repository."""

    def __init__(self, repo: Repository, evidence_dir: str | Path) -> None:
        self.repo = repo
        self.evidence_dir = Path(evidence_dir)

    def visit_opened(
        self, camera_id: str, bay_id: str, visit_id: str, ts: float, data: dict[str, Any]
    ) -> None:
        self.repo.open_visit(
            visit_id, camera_id, bay_id, ts, bool(data.get("arrived_covered", False))
        )

    def decision(
        self,
        camera_id: str,
        bay_id: str,
        visit_id: str,
        decision: Decision,
        evidence_jpeg: bytes | None,
    ) -> None:
        uri = sha = None
        if evidence_jpeg is not None:
            target = self.evidence_dir / visit_id / f"{decision.kind}_{int(decision.ts)}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(evidence_jpeg)
            uri, sha = str(target), hashlib.sha256(evidence_jpeg).hexdigest()
        self.repo.add_decision(visit_id, decision, uri, sha)

    def visit_closed(
        self, camera_id: str, bay_id: str, visit_id: str, ts: float, reason: str
    ) -> None:
        self.repo.close_visit(visit_id, ts, reason)

    def alert(self, camera_id: str, kind: str, message: str, ts: float) -> None:
        self.repo.add_alert(camera_id, kind, message, ts)

    def save_state(self, camera_id: str, bay_id: str, snapshot: dict[str, Any]) -> None:
        self.repo.save_state(camera_id, bay_id, snapshot)
