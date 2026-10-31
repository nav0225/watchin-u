"""Where pipeline output goes. The database sink lives in :mod:`watchinu.store`."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Protocol

from watchinu.domain import Decision


class EventSink(Protocol):
    def visit_opened(
        self, camera_id: str, bay_id: str, visit_id: str, ts: float, data: dict[str, Any]
    ) -> None: ...

    def decision(
        self,
        camera_id: str,
        bay_id: str,
        visit_id: str,
        decision: Decision,
        evidence_jpeg: bytes | None,
    ) -> None: ...

    def visit_closed(
        self, camera_id: str, bay_id: str, visit_id: str, ts: float, reason: str
    ) -> None: ...

    def alert(self, camera_id: str, kind: str, message: str, ts: float) -> None: ...

    def save_state(self, camera_id: str, bay_id: str, snapshot: dict[str, Any]) -> None: ...


def decision_to_dict(decision: Decision) -> dict[str, Any]:
    return dataclasses.asdict(decision)


class MemorySink:
    """Collects everything in memory (tests)."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.states: dict[tuple[str, str], dict[str, Any]] = {}

    def visit_opened(
        self, camera_id: str, bay_id: str, visit_id: str, ts: float, data: dict[str, Any]
    ) -> None:
        self.records.append(
            {"type": "opened", "bay": bay_id, "visit_id": visit_id, "ts": ts, **data}
        )

    def decision(
        self,
        camera_id: str,
        bay_id: str,
        visit_id: str,
        decision: Decision,
        evidence_jpeg: bytes | None,
    ) -> None:
        self.records.append(
            {
                "type": "decision",
                "bay": bay_id,
                "visit_id": visit_id,
                "decision": decision,
                "has_evidence": evidence_jpeg is not None,
            }
        )

    def visit_closed(
        self, camera_id: str, bay_id: str, visit_id: str, ts: float, reason: str
    ) -> None:
        self.records.append(
            {"type": "closed", "bay": bay_id, "visit_id": visit_id, "ts": ts, "reason": reason}
        )

    def alert(self, camera_id: str, kind: str, message: str, ts: float) -> None:
        self.records.append({"type": "alert", "kind": kind, "message": message, "ts": ts})

    def save_state(self, camera_id: str, bay_id: str, snapshot: dict[str, Any]) -> None:
        self.states[(camera_id, bay_id)] = snapshot

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r["type"] == kind]


class JsonlSink:
    """Append-only JSON-lines log plus evidence JPEGs on disk (replay / debugging)."""

    def __init__(self, path: str | Path, evidence_dir: str | Path | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_dir = Path(evidence_dir) if evidence_dir is not None else None

    def _write(self, record: dict[str, Any]) -> None:
        record = {"logged_at": time.time(), **record}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def visit_opened(
        self, camera_id: str, bay_id: str, visit_id: str, ts: float, data: dict[str, Any]
    ) -> None:
        self._write(
            {
                "type": "opened",
                "camera": camera_id,
                "bay": bay_id,
                "visit_id": visit_id,
                "ts": ts,
                **data,
            }
        )

    def decision(
        self,
        camera_id: str,
        bay_id: str,
        visit_id: str,
        decision: Decision,
        evidence_jpeg: bytes | None,
    ) -> None:
        evidence = None
        if evidence_jpeg is not None and self.evidence_dir is not None:
            target = self.evidence_dir / visit_id / f"{decision.kind}_{int(decision.ts)}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(evidence_jpeg)
            evidence = str(target)
        self._write(
            {
                "type": "decision",
                "camera": camera_id,
                "bay": bay_id,
                "visit_id": visit_id,
                "decision": decision_to_dict(decision),
                "evidence": evidence,
            }
        )

    def visit_closed(
        self, camera_id: str, bay_id: str, visit_id: str, ts: float, reason: str
    ) -> None:
        self._write(
            {
                "type": "closed",
                "camera": camera_id,
                "bay": bay_id,
                "visit_id": visit_id,
                "ts": ts,
                "reason": reason,
            }
        )

    def alert(self, camera_id: str, kind: str, message: str, ts: float) -> None:
        self._write(
            {"type": "alert", "camera": camera_id, "kind": kind, "message": message, "ts": ts}
        )

    def save_state(self, camera_id: str, bay_id: str, snapshot: dict[str, Any]) -> None:
        return None


class MultiSink:
    def __init__(self, *sinks: EventSink) -> None:
        self.sinks = sinks

    def visit_opened(
        self, camera_id: str, bay_id: str, visit_id: str, ts: float, data: dict[str, Any]
    ) -> None:
        for s in self.sinks:
            s.visit_opened(camera_id, bay_id, visit_id, ts, data)

    def decision(
        self,
        camera_id: str,
        bay_id: str,
        visit_id: str,
        decision: Decision,
        evidence_jpeg: bytes | None,
    ) -> None:
        for s in self.sinks:
            s.decision(camera_id, bay_id, visit_id, decision, evidence_jpeg)

    def visit_closed(
        self, camera_id: str, bay_id: str, visit_id: str, ts: float, reason: str
    ) -> None:
        for s in self.sinks:
            s.visit_closed(camera_id, bay_id, visit_id, ts, reason)

    def alert(self, camera_id: str, kind: str, message: str, ts: float) -> None:
        for s in self.sinks:
            s.alert(camera_id, kind, message, ts)

    def save_state(self, camera_id: str, bay_id: str, snapshot: dict[str, Any]) -> None:
        for s in self.sinks:
            s.save_state(camera_id, bay_id, snapshot)
