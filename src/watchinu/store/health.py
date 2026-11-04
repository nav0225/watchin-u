"""Database smoke test for a new server: connectivity, migrations, and a full write/read
round trip (timestamps with timezone, JSON columns) inside a transaction that is rolled
back, so nothing is left behind. Run it once after ``db-upgrade`` on every deployment."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.orm import Session

from watchinu.store.db import DecisionRow, OutboxRow, Visit, make_engine


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str = ""


def check_database(url: str, alembic_ini: str | Path) -> list[Check]:
    engine = make_engine(url)
    checks: list[Check] = []
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks.append(Check("connect", True, engine.dialect.name))
    except Exception as exc:
        return [Check("connect", False, f"{type(exc).__name__}: {exc}")]

    head = ScriptDirectory.from_config(Config(str(alembic_ini))).get_current_head()
    with engine.connect() as conn:
        current = MigrationContext.configure(conn).get_current_revision()
    checks.append(
        Check("migrations", current == head, f"database at {current}, code expects {head}")
    )
    if current != head:
        return checks

    now = datetime.now(UTC).replace(microsecond=0)
    vid = "dbcheck" + uuid.uuid4().hex[:20]
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            session = Session(bind=conn)
            session.add(Visit(id=vid, camera_id="dbcheck", bay_id="x", opened_at=now))
            session.flush()
            session.add(
                DecisionRow(
                    visit_id=vid,
                    kind="final",
                    decided_at=now,
                    layout="linear",
                    layout_confidence=1.0,
                    layout_agreement=1.0,
                    n_layout_votes=1,
                    observed_count=1,
                    empty_slots=0,
                    unseen_slots=0,
                    inference_basis="check",
                    needs_review=False,
                    reasons=["ok", "हिंदी"],
                    model_versions={"m": "1"},
                )
            )
            session.add(OutboxRow(topic="dbcheck", payload={"n": 1, "s": "x"}))
            session.flush()
            session.expire_all()
            back = session.get(Visit, vid)
            assert back is not None
            opened = back.opened_at
            aware = opened.tzinfo is not None
            same = (opened if aware else opened.replace(tzinfo=UTC)) == now
            checks.append(
                Check(
                    "timestamps",
                    same,
                    "timezone stored" if aware else "stored as naive UTC (SQLite only)",
                )
            )
            row = session.query(DecisionRow).filter_by(visit_id=vid).one()
            checks.append(
                Check(
                    "json_unicode",
                    row.reasons == ["ok", "हिंदी"] and row.model_versions == {"m": "1"},
                    "JSON and Unicode round trip",
                )
            )
            checks.append(Check("write_permissions", True, "insert/flush succeeded"))
        except Exception as exc:
            checks.append(Check("write_permissions", False, f"{type(exc).__name__}: {exc}"))
        finally:
            trans.rollback()
    return checks
