from __future__ import annotations

from sqlalchemy import select

from watchinu.domain import CountSummary, Decision, Layout
from watchinu.store.db import Base, DecisionRow, make_engine, make_session_factory
from watchinu.store.repository import DbSink, Repository


def repo(tmp_path) -> Repository:
    engine = make_engine(f"sqlite:///{tmp_path / 't.db'}")
    Base.metadata.create_all(engine)
    return Repository(make_session_factory(engine))


def decision(kind: str, layout=Layout.LINEAR, observed=8) -> Decision:
    return Decision(
        kind,
        1_700_000_100.0,
        layout,
        0.9,
        1.0,
        6,
        CountSummary(observed, 0, 2, observed + 2, "basis"),
        False,
    )


def test_visit_lifecycle_append_only_decisions_and_outbox(tmp_path):
    r = repo(tmp_path)
    r.open_visit("v1", "cam01", "left", 1_700_000_000.0, False)
    r.open_visit("v1", "cam01", "left", 1_700_000_000.0, False)  # idempotent
    first = r.add_decision("v1", decision("provisional"))
    second = r.add_decision("v1", decision("final", observed=9))
    r.close_visit("v1", 1_700_003_000.0, "departed")

    latest = r.latest_decision("v1")
    assert latest is not None and latest.id == second
    assert latest.supersedes_id == first
    assert latest.inferred_total == 11
    with r._sf() as s:
        assert len(list(s.scalars(select(DecisionRow)))) == 2  # nothing overwritten

    topics = [o.topic for o in r.pending_outbox()]
    assert topics == ["decision.provisional", "decision.final", "visit.closed"]
    first_out = r.pending_outbox()[0]
    r.mark_sent(first_out.id, error="erp timeout")
    assert r.pending_outbox()[0].attempts == 1
    r.mark_sent(first_out.id)
    assert len(r.pending_outbox()) == 2


def test_machine_state_round_trip(tmp_path):
    r = repo(tmp_path)
    assert r.load_state("cam01", "left") is None
    r.save_state("cam01", "left", {"state": "open"})
    r.save_state("cam01", "left", {"state": "covered"})
    assert r.load_state("cam01", "left") == {"state": "covered"}


def test_db_sink_writes_evidence_with_hash(tmp_path):
    r = repo(tmp_path)
    sink = DbSink(r, tmp_path / "evidence")
    sink.visit_opened("cam01", "left", "v1", 1_700_000_000.0, {"arrived_covered": False})
    sink.decision("cam01", "left", "v1", decision("final"), b"\xff\xd8jpeg")
    files = list((tmp_path / "evidence" / "v1").iterdir())
    assert len(files) == 1 and files[0].read_bytes() == b"\xff\xd8jpeg"
