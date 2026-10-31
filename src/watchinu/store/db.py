"""Relational schema (PostgreSQL in production, SQLite for development and tests).

Design rules:
* ``decisions`` is append-only. A correction is a new row that supersedes the old one;
  a human review is a separate ``reviews`` row. Nothing is ever UPDATEd in place, so the
  full history survives disputes with vendors and customers.
* Images never go into the database: ``evidence`` stores a path/URI plus SHA-256.
* Results reach other systems (ERP, alerts) via the transactional ``outbox``: the
  decision and its outbox row commit together, and a worker delivers with retries.
* All timestamps are timezone-aware UTC.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(UTC)


def from_epoch(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=UTC)


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSON, list[str]: JSON}


class Visit(Base):
    __tablename__ = "visits"
    __table_args__ = (UniqueConstraint("camera_id", "bay_id", "opened_at", name="uq_visit_start"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    bay_id: Mapped[str] = mapped_column(String(64))
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    close_reason: Mapped[str | None] = mapped_column(String(64))
    arrived_covered: Mapped[bool] = mapped_column(Boolean, default=False)
    truck_ref: Mapped[str | None] = mapped_column(String(64))  # from ERP / RFID later
    dispatch_ref: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class DecisionRow(Base):
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    visit_id: Mapped[str] = mapped_column(ForeignKey("visits.id"), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # provisional | final
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    layout: Mapped[str] = mapped_column(String(32))
    layout_confidence: Mapped[float] = mapped_column(Float)
    layout_agreement: Mapped[float] = mapped_column(Float)
    n_layout_votes: Mapped[int] = mapped_column(Integer)
    observed_count: Mapped[int] = mapped_column(Integer)
    empty_slots: Mapped[int] = mapped_column(Integer)
    unseen_slots: Mapped[int] = mapped_column(Integer)
    inferred_total: Mapped[int | None] = mapped_column(Integer)
    inference_basis: Mapped[str] = mapped_column(Text)
    needs_review: Mapped[bool] = mapped_column(Boolean)
    reasons: Mapped[list[str]] = mapped_column(JSON)
    model_versions: Mapped[dict[str, Any]] = mapped_column(JSON)
    supersedes_id: Mapped[int | None] = mapped_column(ForeignKey("decisions.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    visit_id: Mapped[str] = mapped_column(ForeignKey("visits.id"), index=True)
    decision_id: Mapped[int | None] = mapped_column(ForeignKey("decisions.id"))
    kind: Mapped[str] = mapped_column(String(32))
    uri: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Review(Base):
    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    visit_id: Mapped[str] = mapped_column(ForeignKey("visits.id"), index=True)
    decision_id: Mapped[int | None] = mapped_column(ForeignKey("decisions.id"))
    reviewer: Mapped[str] = mapped_column(String(128))
    verdict: Mapped[str] = mapped_column(String(32))  # confirmed | corrected | unclear
    corrected_layout: Mapped[str | None] = mapped_column(String(32))
    corrected_count: Mapped[int | None] = mapped_column(Integer)
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(64))
    message: Mapped[str] = mapped_column(Text)
    raised_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class OutboxRow(Base):
    __tablename__ = "outbox"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)


class MachineState(Base):
    __tablename__ = "machine_state"

    camera_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    bay_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def make_engine(url: str) -> Engine:
    return create_engine(url, future=True, pool_pre_ping=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)
