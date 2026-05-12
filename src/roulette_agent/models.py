"""SQLAlchemy 2.x ORM models for the session, spin, and reflection tables."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    ForeignKey,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    TIMESTAMP,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "session"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    wheel_type: Mapped[str] = mapped_column(String(20), nullable=False, default="american")
    bankroll_init: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    bet_unit: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    bankroll_now: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    excluded_dozens: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    initial_history: Mapped[list] = mapped_column(JSON, nullable=False)
    external_stats: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    external_stats_n_estimate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hyperparams: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")

    spins: Mapped[list[Spin]] = relationship(
        "Spin", back_populates="session", order_by="Spin.spin_index"
    )
    reflections: Mapped[list[Reflection]] = relationship(
        "Reflection", back_populates="session"
    )


class Spin(Base):
    __tablename__ = "spin"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("session.id"), nullable=False, index=True
    )
    spin_index: Mapped[int] = mapped_column(Integer, nullable=False)
    bets: Mapped[list | None] = mapped_column(JSON, nullable=True)
    bets_total: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)
    result_number: Mapped[int] = mapped_column(Integer, nullable=False)
    pnl: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    bankroll_after: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    bias_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped[Session] = relationship("Session", back_populates="spins")


class Reflection(Base):
    __tablename__ = "reflection"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("session.id"), nullable=False, index=True
    )
    at_spin_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    observation: Mapped[str | None] = mapped_column(Text, nullable=True)
    hyperparam_diff: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    notes_diff: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped[Session] = relationship("Session", back_populates="reflections")
