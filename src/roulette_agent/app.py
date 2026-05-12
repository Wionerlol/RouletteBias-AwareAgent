"""FastAPI application: POST /session/new, POST /session/{id}/spin, GET /session/{id}/state endpoints."""

from __future__ import annotations

import os
from typing import Any, Optional

import anthropic
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session as DbSession, sessionmaker

from roulette_agent.agent import RouletteAgent
from roulette_agent.bias_detector import detect_bias
from roulette_agent.models import Base, Session as SessionModel, Spin as SpinModel
from roulette_agent.settler import settle


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def _normalise_db_url(url: str) -> str:
    # Railway injects postgresql:// or postgres://, both route to psycopg2 by
    # default. Rewrite to psycopg3 driver so we don't need psycopg2 installed.
    url = url.replace("postgresql://", "postgresql+psycopg://")
    url = url.replace("postgres://", "postgresql+psycopg://")
    return url


def _make_engine():
    url = _normalise_db_url(os.environ.get("DATABASE_URL", "sqlite:///./dev.db"))
    kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # psycopg3: surface connection errors quickly instead of hanging
        kwargs["connect_args"] = {"connect_timeout": 10}
    return create_engine(url, **kwargs)


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    expected = os.environ.get("API_KEY", "")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class NewSessionRequest(BaseModel):
    wheel_type: str = "american"
    bankroll: float
    bet_unit: float
    excluded_dozens: list[int] = []
    recent_history: list[int]
    external_stats: Optional[dict[str, float]] = None
    external_stats_n_estimate: Optional[int] = None
    strategy_pref: str = "auto"

    @field_validator("recent_history")
    @classmethod
    def history_not_empty(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("recent_history must contain at least one spin")
        for n in v:
            if not 0 <= n <= 37:
                raise ValueError(f"result_number {n} out of range 0..37")
        return v

    @field_validator("wheel_type")
    @classmethod
    def valid_wheel_type(cls, v: str) -> str:
        if v not in ("american", "european"):
            raise ValueError("wheel_type must be 'american' or 'european'")
        return v

    @field_validator("excluded_dozens")
    @classmethod
    def valid_dozens(cls, v: list[int]) -> list[int]:
        for d in v:
            if d not in (1, 2, 3):
                raise ValueError(f"excluded_dozens elements must be 1, 2, or 3; got {d}")
        return v


class SpinRequest(BaseModel):
    result_number: int

    @field_validator("result_number")
    @classmethod
    def valid_result(cls, v: int) -> int:
        if not 0 <= v <= 37:
            raise ValueError(f"result_number {v} out of range 0..37")
        return v


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Roulette Bias-Aware Agent API")


# ---------------------------------------------------------------------------
# GET /health  (no auth — used by Railway health checks)
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "error"
    return {"status": "ok", "db": db_status}


def _anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


def _build_session_state(
    sess: SessionModel,
    recent_history: list[int],
) -> dict:
    return {
        "bankroll": float(sess.bankroll_now),
        "bet_unit": float(sess.bet_unit),
        "wheel_type": str(sess.wheel_type),
        "excluded_dozens": list(sess.excluded_dozens or []),
        "recent_history": recent_history,
        "external_stats": sess.external_stats,
        "external_stats_n_estimate": sess.external_stats_n_estimate,
        "hyperparams": dict(sess.hyperparams or {}),
        "notes": sess.notes or "",
    }


# ---------------------------------------------------------------------------
# POST /session/new
# ---------------------------------------------------------------------------

@app.post("/session/new")
def new_session(
    body: NewSessionRequest,
    _auth: None = Depends(verify_api_key),
    db: DbSession = Depends(get_db),
) -> dict:
    sess = SessionModel(
        wheel_type=body.wheel_type,
        bankroll_init=body.bankroll,
        bet_unit=body.bet_unit,
        bankroll_now=body.bankroll,
        excluded_dozens=body.excluded_dozens,
        initial_history=body.recent_history,
        external_stats=body.external_stats,
        external_stats_n_estimate=body.external_stats_n_estimate,
        hyperparams={},
        notes="",
    )
    db.add(sess)
    db.flush()  # assign id before using it

    bias_report = detect_bias(
        recent_history=body.recent_history,
        wheel_type=body.wheel_type,
        external_stats=body.external_stats,
        external_n_estimate=body.external_stats_n_estimate,
    )

    session_state = _build_session_state(sess, list(body.recent_history))
    agent = RouletteAgent(_anthropic_client(), model="claude-sonnet-4-6")
    decision = agent.decide(session_state)

    db.commit()
    db.refresh(sess)

    return {
        "session_id": sess.id,
        "spin_index": 0,
        "bankroll_now": float(sess.bankroll_now),
        "bias_report": bias_report,
        "next_strategy": decision["bets"],
        "rationale": decision["rationale"],
    }


# ---------------------------------------------------------------------------
# POST /session/{id}/spin
# ---------------------------------------------------------------------------

@app.post("/session/{session_id}/spin")
def spin(
    session_id: str,
    body: SpinRequest,
    _auth: None = Depends(verify_api_key),
    db: DbSession = Depends(get_db),
) -> dict:
    sess = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Load existing spins ordered by spin_index
    existing_spins = (
        db.query(SpinModel)
        .filter(SpinModel.session_id == session_id)
        .order_by(SpinModel.spin_index)
        .all()
    )

    # Rebuild recent_history from DB
    recent_history: list[int] = list(sess.initial_history) + [
        s.result_number for s in existing_spins
    ]

    # Settle the previous spin's bets against this result (if any spins exist)
    pnl_last: float = 0.0
    if existing_spins:
        last_spin = existing_spins[-1]
        last_bets = last_spin.bets or []
        if last_bets:
            outcome = settle(last_bets, body.result_number)
            pnl_last = float(outcome["pnl"])
        sess.bankroll_now = float(sess.bankroll_now) + pnl_last

    # Append current result to history (in memory only — DB rebuilds from spin rows)
    recent_history.append(body.result_number)

    # Detect bias on updated history
    bias_report = detect_bias(
        recent_history=recent_history,
        wheel_type=str(sess.wheel_type),
        external_stats=sess.external_stats,
        external_n_estimate=sess.external_stats_n_estimate,
    )

    # Call agent for next-spin strategy
    session_state = _build_session_state(sess, recent_history)
    agent = RouletteAgent(_anthropic_client(), model="claude-sonnet-4-6")
    decision = agent.decide(session_state)

    bets = decision["bets"]
    bets_total = sum(float(b["amount"]) for b in bets) if bets else 0.0
    spin_index = len(existing_spins) + 1

    spin_row = SpinModel(
        session_id=session_id,
        spin_index=spin_index,
        bets=bets,
        bets_total=bets_total,
        result_number=body.result_number,
        pnl=pnl_last,
        bankroll_after=float(sess.bankroll_now),
        bias_report=bias_report,
        rationale=decision["rationale"],
    )
    db.add(spin_row)
    db.commit()
    db.refresh(sess)

    return {
        "spin_index": spin_index,
        "pnl_last_spin": pnl_last,
        "bankroll_now": float(sess.bankroll_now),
        "bias_report": bias_report,
        "next_strategy": bets,
        "rationale": decision["rationale"],
    }


# ---------------------------------------------------------------------------
# GET /session/{id}/state
# ---------------------------------------------------------------------------

@app.get("/session/{session_id}/state")
def get_state(
    session_id: str,
    last_k: int = 10,
    _auth: None = Depends(verify_api_key),
    db: DbSession = Depends(get_db),
) -> dict:
    sess = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")

    total_spins = (
        db.query(SpinModel).filter(SpinModel.session_id == session_id).count()
    )

    recent_spins = (
        db.query(SpinModel)
        .filter(SpinModel.session_id == session_id)
        .order_by(SpinModel.spin_index.desc())
        .limit(last_k)
        .all()
    )
    recent_spins = list(reversed(recent_spins))

    last_bias_report = recent_spins[-1].bias_report if recent_spins else None

    return {
        "session_id": session_id,
        "wheel_type": str(sess.wheel_type),
        "bankroll_init": float(sess.bankroll_init),
        "bankroll_now": float(sess.bankroll_now),
        "bet_unit": float(sess.bet_unit),
        "excluded_dozens": list(sess.excluded_dozens or []),
        "spin_count": total_spins,
        "recent_k_spins": [
            {
                "spin_index": s.spin_index,
                "result_number": s.result_number,
                "bets": s.bets,
                "pnl": float(s.pnl) if s.pnl is not None else None,
                "bankroll_after": float(s.bankroll_after),
                "rationale": s.rationale,
            }
            for s in recent_spins
        ],
        "notes": sess.notes or "",
        "hyperparams": dict(sess.hyperparams or {}),
        "last_bias_report": last_bias_report,
    }
