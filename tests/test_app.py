"""Integration tests for FastAPI app endpoints (TestClient + mocked agent)."""

from __future__ import annotations

import os

# Set env vars BEFORE importing the app so the module-level engine is configured correctly.
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_app.db")
os.environ.setdefault("API_KEY", "test-key-123")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-fake-key")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch, MagicMock

from roulette_agent.app import app, get_db
from roulette_agent.models import Base

# ---------------------------------------------------------------------------
# Test DB setup
# ---------------------------------------------------------------------------

_TEST_DB_URL = "sqlite:///./test_app.db"
_test_engine = create_engine(_TEST_DB_URL, connect_args={"check_same_thread": False})
_TestSession = sessionmaker(bind=_test_engine, autocommit=False, autoflush=False)


def _override_get_db():
    db = _TestSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db


@pytest.fixture(autouse=True)
def reset_db():
    Base.metadata.create_all(_test_engine)
    yield
    Base.metadata.drop_all(_test_engine)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HEADERS = {"X-API-Key": "test-key-123"}

_MOCK_DECISION = {
    "bets": [{"type": "red", "numbers": None, "amount": 10.0}],
    "rationale": "Mocked agent decision.",
}

_NEW_SESSION_BODY = {
    "wheel_type": "american",
    "bankroll": 500.0,
    "bet_unit": 5.0,
    "excluded_dozens": [],
    "recent_history": [17, 5, 32, 17, 11, 17, 5, 17, 32, 17],
    "external_stats": {"black_pct": 0.60},
    "external_stats_n_estimate": 100,
}


def _patched_agent():
    """Context manager that patches RouletteAgent so it returns _MOCK_DECISION."""
    mock_agent_cls = MagicMock()
    mock_agent_cls.return_value.decide.return_value = _MOCK_DECISION
    return patch("roulette_agent.app.RouletteAgent", mock_agent_cls)


# ---------------------------------------------------------------------------
# A. Full flow: new session → 3 spins
# ---------------------------------------------------------------------------

class TestFullFlow:
    def test_new_then_three_spins(self):
        client = TestClient(app)

        with _patched_agent():
            # 1. Create session
            resp = client.post("/session/new", headers=HEADERS, json=_NEW_SESSION_BODY)
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert "session_id" in data
            assert data["spin_index"] == 0
            assert data["bankroll_now"] == 500.0
            assert "bias_report" in data
            assert "next_strategy" in data
            assert "rationale" in data

            session_id = data["session_id"]

            # 2-4. Three spins
            spin_results = [1, 17, 5]  # 1=red win, 17=black loss, 5=red win
            prev_bankroll = 500.0

            for i, result in enumerate(spin_results, start=1):
                resp = client.post(
                    f"/session/{session_id}/spin",
                    headers=HEADERS,
                    json={"result_number": result},
                )
                assert resp.status_code == 200, resp.text
                data = resp.json()

                assert data["spin_index"] == i
                assert "pnl_last_spin" in data
                assert "bankroll_now" in data
                assert "bias_report" in data
                assert "next_strategy" in data
                assert isinstance(data["next_strategy"], list)
                prev_bankroll = data["bankroll_now"]

            # 5. Check state
            resp = client.get(f"/session/{session_id}/state", headers=HEADERS)
            assert resp.status_code == 200, resp.text
            state = resp.json()
            assert state["session_id"] == session_id
            assert state["spin_count"] == 3
            assert len(state["recent_k_spins"]) == 3
            assert state["bankroll_now"] == pytest.approx(prev_bankroll)

    def test_pnl_accounting_correct(self):
        """Initial bets from /session/new must be settled on the first spin."""
        client = TestClient(app)

        with _patched_agent():
            resp = client.post("/session/new", headers=HEADERS, json=_NEW_SESSION_BODY)
            session_id = resp.json()["session_id"]

            # Spin 1: settles initial bets (red $10) against result 17 (BLACK) → -$10
            resp = client.post(
                f"/session/{session_id}/spin",
                headers=HEADERS,
                json={"result_number": 17},  # 17 is BLACK → red loses
            )
            assert resp.json()["pnl_last_spin"] == pytest.approx(-10.0)
            br1 = resp.json()["bankroll_now"]
            assert br1 == pytest.approx(490.0)

            # Spin 2: settles spin-1 bets (red $10) against result 1 (RED) → +$10
            resp = client.post(
                f"/session/{session_id}/spin",
                headers=HEADERS,
                json={"result_number": 1},  # 1 is RED → red wins
            )
            assert resp.json()["pnl_last_spin"] == pytest.approx(10.0)
            assert resp.json()["bankroll_now"] == pytest.approx(br1 + 10.0)


# ---------------------------------------------------------------------------
# B. Auth: 401 when X-API-Key header is missing or wrong
# ---------------------------------------------------------------------------

class TestAuth:
    def test_missing_api_key_returns_401_new(self):
        client = TestClient(app)
        resp = client.post("/session/new", json=_NEW_SESSION_BODY)
        assert resp.status_code == 401

    def test_wrong_api_key_returns_401_new(self):
        client = TestClient(app)
        resp = client.post(
            "/session/new",
            headers={"X-API-Key": "wrong-key"},
            json=_NEW_SESSION_BODY,
        )
        assert resp.status_code == 401

    def test_missing_api_key_returns_401_spin(self):
        client = TestClient(app)
        resp = client.post("/session/nonexistent/spin", json={"result_number": 5})
        assert resp.status_code == 401

    def test_missing_api_key_returns_401_state(self):
        client = TestClient(app)
        resp = client.get("/session/nonexistent/state")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# C. 404 for unknown session_id
# ---------------------------------------------------------------------------

class TestNotFound:
    def test_spin_unknown_session_404(self):
        client = TestClient(app)
        resp = client.post(
            "/session/00000000-0000-0000-0000-000000000000/spin",
            headers=HEADERS,
            json={"result_number": 5},
        )
        assert resp.status_code == 404

    def test_state_unknown_session_404(self):
        client = TestClient(app)
        resp = client.get(
            "/session/00000000-0000-0000-0000-000000000000/state",
            headers=HEADERS,
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# D. 422 for invalid result_number (out of range)
# ---------------------------------------------------------------------------

class TestValidation:
    def test_result_number_39_is_422(self):
        client = TestClient(app)

        with _patched_agent():
            resp = client.post("/session/new", headers=HEADERS, json=_NEW_SESSION_BODY)
            session_id = resp.json()["session_id"]

        resp = client.post(
            f"/session/{session_id}/spin",
            headers=HEADERS,
            json={"result_number": 39},
        )
        assert resp.status_code == 422

    def test_result_number_negative_is_422(self):
        client = TestClient(app)

        with _patched_agent():
            resp = client.post("/session/new", headers=HEADERS, json=_NEW_SESSION_BODY)
            session_id = resp.json()["session_id"]

        resp = client.post(
            f"/session/{session_id}/spin",
            headers=HEADERS,
            json={"result_number": -1},
        )
        assert resp.status_code == 422

    def test_empty_recent_history_is_422(self):
        client = TestClient(app)
        body = dict(_NEW_SESSION_BODY, recent_history=[])
        resp = client.post("/session/new", headers=HEADERS, json=body)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# E. Nullable fields work correctly
# ---------------------------------------------------------------------------

class TestNullableFields:
    def test_external_stats_none(self):
        """Session creation works when external_stats is omitted (null)."""
        client = TestClient(app)
        body = {
            "bankroll": 400.0,
            "bet_unit": 5.0,
            "recent_history": [1, 2, 3, 5, 7],
            # external_stats omitted → defaults to None
        }
        with _patched_agent():
            resp = client.post("/session/new", headers=HEADERS, json=body)
        assert resp.status_code == 200, resp.text
        assert resp.json()["bankroll_now"] == 400.0

    def test_external_stats_provided_but_n_estimate_null(self):
        """Session works when external_stats present but external_stats_n_estimate is null."""
        client = TestClient(app)
        body = {
            "bankroll": 400.0,
            "bet_unit": 5.0,
            "recent_history": [1, 2, 3, 5, 7],
            "external_stats": {"black_pct": 0.55},
            "external_stats_n_estimate": None,
        }
        with _patched_agent():
            resp = client.post("/session/new", headers=HEADERS, json=body)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["spin_index"] == 0
        assert "bias_report" in data

    def test_external_stats_none_spin_works(self):
        """Spin endpoint works on a session with no external_stats."""
        client = TestClient(app)
        body = {
            "bankroll": 400.0,
            "bet_unit": 5.0,
            "recent_history": [1, 2, 3, 5, 7],
        }
        with _patched_agent():
            resp = client.post("/session/new", headers=HEADERS, json=body)
            session_id = resp.json()["session_id"]
            resp = client.post(
                f"/session/{session_id}/spin",
                headers=HEADERS,
                json={"result_number": 17},
            )
        assert resp.status_code == 200, resp.text
