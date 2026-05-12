"""Tests for roulette_agent.settler."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from roulette_agent.settler import settle


class TestSettleHits:
    def test_straight_hit(self):
        bets = [{"type": "straight", "numbers": [17], "amount": 10.0}]
        r = settle(bets, 17)
        assert r["result_number"] == 17
        assert r["total_staked"] == pytest.approx(10.0)
        assert r["total_payout"] == pytest.approx(360.0)  # 10 × 36
        assert r["pnl"] == pytest.approx(350.0)           # 10 × 35
        assert r["detail"][0]["won"] is True

    def test_straight_miss(self):
        bets = [{"type": "straight", "numbers": [17], "amount": 10.0}]
        r = settle(bets, 18)
        assert r["total_payout"] == pytest.approx(0.0)
        assert r["pnl"] == pytest.approx(-10.0)
        assert r["detail"][0]["won"] is False
        assert r["detail"][0]["payout"] == pytest.approx(0.0)

    def test_corner_hit_spec_sanity_check(self):
        """Spec says: corner [28,29,31,32] result=31 → pnl = 15 × 8 = 120."""
        bets = [{"type": "corner", "numbers": [28, 29, 31, 32], "amount": 15.0}]
        r = settle(bets, 31)
        assert r["pnl"] == pytest.approx(120.0)

    def test_red_hit(self):
        bets = [{"type": "red", "numbers": None, "amount": 50.0}]
        r = settle(bets, 7)  # 7 is red
        assert r["pnl"] == pytest.approx(50.0)

    def test_black_hit(self):
        bets = [{"type": "black", "numbers": None, "amount": 20.0}]
        r = settle(bets, 2)  # 2 is black
        assert r["pnl"] == pytest.approx(20.0)

    def test_odd_hit(self):
        bets = [{"type": "odd", "numbers": None, "amount": 10.0}]
        r = settle(bets, 17)  # 17 is odd
        assert r["pnl"] == pytest.approx(10.0)

    def test_dozen_hit(self):
        bets = [{"type": "dozen", "numbers": [2], "amount": 12.0}]
        r = settle(bets, 17)  # dozen 2 = 13-24
        assert r["pnl"] == pytest.approx(24.0)  # 12 × 2

    def test_five_line_hit_on_zero(self):
        bets = [{"type": "five_line", "numbers": [0, 37, 1, 2, 3], "amount": 5.0}]
        r = settle(bets, 0)
        assert r["pnl"] == pytest.approx(30.0)  # 5 × 6

    def test_five_line_hit_on_double_zero(self):
        bets = [{"type": "five_line", "numbers": [0, 37, 1, 2, 3], "amount": 5.0}]
        r = settle(bets, 37)
        assert r["detail"][0]["won"] is True

    def test_six_line_hit(self):
        bets = [{"type": "six_line", "numbers": [1, 2, 3, 4, 5, 6], "amount": 10.0}]
        r = settle(bets, 4)
        assert r["pnl"] == pytest.approx(50.0)  # 10 × 5


class TestSettleLosses:
    def test_red_miss_on_green_zero(self):
        """0 is green → red bet loses."""
        bets = [{"type": "red", "numbers": None, "amount": 50.0}]
        r = settle(bets, 0)
        assert r["detail"][0]["won"] is False
        assert r["pnl"] == pytest.approx(-50.0)

    def test_double_zero_loses_all_outside_bets(self):
        """37 (00) is green → red, odd, low all lose."""
        bets = [
            {"type": "red",  "numbers": None, "amount": 10.0},
            {"type": "odd",  "numbers": None, "amount": 10.0},
            {"type": "low",  "numbers": None, "amount": 10.0},
        ]
        r = settle(bets, 37)
        assert all(d["won"] is False for d in r["detail"])
        assert r["pnl"] == pytest.approx(-30.0)

    def test_corner_miss(self):
        bets = [{"type": "corner", "numbers": [28, 29, 31, 32], "amount": 15.0}]
        r = settle(bets, 17)
        assert r["pnl"] == pytest.approx(-15.0)


class TestSettleMixed:
    def test_partial_win(self):
        """One win, one loss in same spin. Result=1: red+odd → red wins, even loses."""
        bets = [
            {"type": "red",  "numbers": None, "amount": 20.0},  # 1 is red  → win
            {"type": "even", "numbers": None, "amount": 10.0},  # 1 is odd  → lose
        ]
        r = settle(bets, 1)
        assert r["detail"][0]["won"] is True
        assert r["detail"][1]["won"] is False
        assert r["pnl"] == pytest.approx(10.0)

    def test_total_payout_equals_sum_of_detail_payouts(self):
        bets = [
            {"type": "red",    "numbers": None, "amount": 10.0},
            {"type": "dozen",  "numbers": [1],  "amount": 5.0},
            {"type": "column", "numbers": [1],  "amount": 8.0},
        ]
        r = settle(bets, 3)  # 3 is red, dozen 1, column 3 — red wins, dozen1 wins
        assert r["total_payout"] == pytest.approx(
            sum(d["payout"] for d in r["detail"])
        )

    def test_detail_length_matches_bets(self):
        bets = [
            {"type": "red",      "numbers": None, "amount": 10.0},
            {"type": "straight", "numbers": [0],  "amount":  5.0},
            {"type": "dozen",    "numbers": [1],  "amount":  8.0},
        ]
        r = settle(bets, 5)
        assert len(r["detail"]) == 3

    def test_empty_bets(self):
        r = settle([], 17)
        assert r["total_staked"] == pytest.approx(0.0)
        assert r["total_payout"] == pytest.approx(0.0)
        assert r["pnl"] == pytest.approx(0.0)
        assert r["detail"] == []


# ---------------------------------------------------------------------------
# Hypothesis: invariants for random outside bets and results
# ---------------------------------------------------------------------------

@st.composite
def outside_bet(draw: st.DrawFn) -> dict:
    bet_type = draw(st.sampled_from(["red", "black", "odd", "even", "low", "high"]))
    amount = float(draw(st.integers(min_value=1, max_value=500)))
    return {"type": bet_type, "numbers": None, "amount": amount}


@given(
    bets=st.lists(outside_bet(), min_size=1, max_size=6),
    result=st.integers(min_value=0, max_value=37),
)
def test_hypothesis_settle_invariants(bets: list[dict], result: int) -> None:
    r = settle(bets, result)
    total_staked = sum(b["amount"] for b in bets)

    # Accounting identities
    assert r["total_staked"] == pytest.approx(total_staked)
    assert r["total_payout"] == pytest.approx(sum(d["payout"] for d in r["detail"]))
    assert r["pnl"] == pytest.approx(r["total_payout"] - r["total_staked"])

    # Boundary: can't lose more than staked; can't get negative payout
    assert r["total_payout"] >= -1e-9
    assert r["pnl"] >= -total_staked - 1e-9

    # Structure
    assert len(r["detail"]) == len(bets)

    # Per-bet: losing bets return 0; winning bets return positive
    for d in r["detail"]:
        if d["won"]:
            assert d["payout"] > 0
        else:
            assert d["payout"] == pytest.approx(0.0)
