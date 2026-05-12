"""Tests for roulette_agent.optimizer."""

import pytest

from roulette_agent.optimizer import (
    enumerate_legal_bets,
    excluded_numbers,
    fixed_baseline_allocation,
    greedy_ev_allocation,
    kelly_allocation,
)


# ---------------------------------------------------------------------------
# excluded_numbers
# ---------------------------------------------------------------------------


class TestExcludedNumbers:
    def test_empty_returns_empty(self):
        assert excluded_numbers([]) == set()

    def test_dozen_1(self):
        assert excluded_numbers([1]) == set(range(1, 13))

    def test_dozen_3(self):
        assert excluded_numbers([3]) == set(range(25, 37))

    def test_dozen_2(self):
        assert excluded_numbers([2]) == set(range(13, 25))

    def test_dozens_1_and_2(self):
        result = excluded_numbers([1, 2])
        assert result == set(range(1, 25))

    def test_dozens_2_and_3(self):
        result = excluded_numbers([2, 3])
        assert result == set(range(13, 37))

    def test_dozens_1_and_3_raises(self):
        with pytest.raises(ValueError):
            excluded_numbers([1, 3])

    def test_invalid_dozen_raises(self):
        with pytest.raises(ValueError):
            excluded_numbers([4])

    def test_invalid_zero_raises(self):
        with pytest.raises(ValueError):
            excluded_numbers([0])


# ---------------------------------------------------------------------------
# enumerate_legal_bets
# ---------------------------------------------------------------------------


class TestEnumerateLegalBets:
    def test_no_exclusion_nonempty(self):
        bets = enumerate_legal_bets([])
        assert len(bets) > 0

    def test_excluded_dozen_3_no_overlap(self):
        excl = set(range(25, 37))
        bets = enumerate_legal_bets([3])
        for b in bets:
            assert not b["covered"] & excl, (
                f"Bet {b['type']} {b['numbers']} overlaps excluded set"
            )

    def test_excluded_dozen_1_no_overlap(self):
        excl = set(range(1, 13))
        bets = enumerate_legal_bets([1])
        for b in bets:
            assert not b["covered"] & excl

    def test_invalid_dozens_propagates_error(self):
        with pytest.raises(ValueError):
            enumerate_legal_bets([1, 3])

    def test_all_bets_have_required_keys(self):
        for b in enumerate_legal_bets([]):
            assert "type" in b
            assert "numbers" in b
            assert "covered" in b

    def test_straight_bets_present_without_exclusion(self):
        types = {b["type"] for b in enumerate_legal_bets([])}
        assert "straight" in types

    def test_outside_bets_present_without_exclusion(self):
        types = {b["type"] for b in enumerate_legal_bets([])}
        for t in ("red", "black", "odd", "even", "low", "high"):
            assert t in types


# ---------------------------------------------------------------------------
# greedy_ev_allocation
# ---------------------------------------------------------------------------


def _uniform_p(size: int = 38) -> dict[int, float]:
    return {n: 1.0 / size for n in range(size)}


def _biased_p(hot: int, hot_prob: float, size: int = 38) -> dict[int, float]:
    remaining = (1.0 - hot_prob) / (size - 1)
    return {n: (hot_prob if n == hot else remaining) for n in range(size)}


class TestGreedyEvAllocation:
    def test_uniform_p_returns_empty(self):
        result = greedy_ev_allocation(_uniform_p(), 1000.0, 10.0, [])
        assert result == []

    def test_biased_p_returns_nonempty(self):
        p = _biased_p(17, 0.5)
        result = greedy_ev_allocation(p, 1000.0, 10.0, [])
        assert len(result) > 0

    def test_amounts_are_multiples_of_bet_unit(self):
        p = _biased_p(17, 0.5)
        bet_unit = 7.0
        result = greedy_ev_allocation(p, 1000.0, bet_unit, [])
        for b in result:
            assert b["amount"] % bet_unit == pytest.approx(0.0)

    def test_total_staked_lte_bankroll(self):
        p = _biased_p(17, 0.5)
        bankroll = 500.0
        result = greedy_ev_allocation(p, bankroll, 10.0, [])
        total = sum(b["amount"] for b in result)
        assert total <= bankroll + 1e-9

    def test_excluded_dozen_respected(self):
        p = _biased_p(31, 0.5)  # 31 is in dozen 3
        result = greedy_ev_allocation(p, 1000.0, 10.0, [3])
        excl = set(range(25, 37))
        for b in result:
            from roulette_agent.layout import get_covered_numbers
            covered = get_covered_numbers(b["type"], b["numbers"])
            assert not covered & excl

    def test_top_k_respected(self):
        p = _biased_p(17, 0.8)
        result = greedy_ev_allocation(p, 10000.0, 10.0, [], top_k=3)
        assert len(result) <= 3

    def test_result_keys(self):
        p = _biased_p(17, 0.5)
        for b in greedy_ev_allocation(p, 1000.0, 10.0, []):
            assert "type" in b and "numbers" in b and "amount" in b

    def test_zero_bankroll_returns_empty(self):
        p = _biased_p(17, 0.5)
        result = greedy_ev_allocation(p, 0.0, 10.0, [])
        assert result == []


# ---------------------------------------------------------------------------
# kelly_allocation
# ---------------------------------------------------------------------------


class TestKellyAllocation:
    def test_uniform_p_returns_empty(self):
        result = kelly_allocation(_uniform_p(), 1000.0, 10.0, [])
        assert result == []

    def test_biased_p_bets_on_hot_number(self):
        p = _biased_p(31, 0.5)
        result = kelly_allocation(p, 1000.0, 10.0, [])
        covered_sets = []
        for b in result:
            from roulette_agent.layout import get_covered_numbers
            covered_sets.append(get_covered_numbers(b["type"], b["numbers"]))
        # At least one bet covers the hot number
        assert any(31 in c for c in covered_sets)

    def test_amounts_are_multiples_of_bet_unit(self):
        p = _biased_p(17, 0.5)
        bet_unit = 5.0
        result = kelly_allocation(p, 1000.0, bet_unit, [])
        for b in result:
            assert b["amount"] % bet_unit == pytest.approx(0.0)

    def test_no_bet_exceeds_half_bankroll(self):
        p = _biased_p(17, 0.99)
        bankroll = 200.0
        result = kelly_allocation(p, bankroll, 1.0, [])
        for b in result:
            assert b["amount"] <= bankroll * 0.5 + 1e-9

    def test_excluded_dozen_respected(self):
        p = _biased_p(31, 0.5)
        result = kelly_allocation(p, 1000.0, 10.0, [3])
        excl = set(range(25, 37))
        for b in result:
            from roulette_agent.layout import get_covered_numbers
            covered = get_covered_numbers(b["type"], b["numbers"])
            assert not covered & excl

    def test_result_keys(self):
        p = _biased_p(17, 0.5)
        for b in kelly_allocation(p, 1000.0, 10.0, []):
            assert "type" in b and "numbers" in b and "amount" in b


# ---------------------------------------------------------------------------
# fixed_baseline_allocation
# ---------------------------------------------------------------------------


class TestFixedBaselineAllocation:
    def test_uniform_returns_single_fallback(self):
        result = fixed_baseline_allocation(_uniform_p(), 1000.0, 10.0, [])
        # Under uniform p all outside bets are negative EV → fallback to 1 bet
        assert len(result) == 1
        assert result[0]["amount"] == pytest.approx(10.0)

    def test_biased_returns_positive_ev_bets(self):
        # p(17)=0.5 → odd and other groups may be positive EV
        p = _biased_p(17, 0.5)  # 17 is odd → odd has high p_win
        result = fixed_baseline_allocation(p, 1000.0, 10.0, [])
        assert len(result) >= 1

    def test_all_amounts_equal_bet_unit(self):
        p = _biased_p(17, 0.5)
        bet_unit = 25.0
        result = fixed_baseline_allocation(p, 1000.0, bet_unit, [])
        for b in result:
            assert b["amount"] == pytest.approx(bet_unit)

    def test_insufficient_bankroll_returns_empty(self):
        result = fixed_baseline_allocation(_uniform_p(), 5.0, 10.0, [])
        assert result == []

    def test_only_outside_bets_returned(self):
        outside = {"red", "black", "odd", "even", "low", "high"}
        result = fixed_baseline_allocation(_uniform_p(), 1000.0, 10.0, [])
        for b in result:
            assert b["type"] in outside

    def test_result_keys(self):
        for b in fixed_baseline_allocation(_uniform_p(), 1000.0, 10.0, []):
            assert "type" in b and "numbers" in b and "amount" in b
