"""Tests for roulette_agent.belief."""

import pytest

from roulette_agent.belief import compute_belief


class TestComputeBeliefStructure:
    def test_returns_all_pockets_american(self):
        r = compute_belief([], {"weight": 0.0})
        assert set(r.keys()) == set(range(38))

    def test_returns_all_pockets_european(self):
        r = compute_belief([], {"weight": 0.0}, wheel_type="european")
        assert set(r.keys()) == set(range(37))

    def test_probabilities_sum_to_one(self):
        r = compute_belief([1, 2, 3, 17, 17, 17], {"weight": 0.5})
        assert sum(r.values()) == pytest.approx(1.0, abs=1e-9)

    def test_probabilities_sum_to_one_zero_weight(self):
        r = compute_belief([17] * 100, {"weight": 0.0})
        assert sum(r.values()) == pytest.approx(1.0, abs=1e-9)

    def test_probabilities_sum_to_one_full_weight(self):
        r = compute_belief([17] * 100, {"weight": 1.0})
        assert sum(r.values()) == pytest.approx(1.0, abs=1e-9)

    def test_all_probabilities_positive(self):
        r = compute_belief([17] * 200, {"weight": 0.8})
        assert all(v > 0 for v in r.values())


class TestComputeBeliefZeroWeight:
    def test_uniform_when_weight_zero(self):
        r = compute_belief([17] * 200, {"weight": 0.0})
        expected = 1.0 / 38
        for v in r.values():
            assert v == pytest.approx(expected)

    def test_uniform_empty_history(self):
        r = compute_belief([], {"weight": 0.0})
        expected = 1.0 / 38
        for v in r.values():
            assert v == pytest.approx(expected)


class TestComputeBeliefBiasShift:
    def test_hot_number_gets_higher_prob_with_weight(self):
        history = [17] * 200
        r_biased = compute_belief(history, {"weight": 0.8})
        r_uniform = compute_belief(history, {"weight": 0.0})
        assert r_biased[17] > r_uniform[17]

    def test_cold_number_gets_lower_prob_with_weight(self):
        history = [17] * 200  # 0 never appeared
        r_biased = compute_belief(history, {"weight": 0.8})
        r_uniform = compute_belief(history, {"weight": 0.0})
        assert r_biased[0] < r_uniform[0]

    def test_relative_ordering_preserved(self):
        history = [17] * 100 + [5] * 50
        r = compute_belief(history, {"weight": 0.6})
        assert r[17] > r[5] > r[0]

    def test_weight_missing_defaults_to_zero(self):
        r = compute_belief([17] * 50, {})
        expected = 1.0 / 38
        for v in r.values():
            assert v == pytest.approx(expected)


class TestComputeBeliefEuropean:
    def test_european_sums_to_one(self):
        r = compute_belief([1, 2, 3], {"weight": 0.4}, wheel_type="european")
        assert sum(r.values()) == pytest.approx(1.0, abs=1e-9)

    def test_european_uniform_when_weight_zero(self):
        r = compute_belief([], {"weight": 0.0}, wheel_type="european")
        expected = 1.0 / 37
        for v in r.values():
            assert v == pytest.approx(expected)


class TestComputeBeliefKPrior:
    def test_large_k_prior_approaches_uniform(self):
        history = [17] * 100
        r = compute_belief(history, {"weight": 1.0}, k_prior=1e6)
        expected = 1.0 / 38
        for v in r.values():
            assert v == pytest.approx(expected, rel=1e-3)

    def test_small_k_prior_amplifies_counts(self):
        history = [17] * 100
        r_small = compute_belief(history, {"weight": 1.0}, k_prior=0.01)
        r_large = compute_belief(history, {"weight": 1.0}, k_prior=100.0)
        assert r_small[17] > r_large[17]
