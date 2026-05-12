"""Tests for roulette_agent.bias_detector."""

import numpy as np
import pytest

from roulette_agent.bias_detector import (
    VERDICT_WEIGHTS,
    binomial_test,
    chi_square_sector,
    chi_square_single,
    detect_bias,
    external_consistency_check,
    hot_numbers_test,
)


# ---------------------------------------------------------------------------
# A. Fair-roulette false-positive rate
# ---------------------------------------------------------------------------


def test_fair_roulette_false_positive_rate():
    """On a fair American wheel, moderate/strong verdicts must be rare (<5%)."""
    rng = np.random.default_rng(42)
    history = rng.integers(0, 38, size=10_000).tolist()

    moderate_or_strong = 0
    total_calls = 0
    for end in range(100, 10_001, 100):
        r = detect_bias(history[:end])
        if r["verdict"] in ("moderate", "strong"):
            moderate_or_strong += 1
        total_calls += 1

    rate = moderate_or_strong / total_calls
    assert rate < 0.05, (
        f"False-positive rate {rate:.1%} too high — thresholds may be too loose "
        f"({moderate_or_strong}/{total_calls} moderate/strong on fair wheel)"
    )


# ---------------------------------------------------------------------------
# B. Biased wheel detection
# ---------------------------------------------------------------------------


def _biased_history(n: int, seed: int = 7) -> list[int]:
    """Generate history from a wheel where number 35 appears at 3× fair rate."""
    rng = np.random.default_rng(seed)
    probs = np.full(38, (35 / 38) / 37)
    probs[35] = 3 / 38
    return rng.choice(38, size=n, p=probs).tolist()


def test_biased_detected_at_n2000():
    """With N=2000, a 3× bias on number 35 must be at least 'weak'."""
    history = _biased_history(2000)
    r = detect_bias(history)
    assert r["verdict"] in ("weak", "moderate", "strong"), (
        f"Expected at least weak at N=2000, got {r['verdict']}"
    )


def test_biased_detected_at_n500_majority_of_seeds():
    """With N=500, bias should be detected for most seeds."""
    hits = 0
    for seed in range(30):
        r = detect_bias(_biased_history(500, seed=seed))
        if r["verdict"] != "no_evidence":
            hits += 1
    assert hits / 30 >= 0.50, f"Only {hits}/30 seeds detected bias at N=500"


def test_biased_n100_no_guarantee():
    """At N=100, no guarantee of detection — just verify the call works."""
    r = detect_bias(_biased_history(100))
    assert r["verdict"] in ("no_evidence", "weak", "moderate", "strong")


# ---------------------------------------------------------------------------
# C. External stats downgrade
# ---------------------------------------------------------------------------


def test_external_inconsistent_downgrades_weak_to_no_evidence():
    """Internal weak signal + inconsistent external stats → verdict demoted to no_evidence."""
    # Build history: 60 non-green spins, 40 red, 20 black → red-biased internally
    history = [1] * 40 + [2] * 20  # 1=red, 2=black, no green
    # Verify internal verdict is weak before applying external
    r_internal = detect_bias(history)
    assert r_internal["verdict"] == "weak", (
        f"Expected internal weak, got {r_internal['verdict']} (p={r_internal['tests']['binomial_red']['p_value']:.4f})"
    )
    # External says heavily black (inconsistent with internal red bias)
    ext = {"black_pct": 0.75}
    r = detect_bias(history, external_stats=ext)
    assert r["verdict"] == "no_evidence", (
        f"Expected no_evidence after downgrade, got {r['verdict']}"
    )


def test_external_consistent_does_not_change_verdict():
    """Consistent external stats must not change the verdict in either direction."""
    history = [1] * 40 + [2] * 20  # red-biased internally (weak)
    ext = {"red_pct": 0.68}  # same direction → consistent
    r = detect_bias(history, external_stats=ext)
    assert r["verdict"] == "weak"


# ---------------------------------------------------------------------------
# D. External stats cannot upgrade verdict
# ---------------------------------------------------------------------------


def test_external_cannot_upgrade_no_evidence():
    """Strong external stats with small internal N must NOT upgrade verdict."""
    rng = np.random.default_rng(99)
    history = rng.integers(0, 38, size=20).tolist()
    # Confirm no_evidence before adding external
    assert detect_bias(history)["verdict"] == "no_evidence"
    # Dramatic external signal + large N estimate
    ext = {"black_pct": 0.95}
    r = detect_bias(history, external_stats=ext, external_n_estimate=1000)
    assert r["verdict"] == "no_evidence", (
        f"External stats must NEVER upgrade verdict; got {r['verdict']}"
    )


def test_external_cannot_upgrade_when_internal_weak():
    """Even with 'consistent' external stats, verdict stays at weak, never jumps to moderate/strong."""
    history = [1] * 40 + [2] * 20  # internal weak (red)
    ext = {"red_pct": 0.80}  # strongly consistent
    r = detect_bias(history, external_stats=ext, external_n_estimate=5000)
    assert r["verdict"] == "weak"


# ---------------------------------------------------------------------------
# E. Edge cases
# ---------------------------------------------------------------------------


def test_empty_history():
    r = detect_bias([])
    assert r["verdict"] == "no_evidence"
    assert r["weight"] == pytest.approx(0.0)
    assert r["external_check"] is None


def test_single_spin():
    r = detect_bias([17])
    assert r["verdict"] == "no_evidence"
    assert r["weight"] == pytest.approx(0.0)


def test_all_same_number_extreme_bias():
    """Entire history is the same number — must flag strong bias for large N."""
    history = [17] * 600
    r = detect_bias(history)
    assert r["verdict"] in ("moderate", "strong"), (
        f"Expected moderate/strong for all-same history, got {r['verdict']}"
    )


def test_external_stats_none_gives_none_check():
    r = detect_bias([17, 5, 3], external_stats=None)
    assert r["external_check"] is None


def test_external_stats_without_n_estimate_gives_empty_aux():
    r = detect_bias([17, 5, 3], external_stats={"black_pct": 0.62}, external_n_estimate=None)
    assert r["external_check"] is not None
    assert r["external_check"]["auxiliary_p_values"] == {}


def test_external_stats_with_n_estimate_fills_aux():
    r = detect_bias([17] * 50, external_stats={"black_pct": 0.62}, external_n_estimate=200)
    aux = r["external_check"]["auxiliary_p_values"]
    assert "black" in aux
    assert 0 <= aux["black"] <= 1


def test_weight_matches_verdict():
    for v, expected_w in VERDICT_WEIGHTS.items():
        r = detect_bias([])  # always no_evidence for empty
        if v == "no_evidence":
            assert r["weight"] == pytest.approx(expected_w)


# ---------------------------------------------------------------------------
# Unit tests for individual test functions
# ---------------------------------------------------------------------------


class TestChiSquareSingle:
    def test_zero_history(self):
        r = chi_square_single([])
        assert r["usable"] is False
        assert r["p_value"] == pytest.approx(1.0)

    def test_usable_threshold(self):
        # N=190 → E_per=5.0 → usable
        history = list(range(38)) * 5  # 190 spins, perfectly uniform
        r = chi_square_single(history)
        assert r["usable"] is True
        # Perfectly uniform → p should be high
        assert r["p_value"] > 0.5

    def test_not_usable_small_n(self):
        r = chi_square_single([1, 2, 3])
        assert r["usable"] is False


class TestChiSquareSector:
    def test_zero_history(self):
        r = chi_square_sector([])
        assert r["usable"] is False

    def test_usable_at_n50(self):
        rng = np.random.default_rng(0)
        h = rng.integers(0, 38, 50).tolist()
        r = chi_square_sector(h)
        assert r["usable"] is True

    def test_sector_count(self):
        r = chi_square_sector([1], n_sectors=8)
        assert r["n_sectors"] == 8


class TestBinomialTest:
    def test_invalid_group(self):
        with pytest.raises(ValueError):
            binomial_test([], "purple")

    def test_zero_history(self):
        r = binomial_test([], "red")
        assert r["usable"] is False
        assert r["p_value"] == pytest.approx(1.0)

    def test_balanced_history_high_p(self):
        # 15 red + 15 black → close to 0.5 → p should be high
        history = [1] * 15 + [2] * 15
        r = binomial_test(history, "red")
        assert r["p_value"] > 0.5
        assert r["usable"] is True

    def test_extreme_imbalance_low_p(self):
        # 50 red + 5 black → clearly imbalanced
        history = [1] * 50 + [2] * 5
        r = binomial_test(history, "red")
        assert r["p_value"] < 0.01

    def test_green_excluded_from_n_effective(self):
        # history = [0, 0, ...] (all green) → n_effective=0
        r = binomial_test([0] * 20, "red")
        assert r["n_effective"] == 0
        assert r["usable"] is False


class TestHotNumbersTest:
    def test_empty_history(self):
        assert hot_numbers_test([]) == []

    def test_returns_at_most_5(self):
        rng = np.random.default_rng(1)
        h = rng.integers(0, 38, 100).tolist()
        results = hot_numbers_test(h)
        assert len(results) <= 5

    def test_bonferroni_is_min_1(self):
        # All 500 spins are 17 → p_bonferroni for 17 should be 1.0 (capped)
        results = hot_numbers_test([17] * 500)
        # 17 has the lowest uncorrected p (way below 1/38); Bonferroni-capped at 1
        hit_17 = next(r for r in results if r["n"] == 17)
        assert hit_17["p_bonferroni"] <= 1.0

    def test_sorted_by_p_uncorrected(self):
        h = [17] * 200 + list(range(38)) * 2
        results = hot_numbers_test(h)
        p_vals = [r["p_uncorrected"] for r in results]
        assert p_vals == sorted(p_vals)


class TestExternalConsistencyCheck:
    def test_none_external_returns_none(self):
        assert external_consistency_check([1, 2, 3], None, None) is None

    def test_consistent_status(self):
        # Internal: all red → internal_pct(red) ≈ 1.0 > 0.5
        # External: red_pct = 0.70 > 0.5 → consistent
        r = external_consistency_check([1] * 40, {"red_pct": 0.70}, None)
        assert r["status"] == "consistent"

    def test_inconsistent_status(self):
        # Internal: all red (red_pct ≈ 1.0)
        # External: black_pct = 0.80 > 0.5 → internal_pct(black)≈0 < 0.5 → inconsistent
        r = external_consistency_check([1] * 40, {"black_pct": 0.80}, None)
        assert r["status"] == "inconsistent"

    def test_unknown_when_diff_small(self):
        # Internal: 20 red + 20 black → pct(black) = 0.5
        # External: black_pct = 0.52 → diff = 0.02 < 0.05 → unknown
        r = external_consistency_check([1] * 20 + [2] * 20, {"black_pct": 0.52}, None)
        assert r["status"] == "unknown"

    def test_auxiliary_p_values_populated_with_estimate(self):
        r = external_consistency_check(
            [1] * 30, {"red_pct": 0.70}, external_n_estimate=100
        )
        assert "red" in r["auxiliary_p_values"]
        assert 0 <= r["auxiliary_p_values"]["red"] <= 1

    def test_auxiliary_p_values_empty_without_estimate(self):
        r = external_consistency_check([1] * 30, {"red_pct": 0.70}, None)
        assert r["auxiliary_p_values"] == {}
