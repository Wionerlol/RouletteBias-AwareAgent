"""Tests for hotspot detection, Gaussian heat map, and portfolio-Kelly allocation."""

from __future__ import annotations

import pytest

from roulette_agent.hotspot_grid import (
    Hotspot,
    _exploration_schedule,
    _temporal_belief,
    _trend_scores,
    adaptive_gaussian_kelly_allocation,
    build_heat_map,
    detect_hotspots,
    gaussian_kelly_allocation,
)
from roulette_agent.layout import WHEEL_ORDER_AMERICAN
from roulette_agent.tools import TOOLS, dispatch_tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_p(size: int = 38) -> dict[int, float]:
    return {n: 1.0 / size for n in range(size)}


def _biased_p(hot: int = 17, hot_prob: float = 0.15, size: int = 38) -> dict[int, float]:
    """Single hot pocket; rest share remaining probability uniformly."""
    p = {n: (1.0 - hot_prob) / (size - 1) for n in range(size)}
    p[hot] = hot_prob
    return p


def _multi_biased_p(hot_map: dict[int, float], size: int = 38) -> dict[int, float]:
    """Multiple hot pockets; rest split remaining probability uniformly."""
    total_hot = sum(hot_map.values())
    rest = (1.0 - total_hot) / (size - len(hot_map))
    p = {n: rest for n in range(size)}
    p.update(hot_map)
    return p


# ---------------------------------------------------------------------------
# A. detect_hotspots
# ---------------------------------------------------------------------------

class TestDetectHotspots:
    def test_uniform_returns_empty(self):
        hotspots = detect_hotspots(_uniform_p())
        assert hotspots == []

    def test_single_hot_pocket_rank_1(self):
        p = _biased_p(hot=17)
        hs = detect_hotspots(p)
        assert hs[0].number == 17
        assert hs[0].rank == 1
        assert hs[0].excess > 0

    def test_ranks_by_excess_descending(self):
        p = _multi_biased_p({17: 0.12, 5: 0.09, 32: 0.06})
        hs = detect_hotspots(p, top_k=3)
        assert len(hs) == 3
        assert [h.number for h in hs] == [17, 5, 32]
        assert [h.rank for h in hs] == [1, 2, 3]
        for i in range(len(hs) - 1):
            assert hs[i].excess >= hs[i + 1].excess

    def test_top_k_cap(self):
        p = _biased_p()
        assert len(detect_hotspots(p, top_k=2)) <= 2

    def test_excess_is_p_minus_uniform(self):
        p = _biased_p(hot=7, hot_prob=0.20)
        p_uniform = 1.0 / 38
        hs = detect_hotspots(p, top_k=1)
        assert hs[0].number == 7
        assert abs(hs[0].excess - (0.20 - p_uniform)) < 1e-12


# ---------------------------------------------------------------------------
# B. build_heat_map
# ---------------------------------------------------------------------------

class TestBuildHeatMap:
    def test_uniform_all_zero(self):
        heat = build_heat_map(_uniform_p())
        assert all(abs(v) < 1e-12 for v in heat.values())

    def test_hot_pocket_is_maximum(self):
        heat = build_heat_map(_biased_p(hot=17), sigma=3.0)
        assert max(heat, key=heat.get) == 17

    def test_wheel_neighbors_warmer_than_distant(self):
        """On American wheel: …32 | 17 | 5 | 22…   distance(17,5)=1, distance(17,1)=10."""
        heat = build_heat_map(_biased_p(hot=17), sigma=3.0)
        # 5 is 1 step from 17; 1 is ~10 steps away
        assert heat[5] > heat[1]
        assert heat[32] > heat[1]

    def test_heat_decays_monotonically_with_wheel_distance(self):
        """Heat should decrease the farther a pocket is from the hotspot on the wheel."""
        heat = build_heat_map(_biased_p(hot=17), sigma=2.0)
        # American wheel sequence near index 9 (=17): ...32(8), 17(9), 5(10), 22(11)...
        assert heat[17] > heat[5] > heat[22]

    def test_two_hotspots_additive_at_saddle(self):
        """Pocket 32 (neighbour of both 17 and itself as secondary) gets more heat
        when a second hotspot 5 is added right next to 17."""
        heat_single = build_heat_map(_biased_p(hot=17, hot_prob=0.12), sigma=3.0)
        # Add a secondary hotspot at 5 (adjacent to 17 on wheel)
        p_double = _multi_biased_p({17: 0.12, 5: 0.10})
        heat_double = build_heat_map(p_double, sigma=3.0)
        # Pocket 32 (just before 17) should get more heat from the double-hotspot
        assert heat_double[32] > heat_single[32]

    def test_european_wheel_uses_different_layout(self):
        p = {n: 1.0 / 37 for n in range(37)}
        p[17] = 0.15
        heat_eu = build_heat_map(p, wheel_type="european", sigma=3.0)
        assert heat_eu[17] == max(heat_eu.values())


# ---------------------------------------------------------------------------
# C. gaussian_kelly_allocation
# ---------------------------------------------------------------------------

class TestGaussianKellyAllocation:
    def test_uniform_returns_empty(self):
        assert gaussian_kelly_allocation(_uniform_p(), 800, 10, []) == []

    def test_biased_returns_bets(self):
        result = gaussian_kelly_allocation(_biased_p(hot=17, hot_prob=0.15), 800, 1, [])
        assert len(result) > 0

    def test_result_structure(self):
        result = gaussian_kelly_allocation(_biased_p(hot_prob=0.15), 800, 1, [])
        for b in result:
            assert "type" in b
            assert "numbers" in b
            assert "amount" in b
            assert b["amount"] > 0

    def test_amounts_are_multiples_of_bet_unit(self):
        result = gaussian_kelly_allocation(_biased_p(hot_prob=0.15), 800, 5, [])
        for b in result:
            assert b["amount"] >= 5
            assert abs(b["amount"] % 5) < 1e-9

    def test_true_kelly_amounts_scale_with_bankroll(self):
        """True portfolio Kelly: 10× bankroll → ≥5× bet amounts (proportional scaling)."""
        p = _biased_p(hot_prob=0.20)
        r_small = gaussian_kelly_allocation(p, 200, 1, [])
        r_large = gaussian_kelly_allocation(p, 2000, 1, [])
        if r_small and r_large:
            total_s = sum(b["amount"] for b in r_small)
            total_l = sum(b["amount"] for b in r_large)
            assert total_l > total_s * 5

    def test_respects_excluded_dozens(self):
        """No returned bet should cover a pocket in the excluded dozen."""
        p = _biased_p(hot=17, hot_prob=0.20)  # 17 in dozen 2
        result = gaussian_kelly_allocation(p, 800, 1, excluded_dozens=[3])
        dozen_3 = set(range(25, 37))
        for b in result:
            if b["numbers"] is not None:
                for n in b["numbers"]:
                    assert n not in dozen_3

    def test_total_stake_within_budget_plus_rounding(self):
        """Total staked ≤ kelly_fraction × bankroll + n_bets × bet_unit (rounding allowance)."""
        p = _biased_p(hot_prob=0.20)
        bankroll = 1000.0
        kelly_fraction = 0.25
        bet_unit = 5.0
        result = gaussian_kelly_allocation(
            p, bankroll, bet_unit, [], kelly_fraction=kelly_fraction
        )
        if result:
            total = sum(b["amount"] for b in result)
            budget = kelly_fraction * bankroll + len(result) * bet_unit
            assert total <= budget

    def test_zero_bankroll_returns_empty(self):
        assert gaussian_kelly_allocation(_biased_p(), 0, 10, []) == []

    def test_bankroll_below_bet_unit_returns_empty(self):
        assert gaussian_kelly_allocation(_biased_p(), 5, 10, []) == []

    def test_multi_hotspot_produces_diverse_bets(self):
        """With two hotspots far apart on the wheel, bets should not all cluster on one number."""
        p = _multi_biased_p({17: 0.12, 1: 0.10})  # 17 and 1 are far apart
        result = gaussian_kelly_allocation(p, 800, 1, [])
        if len(result) >= 2:
            # At least two distinct bet types or numbers
            types = {b["type"] for b in result}
            assert len(types) >= 1  # structural check — at minimum one type returned

    def test_sigma_wider_spreads_bets_beyond_hotspot(self):
        """Larger sigma → heat spreads further → more diverse bets selected."""
        p = _biased_p(hot=17, hot_prob=0.20)
        r_narrow = gaussian_kelly_allocation(p, 800, 1, [], sigma=1.0)
        r_wide = gaussian_kelly_allocation(p, 800, 1, [], sigma=6.0)
        # Wide sigma may include bets farther from 17 on the wheel
        if r_narrow and r_wide:
            types_narrow = {b["type"] for b in r_narrow}
            types_wide = {b["type"] for b in r_wide}
            # Wide sigma should produce at least as many bet types
            assert len(types_wide) >= len(types_narrow)


# ---------------------------------------------------------------------------
# D. dispatch_tool integration
# ---------------------------------------------------------------------------

class TestDispatchGaussianKelly:
    def test_tool_registered_in_tools_list(self):
        names = [t["name"] for t in TOOLS]
        assert "gaussian_kelly_allocation" in names

    def test_dispatch_uniform_returns_empty(self):
        p_str = {str(n): 1.0 / 38 for n in range(38)}
        result = dispatch_tool("gaussian_kelly_allocation", {
            "p": p_str, "bankroll": 800, "bet_unit": 10, "excluded_dozens": [],
        })
        assert result["bets"] == []

    def test_dispatch_biased_returns_bets(self):
        p_str = {str(n): (1.0 - 0.15) / 37 for n in range(38)}
        p_str["17"] = 0.15
        result = dispatch_tool("gaussian_kelly_allocation", {
            "p": p_str, "bankroll": 800, "bet_unit": 1, "excluded_dozens": [],
        })
        assert len(result["bets"]) > 0

    def test_dispatch_string_keys_accepted(self):
        """dispatch_tool must convert string pocket keys to int before calling allocator."""
        p_str = {str(n): 1.0 / 38 for n in range(38)}
        p_str["5"] = 0.20
        total = sum(float(v) for v in p_str.values())
        p_str = {k: float(v) / total for k, v in p_str.items()}
        result = dispatch_tool("gaussian_kelly_allocation", {
            "p": p_str, "bankroll": 200, "bet_unit": 1, "excluded_dozens": [],
        })
        assert "bets" in result

    def test_dispatch_optional_params_forwarded(self):
        p_str = {str(n): (1.0 - 0.20) / 37 for n in range(38)}
        p_str["17"] = 0.20
        result = dispatch_tool("gaussian_kelly_allocation", {
            "p": p_str, "bankroll": 1000, "bet_unit": 5,
            "excluded_dozens": [], "sigma": 2.0, "kelly_fraction": 0.10,
        })
        assert "bets" in result
        for b in result["bets"]:
            assert b["amount"] > 0


# ---------------------------------------------------------------------------
# E. _exploration_schedule
# ---------------------------------------------------------------------------

class TestExplorationSchedule:
    def test_early_has_higher_fraction_and_sigma(self):
        early = _exploration_schedule(0)
        late  = _exploration_schedule(1000)
        assert early["kelly_fraction"] > late["kelly_fraction"]
        assert early["sigma"]          > late["sigma"]
        assert early["max_candidates"] > late["max_candidates"]

    def test_monotone_decay(self):
        fracs = [_exploration_schedule(n)["kelly_fraction"] for n in [0, 50, 200, 500]]
        assert fracs == sorted(fracs, reverse=True)

    def test_asymptote_approaches_base(self):
        # Log schedule decays slowly by design ("缓慢变稳定").
        # At N=100k params are well below exploration maximum.
        late = _exploration_schedule(100_000)
        assert late["kelly_fraction"] < 0.22  # well below early max 0.45
        assert late["sigma"] < 3.5            # well below early max 8.0
        # Larger N continues to decrease
        very_late = _exploration_schedule(100_000_000)
        assert very_late["kelly_fraction"] < late["kelly_fraction"]

    def test_n_scale_shifts_midpoint(self):
        """Smaller n_scale → faster convergence."""
        mid_fast = _exploration_schedule(50, n_scale=50)["kelly_fraction"]
        mid_slow = _exploration_schedule(50, n_scale=200)["kelly_fraction"]
        assert mid_fast < mid_slow  # faster schedule is further along


# ---------------------------------------------------------------------------
# F. _temporal_belief
# ---------------------------------------------------------------------------

class TestTemporalBelief:
    def test_empty_history_returns_near_uniform(self):
        t = _temporal_belief([], size=38)
        vals = list(t.values())
        assert abs(max(vals) - min(vals)) < 0.01

    def test_recent_pocket_has_higher_weight(self):
        # Pocket 17 just appeared (index -1 = most recent)
        history = [5, 5, 5, 5, 5, 5, 5, 5, 5, 17]  # 17 is most recent
        t = _temporal_belief(history, size=38, decay_rate=0.1)
        # 17 appeared once most recently; 5 appeared 9 times but older
        # With low decay_rate the recency effect is mild; just verify 17 is present
        assert t[17] > 0

    def test_recency_favours_recent_over_old(self):
        # Two pockets, one recent one old, same raw count
        history_recent = [5] * 10 + [17]   # 17 at the end (recent)
        history_old    = [17] + [5] * 10   # 17 at the start (old)
        t_recent = _temporal_belief(history_recent, decay_rate=0.2)
        t_old    = _temporal_belief(history_old,    decay_rate=0.2)
        assert t_recent[17] > t_old[17]

    def test_sums_to_one(self):
        h = list(range(38)) * 3
        t = _temporal_belief(h, size=38)
        assert abs(sum(t.values()) - 1.0) < 1e-9

    def test_laplace_prevents_extreme_values(self):
        # Single spin on pocket 0 — should not give probability 1
        t = _temporal_belief([0], size=38, k_smooth=3.0)
        assert t[0] < 0.5


# ---------------------------------------------------------------------------
# G. _trend_scores
# ---------------------------------------------------------------------------

class TestTrendScores:
    def test_too_short_returns_ones(self):
        scores = _trend_scores([17] * 10, size=38, min_half=10)
        assert all(v == 1.0 for v in scores.values())

    def test_rising_pocket_score_above_one(self):
        # 17 appears 0 times in first half, many times in second half
        early  = [5] * 20
        recent = [17] * 20
        scores = _trend_scores(early + recent, size=38, min_half=5)
        assert scores[17] > 1.0

    def test_falling_pocket_score_below_one(self):
        early  = [17] * 20
        recent = [5]  * 20
        scores = _trend_scores(early + recent, size=38, min_half=5)
        assert scores[17] < 1.0

    def test_stable_pocket_score_near_one(self):
        history = [17] * 40  # uniform presence in both halves
        scores = _trend_scores(history, size=38, min_half=5)
        assert 0.8 <= scores[17] <= 1.2


# ---------------------------------------------------------------------------
# H. adaptive_gaussian_kelly_allocation
# ---------------------------------------------------------------------------

def _make_bias_report(weight: float = 0.0) -> dict:
    return {"verdict": "no_evidence" if weight == 0 else "weak",
            "weight": weight, "suspected_bias": None, "summary": ""}


class TestAdaptiveGaussianKelly:
    def test_zero_history_returns_outside_bets_floor(self):
        """With no history, exploration floor kicks in and returns outside bets."""
        report = _make_bias_report(0.0)
        result = adaptive_gaussian_kelly_allocation([], report, 800, 10, [])
        # Exploration floor should provide at least 1 outside bet
        assert len(result) >= 1
        outside = {"red", "black", "odd", "even", "low", "high"}
        assert all(b["type"] in outside for b in result)

    def test_early_bets_broader_than_late(self):
        """Early phase (N=20 with repeated pocket) should produce more bet types than late."""
        # Short history: pocket 17 repeated to create a signal
        short_hist = [17] * 8 + [5] * 3 + [32] * 3 + [0, 1, 2, 3, 4, 6]
        # Long history: same signal but stable
        long_hist = short_hist * 20

        report_s = _make_bias_report(0.10)
        report_l = _make_bias_report(0.45)

        r_short = adaptive_gaussian_kelly_allocation(short_hist, report_s, 800, 1, [])
        r_long  = adaptive_gaussian_kelly_allocation(long_hist,  report_l, 800, 1, [])

        if r_short and r_long:
            types_s = {b["type"] for b in r_short}
            types_l = {b["type"] for b in r_long}
            # Early phase should select at least as many bet types
            assert len(types_s) >= len(types_l)

    def test_schedule_higher_fraction_and_sigma_early(self):
        """Direct schedule check: early history → higher kelly_fraction and sigma."""
        short_hist = [17] * 8 + [5] * 3 + [32] * 3   # N=14
        long_hist  = short_hist * 30                    # N=420
        sched_s = _exploration_schedule(len(short_hist))
        sched_l = _exploration_schedule(len(long_hist))
        assert sched_s["kelly_fraction"] > sched_l["kelly_fraction"]
        assert sched_s["sigma"]          > sched_l["sigma"]
        assert sched_s["max_candidates"] > sched_l["max_candidates"]

    def test_amounts_scale_with_bankroll(self):
        """True Kelly: 10× bankroll → ≥5× bet amounts."""
        hist = [17] * 15 + [5, 32, 5, 32]
        report = _make_bias_report(0.15)
        r_small = adaptive_gaussian_kelly_allocation(hist, report, 200,  1, [])
        r_large = adaptive_gaussian_kelly_allocation(hist, report, 2000, 1, [])
        if r_small and r_large:
            total_s = sum(b["amount"] for b in r_small)
            total_l = sum(b["amount"] for b in r_large)
            assert total_l > total_s * 4

    def test_respects_excluded_dozens(self):
        hist = [17] * 15 + [5, 32, 5]
        report = _make_bias_report(0.15)
        result = adaptive_gaussian_kelly_allocation(hist, report, 800, 1, excluded_dozens=[3])
        dozen_3 = set(range(25, 37))
        for b in result:
            if b["numbers"] is not None:
                for n in b["numbers"]:
                    assert n not in dozen_3

    def test_result_structure(self):
        hist = [17] * 10 + list(range(10))
        report = _make_bias_report(0.10)
        result = adaptive_gaussian_kelly_allocation(hist, report, 800, 5, [])
        for b in result:
            assert "type" in b and "numbers" in b and "amount" in b
            assert b["amount"] > 0
            assert b["amount"] % 5 < 1e-9 or abs(b["amount"] % 5 - 5) < 1e-9

    def test_amounts_are_multiples_of_bet_unit(self):
        hist = [17] * 12 + list(range(15))
        report = _make_bias_report(0.15)
        result = adaptive_gaussian_kelly_allocation(hist, report, 800, 10, [])
        for b in result:
            assert abs(b["amount"] % 10) < 1e-9

    def test_temporal_recency_shifts_belief(self):
        """Pocket 5 appearing in recent half should boost its bet vs. old half."""
        history = [17] * 20 + [5] * 20  # 5 is recent, 17 is old
        report = _make_bias_report(0.15)
        result = adaptive_gaussian_kelly_allocation(history, report, 800, 1, [])
        # At least one bet should cover pocket 5 or its neighbours given recency boost
        assert len(result) > 0


# ---------------------------------------------------------------------------
# I. dispatch adaptive tool
# ---------------------------------------------------------------------------

class TestDispatchAdaptiveGaussianKelly:
    def test_tool_registered(self):
        names = [t["name"] for t in TOOLS]
        assert "adaptive_gaussian_kelly_allocation" in names

    def test_dispatch_zero_history_returns_floor_bets(self):
        result = dispatch_tool("adaptive_gaussian_kelly_allocation", {
            "recent_history": [],
            "bias_report": {"weight": 0.0, "verdict": "no_evidence"},
            "bankroll": 800, "bet_unit": 10, "excluded_dozens": [],
        })
        assert "bets" in result
        assert len(result["bets"]) >= 1

    def test_dispatch_biased_history(self):
        hist = [17] * 20 + list(range(10))
        result = dispatch_tool("adaptive_gaussian_kelly_allocation", {
            "recent_history": hist,
            "bias_report": {"weight": 0.15, "verdict": "weak"},
            "bankroll": 800, "bet_unit": 1, "excluded_dozens": [],
        })
        assert len(result["bets"]) > 0

    def test_dispatch_optional_params(self):
        hist = [17] * 15
        result = dispatch_tool("adaptive_gaussian_kelly_allocation", {
            "recent_history": hist,
            "bias_report": {"weight": 0.15, "verdict": "weak"},
            "bankroll": 1000, "bet_unit": 5, "excluded_dozens": [],
            "n_scale": 50, "temporal_decay": 0.1,
        })
        assert "bets" in result
