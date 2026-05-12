"""Monte Carlo simulator: runs roulette strategies and returns bankroll trajectories."""

from __future__ import annotations

import numpy as np

from roulette_agent.belief import compute_belief
from roulette_agent.bias_detector import detect_bias
from roulette_agent.optimizer import (
    fixed_baseline_allocation,
    greedy_ev_allocation,
    kelly_allocation,
)
from roulette_agent.settler import settle


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_spin_probs(size: int, bias_inject: dict | None) -> list[float]:
    if bias_inject is None:
        return [1.0 / size] * size
    probs = [0.0] * size
    injected: set[int] = set()
    total_injected = 0.0
    for pocket, prob in bias_inject.items():
        idx = int(pocket)
        probs[idx] = float(prob)
        total_injected += float(prob)
        injected.add(idx)
    remaining = 1.0 - total_injected
    n_rem = size - len(injected)
    if n_rem > 0:
        per = remaining / n_rem
        for i in range(size):
            if i not in injected:
                probs[i] = per
    total = sum(probs)
    return [p / total for p in probs]


def _get_bets(
    strategy_name: str,
    p: dict[int, float],
    bankroll: float,
    bet_unit: float,
    excluded_dozens: list[int],
    weight: float = 0.0,
) -> list[dict]:
    # When no bias is detected, greedy/kelly will find no positive-EV bet — skip allocation.
    if weight == 0.0 and strategy_name in ("greedy_ev", "kelly"):
        return []
    if strategy_name == "greedy_ev":
        return greedy_ev_allocation(p, bankroll, bet_unit, excluded_dozens)
    if strategy_name == "kelly":
        return kelly_allocation(p, bankroll, bet_unit, excluded_dozens)
    if strategy_name == "fixed_baseline":
        return fixed_baseline_allocation(p, bankroll, bet_unit, excluded_dozens)
    if strategy_name == "always_red":
        if bankroll >= bet_unit:
            return [{"type": "red", "numbers": None, "amount": bet_unit}]
        return []
    if strategy_name == "no_bet":
        return []
    raise ValueError(f"Unknown strategy: {strategy_name!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate(
    strategy_name: str,
    n_spins: int,
    initial_bankroll: float,
    bet_unit: float,
    wheel_type: str = "american",
    excluded_dozens: list[int] | None = None,
    bias_inject: dict | None = None,
    seed: int | None = None,
    _refresh_every: int = 1,
) -> dict:
    """Run a single simulation and return trajectory data.

    strategy_name : "greedy_ev" | "kelly" | "fixed_baseline" | "always_red" | "no_bet"
    bias_inject   : e.g. {35: 3/38} — pocket 35 gets p=3/38; rest share remaining prob.
    _refresh_every: recompute bias/belief every N spins (private; 1 = every spin).

    Returns dict with keys:
      bankroll_curve (len n_spins+1), spin_history, total_pnl,
      final_bankroll, ruined, bets_log (len n_spins).
    """
    if excluded_dozens is None:
        excluded_dozens = []

    rng = np.random.default_rng(seed)
    size = 38 if wheel_type == "american" else 37
    pockets = np.arange(size)
    spin_probs = np.array(_build_spin_probs(size, bias_inject))

    bankroll = float(initial_bankroll)
    bankroll_curve: list[float] = [bankroll]
    spin_history: list[int] = []
    bets_log: list[list[dict]] = []

    bias_report: dict = {"weight": 0.0}
    p: dict[int, float] = {n: 1.0 / size for n in range(size)}
    ruined = False

    for i in range(n_spins):
        if i % _refresh_every == 0:
            bias_report = detect_bias(spin_history, wheel_type)
            p = compute_belief(spin_history, bias_report, wheel_type)

        bets = _get_bets(strategy_name, p, bankroll, bet_unit, excluded_dozens,
                         weight=bias_report.get("weight", 0.0))
        bets_log.append(bets)

        result = int(rng.choice(pockets, p=spin_probs))
        spin_history.append(result)

        if bets:
            outcome = settle(bets, result)
            bankroll += outcome["pnl"]

        bankroll = max(0.0, bankroll)
        bankroll_curve.append(bankroll)

        if bankroll <= 0:
            remaining = n_spins - i - 1
            bankroll_curve.extend([0.0] * remaining)
            bets_log.extend([[] for _ in range(remaining)])
            ruined = True
            break

    return {
        "bankroll_curve": bankroll_curve,
        "spin_history": spin_history,
        "total_pnl": bankroll - initial_bankroll,
        "final_bankroll": bankroll,
        "ruined": ruined,
        "bets_log": bets_log,
    }


def compare_strategies(
    strategies: list[str],
    n_spins: int,
    n_runs: int,
    initial_bankroll: float,
    bet_unit: float,
    wheel_type: str = "american",
    seed: int | None = None,
) -> dict:
    """Run n_runs simulations for each strategy and aggregate statistics.

    Returns a dict keyed by strategy name, each containing:
    mean/median/std/max/min final_bankroll and ruin_rate.
    Uses _refresh_every=100 internally for performance over many runs.
    """
    results: dict[str, dict] = {}
    for s_idx, strategy in enumerate(strategies):
        finals: list[float] = []
        ruin_count = 0
        for run in range(n_runs):
            run_seed = (
                None if seed is None
                else seed * 100_000 + s_idx * 10_000 + run
            )
            r = simulate(
                strategy_name=strategy,
                n_spins=n_spins,
                initial_bankroll=initial_bankroll,
                bet_unit=bet_unit,
                wheel_type=wheel_type,
                seed=run_seed,
                _refresh_every=100,
            )
            finals.append(r["final_bankroll"])
            if r["ruined"]:
                ruin_count += 1

        arr = np.array(finals, dtype=float)
        results[strategy] = {
            "mean_final_bankroll":   float(np.mean(arr)),
            "median_final_bankroll": float(np.median(arr)),
            "std_final_bankroll":    float(np.std(arr)),
            "max_final_bankroll":    float(np.max(arr)),
            "min_final_bankroll":    float(np.min(arr)),
            "ruin_rate":             ruin_count / n_runs,
            "n_runs":                n_runs,
        }

    return results
