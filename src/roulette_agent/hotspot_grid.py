"""Gaussian multi-hotspot heat map and true portfolio-Kelly allocation.

Three public allocators
-----------------------
gaussian_kelly_allocation
    Pure portfolio Kelly on a Gaussian heat map.  Requires a pre-built belief
    distribution p and a bias_report weight > 0 to return any bets.

adaptive_gaussian_kelly_allocation  ← primary for live play
    Wraps gaussian_kelly_allocation with two enhancements:

    1. Log-scheduled exploration → exploitation
       As history length N grows, parameters decay logarithmically:
         kelly_fraction : 0.45 (N=0) → 0.15 (N=∞)   large stake early, stable late
         sigma          : 8.0  (N=0) → 2.5 (N=∞)    wide coverage early, tight late
         max_candidates : 35   (N=0) → 15 (N=∞)     more bet types early
       When kelly returns no bets and N < n_scale, a broad set of outside bets is
       placed at amplified amounts (3× bet_unit decaying to 1×) as a floor.

    2. Temporal belief blending
       Three belief components are blended in proportion that all decay with N:
         p_static   : standard Dirichlet posterior from compute_belief
         p_temporal : exponentially-weighted recent frequencies (recency bias)
         p_explore  : raw Laplace-smoothed counts (fires before bias detector)
       Trend multiplier boosts the Gaussian heat of pockets whose frequency is
       rising in the second half of history versus the first half.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize

from roulette_agent.belief import compute_belief
from roulette_agent.layout import BET_TYPES, WHEEL_ORDER_AMERICAN, WHEEL_ORDER_EUROPEAN
from roulette_agent.optimizer import enumerate_legal_bets, fixed_baseline_allocation


# ---------------------------------------------------------------------------
# Hotspot detection
# ---------------------------------------------------------------------------

class Hotspot(NamedTuple):
    number: int
    rank: int      # 1 = primary, 2 = secondary, …
    excess: float  # p[n] − p_uniform


def detect_hotspots(p: dict[int, float], top_k: int = 5) -> list[Hotspot]:
    """Return up to top_k pockets ranked by probability excess above uniform."""
    p_uniform = 1.0 / len(p)
    ranked = sorted(
        ((n, prob - p_uniform) for n, prob in p.items() if prob > p_uniform),
        key=lambda x: x[1],
        reverse=True,
    )
    return [Hotspot(n, i + 1, e) for i, (n, e) in enumerate(ranked[:top_k])]


# ---------------------------------------------------------------------------
# Gaussian heat map in wheel space
# ---------------------------------------------------------------------------

def _get_wheel(wheel_type: str) -> list[int]:
    return WHEEL_ORDER_AMERICAN if wheel_type == "american" else WHEEL_ORDER_EUROPEAN


def _arc_distance(ia: int, ib: int, n_slots: int) -> int:
    d = abs(ia - ib)
    return min(d, n_slots - d)


def build_heat_map(
    p: dict[int, float],
    wheel_type: str = "american",
    sigma: float = 3.0,
    top_k: int = 5,
) -> dict[int, float]:
    """
    Gaussian-smoothed probability heat map in physical wheel space.

    Each hotspot radiates heat to its wheel neighbours via exp(-d²/2σ²).
    Multiple hotspot halos add, producing elevated saddle regions between
    co-located hot zones and attenuated heat between opposing ones.
    """
    wheel = _get_wheel(wheel_type)
    pos_idx: dict[int, int] = {n: i for i, n in enumerate(wheel)}
    n_slots = len(wheel)
    hotspots = detect_hotspots(p, top_k=top_k)

    heat: dict[int, float] = {}
    for pocket in wheel:
        h = 0.0
        for hs in hotspots:
            dist = _arc_distance(pos_idx[pocket], pos_idx[hs.number], n_slots)
            h += hs.excess * math.exp(-dist * dist / (2.0 * sigma * sigma))
        heat[pocket] = h
    return heat


# ---------------------------------------------------------------------------
# Payout matrix builder
# ---------------------------------------------------------------------------

def _build_payout_matrix(
    candidates: list[dict],
    outcomes: list[int],
) -> np.ndarray:
    """Shape (n_bets, n_outcomes). Net return per unit stake for each outcome."""
    outcome_idx = {o: i for i, o in enumerate(outcomes)}
    mat = np.full((len(candidates), len(outcomes)), -1.0)
    for b, bet in enumerate(candidates):
        payout = float(BET_TYPES[bet["type"]].payout)
        for n in bet["covered"]:
            if n in outcome_idx:
                mat[b, outcome_idx[n]] = payout
    return mat


# ---------------------------------------------------------------------------
# True portfolio-Kelly optimiser
# ---------------------------------------------------------------------------

def _portfolio_kelly_fractions(
    p_vec: np.ndarray,
    payout_matrix: np.ndarray,
    max_fraction: float,
) -> np.ndarray:
    """
    True multi-bet Kelly via convex optimisation (SLSQP).

    Maximise  sum_n p[n] · log(1 + sum_b f_b · R[b,n])
    subject to  f_b ≥ 0,  sum_b f_b ≤ max_fraction.
    """
    n_bets = payout_matrix.shape[0]
    R_T = payout_matrix.T

    def neg_log_growth(f: np.ndarray) -> float:
        bf = 1.0 + R_T @ f
        if np.any(bf <= 1e-12):
            return 1e10
        return -float(np.dot(p_vec, np.log(bf)))

    def gradient(f: np.ndarray) -> np.ndarray:
        bf = 1.0 + R_T @ f
        if np.any(bf <= 1e-12):
            return np.zeros(n_bets)
        return -(payout_matrix @ (p_vec / bf))

    x0 = np.full(n_bets, max_fraction / n_bets * 0.05)
    bounds = [(0.0, max_fraction)] * n_bets
    constraint = {
        "type": "ineq",
        "fun": lambda f: max_fraction - f.sum(),
        "jac": lambda _: -np.ones(n_bets),
    }
    try:
        res = minimize(
            neg_log_growth, x0, jac=gradient,
            method="SLSQP", bounds=bounds, constraints=constraint,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        return np.clip(res.x, 0.0, max_fraction)
    except Exception:
        return np.zeros(n_bets)


# ---------------------------------------------------------------------------
# Shared core scorer / allocator (used by both public functions)
# ---------------------------------------------------------------------------

def _run_gaussian_kelly(
    p: dict[int, float],
    heat: dict[int, float],
    bankroll: float,
    bet_unit: float,
    legal: list[dict],
    max_candidates: int,
    kelly_fraction: float,
) -> list[dict]:
    """Score legal bets, select top candidates, solve portfolio Kelly, round amounts."""
    scored: list[tuple[float, dict]] = []
    for bet in legal:
        covered = bet["covered"]
        payout = BET_TYPES[bet["type"]].payout
        p_win = sum(p.get(n, 0.0) for n in covered)
        ev = payout * p_win - (1.0 - p_win)
        if ev <= 0:
            continue
        f_star = ev / payout
        if f_star <= 0:
            continue
        heat_score = sum(heat.get(n, 0.0) for n in covered) / len(covered)
        scored.append((heat_score * f_star, bet))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = [bet for _, bet in scored[:max_candidates]]

    outcomes = sorted(p.keys())
    p_vec = np.array([p[n] for n in outcomes])
    payout_matrix = _build_payout_matrix(candidates, outcomes)
    fractions = _portfolio_kelly_fractions(p_vec, payout_matrix, kelly_fraction)

    result: list[dict] = []
    for bet, frac in zip(candidates, fractions):
        raw = frac * bankroll
        amount = round(raw / bet_unit) * bet_unit
        if amount < bet_unit:
            continue
        result.append({"type": bet["type"], "numbers": bet["numbers"], "amount": float(amount)})
    return result


# ---------------------------------------------------------------------------
# gaussian_kelly_allocation  (unchanged public API)
# ---------------------------------------------------------------------------

def gaussian_kelly_allocation(
    p: dict[int, float],
    bankroll: float,
    bet_unit: float,
    excluded_dozens: list[int],
    wheel_type: str = "american",
    sigma: float = 3.0,
    top_k_hotspots: int = 5,
    max_candidates: int = 20,
    kelly_fraction: float = 0.25,
) -> list[dict]:
    """
    Gaussian multi-hotspot grid analysis + true portfolio-Kelly allocation.

    Pipeline:
    1. Build Gaussian heat map on physical wheel from the top-k hotspots.
    2. Enumerate all legal bets; compute heat_score × individual f* for each.
    3. Keep top max_candidates positive-EV bets as portfolio candidates.
    4. Solve true portfolio-Kelly (SLSQP, log-growth maximisation).
    5. Amounts = fraction × bankroll, rounded to nearest bet_unit; drop < bet_unit.

    Returns [] when no bet has positive EV (uniform / no-evidence wheel).
    """
    if bankroll < bet_unit:
        return []
    heat = build_heat_map(p, wheel_type=wheel_type, sigma=sigma, top_k=top_k_hotspots)
    legal = enumerate_legal_bets(excluded_dozens)
    return _run_gaussian_kelly(p, heat, bankroll, bet_unit, legal, max_candidates, kelly_fraction)


# ---------------------------------------------------------------------------
# Adaptive components
# ---------------------------------------------------------------------------

def _exploration_schedule(n_history: int, n_scale: float = 100.0) -> dict:
    """
    Log-decaying schedule from exploration (small N) to exploitation (large N).

    Formula: param = base + extra / (1 + log(1 + N/n_scale))
      N=0      → t=0,    decay=1.0   → params at exploration maximum
      N=n_scale → t≈0.69, decay≈0.59 → ~60% of the way to stable
      N=∞       → t→∞,   decay→0    → params at stable minimum
    """
    t = math.log(1.0 + n_history / n_scale)
    decay = 1.0 / (1.0 + t)
    return {
        "kelly_fraction":  0.15 + 0.30 * decay,    # 0.45 → 0.15
        "sigma":           2.5  + 5.5  * decay,     # 8.0  → 2.5
        "max_candidates":  int(15 + 20 * decay),    # 35   → 15
        "decay":           decay,
    }


def _temporal_belief(
    recent_history: list[int],
    size: int = 38,
    decay_rate: float = 0.05,
    k_smooth: float = 3.0,
) -> dict[int, float]:
    """
    Exponentially-weighted pocket frequency distribution.

    Most-recent spin has weight 1; a spin k steps back has weight exp(-decay_rate·k).
    Laplace smoothing (k_smooth) prevents extreme probabilities from single observations.
    """
    raw: dict[int, float] = {n: 0.0 for n in range(size)}
    n_spins = len(recent_history)
    for i, pocket in enumerate(recent_history):
        age = n_spins - 1 - i
        w = math.exp(-decay_rate * age)
        if 0 <= pocket < size:
            raw[pocket] += w

    # Laplace smoothing
    smooth_add = k_smooth / size
    smoothed = {n: raw[n] + smooth_add for n in range(size)}
    total = sum(smoothed.values())
    return {n: v / total for n, v in smoothed.items()}


def _trend_scores(
    recent_history: list[int],
    size: int = 38,
    min_half: int = 10,
) -> dict[int, float]:
    """
    Return p_recent[n] / p_early[n] for each pocket (ratio ≥ 1 = trending up).

    Splits history into two equal halves; uses Laplace smoothing to avoid 0/0.
    Returns all-ones dict when history is too short (< 2 × min_half).
    """
    n = len(recent_history)
    if n < 2 * min_half:
        return {pocket: 1.0 for pocket in range(size)}

    half = n // 2
    early  = Counter(recent_history[:half])
    recent = Counter(recent_history[half:])

    k = 1.0  # Laplace prior per pocket
    denom_early  = half  + size * k
    denom_recent = n - half + size * k

    scores: dict[int, float] = {}
    for pocket in range(size):
        p_early  = (early.get(pocket,  0) + k) / denom_early
        p_recent = (recent.get(pocket, 0) + k) / denom_recent
        scores[pocket] = p_recent / p_early
    return scores


def _blend_beliefs(
    p_static:   dict[int, float],
    p_temporal: dict[int, float],
    p_explore:  dict[int, float],
    w_temporal: float,
    w_explore:  float,
) -> dict[int, float]:
    """Linear blend of three beliefs, renormalised to sum=1."""
    w_static = max(0.0, 1.0 - w_temporal - w_explore)
    size = len(p_static)
    blended = {
        n: w_static * p_static[n] + w_temporal * p_temporal[n] + w_explore * p_explore[n]
        for n in range(size)
    }
    total = sum(blended.values())
    return {n: v / total for n, v in blended.items()}


# ---------------------------------------------------------------------------
# adaptive_gaussian_kelly_allocation  (primary live-play allocator)
# ---------------------------------------------------------------------------

def adaptive_gaussian_kelly_allocation(
    recent_history: list[int],
    bias_report: dict,
    bankroll: float,
    bet_unit: float,
    excluded_dozens: list[int],
    wheel_type: str = "american",
    k_prior: float = 10.0,
    n_scale: float = 100.0,
    temporal_decay: float = 0.05,
    temporal_k_smooth: float = 3.0,
    trend_blend: float = 0.20,
) -> list[dict]:
    """
    Adaptive exploration→exploitation Gaussian Kelly with temporal analysis.

    Exploration schedule (logarithmic decay with N):
    ┌─────────────────┬──────────┬──────────┬──────────┐
    │  Parameter      │  N = 0   │ N = 100  │  N → ∞   │
    ├─────────────────┼──────────┼──────────┼──────────┤
    │ kelly_fraction  │  0.45    │  0.32    │  0.15    │
    │ sigma (steps)   │  8.0     │  5.7     │  2.5     │
    │ max_candidates  │  35      │  27      │  15      │
    └─────────────────┴──────────┴──────────┴──────────┘
    All decay as  base + extra / (1 + log(1 + N/n_scale)).

    Belief blending (components that fade as N → ∞):
    • p_static   – Dirichlet posterior from compute_belief (always present)
    • p_explore  – raw Laplace-smoothed counts; fires before bias detector fires
    • p_temporal – exponentially-weighted recent frequencies; emphasises last ~20 spins

    Trend boost: pockets whose frequency is rising in the second half of history
    have their Gaussian heat amplified by (1 + trend_blend × (trend_ratio − 1)).

    Exploration floor: when N < n_scale and kelly returns [], place amplified outside
    bets at  bet_unit × (1 + 2 × decay) rounded to nearest bet_unit, giving broad
    coverage at a declining premium until the history is long enough for pure Kelly.
    """
    if bankroll < bet_unit:
        return []

    n = len(recent_history)
    size = 38 if wheel_type == "american" else 37

    # ── 1. Adaptive schedule ─────────────────────────────────────────────────
    sched = _exploration_schedule(n, n_scale)
    kelly_fraction = sched["kelly_fraction"]
    sigma          = sched["sigma"]
    max_candidates = sched["max_candidates"]
    decay          = sched["decay"]

    # ── 2. Belief components ─────────────────────────────────────────────────
    p_static = compute_belief(recent_history, bias_report, wheel_type, k_prior)

    # Exploration belief: raw frequency with Laplace smoothing (no bias test required)
    raw_counts = Counter(recent_history)
    pseudo_k = 0.5
    denom_exp = n + size * pseudo_k
    p_explore = {i: (raw_counts.get(i, 0) + pseudo_k) / denom_exp for i in range(size)}
    p_e_total = sum(p_explore.values())
    p_explore = {i: v / p_e_total for i, v in p_explore.items()}

    # Temporal belief: exponentially-weighted recency
    p_temporal = _temporal_belief(recent_history, size, temporal_decay, temporal_k_smooth)

    # Blend weights decay with N; explore/temporal ramp up only after N ≥ 5
    ramp = min(1.0, n / max(1, n_scale * 0.05))   # 0→1 over first 5% of n_scale
    w_explore  = 0.10 * decay * ramp
    w_temporal = 0.25 * decay * min(1.0, n / max(1, n_scale * 0.20))  # ramp over 20%

    p_eff = _blend_beliefs(p_static, p_temporal, p_explore, w_temporal, w_explore)

    # ── 3. Trend-boosted heat map ─────────────────────────────────────────────
    heat = build_heat_map(p_eff, wheel_type=wheel_type, sigma=sigma)
    trend = _trend_scores(recent_history, size)
    t_strength = trend_blend * decay  # trend boost decays with N
    heat_boosted = {
        pocket: h * max(0.01, 1.0 + t_strength * (trend.get(pocket, 1.0) - 1.0))
        for pocket, h in heat.items()
    }

    # ── 4. Core Kelly allocation ──────────────────────────────────────────────
    legal = enumerate_legal_bets(excluded_dozens)
    bets = _run_gaussian_kelly(
        p_eff, heat_boosted, bankroll, bet_unit,
        legal, max_candidates, kelly_fraction,
    )

    # ── 5. Exploration floor ──────────────────────────────────────────────────
    # When Kelly finds nothing (uniform history) and we're still exploring,
    # place amplified outside bets for broad coverage.
    if not bets and n < n_scale:
        amplified = round(bet_unit * (1.0 + 2.0 * decay) / bet_unit) * bet_unit
        amplified = max(bet_unit, amplified)
        outside_types = {"red", "black", "odd", "even", "low", "high"}
        bets = [
            {"type": b["type"], "numbers": None, "amount": float(amplified)}
            for b in legal if b["type"] in outside_types
        ]

    return bets
