"""Anthropic tool definitions (JSON schema) and dispatch router for all agent-callable Python functions."""

from __future__ import annotations

import json

from roulette_agent.belief import compute_belief
from roulette_agent.bias_detector import detect_bias as _detect_bias
from roulette_agent.hotspot_grid import (
    adaptive_gaussian_kelly_allocation as _adaptive_gaussian_kelly,
    gaussian_kelly_allocation as _gaussian_kelly,
)
from roulette_agent.optimizer import (
    fixed_baseline_allocation as _fixed_baseline,
    greedy_ev_allocation as _greedy_ev,
    kelly_allocation as _kelly,
)
from roulette_agent.settler import settle as _settle

# ---------------------------------------------------------------------------
# Tool schemas  (Anthropic tool_use format)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "detect_bias",
        "description": (
            "Run statistical bias tests (χ², binomial, hot-numbers) on the recent spin history. "
            "Returns verdict ('no_evidence'|'weak'|'moderate'|'strong'), weight ∈ [0,1], "
            "suspected_bias, and a human-readable summary. "
            "External stats can downgrade a weak verdict but NEVER upgrade."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "recent_history": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of recent result numbers (0–37 for American, 0–36 for European).",
                },
                "wheel_type": {
                    "type": "string",
                    "enum": ["american", "european"],
                    "description": "Wheel variant. Default 'american'.",
                },
                "external_stats": {
                    "type": "object",
                    "description": (
                        "Optional display-screen aggregates, e.g. {\"red_pct\": 0.55}. "
                        "For reference only — cannot raise the verdict."
                    ),
                    "additionalProperties": {"type": "number"},
                },
                "external_n_estimate": {
                    "type": "integer",
                    "description": "Estimated N behind external_stats. Used only for auxiliary p-values.",
                },
            },
            "required": ["recent_history"],
        },
    },
    {
        "name": "compute_belief",
        "description": (
            "Build a Dirichlet-mixture belief distribution over all pockets: "
            "p(n) = w·p_bias(n) + (1−w)·p_uniform(n). "
            "Returns a dict mapping pocket number → probability. "
            "Pass the bias_report returned by detect_bias."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "recent_history": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Same history used for detect_bias.",
                },
                "bias_report": {
                    "type": "object",
                    "description": "Full dict returned by detect_bias (must contain 'weight').",
                },
                "wheel_type": {
                    "type": "string",
                    "enum": ["american", "european"],
                },
                "k_prior": {
                    "type": "number",
                    "description": "Dirichlet concentration prior. Default 10.0.",
                },
            },
            "required": ["recent_history", "bias_report"],
        },
    },
    {
        "name": "kelly_allocation",
        "description": (
            "Fractional Kelly bet sizing from a shared budget (fraction × bankroll). "
            "Returns only bets with positive Kelly fraction. "
            "Returns [] on a fair wheel (all fractions ≤ 0)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "p": {
                    "type": "object",
                    "description": "Belief distribution from compute_belief (pocket → probability).",
                    "additionalProperties": {"type": "number"},
                },
                "bankroll": {"type": "number", "description": "Current bankroll."},
                "bet_unit": {"type": "number", "description": "Minimum bet increment."},
                "excluded_dozens": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Dozens (1, 2, or 3) to exclude. [1,3] raises ValueError.",
                },
                "fraction": {
                    "type": "number",
                    "description": "Kelly fraction ∈ (0,1]. Default 0.25.",
                },
            },
            "required": ["p", "bankroll", "bet_unit", "excluded_dozens"],
        },
    },
    {
        "name": "greedy_ev_allocation",
        "description": (
            "Select up to top_k positive-EV bets, each sized at bet_unit. "
            "Returns [] when all EVs are ≤ 0 (fair wheel)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "p": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Belief distribution from compute_belief.",
                },
                "bankroll": {"type": "number"},
                "bet_unit": {"type": "number"},
                "excluded_dozens": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "max_bet_fraction": {
                    "type": "number",
                    "description": "Max fraction of bankroll per bet. Default 0.1.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of bets to return. Default 5.",
                },
            },
            "required": ["p", "bankroll", "bet_unit", "excluded_dozens"],
        },
    },
    {
        "name": "fixed_baseline_allocation",
        "description": (
            "Flat bet_unit on each positive-EV outside bet (red/black/odd/even/low/high). "
            "Falls back to the single best-EV outside bet when none are positive-EV. "
            "Returns [] when bankroll < bet_unit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "p": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Belief distribution from compute_belief.",
                },
                "bankroll": {"type": "number"},
                "bet_unit": {"type": "number"},
                "excluded_dozens": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
            "required": ["p", "bankroll", "bet_unit", "excluded_dozens"],
        },
    },
    {
        "name": "adaptive_gaussian_kelly_allocation",
        "description": (
            "PRIMARY live-play allocator. Two-step call: detect_bias → this tool. "
            "Combines Gaussian multi-hotspot Kelly with two adaptive enhancements:\n"
            "1. Log-scheduled exploration→exploitation: early (N<100) uses kelly_fraction≈0.45, "
            "sigma≈8 pocket-steps, 35 candidates — large stake, broadly distributed. "
            "As N grows all params decay to stable minimums (0.15 / 2.5 / 15). "
            "Formula: param = base + extra / (1 + log(1 + N/n_scale)).\n"
            "2. Temporal belief blending: mixes static Dirichlet belief with (a) an "
            "exploration prior (raw counts before bias detector fires) and (b) an "
            "exponentially-weighted recent-frequency belief (recency bias). "
            "Trend multiplier boosts Gaussian heat for rising pockets. "
            "Exploration floor: when Kelly finds no bets and N < n_scale, returns "
            "amplified outside bets (amount decays from 3× to 1× bet_unit)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "recent_history": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Full list of recent spin results (same list passed to detect_bias).",
                },
                "bias_report": {
                    "type": "object",
                    "description": "Full dict returned by detect_bias.",
                },
                "bankroll": {"type": "number"},
                "bet_unit": {
                    "type": "number",
                    "description": "Minimum bet resolution. Amounts are proportional to bankroll (true Kelly).",
                },
                "excluded_dozens": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "wheel_type": {
                    "type": "string",
                    "enum": ["american", "european"],
                },
                "n_scale": {
                    "type": "number",
                    "description": "Log-schedule half-life in spins. Default 100.",
                },
                "temporal_decay": {
                    "type": "number",
                    "description": "Exponential decay rate for temporal weighting. Default 0.05.",
                },
                "trend_blend": {
                    "type": "number",
                    "description": "Maximum trend-boost multiplier on heat scores. Default 0.20.",
                },
            },
            "required": ["recent_history", "bias_report", "bankroll", "bet_unit", "excluded_dozens"],
        },
    },
    {
        "name": "gaussian_kelly_allocation",
        "description": (
            "PRIMARY allocator when bias is detected (weight ≥ 0.10). "
            "Runs a full Gaussian multi-hotspot analysis on the physical wheel layout: "
            "identifies primary / secondary / tertiary hotspots, builds a Gaussian heat map "
            "(bandwidth = sigma wheel-steps), scores every legal bet type — straight, split, "
            "street, corner, six_line, dozen, column, red/black, odd/even, high/low — by "
            "heat_score × individual Kelly f*, then solves the true multi-bet portfolio-Kelly "
            "problem via SLSQP convex optimisation (maximises expected log-bankroll-growth). "
            "Bet amounts scale proportionally with bankroll (true Kelly behaviour). "
            "Returns [] when no bet has positive EV (uniform / no-evidence wheel)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "p": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Belief distribution from compute_belief (pocket → probability).",
                },
                "bankroll": {"type": "number", "description": "Current bankroll."},
                "bet_unit": {
                    "type": "number",
                    "description": "Minimum bet resolution. Amounts are rounded to the nearest bet_unit.",
                },
                "excluded_dozens": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Dozens (1, 2, or 3) to exclude.",
                },
                "wheel_type": {
                    "type": "string",
                    "enum": ["american", "european"],
                    "description": "Wheel variant. Default 'american'.",
                },
                "sigma": {
                    "type": "number",
                    "description": (
                        "Gaussian kernel bandwidth in pocket steps on the physical wheel. "
                        "Larger sigma spreads heat to more distant neighbours. Default 3.0."
                    ),
                },
                "top_k_hotspots": {
                    "type": "integer",
                    "description": "Number of top hotspots to include in the heat map. Default 5.",
                },
                "max_candidates": {
                    "type": "integer",
                    "description": "Maximum bet candidates fed into the portfolio optimiser. Default 20.",
                },
                "kelly_fraction": {
                    "type": "number",
                    "description": (
                        "Total risk budget as a fraction of bankroll (sum of all bet fractions ≤ this). "
                        "Default 0.25."
                    ),
                },
            },
            "required": ["p", "bankroll", "bet_unit", "excluded_dozens"],
        },
    },
    {
        "name": "settle",
        "description": (
            "Settle a list of bets against a result number. "
            "Returns total_staked, total_payout, pnl, and per-bet detail."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "bets": {
                    "type": "array",
                    "description": "List of bet dicts: {type, numbers, amount}.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "numbers": {
                                "oneOf": [
                                    {"type": "array", "items": {"type": "integer"}},
                                    {"type": "null"},
                                ]
                            },
                            "amount": {"type": "number"},
                        },
                        "required": ["type", "numbers", "amount"],
                    },
                },
                "result_number": {
                    "type": "integer",
                    "description": "The winning pocket number.",
                },
            },
            "required": ["bets", "result_number"],
        },
    },
]

# ---------------------------------------------------------------------------
# Dispatch router
# ---------------------------------------------------------------------------

def dispatch_tool(name: str, args: dict, context: dict | None = None) -> dict:
    """Route a tool call from Claude to the corresponding Python function.

    context carries session-level config injected by the server (not by the agent):
      custom_payouts   – dict[bet_type → int net payout] or None
      custom_wheel_order – list[int] physical wheel order or None
    """
    ctx = context or {}
    custom_payouts: dict | None = ctx.get("custom_payouts")
    wheel_order: list | None = ctx.get("custom_wheel_order")

    if name == "detect_bias":
        return _detect_bias(
            recent_history=args["recent_history"],
            wheel_type=args.get("wheel_type", "american"),
            external_stats=args.get("external_stats"),
            external_n_estimate=args.get("external_n_estimate"),
            wheel_order=wheel_order,
        )

    if name == "compute_belief":
        p = compute_belief(
            recent_history=args["recent_history"],
            bias_report=args["bias_report"],
            wheel_type=args.get("wheel_type", "american"),
            k_prior=args.get("k_prior", 10.0),
        )
        # Convert int keys to strings for JSON round-trip
        return {str(k): v for k, v in p.items()}

    if name == "kelly_allocation":
        p_raw = args["p"]
        p = {int(k): float(v) for k, v in p_raw.items()}
        return {
            "bets": _kelly(
                p=p,
                bankroll=float(args["bankroll"]),
                bet_unit=float(args["bet_unit"]),
                excluded_dozens=args.get("excluded_dozens", []),
                fraction=float(args.get("fraction", 0.25)),
                custom_payouts=custom_payouts,
            )
        }

    if name == "greedy_ev_allocation":
        p_raw = args["p"]
        p = {int(k): float(v) for k, v in p_raw.items()}
        return {
            "bets": _greedy_ev(
                p=p,
                bankroll=float(args["bankroll"]),
                bet_unit=float(args["bet_unit"]),
                excluded_dozens=args.get("excluded_dozens", []),
                max_bet_fraction=float(args.get("max_bet_fraction", 0.1)),
                top_k=int(args.get("top_k", 5)),
                custom_payouts=custom_payouts,
            )
        }

    if name == "fixed_baseline_allocation":
        p_raw = args["p"]
        p = {int(k): float(v) for k, v in p_raw.items()}
        return {
            "bets": _fixed_baseline(
                p=p,
                bankroll=float(args["bankroll"]),
                bet_unit=float(args["bet_unit"]),
                excluded_dozens=args.get("excluded_dozens", []),
                custom_payouts=custom_payouts,
            )
        }

    if name == "adaptive_gaussian_kelly_allocation":
        return {
            "bets": _adaptive_gaussian_kelly(
                recent_history=args["recent_history"],
                bias_report=args["bias_report"],
                bankroll=float(args["bankroll"]),
                bet_unit=float(args["bet_unit"]),
                excluded_dozens=args.get("excluded_dozens", []),
                wheel_type=args.get("wheel_type", "american"),
                n_scale=float(args.get("n_scale", 100.0)),
                temporal_decay=float(args.get("temporal_decay", 0.05)),
                trend_blend=float(args.get("trend_blend", 0.20)),
                wheel_order=wheel_order,
                custom_payouts=custom_payouts,
            )
        }

    if name == "gaussian_kelly_allocation":
        p_raw = args["p"]
        p = {int(k): float(v) for k, v in p_raw.items()}
        return {
            "bets": _gaussian_kelly(
                p=p,
                bankroll=float(args["bankroll"]),
                bet_unit=float(args["bet_unit"]),
                excluded_dozens=args.get("excluded_dozens", []),
                wheel_type=args.get("wheel_type", "american"),
                sigma=float(args.get("sigma", 3.0)),
                top_k_hotspots=int(args.get("top_k_hotspots", 5)),
                max_candidates=int(args.get("max_candidates", 20)),
                kelly_fraction=float(args.get("kelly_fraction", 0.25)),
                wheel_order=wheel_order,
                custom_payouts=custom_payouts,
            )
        }

    if name == "settle":
        return _settle(
            bets=args["bets"],
            result_number=int(args["result_number"]),
            custom_payouts=custom_payouts,
        )

    raise ValueError(f"Unknown tool: {name!r}")
