"""Anthropic tool definitions (JSON schema) and dispatch router for all agent-callable Python functions."""

from __future__ import annotations

import json

from roulette_agent.belief import compute_belief
from roulette_agent.bias_detector import detect_bias as _detect_bias
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

def dispatch_tool(name: str, args: dict) -> dict:
    """Route a tool call from Claude to the corresponding Python function."""
    if name == "detect_bias":
        return _detect_bias(
            recent_history=args["recent_history"],
            wheel_type=args.get("wheel_type", "american"),
            external_stats=args.get("external_stats"),
            external_n_estimate=args.get("external_n_estimate"),
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
            )
        }

    if name == "settle":
        return _settle(
            bets=args["bets"],
            result_number=int(args["result_number"]),
        )

    raise ValueError(f"Unknown tool: {name!r}")
