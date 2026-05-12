"""Bias-aware belief model: Dirichlet posterior mixed with uniform prior, weighted by the bias detector's confidence."""

from __future__ import annotations


def _wheel_size(wheel_type: str) -> int:
    return 38 if wheel_type == "american" else 37


def compute_belief(
    recent_history: list[int],
    bias_report: dict,
    wheel_type: str = "american",
    k_prior: float = 10.0,
) -> dict[int, float]:
    """Bias-aware mixture (README §3.1).

    p(n) = w * p_bias(n) + (1 - w) * p_uniform(n)

    p_bias(n) = (count(n) + k_prior) / (N + wheel_size * k_prior)   [Dirichlet posterior]
    p_uniform  = 1 / wheel_size                                       [fair wheel]
    w          = bias_report["weight"]  ∈ [0, 1]

    Only recent_history feeds count(n); external_stats is never used here.
    """
    size = _wheel_size(wheel_type)
    w = float(bias_report.get("weight", 0.0))
    N = len(recent_history)
    p_uniform = 1.0 / size

    # Dirichlet count vector — all pockets start at zero
    counts: dict[int, int] = {n: 0 for n in range(size)}
    for n in recent_history:
        if n in counts:
            counts[n] += 1

    denom = N + size * k_prior

    result: dict[int, float] = {}
    for n in range(size):
        p_bias = (counts[n] + k_prior) / denom
        result[n] = w * p_bias + (1.0 - w) * p_uniform

    return result
