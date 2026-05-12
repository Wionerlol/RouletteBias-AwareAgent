"""Statistical bias detection: χ² and binomial tests on the internal history, plus external-stats sanity checks."""

from __future__ import annotations

from scipy import stats

from roulette_agent.layout import (
    WHEEL_ORDER_AMERICAN,
    WHEEL_ORDER_EUROPEAN,
    color,
    column,
    dozen,
    high_low,
    parity,
)
from roulette_agent.stats import frequency_counts

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERDICT_THRESHOLDS: dict[str, dict] = {
    "weak":     {"p_max": 0.1,   "p_min": 0.01,  "n_min": 30},
    "moderate": {"p_max": 0.01,  "p_min": 0.001, "n_min": 200},
    "strong":   {"p_max": 0.001, "p_min": 0.0,   "n_min": 500},
}

VERDICT_WEIGHTS: dict[str, float] = {
    "no_evidence": 0.0,
    "weak":        0.15,
    "moderate":    0.45,
    "strong":      0.80,
}

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _green(wheel_type: str) -> set[int]:
    return {0, 37} if wheel_type == "american" else {0}


def _wheel_size(wheel_type: str) -> int:
    return 38 if wheel_type == "american" else 37


def _wheel_order(wheel_type: str) -> list[int]:
    return WHEEL_ORDER_AMERICAN if wheel_type == "american" else WHEEL_ORDER_EUROPEAN


def _internal_pct(group: str, non_green_history: list[int]) -> float:
    N = len(non_green_history)
    if N == 0:
        return 0.0
    dispatch = {
        "red":   lambda n: color(n) == "red",
        "black": lambda n: color(n) == "black",
        "odd":   lambda n: parity(n) == "odd",
        "even":  lambda n: parity(n) == "even",
        "low":   lambda n: high_low(n) == "low",
        "high":  lambda n: high_low(n) == "high",
    }
    return sum(1 for n in non_green_history if dispatch[group](n)) / N


# ---------------------------------------------------------------------------
# Internal statistical tests  (use ONLY recent_history)
# ---------------------------------------------------------------------------


def chi_square_single(recent_history: list[int], wheel_type: str = "american") -> dict:
    """χ² goodness-of-fit over all pockets (df = wheel_size - 1)."""
    N = len(recent_history)
    size = _wheel_size(wheel_type)
    exp_per = N / size if N > 0 else 0.0
    usable = exp_per >= 5  # requires N >= 190 for American
    reason = "ok" if usable else f"E_per_pocket={exp_per:.2f} < 5 (need N≥{5*size})"

    if N == 0:
        return {"stat": 0.0, "df": size - 1, "p_value": 1.0, "usable": False, "reason": "N=0"}

    freq = frequency_counts(recent_history, wheel_type)
    observed = [freq[n] for n in sorted(freq)]
    expected = [exp_per] * size
    stat, p_value = stats.chisquare(observed, expected)
    return {"stat": float(stat), "df": size - 1, "p_value": float(p_value),
            "usable": usable, "reason": reason}


def chi_square_sector(
    recent_history: list[int],
    wheel_type: str = "american",
    n_sectors: int = 8,
) -> dict:
    """χ² goodness-of-fit over wheel sectors (physical order)."""
    N = len(recent_history)
    wheel = _wheel_order(wheel_type)
    size = len(wheel)
    usable = N >= 50
    reason = "ok" if usable else f"N={N} < 50"

    # Partition wheel into n_sectors approximately equal segments
    base, rem = divmod(size, n_sectors)
    sectors: list[list[int]] = []
    start = 0
    for i in range(n_sectors):
        end = start + base + (1 if i < rem else 0)
        sectors.append(wheel[start:end])
        start = end

    number_to_sector = {n: s_idx for s_idx, seg in enumerate(sectors) for n in seg}
    observed = [0] * n_sectors
    for n in recent_history:
        if n in number_to_sector:
            observed[number_to_sector[n]] += 1

    sector_sizes = [len(s) for s in sectors]
    expected = [N * (sz / size) for sz in sector_sizes]

    if N == 0:
        return {"stat": 0.0, "df": n_sectors - 1, "p_value": 1.0,
                "usable": False, "reason": "N=0", "n_sectors": n_sectors}

    stat, p_value = stats.chisquare(observed, expected)
    return {"stat": float(stat), "df": n_sectors - 1, "p_value": float(p_value),
            "usable": usable, "reason": reason, "n_sectors": n_sectors}


def binomial_test(
    recent_history: list[int],
    group: str,
    wheel_type: str = "american",
) -> dict:
    """Two-tailed binomial test for a binary group.

    Uses non-green spins as effective N, p0=0.5 (each group covers exactly half
    of the 36 non-green outcomes in both American and European roulette).
    """
    valid = {"red", "black", "odd", "even", "low", "high"}
    if group not in valid:
        raise ValueError(f"group must be one of {valid}")

    green = _green(wheel_type)
    ng_history = [n for n in recent_history if n not in green]
    n_eff = len(ng_history)
    usable = n_eff >= 30

    if n_eff == 0:
        return {"group": group, "observed_pct": 0.0, "n_effective": 0,
                "p_value": 1.0, "usable": False}

    obs_pct = _internal_pct(group, ng_history)
    observed = round(obs_pct * n_eff)
    result = stats.binomtest(observed, n_eff, 0.5, alternative="two-sided")
    return {
        "group": group,
        "observed_pct": round(obs_pct, 4),
        "n_effective": n_eff,
        "p_value": float(result.pvalue),
        "usable": usable,
    }


def hot_numbers_test(
    recent_history: list[int],
    wheel_type: str = "american",
) -> list[dict]:
    """Per-number binomial tests (H0: p=1/wheel_size). Returns top-5 by p_uncorrected."""
    N = len(recent_history)
    size = _wheel_size(wheel_type)
    p0 = 1.0 / size

    if N == 0:
        return []

    freq = frequency_counts(recent_history, wheel_type)
    results = []
    for n in sorted(freq):
        obs = freq[n]
        exp = N * p0
        res = stats.binomtest(obs, N, p0, alternative="two-sided")
        p_unc = float(res.pvalue)
        results.append({
            "n": n,
            "observed": obs,
            "expected": round(exp, 2),
            "p_uncorrected": p_unc,
            "p_bonferroni": min(1.0, p_unc * size),
        })

    results.sort(key=lambda x: x["p_uncorrected"])
    return results[:5]


def _chi_square_group(
    recent_history: list[int],
    group_type: str,
    wheel_type: str = "american",
) -> dict:
    """χ² test for dozen or column distribution over non-green spins (3 categories)."""
    green = _green(wheel_type)
    ng = [n for n in recent_history if n not in green]
    N = len(ng)
    n_min = 60
    usable = N >= n_min
    reason = "ok" if usable else f"N_nong={N} < {n_min}"

    if N == 0:
        return {"stat": 0.0, "df": 2, "p_value": 1.0, "usable": False, "reason": "N=0"}

    fn = dozen if group_type == "dozen" else column
    observed = [sum(1 for n in ng if fn(n) == c) for c in (1, 2, 3)]
    expected = [N / 3.0] * 3
    stat, p_value = stats.chisquare(observed, expected)
    return {"stat": float(stat), "df": 2, "p_value": float(p_value),
            "usable": usable, "reason": reason}


# ---------------------------------------------------------------------------
# External sanity check  (never upgrades verdict)
# ---------------------------------------------------------------------------


def external_consistency_check(
    recent_history: list[int],
    external_stats: dict | None,
    external_n_estimate: int | None,
    wheel_type: str = "american",
) -> dict | None:
    """Compare external display-screen aggregates against the internal history.

    Can only be used to *downgrade* a weak verdict — never to upgrade.
    Returns None when external_stats is None.
    """
    if external_stats is None:
        return None

    green = _green(wheel_type)
    ng_history = [n for n in recent_history if n not in green]
    n_internal = len(recent_history)

    known_groups = {"red", "black", "odd", "even", "low", "high"}
    details: dict[str, dict] = {}
    group_status: dict[str, str] = {}

    for key, ext_pct in external_stats.items():
        group = key.replace("_pct", "")
        if group not in known_groups:
            continue
        int_p = _internal_pct(group, ng_history)
        diff = float(ext_pct) - int_p
        abs_diff = abs(diff)

        if abs_diff < 0.05:
            status = "unknown"
        elif (int_p > 0.5) == (float(ext_pct) > 0.5):
            status = "consistent"
        else:
            status = "inconsistent"

        details[group] = {
            "internal_pct": round(int_p, 4),
            "external_pct": round(float(ext_pct), 4),
            "diff": round(diff, 4),
        }
        group_status[group] = status

    if not details:
        return {"status": "unknown", "details": {}, "auxiliary_p_values": {},
                "note": "外部统计无可识别字段。"}

    statuses = list(group_status.values())
    if "inconsistent" in statuses:
        overall = "inconsistent"
    elif "consistent" in statuses:
        overall = "consistent"
    else:
        overall = "unknown"

    # Auxiliary p-values — only when user supplied an N estimate
    aux_p: dict[str, float] = {}
    if external_n_estimate is not None and external_n_estimate > 0:
        for g, info in details.items():
            obs_ext = round(info["external_pct"] * external_n_estimate)
            res = stats.binomtest(obs_ext, external_n_estimate, 0.5, alternative="two-sided")
            aux_p[g] = round(float(res.pvalue), 6)

    inconsistent = [g for g, s in group_status.items() if s == "inconsistent"]
    consistent   = [g for g, s in group_status.items() if s == "consistent"]
    if overall == "inconsistent":
        note = (f"显示屏统计 ({', '.join(inconsistent)}) 与内部 {n_internal} 轮方向相反，"
                "数据可能陈旧，已降低信任度。")
    elif overall == "consistent":
        note = (f"显示屏统计 ({', '.join(consistent)}) 与内部 {n_internal} 轮方向一致，"
                "样本量未知，仅供参考。")
    else:
        note = f"显示屏统计与内部 {n_internal} 轮差异 <5%，方向不明确。"

    return {"status": overall, "details": details, "auxiliary_p_values": aux_p, "note": note}


# ---------------------------------------------------------------------------
# Verdict helper
# ---------------------------------------------------------------------------


def _initial_verdict(tests: dict, N: int) -> tuple[str, float | None, str | None]:
    """Determine the strongest verdict supported by internal tests.

    Returns (verdict, best_p_value, best_test_name).
    """
    candidates: list[tuple[float, str]] = []

    def _add(test_key: str) -> None:
        t = tests[test_key]
        if isinstance(t, dict) and t.get("usable"):
            candidates.append((t["p_value"], test_key))

    for key in ("chi2_single", "chi2_sector_8", "binomial_red", "binomial_odd",
                "binomial_low", "dozen", "column"):
        _add(key)

    # Hot numbers: use Bonferroni-corrected p, only when N >= 100
    if N >= 100 and tests.get("hot_numbers"):
        p_bon = tests["hot_numbers"][0]["p_bonferroni"]
        candidates.append((p_bon, "hot_numbers"))

    if not candidates:
        return "no_evidence", None, None

    best_p, best_test = min(candidates, key=lambda x: x[0])

    for level in ("strong", "moderate", "weak"):
        t = VERDICT_THRESHOLDS[level]
        if N >= t["n_min"] and best_p < t["p_max"]:
            return level, best_p, best_test

    return "no_evidence", best_p, best_test


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def detect_bias(
    recent_history: list[int],
    wheel_type: str = "american",
    external_stats: dict | None = None,
    external_n_estimate: int | None = None,
) -> dict:
    """Run all statistical tests and return a complete bias report.

    Only recent_history feeds the formal tests; external_stats is a sanity
    check that can downgrade (but never upgrade) a weak verdict.
    """
    N = len(recent_history)

    tests: dict = {
        "chi2_single":   chi_square_single(recent_history, wheel_type),
        "chi2_sector_8": chi_square_sector(recent_history, wheel_type, n_sectors=8),
        "binomial_red":  binomial_test(recent_history, "red",  wheel_type),
        "binomial_odd":  binomial_test(recent_history, "odd",  wheel_type),
        "binomial_low":  binomial_test(recent_history, "low",  wheel_type),
        "dozen":         _chi_square_group(recent_history, "dozen",  wheel_type),
        "column":        _chi_square_group(recent_history, "column", wheel_type),
        "hot_numbers":   hot_numbers_test(recent_history, wheel_type),
    }

    ext_check = external_consistency_check(
        recent_history, external_stats, external_n_estimate, wheel_type
    )

    verdict, best_p, best_test = _initial_verdict(tests, N)

    # External downgrade: inconsistent external data demotes a weak signal
    if (ext_check is not None
            and ext_check["status"] == "inconsistent"
            and verdict == "weak"):
        verdict = "no_evidence"

    weight = VERDICT_WEIGHTS[verdict]

    # Suspected bias label
    _bias_labels = {
        "binomial_red":   ("color",         "red/black imbalance"),
        "binomial_odd":   ("parity",        "odd/even imbalance"),
        "binomial_low":   ("range",         "low/high imbalance"),
        "chi2_single":    ("single_number", "individual pocket frequency anomaly"),
        "chi2_sector_8":  ("sector",        "wheel sector bias"),
        "dozen":          ("dozen",         "dozen distribution anomaly"),
        "column":         ("column",        "column distribution anomaly"),
        "hot_numbers":    ("single_number", f"hot number detected"),
    }
    if verdict != "no_evidence" and best_test in _bias_labels:
        b_type, b_detail = _bias_labels[best_test]
        if best_test == "hot_numbers" and tests["hot_numbers"]:
            b_detail = f"number {tests['hot_numbers'][0]['n']} appears unusually often"
        suspected_bias = {"type": b_type, "details": b_detail}
    else:
        suspected_bias = {"type": None, "details": None}

    # Human-readable summary
    p_str = f"p={best_p:.4f}" if best_p is not None else "p=n/a"
    summaries = {
        "no_evidence": f"N={N}，无统计证据，按公平轮盘对待。",
        "weak":        f"N={N}，弱偏向信号（{p_str}），weight={weight}，保持保守。",
        "moderate":    f"N={N}，中等偏向证据（{p_str}），weight={weight}，信念模型小幅偏移。",
        "strong":      f"N={N}，强偏向证据（{p_str}），weight={weight}，信念模型显著偏移。",
    }

    return {
        "n_internal": N,
        "wheel_type": wheel_type,
        "tests": tests,
        "external_check": ext_check,
        "verdict": verdict,
        "weight": weight,
        "suspected_bias": suspected_bias,
        "summary": summaries[verdict],
    }
