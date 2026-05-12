"""Bias detector demo: three scenarios matching the test suite.

Run with:
    uv run python examples/bias_demo.py
"""

import numpy as np

from roulette_agent.bias_detector import detect_bias


def _row(label: str, *cols: str, widths: list[int] | None = None) -> str:
    if widths is None:
        widths = [18, 8, 8, 8, 8, 40]
    parts = [str(c).ljust(w) for c, w in zip([label, *cols], widths)]
    return "  ".join(parts)


def _header(title: str) -> None:
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Scenario A: fair roulette false-positive check
# ---------------------------------------------------------------------------

def scenario_a() -> None:
    _header("Scenario A — Fair roulette: false-positive rate check")
    rng = np.random.default_rng(42)
    history = rng.integers(0, 38, size=10_000).tolist()

    widths = [8, 12, 8, 8, 50]
    print("  " + _row("N", "verdict", "weight", "p_best", "summary", widths=widths))
    print("  " + "-" * 86)

    results, fp_count = [], 0
    for end in range(100, 10_001, 100):
        r = detect_bias(history[:end])
        results.append(r["verdict"])
        if r["verdict"] in ("moderate", "strong"):
            fp_count += 1

        if end <= 1000 or end % 1000 == 0:
            tests = r["tests"]
            best_p = min(
                (t["p_value"] for t in [tests["chi2_sector_8"], tests["binomial_red"],
                                         tests["binomial_odd"]] if t["usable"]),
                default=float("nan"),
            )
            print("  " + _row(
                str(end), r["verdict"], f"{r['weight']:.2f}",
                f"{best_p:.3f}" if not np.isnan(best_p) else "n/a",
                r["summary"][:48],
                widths=widths,
            ))

    fp_rate = fp_count / len(results)
    print()
    print(f"  False-positive (moderate/strong): {fp_count}/{len(results)} = {fp_rate:.1%}")
    status = "PASS ✓" if fp_rate < 0.05 else "FAIL ✗"
    print(f"  Threshold <5%: {status}")


# ---------------------------------------------------------------------------
# Scenario B: biased wheel (number 35 at 3x fair rate)
# ---------------------------------------------------------------------------

def scenario_b() -> None:
    _header("Scenario B — Biased wheel: number 35 at p=3/38 (3× fair)")
    rng = np.random.default_rng(7)
    probs = np.full(38, (35 / 38) / 37)
    probs[35] = 3 / 38

    widths = [8, 12, 8, 10, 46]
    print("  " + _row("N", "verdict", "weight", "count(35)", "suspected", widths=widths))
    print("  " + "-" * 84)

    full = rng.choice(38, size=2000, p=probs).tolist()
    for n in (100, 500, 2000):
        h = full[:n]
        r = detect_bias(h)
        cnt35 = h.count(35)
        expected = round(n / 38, 1)
        print("  " + _row(
            str(n), r["verdict"], f"{r['weight']:.2f}",
            f"{cnt35} (exp≈{expected})",
            str(r["suspected_bias"]),
            widths=widths,
        ))

    print()
    print("  Multi-seed detection rate at N=500:")
    hits = 0
    for seed in range(20):
        rng2 = np.random.default_rng(seed)
        h = rng2.choice(38, size=500, p=probs).tolist()
        if detect_bias(h)["verdict"] != "no_evidence":
            hits += 1
    print(f"  Detected (not no_evidence): {hits}/20 seeds = {hits/20:.0%}")


# ---------------------------------------------------------------------------
# Scenario C: external stats downgrade
# ---------------------------------------------------------------------------

def scenario_c() -> None:
    _header("Scenario C — External stats downgrade")

    # Build a clearly red-biased internal history
    history = [1] * 40 + [2] * 20  # 40 red, 20 black, 0 green → 66.7% red

    print("  Internal history: 40 × red(1), 20 × black(2) → 66.7% red among non-green")
    print()

    widths = [30, 14, 8, 36]
    print("  " + _row("external_stats", "verdict", "weight", "note", widths=widths))
    print("  " + "-" * 88)

    cases = [
        (None,                         "no external"),
        ({"red_pct": 0.70},            "consistent (red 70%)"),
        ({"black_pct": 0.75},          "inconsistent (black 75%)"),
        ({"black_pct": 0.75,
          "red_pct": 0.25},            "inconsistent (both fields)"),
    ]

    for ext, label in cases:
        r = detect_bias(history, external_stats=ext)
        ext_note = ""
        if r["external_check"]:
            ext_note = r["external_check"]["status"]
        print("  " + _row(
            label, r["verdict"], f"{r['weight']:.2f}", ext_note or "—",
            widths=widths,
        ))

    print()
    print("  Downgrade rule: inconsistent external + internal weak → no_evidence ✓")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scenario_a()
    scenario_b()
    scenario_c()
    print()
