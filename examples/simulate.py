"""Strategy comparison demo: Monte Carlo over fair and biased wheels.

Run with:
    uv run python examples/simulate.py
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roulette_agent.simulator import compare_strategies, simulate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STRATEGIES = ["greedy_ev", "kelly", "fixed_baseline", "always_red"]
N_SPINS = 200
N_RUNS = 1000
INITIAL_BANKROLL = 800.0
BET_UNIT = 10.0
SEED = 42
N_SAMPLE_PATHS = 20
OUT_DIR = os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# A. Comparison table
# ---------------------------------------------------------------------------

def run_comparison() -> dict:
    print("=" * 72)
    print("  Strategy comparison — fair American wheel")
    print(f"  {N_RUNS} runs × {N_SPINS} spins  |  bankroll={INITIAL_BANKROLL}  bet_unit={BET_UNIT}")
    print("=" * 72)

    t0 = time.time()
    stats = compare_strategies(
        strategies=STRATEGIES,
        n_spins=N_SPINS,
        n_runs=N_RUNS,
        initial_bankroll=INITIAL_BANKROLL,
        bet_unit=BET_UNIT,
        seed=SEED,
    )
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s\n")

    col = [18, 9, 9, 9, 8, 9, 9]
    header = (
        f"  {'Strategy':<{col[0]}}"
        f"{'Mean':>{col[1]}}"
        f"{'Median':>{col[2]}}"
        f"{'Std':>{col[3]}}"
        f"{'Ruin%':>{col[4]}}"
        f"{'Max':>{col[5]}}"
        f"{'Min':>{col[6]}}"
    )
    print(header)
    print("  " + "-" * (sum(col) + 2))
    for s in STRATEGIES:
        st = stats[s]
        print(
            f"  {s:<{col[0]}}"
            f"{st['mean_final_bankroll']:>{col[1]}.1f}"
            f"{st['median_final_bankroll']:>{col[2]}.1f}"
            f"{st['std_final_bankroll']:>{col[3]}.1f}"
            f"{st['ruin_rate']:>{col[4]}.1%}"
            f"{st['max_final_bankroll']:>{col[5]}.1f}"
            f"{st['min_final_bankroll']:>{col[6]}.1f}"
        )
    return stats


# ---------------------------------------------------------------------------
# B. Bankroll curve plots (20 sample paths + mean)
# ---------------------------------------------------------------------------

def plot_strategies() -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Bankroll Curves — Fair American Wheel\n"
        f"({N_RUNS} runs, {N_SPINS} spins, initial={INITIAL_BANKROLL}, unit={BET_UNIT})",
        fontsize=12,
    )

    rng = np.random.default_rng(SEED + 1000)
    seeds = rng.integers(0, 1_000_000, size=N_SAMPLE_PATHS).tolist()
    x = np.arange(N_SPINS + 1)

    for ax, strategy in zip(axes.flat, STRATEGIES):
        curves = []
        for s in seeds:
            r = simulate(
                strategy_name=strategy,
                n_spins=N_SPINS,
                initial_bankroll=INITIAL_BANKROLL,
                bet_unit=BET_UNIT,
                seed=int(s),
                _refresh_every=10,
            )
            curves.append(r["bankroll_curve"])

        arr = np.array(curves)
        for curve in arr:
            ax.plot(x, curve, alpha=0.25, linewidth=0.8, color="steelblue")
        ax.plot(x, arr.mean(axis=0), color="navy", linewidth=2, label="mean")
        ax.axhline(INITIAL_BANKROLL, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(0, color="red", linestyle=":", linewidth=0.8, alpha=0.6)

        ax.set_title(strategy, fontsize=11)
        ax.set_xlabel("Spin")
        ax.set_ylabel("Bankroll")
        ax.set_ylim(-30, INITIAL_BANKROLL * 1.6)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "strategies_fair.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\n  [chart] {path}")
    return path


# ---------------------------------------------------------------------------
# C. Kelly on biased wheel
# ---------------------------------------------------------------------------

def plot_kelly_biased() -> str:
    # 500 spins: long enough for bias detection to fire (~300 spins), short enough
    # that compounding stays in a displayable range.
    BIASED_SPINS = 500
    BIASED_UNIT = 1.0

    print("\n" + "=" * 72)
    print("  Kelly on biased wheel: number 35 at 3× fair rate (p = 3/38)")
    print(f"  ({BIASED_SPINS} spins, bet_unit={BIASED_UNIT} — small unit so Kelly fraction clears floor)")
    print("=" * 72)

    bias = {35: 3 / 38}
    rng = np.random.default_rng(SEED + 2000)
    seeds = rng.integers(0, 1_000_000, size=N_SAMPLE_PATHS).tolist()
    x = np.arange(BIASED_SPINS + 1)

    curves_fair: list[list[float]] = []
    curves_biased: list[list[float]] = []
    for s in seeds:
        rf = simulate("kelly", BIASED_SPINS, INITIAL_BANKROLL, BIASED_UNIT,
                      seed=int(s), _refresh_every=10)
        rb = simulate("kelly", BIASED_SPINS, INITIAL_BANKROLL, BIASED_UNIT,
                      bias_inject=bias, seed=int(s), _refresh_every=10)
        curves_fair.append(rf["bankroll_curve"])
        curves_biased.append(rb["bankroll_curve"])

    fair_arr = np.array(curves_fair, dtype=float)
    biased_arr = np.array(curves_biased, dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Kelly Strategy: Fair vs Biased Wheel (35 at 3× rate)\n"
        f"{BIASED_SPINS} spins, bet_unit={BIASED_UNIT}",
        fontsize=12,
    )

    for ax, arr, label, color in [
        (ax1, fair_arr,   "Fair wheel",              "steelblue"),
        (ax2, biased_arr, "Biased wheel (35 at 3×)", "darkorange"),
    ]:
        for curve in arr:
            ax.plot(x, curve, alpha=0.25, linewidth=0.8, color=color)
        mean_curve = np.clip(arr, 0, None).mean(axis=0)
        ax.plot(x, mean_curve, color="black", linewidth=2, label="mean")
        ax.axhline(INITIAL_BANKROLL, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(0, color="red", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Spin")
        ax.set_ylabel("Bankroll (log scale)")
        # Log scale: clip floor to 1 so zeros don't break the axis
        clipped = np.clip(arr, 1, None)
        ymax = float(np.percentile(clipped, 95)) * 3
        ax.set_yscale("log")
        ax.set_ylim(1, max(ymax, INITIAL_BANKROLL * 2))
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "kelly_biased.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\n  [chart] {path}")

    # Summary stats — use median (robust to outliers) and format with engineering suffix
    def _fmt(v: float) -> str:
        if v >= 1e9:   return f"{v/1e9:.1f}B"
        if v >= 1e6:   return f"{v/1e6:.1f}M"
        if v >= 1e3:   return f"{v/1e3:.1f}k"
        return f"{v:.1f}"

    final_fair   = fair_arr[:, -1]
    final_biased = biased_arr[:, -1]
    print(f"\n  {'':26} {'Fair':>10} {'Biased':>10}")
    print(f"  {'-'*46}")
    print(f"  {'Mean final bankroll':26} {_fmt(np.mean(final_fair)):>10} {_fmt(np.mean(final_biased)):>10}")
    print(f"  {'Median':26} {_fmt(np.median(final_fair)):>10} {_fmt(np.median(final_biased)):>10}")
    print(f"  {'Std':26} {_fmt(np.std(final_fair)):>10} {_fmt(np.std(final_biased)):>10}")
    print(f"  {'Ruin rate':26} {np.mean(final_fair <= 0):>10.1%} {np.mean(final_biased <= 0):>10.1%}")

    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_comparison()
    plot_strategies()
    plot_kelly_biased()
    print()
