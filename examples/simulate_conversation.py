"""10-round realistic user-agent conversation simulation.

Scenario: a player has already observed 20 spins of a biased American wheel
(pocket 17 appearing ~50%, with secondary heat on 5 and 32). They provide
external display-screen stats and interact with the agent over 10 more spins.
"""

from __future__ import annotations

import os
import random
import textwrap
from pathlib import Path

import anthropic

from roulette_agent.agent import RouletteAgent
from roulette_agent.layout import BLACK, RED
from roulette_agent.settler import settle


# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------

def _load_env() -> None:
    for p in (Path(__file__).parents[1] / ".env", Path.cwd() / ".env"):
        if p.is_file():
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
            return

_load_env()


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

# 20 spins of initial history: pocket 17 is heavily biased (10/20 = 50%)
# Pockets 5 and 32 are secondary hotspots (both physically adjacent to 17 on
# the American wheel: ...32 | 17 | 5...)
INITIAL_HISTORY: list[int] = [
    17,  5, 17, 32, 17, 11, 17,  5, 17, 32,
    17,  0, 17,  5, 17, 22, 17,  8, 17, 32,
]

# External display-screen stats (from ~200 spins on this table).
# 17 is BLACK/ODD → inflated black_pct and odd_pct are consistent.
EXTERNAL_STATS: dict[str, float] = {
    "black_pct": 0.62,
    "odd_pct":   0.58,
    "high_pct":  0.52,
}
EXTERNAL_N = 200

# Generate 11 future spins from the same biased wheel (to settle rounds 1-10)
random.seed(2025)
_wheel   = list(range(38))
_weights = [5 if n == 17 else (3 if n in (5, 32) else 1) for n in _wheel]
_future  = random.choices(_wheel, weights=_weights, k=11)

# observed[i]  = "the spin that just happened" (reported by the user; added to history)
# settle_on[i] = the NEXT spin (the one the agent's bets are settled against)
OBSERVED_SPINS: list[int] = _future[:10]
SETTLE_SPINS:   list[int] = _future[1:11]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _desc(n: int) -> str:
    color   = "红" if n in RED else ("绿" if n in (0, 37) else "黑")
    parity  = "" if n in (0, 37) else ("奇" if n % 2 == 1 else "偶")
    label   = "00" if n == 37 else str(n)
    return f"{label}({color}{parity})"


def _wrap(text: str, indent: str = "          ") -> str:
    return textwrap.fill(text, width=66, subsequent_indent=indent)


LINE  = "─" * 72
DLINE = "═" * 72


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def main() -> None:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    agent  = RouletteAgent(client, model="claude-sonnet-4-6")

    session: dict = {
        "bankroll":                  800.0,
        "bet_unit":                  5.0,
        "wheel_type":                "american",
        "excluded_dozens":           [],
        "recent_history":            list(INITIAL_HISTORY),
        "external_stats":            EXTERNAL_STATS,
        "external_stats_n_estimate": EXTERNAL_N,
        "hyperparams":               {},
        "notes":                     "",
    }

    # ── Banner ──────────────────────────────────────────────────────────────
    black_n = sum(1 for n in INITIAL_HISTORY if n in BLACK)
    odd_n   = sum(1 for n in INITIAL_HISTORY if 1 <= n <= 36 and n % 2 == 1)

    print(DLINE)
    print("  ROULETTE BIAS-AWARE AGENT  |  10轮真实场景模拟")
    print(DLINE)
    print(f"  轮盘类型 : American (38格)")
    print(f"  初始筹码 : ${session['bankroll']:.0f}   下注单位: ${session['bet_unit']:.0f}")
    print()
    print(f"  初始历史 (20转):")
    print(f"    {INITIAL_HISTORY}")
    print(f"    17号: {INITIAL_HISTORY.count(17)}次  5号: {INITIAL_HISTORY.count(5)}次  "
          f"32号: {INITIAL_HISTORY.count(32)}次  黑: {black_n}/20  奇: {odd_n}/20")
    print()
    print(f"  外部显示屏统计 (≈{EXTERNAL_N}转):")
    for k, v in EXTERNAL_STATS.items():
        print(f"    {k}: {v:.0%}")
    print()
    print(f"  即将揭晓的10转结果 (已锁定，仅用于结算):")
    print(f"    观察: {[_desc(n) for n in OBSERVED_SPINS]}")
    print(f"    结算: {[_desc(n) for n in SETTLE_SPINS]}")
    print(DLINE)

    pnl_log: list[float] = []

    for rnd in range(1, 11):
        observed  = OBSERVED_SPINS[rnd - 1]
        settle_on = SETTLE_SPINS[rnd - 1]

        session["recent_history"].append(observed)
        n_hist = len(session["recent_history"])

        # ── 用户发言 ─────────────────────────────────────────────────────────
        print(f"\n{LINE}")
        print(f"  第 {rnd:02d} 轮  ·  历史 {n_hist} 转  ·  筹码 ${session['bankroll']:.0f}")
        print(LINE)

        print(f"\n  [用户]")
        ext_summary = "  ".join(f"{k}={v:.0%}" for k, v in EXTERNAL_STATS.items())
        print(f"    上一转结果: {_desc(observed)}")
        print(f"    历史现有 {n_hist} 转数据。"
              f"显示屏统计: {ext_summary} (N≈{EXTERNAL_N})。")
        print(f"    当前筹码 ${session['bankroll']:.0f}，"
              f"下注单位 ${session['bet_unit']:.0f}。请分析偏差情况并给出下注建议。")

        # ── Agent 分析与建议 ──────────────────────────────────────────────────
        print(f"\n  [Agent]")
        decision  = agent.decide(session)
        bets      = decision.get("bets", [])
        rationale = decision.get("rationale", "")

        # Print wrapped rationale
        for line in _wrap(rationale, indent="    ").splitlines():
            print(f"    {line.lstrip()}")

        print()
        if bets:
            total_staked = sum(b["amount"] for b in bets)
            pct_bankroll = total_staked / session["bankroll"]
            print(f"    下注建议: {len(bets)} 注  总计 ${total_staked:.0f}"
                  f"  ({pct_bankroll:.1%} 筹码)")
            for b in bets:
                nums = str(b["numbers"]) if b["numbers"] is not None else "—"
                print(f"      {b['type']:16s}  {nums:22s}  ${b['amount']:.0f}")
        else:
            print(f"    建议: 本轮不下注。")

        # ── 结算 ─────────────────────────────────────────────────────────────
        print()
        print(f"  [结算]  下一转: {_desc(settle_on)}", end="")
        if bets:
            outcome  = settle(bets, settle_on)
            pnl      = outcome["pnl"]
            session["bankroll"] = round(session["bankroll"] + pnl, 2)
            pnl_log.append(pnl)
            if pnl > 0:
                print(f"  →  盈利 +${pnl:.0f}  ✓   新筹码 ${session['bankroll']:.0f}")
            elif pnl < 0:
                print(f"  →  亏损 ${pnl:.0f}   新筹码 ${session['bankroll']:.0f}")
            else:
                print(f"  →  平局    筹码不变 ${session['bankroll']:.0f}")
        else:
            pnl_log.append(0.0)
            print(f"  →  本轮无注，跳过结算")

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    initial_br = 800.0
    total_pnl  = session["bankroll"] - initial_br
    wins  = sum(1 for p in pnl_log if p > 0)
    loss  = sum(1 for p in pnl_log if p < 0)
    skips = sum(1 for p in pnl_log if p == 0)

    print(f"\n{DLINE}")
    print(f"  10轮汇总")
    print(DLINE)
    print(f"  初始筹码: ${initial_br:.0f}   最终筹码: ${session['bankroll']:.0f}   "
          f"总PnL: {total_pnl:+.0f} ({total_pnl/initial_br:+.1%})")
    print(f"  胜: {wins}轮  负: {loss}轮  无注: {skips}轮")
    print(f"  最终历史长度: {len(session['recent_history'])} 转")
    print(DLINE)


if __name__ == "__main__":
    main()
