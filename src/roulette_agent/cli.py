"""CLI entry point.

Usage
-----
python -m roulette_agent.cli init \\
    --bankroll 800 \\
    --recent-history "35,17,35,29,22,31,35,11,29,4,13,26,8,31,20" \\
    --external-stats '{"black_pct":0.62,"odd_pct":0.59}' \\
    --excluded-dozens 3

python -m roulette_agent.cli spin <session_id> <result_number>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import anthropic

from roulette_agent.agent import RouletteAgent

# ---------------------------------------------------------------------------
# Session storage
# ---------------------------------------------------------------------------

_SESSION_DIR = Path.home() / ".roulette_agent" / "sessions"


def _session_path(session_id: str) -> Path:
    return _SESSION_DIR / f"{session_id}.json"


def _load_session(session_id: str) -> dict:
    path = _session_path(session_id)
    if not path.exists():
        print(f"Session not found: {session_id}", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def _save_session(session: dict) -> None:
    _SESSION_DIR.mkdir(parents=True, exist_ok=True)
    _session_path(session["session_id"]).write_text(
        json.dumps(session, indent=2)
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> None:
    session_id = str(uuid.uuid4())

    history: list[int] = []
    if args.recent_history:
        history = [int(x.strip()) for x in args.recent_history.split(",") if x.strip()]

    excluded: list[int] = []
    if args.excluded_dozens:
        excluded = [int(x) for x in args.excluded_dozens]

    ext_stats = None
    if args.external_stats:
        ext_stats = json.loads(args.external_stats)

    session = {
        "session_id": session_id,
        "bankroll": float(args.bankroll),
        "bet_unit": float(args.bet_unit),
        "wheel_type": args.wheel_type,
        "excluded_dozens": excluded,
        "recent_history": history,
        "external_stats": ext_stats,
        "external_stats_n_estimate": args.external_n_estimate,
        "hyperparams": {},
        "notes": "",
        "spin_log": [],
    }
    _save_session(session)
    print(f"Session created: {session_id}")
    print(f"  bankroll={session['bankroll']}  bet_unit={session['bet_unit']}")
    print(f"  wheel={session['wheel_type']}  excluded_dozens={session['excluded_dozens']}")
    print(f"  history length={len(history)}")
    if ext_stats:
        print(f"  external_stats={ext_stats}")


def cmd_spin(args: argparse.Namespace) -> None:
    session = _load_session(args.session_id)
    result_number = int(args.result_number)

    # Append result to history
    session["recent_history"].append(result_number)

    # Build agent and run
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    agent = RouletteAgent(client, model=args.model)

    print(f"\n{'='*60}")
    print(f"Spin result: {result_number}  |  History length: {len(session['recent_history'])}")
    print(f"Bankroll: {session['bankroll']}")
    print(f"{'='*60}")
    print("Running agent…")

    decision = agent.decide(session)

    bets = decision.get("bets", [])
    rationale = decision.get("rationale", "")

    print(f"\nRationale: {rationale}")
    print(f"\nRecommended bets ({len(bets)}):")
    if bets:
        for b in bets:
            nums = b.get("numbers")
            print(f"  {b['type']:16s}  numbers={nums}  amount={b['amount']}")
    else:
        print("  (no bet recommended)")

    # Log the spin
    session["spin_log"].append({
        "result_number": result_number,
        "decision": decision,
        "bankroll_before": session["bankroll"],
    })

    # Update bankroll if bets were placed (settle against the result)
    if bets:
        from roulette_agent.settler import settle
        outcome = settle(bets, result_number)
        session["bankroll"] = round(session["bankroll"] + outcome["pnl"], 2)
        print(f"\nSettlement: pnl={outcome['pnl']:+.2f}  "
              f"new bankroll={session['bankroll']}")

    _save_session(session)
    print(f"\nSession saved: {args.session_id}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m roulette_agent.cli",
        description="Roulette Bias-Aware Agent CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- init ---
    p_init = sub.add_parser("init", help="Create a new session")
    p_init.add_argument("--bankroll", type=float, default=800.0)
    p_init.add_argument("--bet-unit", type=float, default=10.0)
    p_init.add_argument("--wheel-type", default="american",
                        choices=["american", "european"])
    p_init.add_argument("--recent-history", default="",
                        help="Comma-separated spin results, e.g. '35,17,22'")
    p_init.add_argument("--external-stats", default=None,
                        help="JSON string, e.g. '{\"black_pct\":0.62}'")
    p_init.add_argument("--external-n-estimate", type=int, default=None)
    p_init.add_argument("--excluded-dozens", type=int, nargs="*", default=[])

    # --- spin ---
    p_spin = sub.add_parser("spin", help="Record a spin result and get next bet advice")
    p_spin.add_argument("session_id")
    p_spin.add_argument("result_number", type=int)
    p_spin.add_argument("--model", default="claude-sonnet-4-5-20250929")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "spin":
        cmd_spin(args)


if __name__ == "__main__":
    main()
