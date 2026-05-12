"""Spin settlement: given a list of bets and the winning number, compute pnl and per-bet detail."""

from roulette_agent.layout import BET_TYPES, get_covered_numbers


def settle(bets: list[dict], result_number: int) -> dict:
    """
    bets: [{"type": str, "numbers": list[int] | None, "amount": float}, ...]
    result_number: 0..37  (37 == 00)

    Returns {
        "result_number": int,
        "total_staked": float,
        "total_payout": float,   # returned cash including original stake on wins
        "pnl": float,            # total_payout - total_staked
        "detail": [
            {"bet": <original dict>, "won": bool, "payout": float},
            ...
        ]
    }

    Win: payout = amount * (net_payout + 1)
    Loss: payout = 0  (stake is forfeit)
    """
    total_staked = 0.0
    total_payout = 0.0
    detail: list[dict] = []

    for bet in bets:
        bet_type = bet["type"]
        numbers = bet.get("numbers")
        amount = float(bet["amount"])
        total_staked += amount

        covered = get_covered_numbers(bet_type, numbers)
        won = result_number in covered
        payout = amount * (BET_TYPES[bet_type].payout + 1) if won else 0.0
        total_payout += payout

        detail.append({"bet": bet, "won": won, "payout": payout})

    return {
        "result_number": result_number,
        "total_staked": total_staked,
        "total_payout": total_payout,
        "pnl": total_payout - total_staked,
        "detail": detail,
    }
