"""Bet allocation strategies: EV-greedy, Fractional Kelly, and fixed-baseline; respects excluded_dozens constraints."""

from __future__ import annotations

import math

from roulette_agent.layout import BET_TYPES, get_covered_numbers, is_valid_bet

_DOZEN_RANGES: dict[int, set[int]] = {
    1: set(range(1, 13)),
    2: set(range(13, 25)),
    3: set(range(25, 37)),
}

_LEGAL_DOZENS = {1, 2, 3}


def excluded_numbers(excluded_dozens: list[int]) -> set[int]:
    """Return the set of pocket numbers covered by the excluded dozens.

    Raises ValueError if excluded_dozens contains values outside {1, 2, 3} or
    if it contains both 1 and 3 simultaneously (non-contiguous exclusion).
    """
    if not excluded_dozens:
        return set()

    bad = set(excluded_dozens) - _LEGAL_DOZENS
    if bad:
        raise ValueError(f"Invalid dozens: {bad}. Must be 1, 2, or 3.")

    if 1 in excluded_dozens and 3 in excluded_dozens:
        raise ValueError("Cannot exclude dozens 1 and 3 simultaneously (non-contiguous).")

    result: set[int] = set()
    for d in excluded_dozens:
        result |= _DOZEN_RANGES[d]
    return result


def enumerate_legal_bets(excluded_dozens: list[int]) -> list[dict]:
    """Return all (bet_type, numbers) combos whose covered numbers don't intersect excluded set.

    Each entry: {"type": str, "numbers": list[int]|None, "covered": set[int]}
    """
    excl = excluded_numbers(excluded_dozens)

    legal: list[dict] = []
    for bet_name, bet in BET_TYPES.items():
        if bet_name in ("straight", "split", "street", "corner", "six_line",
                        "five_line", "dozen", "column"):
            _enumerate_structured(bet_name, excl, legal)
        else:
            # Outside bets with no number parameter (red/black/odd/even/low/high)
            try:
                covered = get_covered_numbers(bet_name, None)
            except Exception:
                continue
            if not covered & excl:
                legal.append({"type": bet_name, "numbers": None, "covered": covered})

    return legal


def _enumerate_structured(bet_name: str, excl: set[int], out: list[dict]) -> None:
    """Enumerate all valid placements for structured bets, skipping excluded pockets."""
    if bet_name == "straight":
        for n in range(38):
            if n not in excl and is_valid_bet("straight", [n]):
                out.append({"type": "straight", "numbers": [n],
                            "covered": get_covered_numbers("straight", [n])})

    elif bet_name == "split":
        # Adjacent pairs: horizontal (same row) and vertical (same column)
        from roulette_agent.layout import grid_pos
        for n in range(1, 37):
            col, row = grid_pos(n)  # type: ignore[misc]
            # horizontal neighbour
            nbr = n + 3
            if nbr <= 36 and is_valid_bet("split", [n, nbr]):
                covered = get_covered_numbers("split", [n, nbr])
                if not covered & excl:
                    out.append({"type": "split", "numbers": [n, nbr], "covered": covered})
            # vertical neighbour
            nbr = n + 1
            if nbr <= 36 and is_valid_bet("split", [n, nbr]):
                covered = get_covered_numbers("split", [n, nbr])
                if not covered & excl:
                    out.append({"type": "split", "numbers": [n, nbr], "covered": covered})
        # Zero splits: 0-1, 0-2, 0-3, 37-1, 37-2, 37-3, 0-37
        for pair in ([0,1],[0,2],[0,3],[37,1],[37,2],[37,3],[0,37]):
            if is_valid_bet("split", pair):
                covered = get_covered_numbers("split", pair)
                if not covered & excl:
                    out.append({"type": "split", "numbers": pair, "covered": covered})

    elif bet_name == "street":
        for row in range(12):  # 12 streets
            n = 3 * row + 1
            nums = [n, n+1, n+2]
            if is_valid_bet("street", nums):
                covered = get_covered_numbers("street", nums)
                if not covered & excl:
                    out.append({"type": "street", "numbers": nums, "covered": covered})

    elif bet_name == "corner":
        from roulette_agent.layout import grid_pos
        for col in range(11):
            for row in range(2):
                n = 3 * col + (2 - row)  # top-left of corner
                nums = [n, n+3, n+1, n+4]
                nums_sorted = sorted(nums)
                if is_valid_bet("corner", nums_sorted):
                    covered = get_covered_numbers("corner", nums_sorted)
                    if not covered & excl:
                        out.append({"type": "corner", "numbers": nums_sorted,
                                    "covered": covered})

    elif bet_name == "six_line":
        for row in range(11):  # 11 six-lines
            n = 3 * row + 1
            nums = [n, n+1, n+2, n+3, n+4, n+5]
            if is_valid_bet("six_line", nums):
                covered = get_covered_numbers("six_line", nums)
                if not covered & excl:
                    out.append({"type": "six_line", "numbers": nums, "covered": covered})

    elif bet_name == "five_line":
        nums = [0, 37, 1, 2, 3]
        if is_valid_bet("five_line", nums):
            covered = get_covered_numbers("five_line", nums)
            if not covered & excl:
                out.append({"type": "five_line", "numbers": nums, "covered": covered})

    elif bet_name == "dozen":
        for d in (1, 2, 3):
            if is_valid_bet("dozen", [d]):
                covered = get_covered_numbers("dozen", [d])
                if not covered & excl:
                    out.append({"type": "dozen", "numbers": [d], "covered": covered})

    elif bet_name == "column":
        for c in (1, 2, 3):
            if is_valid_bet("column", [c]):
                covered = get_covered_numbers("column", [c])
                if not covered & excl:
                    out.append({"type": "column", "numbers": [c], "covered": covered})


def _expected_value_from_p(bet: dict, p: dict[int, float]) -> float:
    """EV = payout * p_win - (1 - p_win), where p_win = sum of p over covered numbers."""
    payout = BET_TYPES[bet["type"]].payout
    p_win = sum(p.get(n, 0.0) for n in bet["covered"])
    return payout * p_win - (1.0 - p_win)


def greedy_ev_allocation(
    p: dict[int, float],
    bankroll: float,
    bet_unit: float,
    excluded_dozens: list[int],
    max_bet_fraction: float = 0.1,
    top_k: int = 5,
) -> list[dict]:
    """Select up to top_k positive-EV bets, each sized at bet_unit, capped at bankroll × max_bet_fraction.

    Returns list of {"type", "numbers", "amount"}.  Empty when no positive-EV bet exists.
    """
    legal = enumerate_legal_bets(excluded_dozens)
    ev_bets = [(b, _expected_value_from_p(b, p)) for b in legal]
    ev_bets = [(b, ev) for b, ev in ev_bets if ev > 0]
    ev_bets.sort(key=lambda x: x[1], reverse=True)

    max_amount = bankroll * max_bet_fraction
    amount = min(bet_unit, max_amount)
    if amount <= 0:
        return []

    result = []
    for b, _ in ev_bets[:top_k]:
        result.append({"type": b["type"], "numbers": b["numbers"], "amount": amount})
    return result


def kelly_allocation(
    p: dict[int, float],
    bankroll: float,
    bet_unit: float,
    excluded_dozens: list[int],
    fraction: float = 0.25,
) -> list[dict]:
    """Fractional Kelly sizing for all positive-Kelly legal bets.

    f* = (b*p_win - p_lose) / b  (full Kelly)
    amount = fraction * f* * bankroll, rounded down to nearest bet_unit.
    Capped at 0.5 × bankroll per bet. Bets with f* <= 0 are skipped.
    """
    legal = enumerate_legal_bets(excluded_dozens)
    result = []

    for b in legal:
        payout = BET_TYPES[b["type"]].payout
        p_win = sum(p.get(n, 0.0) for n in b["covered"])
        p_lose = 1.0 - p_win
        if payout == 0 or p_win <= 0:
            continue
        f_star = (payout * p_win - p_lose) / payout
        if f_star <= 0:
            continue

        raw_amount = fraction * f_star * bankroll
        raw_amount = min(raw_amount, 0.5 * bankroll)
        if bet_unit > 0:
            units = math.floor(raw_amount / bet_unit)
            amount = units * bet_unit
        else:
            amount = raw_amount

        if amount <= 0:
            continue
        result.append({"type": b["type"], "numbers": b["numbers"], "amount": amount})

    return result


def fixed_baseline_allocation(
    p: dict[int, float],
    bankroll: float,
    bet_unit: float,
    excluded_dozens: list[int],
) -> list[dict]:
    """Flat bet_unit on each of the six outside bets that are legal and positive-EV.

    Falls back to the single best-EV outside bet if none are positive-EV under p.
    Returns empty list when bankroll < bet_unit.
    """
    if bankroll < bet_unit:
        return []

    outside_types = ["red", "black", "odd", "even", "low", "high"]
    legal_outside = {b["type"]: b for b in enumerate_legal_bets(excluded_dozens)
                     if b["type"] in outside_types}

    positive = [(t, b) for t, b in legal_outside.items()
                if _expected_value_from_p(b, p) > 0]

    if positive:
        return [{"type": t, "numbers": None, "amount": bet_unit} for t, _ in positive]

    # Fallback: best-EV legal outside bet
    if not legal_outside:
        return []
    best_type, best_b = max(legal_outside.items(),
                            key=lambda kv: _expected_value_from_p(kv[1], p))
    return [{"type": best_type, "numbers": None, "amount": bet_unit}]
