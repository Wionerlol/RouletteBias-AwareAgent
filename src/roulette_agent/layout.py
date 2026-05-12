"""Static roulette data structures: wheel layout, number attributes, bet types, and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------------
# Number sets
# ---------------------------------------------------------------------------

RED: set[int] = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
BLACK: set[int] = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}
GREEN: set[int] = {0, 37}  # 37 represents 00

ALL_NUMBERS: set[int] = RED | BLACK | GREEN  # 0..37

# ---------------------------------------------------------------------------
# Wheel physical orders (clockwise, used for sector bias detection)
# ---------------------------------------------------------------------------

WHEEL_ORDER_AMERICAN: list[int] = [
    0, 28, 9, 26, 30, 11, 7, 20, 32, 17, 5, 22, 34, 15, 3, 24, 36, 13, 1,
    37, 27, 10, 25, 29, 12, 8, 19, 31, 18, 6, 21, 33, 16, 4, 23, 35, 14, 2,
]

WHEEL_ORDER_EUROPEAN: list[int] = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10,
    5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26,
]

# ---------------------------------------------------------------------------
# Number attribute functions
# ---------------------------------------------------------------------------


def color(n: int) -> str:
    if n in RED:
        return "red"
    if n in BLACK:
        return "black"
    return "green"


def parity(n: int) -> str | None:
    if n in GREEN:
        return None
    return "odd" if n % 2 == 1 else "even"


def high_low(n: int) -> str | None:
    if n in GREEN:
        return None
    if 1 <= n <= 18:
        return "low"
    return "high"  # 19–36


def dozen(n: int) -> int | None:
    if n in GREEN:
        return None
    if n <= 12:
        return 1
    if n <= 24:
        return 2
    return 3


def column(n: int) -> int | None:
    if n in GREEN:
        return None
    r = n % 3
    return r if r != 0 else 3


def grid_pos(n: int) -> tuple[int, int] | None:
    """Return (col_idx 0..11, row_idx 0..2) on the 12×3 betting table.

    Layout (row 0 = top):
      row 0: 3, 6, 9, …, 36
      row 1: 2, 5, 8, …, 35
      row 2: 1, 4, 7, …, 34
    """
    if n in GREEN or not (1 <= n <= 36):
        return None
    col_idx = (n - 1) // 3
    row_idx = 2 - (n - 1) % 3
    return (col_idx, row_idx)


# ---------------------------------------------------------------------------
# BetType definition
# ---------------------------------------------------------------------------


@dataclass
class BetType:
    name: str
    payout: int        # net payout on a 1-unit win
    size: int          # canonical number of outcomes covered
    edge: float        # house edge (positive fraction)
    covered_fn: Callable[[list[int] | None], set[int]]


# ---------------------------------------------------------------------------
# covered_fn implementations
# ---------------------------------------------------------------------------


def _covered_straight(numbers: list[int] | None) -> set[int]:
    return {numbers[0]} if numbers else set()


def _covered_split(numbers: list[int] | None) -> set[int]:
    return set(numbers) if numbers else set()


def _covered_street(numbers: list[int] | None) -> set[int]:
    return set(numbers) if numbers else set()


def _covered_corner(numbers: list[int] | None) -> set[int]:
    return set(numbers) if numbers else set()


def _covered_five_line(_: list[int] | None) -> set[int]:
    return {0, 37, 1, 2, 3}


def _covered_six_line(numbers: list[int] | None) -> set[int]:
    return set(numbers) if numbers else set()


def _covered_red(_: list[int] | None) -> set[int]:
    return set(RED)


def _covered_black(_: list[int] | None) -> set[int]:
    return set(BLACK)


def _covered_odd(_: list[int] | None) -> set[int]:
    return {n for n in range(1, 37) if n % 2 == 1}


def _covered_even(_: list[int] | None) -> set[int]:
    return {n for n in range(1, 37) if n % 2 == 0}


def _covered_low(_: list[int] | None) -> set[int]:
    return set(range(1, 19))


def _covered_high(_: list[int] | None) -> set[int]:
    return set(range(19, 37))


def _covered_dozen(numbers: list[int] | None) -> set[int]:
    if not numbers:
        return set()
    d = numbers[0]
    return {n for n in range(1, 37) if dozen(n) == d}


def _covered_column(numbers: list[int] | None) -> set[int]:
    if not numbers:
        return set()
    c = numbers[0]
    return {n for n in range(1, 37) if column(n) == c}


# ---------------------------------------------------------------------------
# BET_TYPES registry
# ---------------------------------------------------------------------------

_EDGE = 2 / 38       # 5.2631…%  (all regular bets on American wheel)
_EDGE_5L = 3 / 38    # 7.8947…%  (five_line only)

BET_TYPES: dict[str, BetType] = {
    "straight":  BetType("straight",  35, 1,  _EDGE,    _covered_straight),
    "split":     BetType("split",     17, 2,  _EDGE,    _covered_split),
    "street":    BetType("street",    11, 3,  _EDGE,    _covered_street),
    "corner":    BetType("corner",     8, 4,  _EDGE,    _covered_corner),
    "five_line": BetType("five_line",  6, 5,  _EDGE_5L, _covered_five_line),
    "six_line":  BetType("six_line",   5, 6,  _EDGE,    _covered_six_line),
    "red":       BetType("red",        1, 18, _EDGE,    _covered_red),
    "black":     BetType("black",      1, 18, _EDGE,    _covered_black),
    "odd":       BetType("odd",        1, 18, _EDGE,    _covered_odd),
    "even":      BetType("even",       1, 18, _EDGE,    _covered_even),
    "low":       BetType("low",        1, 18, _EDGE,    _covered_low),
    "high":      BetType("high",       1, 18, _EDGE,    _covered_high),
    "dozen":     BetType("dozen",      2, 12, _EDGE,    _covered_dozen),
    "column":    BetType("column",     2, 12, _EDGE,    _covered_column),
}


# ---------------------------------------------------------------------------
# Grid adjacency (internal helper)
# ---------------------------------------------------------------------------


def _are_adjacent(n1: int, n2: int) -> bool:
    """Two table cells share an edge."""
    if n1 == n2:
        return False
    # 0 and 37(00) sit side by side in the top row
    if n1 in GREEN and n2 in GREEN:
        return True
    # 0 / 00 are adjacent to 1, 2, 3 on the table
    if n1 in GREEN or n2 in GREEN:
        other = n2 if n1 in GREEN else n1
        return other in {1, 2, 3}
    p1 = grid_pos(n1)
    p2 = grid_pos(n2)
    if p1 is None or p2 is None:
        return False
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_valid_bet(bet_type: str, numbers: list[int] | None) -> bool:
    """Return True iff the bet is geometrically / structurally legal."""
    if bet_type not in BET_TYPES:
        return False

    if bet_type == "straight":
        return (
            numbers is not None
            and len(numbers) == 1
            and numbers[0] in ALL_NUMBERS
        )

    if bet_type == "split":
        if numbers is None or len(numbers) != 2:
            return False
        n1, n2 = numbers
        return (
            n1 in ALL_NUMBERS
            and n2 in ALL_NUMBERS
            and _are_adjacent(n1, n2)
        )

    if bet_type == "street":
        if numbers is None or len(numbers) != 3:
            return False
        s = set(numbers)
        if len(s) != 3 or not all(1 <= n <= 36 for n in s):
            return False
        positions = [grid_pos(n) for n in s]
        if any(p is None for p in positions):
            return False
        cols = {p[0] for p in positions}  # type: ignore[index]
        rows = {p[1] for p in positions}  # type: ignore[index]
        return len(cols) == 1 and rows == {0, 1, 2}

    if bet_type == "corner":
        if numbers is None or len(numbers) != 4:
            return False
        s = set(numbers)
        if len(s) != 4 or not all(1 <= n <= 36 for n in s):
            return False
        positions = [grid_pos(n) for n in s]
        if any(p is None for p in positions):
            return False
        pos_set = set(positions)  # type: ignore[arg-type]
        if len(pos_set) != 4:
            return False
        cols = sorted({p[0] for p in pos_set})
        rows = sorted({p[1] for p in pos_set})
        if len(cols) != 2 or len(rows) != 2:
            return False
        if cols[1] - cols[0] != 1 or rows[1] - rows[0] != 1:
            return False
        expected = {
            (cols[0], rows[0]), (cols[0], rows[1]),
            (cols[1], rows[0]), (cols[1], rows[1]),
        }
        return pos_set == expected

    if bet_type == "five_line":
        # American roulette only — covers 0, 00(=37), 1, 2, 3.
        # TODO: raise ValueError when called in a European context.
        return numbers is not None and set(numbers) == {0, 37, 1, 2, 3}

    if bet_type == "six_line":
        if numbers is None or len(numbers) != 6:
            return False
        s = set(numbers)
        if len(s) != 6 or not all(1 <= n <= 36 for n in s):
            return False
        positions = [grid_pos(n) for n in s]
        if any(p is None for p in positions):
            return False
        cols = sorted({p[0] for p in positions})  # type: ignore[index]
        rows = {p[1] for p in positions}  # type: ignore[index]
        return (
            len(cols) == 2
            and cols[1] - cols[0] == 1
            and rows == {0, 1, 2}
        )

    if bet_type in ("red", "black", "odd", "even", "low", "high"):
        return numbers is None or len(numbers) == 0

    if bet_type in ("dozen", "column"):
        return (
            numbers is not None
            and len(numbers) == 1
            and numbers[0] in {1, 2, 3}
        )

    return False  # unreachable — satisfies type checker


def get_covered_numbers(bet_type: str, numbers: list[int] | None) -> set[int]:
    if bet_type not in BET_TYPES:
        raise ValueError(f"Unknown bet type: {bet_type!r}")
    return BET_TYPES[bet_type].covered_fn(numbers)


def expected_value(
    bet_type: str,
    numbers: list[int] | None,
    p: dict[int, float] | None = None,
) -> float:
    """Subjective EV for a single bet.

    With the default uniform p=1/38 this equals -house_edge for every bet type.
    """
    if bet_type not in BET_TYPES:
        raise ValueError(f"Unknown bet type: {bet_type!r}")
    if p is None:
        p = {n: 1.0 / 38 for n in ALL_NUMBERS}
    bt = BET_TYPES[bet_type]
    covered = bt.covered_fn(numbers)
    p_win = sum(p.get(n, 0.0) for n in covered)
    return (bt.payout + 1) * p_win - 1.0
