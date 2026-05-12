"""Tests for roulette_agent.layout — must all pass before Round 2 begins."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from roulette_agent.layout import (
    ALL_NUMBERS,
    BET_TYPES,
    BLACK,
    GREEN,
    RED,
    WHEEL_ORDER_AMERICAN,
    WHEEL_ORDER_EUROPEAN,
    color,
    column,
    dozen,
    expected_value,
    get_covered_numbers,
    grid_pos,
    high_low,
    is_valid_bet,
    parity,
)


# ---------------------------------------------------------------------------
# Number-set sizes
# ---------------------------------------------------------------------------


class TestNumberSets:
    def test_red_count(self):
        assert len(RED) == 18

    def test_black_count(self):
        assert len(BLACK) == 18

    def test_red_black_disjoint(self):
        assert not (RED & BLACK)

    def test_all_numbers_is_0_to_37(self):
        assert ALL_NUMBERS == set(range(38))

    def test_odd_count(self):
        assert len([n for n in range(1, 37) if parity(n) == "odd"]) == 18

    def test_even_count(self):
        assert len([n for n in range(1, 37) if parity(n) == "even"]) == 18

    def test_low_count(self):
        assert len([n for n in range(1, 37) if high_low(n) == "low"]) == 18

    def test_high_count(self):
        assert len([n for n in range(1, 37) if high_low(n) == "high"]) == 18

    def test_dozen_counts(self):
        for d in (1, 2, 3):
            assert len([n for n in range(1, 37) if dozen(n) == d]) == 12

    def test_column_counts(self):
        for c in (1, 2, 3):
            assert len([n for n in range(1, 37) if column(n) == c]) == 12


# ---------------------------------------------------------------------------
# Wheel order integrity
# ---------------------------------------------------------------------------


class TestWheelOrders:
    def test_american_length(self):
        assert len(WHEEL_ORDER_AMERICAN) == 38

    def test_american_contains_all(self):
        assert set(WHEEL_ORDER_AMERICAN) == set(range(38))

    def test_american_no_duplicates(self):
        assert len(WHEEL_ORDER_AMERICAN) == len(set(WHEEL_ORDER_AMERICAN))

    def test_european_length(self):
        assert len(WHEEL_ORDER_EUROPEAN) == 37

    def test_european_contains_all(self):
        assert set(WHEEL_ORDER_EUROPEAN) == set(range(37))

    def test_european_no_duplicates(self):
        assert len(WHEEL_ORDER_EUROPEAN) == len(set(WHEEL_ORDER_EUROPEAN))

    def test_european_has_no_double_zero(self):
        assert 37 not in WHEEL_ORDER_EUROPEAN


# ---------------------------------------------------------------------------
# Attribute functions — green / boundary values
# ---------------------------------------------------------------------------


class TestAttributeFunctions:
    def test_color_zero(self):
        assert color(0) == "green"

    def test_color_double_zero(self):
        assert color(37) == "green"

    def test_parity_green_is_none(self):
        assert parity(0) is None
        assert parity(37) is None

    def test_high_low_green_is_none(self):
        assert high_low(0) is None
        assert high_low(37) is None

    def test_dozen_green_is_none(self):
        assert dozen(0) is None
        assert dozen(37) is None

    def test_column_green_is_none(self):
        assert column(0) is None
        assert column(37) is None

    def test_grid_pos_green_is_none(self):
        assert grid_pos(0) is None
        assert grid_pos(37) is None

    def test_column_mapping(self):
        # col 1: n % 3 == 1
        for n in (1, 4, 7, 34):
            assert column(n) == 1
        # col 2: n % 3 == 2
        for n in (2, 5, 8, 35):
            assert column(n) == 2
        # col 3: n % 3 == 0
        for n in (3, 6, 9, 36):
            assert column(n) == 3

    def test_dozen_boundaries(self):
        assert dozen(1) == 1
        assert dozen(12) == 1
        assert dozen(13) == 2
        assert dozen(24) == 2
        assert dozen(25) == 3
        assert dozen(36) == 3

    def test_high_low_boundaries(self):
        assert high_low(1) == "low"
        assert high_low(18) == "low"
        assert high_low(19) == "high"
        assert high_low(36) == "high"


# ---------------------------------------------------------------------------
# grid_pos round-trip: (col, row) → n = 3*col + 3 - row
# ---------------------------------------------------------------------------


class TestGridPos:
    def test_round_trip_all(self):
        for n in range(1, 37):
            pos = grid_pos(n)
            assert pos is not None, f"grid_pos({n}) unexpectedly None"
            col_idx, row_idx = pos
            recovered = 3 * col_idx + 3 - row_idx
            assert recovered == n, f"round-trip failed: n={n} → {pos} → {recovered}"

    def test_specific_corners(self):
        assert grid_pos(1) == (0, 2)   # bottom-left
        assert grid_pos(3) == (0, 0)   # top-left
        assert grid_pos(34) == (11, 2) # bottom-right
        assert grid_pos(36) == (11, 0) # top-right

    def test_col_and_row_ranges(self):
        for n in range(1, 37):
            col_idx, row_idx = grid_pos(n)  # type: ignore[misc]
            assert 0 <= col_idx <= 11
            assert 0 <= row_idx <= 2


# ---------------------------------------------------------------------------
# Expected value for every bet type
# ---------------------------------------------------------------------------


class TestExpectedValue:
    # Maps bet_type → a valid numbers argument
    _SAMPLE: dict[str, list[int] | None] = {
        "straight":  [17],
        "split":     [1, 2],
        "street":    [1, 2, 3],
        "corner":    [1, 2, 4, 5],
        "five_line": [0, 37, 1, 2, 3],
        "six_line":  [1, 2, 3, 4, 5, 6],
        "red":       None,
        "black":     None,
        "odd":       None,
        "even":      None,
        "low":       None,
        "high":      None,
        "dozen":     [1],
        "column":    [1],
    }

    def test_ev_equals_neg_edge_for_all_types(self):
        for name, bt in BET_TYPES.items():
            nums = self._SAMPLE[name]
            ev = expected_value(name, nums)
            assert abs(ev - (-bt.edge)) < 1e-9, (
                f"{name}: expected EV={-bt.edge:.10f}, got {ev:.10f}"
            )

    def test_straight_exact(self):
        ev = expected_value("straight", [17])
        assert abs(ev - (-2 / 38)) < 1e-9

    def test_five_line_exact(self):
        ev = expected_value("five_line", [0, 37, 1, 2, 3])
        assert abs(ev - (-3 / 38)) < 1e-9

    def test_custom_p_shifts_ev(self):
        """p heavily favouring 17 should give a positive EV for straight 17."""
        p = {n: 0.0 for n in ALL_NUMBERS}
        p[17] = 1.0
        ev = expected_value("straight", [17], p=p)
        # EV = 36 * 1.0 - 1 = 35.0
        assert abs(ev - 35.0) < 1e-9


# ---------------------------------------------------------------------------
# Bet validity
# ---------------------------------------------------------------------------


class TestIsValidBet:
    # --- straight ---
    def test_straight_valid(self):
        assert is_valid_bet("straight", [17]) is True

    def test_straight_valid_zero(self):
        assert is_valid_bet("straight", [0]) is True

    def test_straight_valid_double_zero(self):
        assert is_valid_bet("straight", [37]) is True

    def test_straight_invalid_none(self):
        assert is_valid_bet("straight", None) is False

    def test_straight_invalid_two_numbers(self):
        assert is_valid_bet("straight", [1, 2]) is False

    # --- split ---
    def test_split_vertical(self):
        assert is_valid_bet("split", [1, 2]) is True

    def test_split_horizontal(self):
        assert is_valid_bet("split", [1, 4]) is True

    def test_split_zero_and_one(self):
        assert is_valid_bet("split", [0, 1]) is True

    def test_split_zero_and_double_zero(self):
        assert is_valid_bet("split", [0, 37]) is True

    def test_split_non_adjacent(self):
        assert is_valid_bet("split", [1, 5]) is False

    def test_split_same_number(self):
        assert is_valid_bet("split", [5, 5]) is False

    # --- street ---
    def test_street_valid(self):
        assert is_valid_bet("street", [1, 2, 3]) is True
        assert is_valid_bet("street", [34, 35, 36]) is True

    def test_street_invalid_mixed_cols(self):
        assert is_valid_bet("street", [1, 2, 4]) is False

    # --- corner ---
    def test_corner_valid_classic(self):
        assert is_valid_bet("corner", [28, 29, 31, 32]) is True

    def test_corner_invalid_not_2x2(self):
        assert is_valid_bet("corner", [1, 2, 3, 4]) is False

    def test_corner_valid_top_left(self):
        # 1:(0,2), 2:(0,1), 4:(1,2), 5:(1,1) → 2×2 ✓
        assert is_valid_bet("corner", [1, 2, 4, 5]) is True

    # --- five_line ---
    def test_five_line_valid(self):
        assert is_valid_bet("five_line", [0, 37, 1, 2, 3]) is True

    def test_five_line_wrong_numbers(self):
        assert is_valid_bet("five_line", [0, 1, 2, 3, 4]) is False

    def test_five_line_none(self):
        assert is_valid_bet("five_line", None) is False

    # --- six_line ---
    def test_six_line_valid(self):
        assert is_valid_bet("six_line", [1, 2, 3, 4, 5, 6]) is True

    def test_six_line_invalid_non_adjacent_streets(self):
        # 1-2-3 and 7-8-9 are not adjacent columns
        assert is_valid_bet("six_line", [1, 2, 3, 7, 8, 9]) is False

    # --- outside bets ---
    def test_red_none(self):
        assert is_valid_bet("red", None) is True

    def test_red_empty_list(self):
        assert is_valid_bet("red", []) is True

    # --- dozen / column ---
    def test_dozen_valid(self):
        assert is_valid_bet("dozen", [1]) is True
        assert is_valid_bet("dozen", [3]) is True

    def test_dozen_invalid_value(self):
        assert is_valid_bet("dozen", [4]) is False

    def test_column_valid(self):
        assert is_valid_bet("column", [2]) is True

    def test_column_invalid_none(self):
        assert is_valid_bet("column", None) is False

    # --- unknown bet type ---
    def test_unknown_type(self):
        assert is_valid_bet("yolo", [1]) is False


# ---------------------------------------------------------------------------
# get_covered_numbers spot-checks
# ---------------------------------------------------------------------------


class TestGetCoveredNumbers:
    def test_red_covers_18(self):
        assert get_covered_numbers("red", None) == RED

    def test_straight_covers_one(self):
        assert get_covered_numbers("straight", [17]) == {17}

    def test_five_line_covers_5(self):
        assert get_covered_numbers("five_line", None) == {0, 37, 1, 2, 3}

    def test_dozen_1_covers_1_to_12(self):
        assert get_covered_numbers("dozen", [1]) == set(range(1, 13))

    def test_column_1_covers_correct(self):
        expected = {n for n in range(1, 37) if n % 3 == 1}
        assert get_covered_numbers("column", [1]) == expected

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_covered_numbers("yolo", None)


# ---------------------------------------------------------------------------
# Hypothesis: attribute consistency for all n in 1..36
# ---------------------------------------------------------------------------


@given(st.integers(min_value=1, max_value=36))
def test_attribute_consistency(n: int) -> None:
    """All attributes are defined and internally consistent for every n in 1..36."""
    assert color(n) in ("red", "black")
    assert parity(n) in ("odd", "even")
    assert high_low(n) in ("low", "high")
    assert dozen(n) in (1, 2, 3)
    assert column(n) in (1, 2, 3)

    # Color consistent with RED/BLACK sets
    assert (color(n) == "red") == (n in RED)
    assert (color(n) == "black") == (n in BLACK)

    # Parity consistent with n % 2
    assert (parity(n) == "odd") == (n % 2 == 1)

    # Dozen boundaries
    d = dozen(n)
    if d == 1:
        assert 1 <= n <= 12
    elif d == 2:
        assert 13 <= n <= 24
    else:
        assert 25 <= n <= 36

    # Column consistent with n % 3
    expected_col = (n % 3) if n % 3 != 0 else 3
    assert column(n) == expected_col

    # grid_pos in valid range
    pos = grid_pos(n)
    assert pos is not None
    col_idx, row_idx = pos
    assert 0 <= col_idx <= 11
    assert 0 <= row_idx <= 2
