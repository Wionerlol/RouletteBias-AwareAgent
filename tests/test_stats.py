"""Tests for roulette_agent.stats."""

import pytest

from roulette_agent.stats import basic_stats, frequency_counts


class TestFrequencyCounts:
    def test_empty_american_has_all_38_keys(self):
        freq = frequency_counts([], "american")
        assert set(freq.keys()) == set(range(38))
        assert all(v == 0 for v in freq.values())

    def test_empty_european_has_37_keys_no_double_zero(self):
        freq = frequency_counts([], "european")
        assert set(freq.keys()) == set(range(37))
        assert 37 not in freq

    def test_counts_accumulate(self):
        freq = frequency_counts([17, 17, 0, 17], "american")
        assert freq[17] == 3
        assert freq[0] == 1
        assert freq[1] == 0

    def test_unseen_numbers_zero(self):
        freq = frequency_counts([5], "american")
        assert freq[5] == 1
        assert sum(v for k, v in freq.items() if k != 5) == 0

    def test_total_count_equals_len_history(self):
        history = [1, 2, 3, 0, 17, 35, 37]
        freq = frequency_counts(history, "american")
        assert sum(freq.values()) == len(history)


class TestBasicStatsEmpty:
    def test_zero_spins(self):
        s = basic_stats([])
        assert s["n_spins"] == 0
        assert s["red_count"] == 0
        assert s["black_count"] == 0
        assert s["green_count"] == 0
        assert s["odd_count"] == 0
        assert s["even_count"] == 0
        assert s["low_count"] == 0
        assert s["high_count"] == 0

    def test_zero_pcts_when_empty(self):
        s = basic_stats([])
        for key in ("red_pct", "black_pct", "odd_pct", "even_pct", "low_pct", "high_pct"):
            assert s[key] == pytest.approx(0.0), f"{key} should be 0"

    def test_dozen_and_column_zero_when_empty(self):
        s = basic_stats([])
        assert s["dozen_counts"] == {1: 0, 2: 0, 3: 0}
        assert s["column_counts"] == {1: 0, 2: 0, 3: 0}


class TestBasicStatsCounts:
    def test_n_spins(self):
        assert basic_stats([1, 2, 3, 0, 17])["n_spins"] == 5

    def test_red_black_green_partition(self):
        history = [0, 37, 1, 2, 7, 14]  # 2 green, rest non-green
        s = basic_stats(history)
        assert s["red_count"] + s["black_count"] + s["green_count"] == 6

    def test_green_count_american(self):
        s = basic_stats([0, 37, 0])
        assert s["green_count"] == 3

    def test_green_count_european(self):
        # European has no 37; only 0 is green
        s = basic_stats([0, 0, 1], "european")
        assert s["green_count"] == 2

    def test_dozen_counts(self):
        history = [1, 5, 12, 13, 24, 25, 36, 0]  # 0 is green → ignored
        s = basic_stats(history)
        assert s["dozen_counts"] == {1: 3, 2: 2, 3: 2}

    def test_column_counts(self):
        # col1: 1,4,7  col2: 2,5  col3: 3
        history = [1, 4, 7, 2, 5, 3]
        s = basic_stats(history)
        assert s["column_counts"] == {1: 3, 2: 2, 3: 1}


class TestBasicStatsDenominators:
    def test_odd_pct_denominator_excludes_green(self):
        """3 odd spins + 1 green → odd_pct = 3/3 = 1.0 (not 3/4)."""
        history = [0, 1, 3, 5]
        s = basic_stats(history)
        assert s["odd_count"] == 3
        assert s["odd_pct"] == pytest.approx(1.0)

    def test_even_pct_denominator_excludes_green(self):
        history = [37, 2, 4]  # 1 green, 2 even
        s = basic_stats(history)
        assert s["even_pct"] == pytest.approx(1.0)

    def test_low_high_pct_denominator_excludes_green(self):
        history = [0, 1, 19]  # 0=green, 1=low, 19=high
        s = basic_stats(history)
        assert s["low_pct"] == pytest.approx(0.5)
        assert s["high_pct"] == pytest.approx(0.5)

    def test_all_green_pcts_are_zero(self):
        """If every result is green, odd/even/low/high pcts are 0 (not NaN)."""
        s = basic_stats([0, 37, 0])
        assert s["odd_pct"] == pytest.approx(0.0)
        assert s["even_pct"] == pytest.approx(0.0)
        assert s["low_pct"] == pytest.approx(0.0)
        assert s["high_pct"] == pytest.approx(0.0)

    def test_red_pct_denominator_is_total_spins(self):
        """red_pct includes green in denominator."""
        history = [0, 1]  # 0=green, 1=red → red_pct = 1/2
        s = basic_stats(history)
        assert s["red_pct"] == pytest.approx(0.5)


class TestHotNumbers:
    def test_ordering_by_count(self):
        history = [17, 17, 17, 5, 5, 1]
        s = basic_stats(history)
        hot = s["hot_numbers_top5"]
        assert hot[0] == (17, 3)
        assert hot[1] == (5, 2)
        assert hot[2] == (1, 1)

    def test_tiebreak_by_smaller_number(self):
        """Equal counts → smaller number ranks first."""
        history = [1, 2]
        s = basic_stats(history)
        top = s["hot_numbers_top5"]
        assert top[0][0] == 1
        assert top[1][0] == 2

    def test_returns_five_items(self):
        s = basic_stats([1, 2, 3])
        assert len(s["hot_numbers_top5"]) == 5

    def test_frequency_key_matches_hot_numbers(self):
        history = [17, 17, 5]
        s = basic_stats(history)
        assert s["frequency"][17] == 2
        assert s["frequency"][5] == 1
        assert s["frequency"][0] == 0
