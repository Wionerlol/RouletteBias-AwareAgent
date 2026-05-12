"""Basic descriptive statistics over a spin history (frequency counts, color/parity breakdowns, hot numbers)."""

from roulette_agent.layout import color, column, dozen, high_low, parity


def _all_numbers(wheel_type: str) -> set[int]:
    return set(range(38)) if wheel_type == "american" else set(range(37))


def _green(wheel_type: str) -> set[int]:
    return {0, 37} if wheel_type == "american" else {0}


def frequency_counts(history: list[int], wheel_type: str = "american") -> dict[int, int]:
    """Count occurrences of each pocket; pockets not seen in history are included with count 0."""
    counts: dict[int, int] = {n: 0 for n in _all_numbers(wheel_type)}
    for n in history:
        if n in counts:
            counts[n] += 1
    return counts


def basic_stats(history: list[int], wheel_type: str = "american") -> dict:
    """Comprehensive summary statistics over a spin history.

    Percentages for odd/even/low/high use the non-green spin count as denominator
    because 0 and 00 are neither odd/even nor low/high.
    Percentages for red/black use total spins as denominator.
    """
    n_spins = len(history)
    wheel_green = _green(wheel_type)

    red_count = sum(1 for n in history if color(n) == "red")
    black_count = sum(1 for n in history if color(n) == "black")
    green_count = sum(1 for n in history if n in wheel_green)
    non_green = n_spins - green_count

    odd_count = sum(1 for n in history if parity(n) == "odd")
    even_count = sum(1 for n in history if parity(n) == "even")
    low_count = sum(1 for n in history if high_low(n) == "low")
    high_count = sum(1 for n in history if high_low(n) == "high")

    dozen_counts = {d: sum(1 for n in history if dozen(n) == d) for d in (1, 2, 3)}
    column_counts = {c: sum(1 for n in history if column(n) == c) for c in (1, 2, 3)}

    freq = frequency_counts(history, wheel_type)
    hot5: list[tuple[int, int]] = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:5]

    def pct(count: int, total: int) -> float:
        return count / total if total > 0 else 0.0

    return {
        "n_spins": n_spins,
        "red_count": red_count,
        "red_pct": pct(red_count, n_spins),
        "black_count": black_count,
        "black_pct": pct(black_count, n_spins),
        "green_count": green_count,
        "odd_count": odd_count,
        "odd_pct": pct(odd_count, non_green),    # denominator excludes green
        "even_count": even_count,
        "even_pct": pct(even_count, non_green),
        "low_count": low_count,
        "low_pct": pct(low_count, non_green),
        "high_count": high_count,
        "high_pct": pct(high_count, non_green),
        "dozen_counts": dozen_counts,
        "column_counts": column_counts,
        "frequency": freq,
        "hot_numbers_top5": hot5,
    }
