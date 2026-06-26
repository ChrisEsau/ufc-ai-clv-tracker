"""Formula helpers for Round Fighter State features.

This module contains pure calculation helpers only.
It should not read or write parquet files.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd


def safe_div(
    numerator: float | int | None,
    denominator: float | int | None,
) -> float | None:
    """Return numerator / denominator, or None when the rate is not observable."""
    if numerator is None or denominator is None:
        return None

    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return None

    if math.isnan(num) or math.isnan(den) or den == 0:
        return None

    return num / den


def ols_slope(
    rounds: Iterable[float | int],
    values: Iterable[float | int | None],
) -> float | None:
    """Return OLS slope of values by round.

    Rules:
    - Requires at least two non-null observations.
    - Ignores null values.
    - Returns None when slope is not observable.
    """
    frame = pd.DataFrame(
        {
            "round": list(rounds),
            "value": list(values),
        }
    )

    frame["round"] = pd.to_numeric(frame["round"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["round", "value"])

    if len(frame) < 2:
        return None

    round_var = frame["round"].var(ddof=0)
    if pd.isna(round_var) or round_var == 0:
        return None

    slope = frame["round"].cov(frame["value"], ddof=0) / round_var
    return float(slope)


def late_ratio(round_values: pd.Series) -> float | None:
    """Return final observed round value divided by round-one value."""
    values = pd.to_numeric(round_values, errors="coerce").dropna()

    if len(values) < 2:
        return None

    first = float(values.iloc[0])
    last = float(values.iloc[-1])

    if first == 0 or math.isnan(first) or math.isnan(last):
        return None

    return last / first


def late_diff(round_values: pd.Series) -> float | None:
    """Return final observed round value minus round-one value."""
    values = pd.to_numeric(round_values, errors="coerce").dropna()

    if len(values) < 2:
        return None

    first = float(values.iloc[0])
    last = float(values.iloc[-1])

    if math.isnan(first) or math.isnan(last):
        return None

    return last - first
