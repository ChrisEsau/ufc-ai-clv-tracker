"""Pair-maximum transform plugin."""

from __future__ import annotations

import pandas as pd


def apply(red: pd.Series, blue: pd.Series, context: dict | None = None) -> pd.Series:
    """Return the row-wise maximum of red and blue values."""

    red_values = pd.to_numeric(red, errors="coerce")
    blue_values = pd.to_numeric(blue, errors="coerce")
    return pd.concat([red_values, blue_values], axis=1).max(axis=1)
