"""Blue-minus-red transform plugin."""

from __future__ import annotations

import pandas as pd


def apply(red: pd.Series, blue: pd.Series, context: dict | None = None) -> pd.Series:
    """Return blue values minus red values."""

    return pd.to_numeric(blue, errors="coerce") - pd.to_numeric(red, errors="coerce")
