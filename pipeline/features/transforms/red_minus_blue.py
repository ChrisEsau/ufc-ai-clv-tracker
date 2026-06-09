"""Red-minus-blue transform plugin."""

from __future__ import annotations

import pandas as pd


def apply(red: pd.Series, blue: pd.Series, context: dict | None = None) -> pd.Series:
    """Return red values minus blue values."""

    return pd.to_numeric(red, errors="coerce") - pd.to_numeric(blue, errors="coerce")
