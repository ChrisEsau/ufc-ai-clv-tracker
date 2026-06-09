"""Safe ratio transform plugin."""

from __future__ import annotations

import numpy as np
import pandas as pd


SAFE_DENOMINATOR_EPSILON = 1e-9


def apply(red: pd.Series, blue: pd.Series, context: dict | None = None) -> pd.Series:
    """Return red values divided by blue values with safe zero handling."""

    red_values = pd.to_numeric(red, errors="coerce")
    blue_values = pd.to_numeric(blue, errors="coerce")
    safe_denominator = blue_values.where(blue_values.abs() > SAFE_DENOMINATOR_EPSILON, np.nan)
    return red_values / safe_denominator
