"""Shared helpers for CLV artifact generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

MONEYLINE = "Moneyline"


def american_to_decimal(odds) -> float:
    """Convert American odds to decimal odds."""

    value = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(value) or float(value) == 0:
        return np.nan
    value = float(value)
    if value > 0:
        return 1 + value / 100
    return 1 + 100 / abs(value)


def american_to_implied_prob(odds) -> float:
    """Convert American odds to implied probability before vig removal."""

    value = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(value) or float(value) == 0:
        return np.nan
    value = float(value)
    if value > 0:
        return 100 / (value + 100)
    return abs(value) / (abs(value) + 100)


def clv_pct(odds_taken, closing_odds) -> float:
    """Return CLV percentage using decimal odds ratio."""

    taken_decimal = american_to_decimal(odds_taken)
    closing_decimal = american_to_decimal(closing_odds)
    if pd.isna(taken_decimal) or pd.isna(closing_decimal) or closing_decimal == 0:
        return np.nan
    return (taken_decimal / closing_decimal - 1) * 100


def normalize_market_type(value) -> str:
    """Normalize market labels for reliable joins."""

    text = "" if pd.isna(value) else str(value).strip().lower()
    if text in {"", "moneyline", "h2h", "money line"}:
        return MONEYLINE
    return str(value).strip().title()


def confidence_tier(probability) -> str:
    """Bucket model confidence for CLV reporting."""

    value = pd.to_numeric(pd.Series([probability]), errors="coerce").iloc[0]
    if pd.isna(value):
        return "Unknown"
    value = float(value)
    if value <= 1:
        value *= 100
    if value >= 70:
        return "Strong Bet (>=70%)"
    if value >= 50:
        return "Lean Bet (50-69%)"
    return "Watchlist (<50%)"


def odds_bucket(odds) -> str:
    """Bucket American odds for CLV reporting."""

    value = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(value):
        return "Unknown"
    value = float(value)
    if value < -500:
        return "Heavy Favorites (<-500)"
    if value <= -150:
        return "Favorites (-500 to -150)"
    if value <= 125:
        return "Small Dogs (-150 to +125)"
    if value <= 300:
        return "Medium Dogs (+125 to +300)"
    return "Large Dogs (+300+)"


def empty_frame(columns: list[str]) -> pd.DataFrame:
    """Return an empty DataFrame with a stable column order."""

    return pd.DataFrame(columns=columns)
