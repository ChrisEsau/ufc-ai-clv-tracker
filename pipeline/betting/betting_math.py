from __future__ import annotations

from typing import Any

import pandas as pd


def american_to_decimal(american_odds: Any) -> float | None:
    odds = pd.to_numeric(pd.Series([american_odds]), errors="coerce").iloc[0]
    if pd.isna(odds) or float(odds) == 0:
        return None
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def kelly_fraction(probability: Any, decimal_odds: Any) -> float:
    p = pd.to_numeric(pd.Series([probability]), errors="coerce").iloc[0]
    d = pd.to_numeric(pd.Series([decimal_odds]), errors="coerce").iloc[0]
    if pd.isna(p) or pd.isna(d) or d <= 1.0:
        return 0.0
    b = float(d) - 1.0
    q = 1.0 - float(p)
    kelly = (b * float(p) - q) / b
    return float(max(kelly, 0.0))


def market_display(market_key: Any) -> str:
    key = str(market_key or "").strip().lower()
    mapping = {
        "h2h": "Moneyline",
        "moneyline": "Moneyline",
        "method": "Method of Victory",
        "goes_distance": "Goes Distance",
        "totals": "Totals",
        "round": "Round Props",
    }
    return mapping.get(key, str(market_key or "").replace("_", " ").title())


def bet_status(row: pd.Series) -> str:
    if not bool(row.get("passes_market_data_filter", False)):
        return "NO_MARKET_DATA"
    if not bool(row.get("passes_edge_filter", False)):
        return "FILTERED_EDGE"
    if not bool(row.get("passes_confidence_filter", False)):
        return "FILTERED_CONFIDENCE"
    if not bool(row.get("passes_odds_filter", False)):
        return "FILTERED_ODDS"
    return "BET_CANDIDATE"
