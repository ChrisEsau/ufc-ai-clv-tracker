from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from pipeline.common.paths import MARKET_OUTCOMES_PATH


def american_to_decimal(odds: float | int | None) -> float | None:
    if odds is None or pd.isna(odds):
        return None
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def american_to_implied(odds: float | int | None) -> float | None:
    dec = american_to_decimal(odds)
    if dec is None or dec <= 0:
        return None
    return 1.0 / dec


def signal_id(*parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def display_fight(row: pd.Series) -> str:
    red = row.get("red_fighter")
    blue = row.get("blue_fighter")
    if pd.notna(red) and pd.notna(blue):
        return f"{red} vs {blue}"
    return str(row.get("fight_display") or row.get("event_name") or "Unknown fight")


def outcome_display(row: pd.Series) -> str:
    for col in ["fighter_name", "provider_selection_name", "outcome_display", "side"]:
        value = row.get(col)
        if pd.notna(value) and str(value).strip():
            return str(value)
    return "Unknown outcome"


def market_display(row: pd.Series) -> str:
    value = row.get("market_display")
    if pd.notna(value) and str(value).strip():
        return str(value)
    value = row.get("market_key")
    return str(value or "Unknown market").replace("_", " ").title()


def base_signal(
    *,
    run_id: str,
    timestamp: str,
    signal_type: str,
    signal_family: str,
    severity: str,
    confidence_score: float,
    is_actionable: bool,
    action_label: str,
    row: pd.Series,
    explanation: str,
    suggested_action: str,
    source_path: Any = MARKET_OUTCOMES_PATH,
) -> dict:
    return {
        "signal_id": signal_id(
            run_id,
            signal_type,
            row.get("fight_id"),
            row.get("market_key"),
            row.get("comparison_key"),
            row.get("bookmaker"),
        ),
        "signal_run_id": run_id,
        "signal_timestamp": timestamp,
        "signal_type": signal_type,
        "signal_family": signal_family,
        "severity": severity,
        "confidence_score": confidence_score,
        "is_actionable": is_actionable,
        "action_label": action_label,
        "fight_id": row.get("fight_id"),
        "event_name": row.get("event_name"),
        "fight_display": display_fight(row),
        "market_key": row.get("market_key"),
        "market_display": market_display(row),
        "outcome_key": row.get("outcome_key"),
        "comparison_key": row.get("comparison_key"),
        "outcome_display": outcome_display(row),
        "side": row.get("side"),
        "fighter_name": row.get("fighter_name"),
        "bookmaker": row.get("bookmaker"),
        "explanation": explanation,
        "suggested_action": suggested_action,
        "source_path": str(source_path),
    }
