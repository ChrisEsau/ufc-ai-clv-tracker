"""Record-state raw fighter feature plugin.

This plugin mirrors the existing V5 fighter-state behavior for record and layoff
features. It is designed for shadow validation before production integration.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline.features.base.elo import safe_div


OUTPUT_COLUMNS = [
    "fights",
    "wins",
    "losses",
    "win_pct",
    "win_streak",
    "loss_streak",
    "days_since_last_fight",
]


def initial_state() -> dict[str, Any]:
    """Return record-specific initial state for one fighter."""

    return {
        "fights": 0,
        "wins": 0,
        "losses": 0,
        "win_streak": 0,
        "loss_streak": 0,
        "last_fight_date": None,
    }


def calculate(
    fighter_history: pd.DataFrame,
    fight_row: pd.Series,
    context: dict | None = None,
) -> dict[str, float]:
    """Return point-in-time record-state features for one fighter."""

    del fighter_history
    context = context or {}
    state = context.get("state", initial_state())
    fight_date = fight_row["date"]

    fights = state.get("fights", 0)
    last_fight_date = state.get("last_fight_date")
    days_since_last_fight = 365 if last_fight_date is None else (fight_date - last_fight_date).days

    return {
        "fights": fights,
        "wins": state.get("wins", 0),
        "losses": state.get("losses", 0),
        "win_pct": safe_div(state.get("wins", 0), fights),
        "win_streak": state.get("win_streak", 0),
        "loss_streak": state.get("loss_streak", 0),
        "days_since_last_fight": days_since_last_fight,
    }


def update_after_fight(
    *,
    state: dict[str, Any],
    fight_date: Any,
    won: bool,
) -> None:
    """Update record-state fields after a completed fight."""

    state["fights"] = state.get("fights", 0) + 1
    if won:
        state["wins"] = state.get("wins", 0) + 1
        state["win_streak"] = state.get("win_streak", 0) + 1
        state["loss_streak"] = 0
    else:
        state["losses"] = state.get("losses", 0) + 1
        state["loss_streak"] = state.get("loss_streak", 0) + 1
        state["win_streak"] = 0

    state["last_fight_date"] = fight_date
