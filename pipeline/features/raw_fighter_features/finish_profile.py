"""Finish-profile raw fighter feature plugin.

Mirrors the V5 prefight finish/method and fight-duration formulas from
pipeline.features.base.prefight_features and update behavior from fighter_state.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline.features.base.elo import safe_div
from pipeline.features.base.fighter_state import FINISH_METHODS, KO_METHODS


OUTPUT_COLUMNS = [
    "finish_rate",
    "ko_rate",
    "sub_win_rate",
    "decision_win_rate",
    "finish_loss_rate",
    "decision_loss_rate",
    "avg_fight_time",
]


def initial_state() -> dict[str, Any]:
    """Return finish-profile initial state for one fighter."""

    return {
        "fights": 0,
        "wins": 0,
        "losses": 0,
        "finish_wins": 0,
        "ko_wins": 0,
        "sub_wins": 0,
        "decision_wins": 0,
        "finish_losses": 0,
        "decision_losses": 0,
        "fight_time_sec": 0,
    }


def calculate(
    fighter_history: pd.DataFrame,
    fight_row: pd.Series,
    context: dict | None = None,
) -> dict[str, float]:
    """Return point-in-time finish-profile features for one fighter."""

    del fighter_history, fight_row
    context = context or {}
    state = context.get("state", initial_state())

    fights = state.get("fights", 0)
    wins = state.get("wins", 0)
    losses = state.get("losses", 0)

    return {
        "finish_rate": safe_div(state.get("finish_wins", 0), wins),
        "ko_rate": safe_div(state.get("ko_wins", 0), wins),
        "sub_win_rate": safe_div(state.get("sub_wins", 0), wins),
        "decision_win_rate": safe_div(state.get("decision_wins", 0), wins),
        "finish_loss_rate": safe_div(state.get("finish_losses", 0), losses),
        "decision_loss_rate": safe_div(state.get("decision_losses", 0), losses),
        "avg_fight_time": safe_div(state.get("fight_time_sec", 0), fights),
    }


def update_after_fight(
    *,
    state: dict[str, Any],
    method: Any,
    won: bool,
    fight_time_sec: float,
) -> None:
    """Update finish-profile state after a completed fight."""

    method = str(method)
    state["fights"] = state.get("fights", 0) + 1
    state["fight_time_sec"] = state.get("fight_time_sec", 0) + fight_time_sec

    if won:
        state["wins"] = state.get("wins", 0) + 1
        if method in KO_METHODS:
            state["finish_wins"] = state.get("finish_wins", 0) + 1
            state["ko_wins"] = state.get("ko_wins", 0) + 1
        elif method == "Submission":
            state["finish_wins"] = state.get("finish_wins", 0) + 1
            state["sub_wins"] = state.get("sub_wins", 0) + 1
        elif "Decision" in method:
            state["decision_wins"] = state.get("decision_wins", 0) + 1
    else:
        state["losses"] = state.get("losses", 0) + 1
        if method in FINISH_METHODS:
            state["finish_losses"] = state.get("finish_losses", 0) + 1
        elif "Decision" in method:
            state["decision_losses"] = state.get("decision_losses", 0) + 1
