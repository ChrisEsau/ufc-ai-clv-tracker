"""Finish-profile raw fighter feature plugin.

Mirrors the V5 prefight finish/method and fight-duration formulas from
pipeline.features.base.prefight_features and update behavior from fighter_state.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import pandas as pd

from pipeline.features.base.elo import safe_div
from pipeline.features.base.fighter_state import FINISH_METHODS, KO_METHODS

RECENT_METHOD_DEFEAT_N = 3
NO_PRIOR_METHOD_DEFEAT_DAYS = 3650

OUTPUT_COLUMNS = [
    "finish_rate",
    "ko_rate",
    "sub_win_rate",
    "decision_win_rate",
    "finish_loss_rate",
    "career_ko_losses",
    "recent_ko_losses_3",
    "last_fight_ko_loss",
    "days_since_last_ko_loss",
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
        "method_defeat_count": 0,
        "last_method_defeat": 0,
        "last_method_defeat_date": None,
        "recent_method_defeats": deque(maxlen=RECENT_METHOD_DEFEAT_N),
        "decision_losses": 0,
        "fight_time_sec": 0,
    }


def calculate(
    fighter_history: pd.DataFrame,
    fight_row: pd.Series,
    context: dict | None = None,
) -> dict[str, float]:
    """Return point-in-time finish-profile features for one fighter."""

    del fighter_history
    context = context or {}
    state = context.get("state", initial_state())

    fights = state.get("fights", 0)
    wins = state.get("wins", 0)
    losses = state.get("losses", 0)
    last_method_defeat_date = state.get("last_method_defeat_date")
    if last_method_defeat_date is None:
        days_since_last_method_defeat = NO_PRIOR_METHOD_DEFEAT_DAYS
    else:
        days_since_last_method_defeat = (fight_row["date"] - last_method_defeat_date).days

    return {
        "finish_rate": safe_div(state.get("finish_wins", 0), wins),
        "ko_rate": safe_div(state.get("ko_wins", 0), wins),
        "sub_win_rate": safe_div(state.get("sub_wins", 0), wins),
        "decision_win_rate": safe_div(state.get("decision_wins", 0), wins),
        "finish_loss_rate": safe_div(state.get("finish_losses", 0), losses),
        "career_ko_losses": state.get("method_defeat_count", 0),
        "recent_ko_losses_3": sum(state.get("recent_method_defeats", [])),
        "last_fight_ko_loss": state.get("last_method_defeat", 0),
        "days_since_last_ko_loss": days_since_last_method_defeat,
        "decision_loss_rate": safe_div(state.get("decision_losses", 0), losses),
        "avg_fight_time": safe_div(state.get("fight_time_sec", 0), fights),
    }


def update_after_fight(
    *,
    state: dict[str, Any],
    method: Any,
    won: bool,
    fight_time_sec: float,
    fight_date: Any,
) -> None:
    """Update finish-profile state after a completed fight."""

    method = str(method)
    method_defeat = (not won) and method in KO_METHODS

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

        if method_defeat:
            state["method_defeat_count"] = state.get("method_defeat_count", 0) + 1
            state["last_method_defeat_date"] = fight_date

    state["last_method_defeat"] = 1 if method_defeat else 0
    state.setdefault("recent_method_defeats", deque(maxlen=RECENT_METHOD_DEFEAT_N)).append(
        1 if method_defeat else 0
    )
