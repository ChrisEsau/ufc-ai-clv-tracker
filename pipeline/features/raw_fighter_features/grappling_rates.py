"""Grappling-rate raw fighter feature plugin.

Mirrors the V5 prefight grappling and control formulas from
pipeline.features.base.prefight_features and update behavior from fighter_state.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline.features.base.elo import safe_div


OUTPUT_COLUMNS = [
    "td_avg",
    "td_acc",
    "td_def",
    "sub_avg",
    "ctrl_per_min",
    "ctrl_against_per_min",
]


def initial_state() -> dict[str, Any]:
    """Return grappling-specific initial state for one fighter."""

    return {
        "td_landed": 0,
        "td_attempted": 0,
        "td_allowed": 0,
        "td_attempted_against": 0,
        "sub_att": 0,
        "ctrl": 0,
        "ctrl_against": 0,
        "fight_time_sec": 0,
    }


def calculate(
    fighter_history: pd.DataFrame,
    fight_row: pd.Series,
    context: dict | None = None,
) -> dict[str, float]:
    """Return point-in-time grappling-rate features for one fighter."""

    del fighter_history, fight_row
    context = context or {}
    state = context.get("state", initial_state())

    fight_time_sec = state.get("fight_time_sec", 0)
    minutes = fight_time_sec / 60
    fifteen_min_units = fight_time_sec / 900

    return {
        "td_avg": safe_div(state.get("td_landed", 0), fifteen_min_units),
        "td_acc": safe_div(state.get("td_landed", 0), state.get("td_attempted", 0)),
        "td_def": 1 - safe_div(
            state.get("td_allowed", 0),
            state.get("td_attempted_against", 0),
        ),
        "sub_avg": safe_div(state.get("sub_att", 0), fifteen_min_units),
        "ctrl_per_min": safe_div(state.get("ctrl", 0), minutes),
        "ctrl_against_per_min": safe_div(state.get("ctrl_against", 0), minutes),
    }


def update_after_fight(
    *,
    state: dict[str, Any],
    own: dict[str, float],
    opp: dict[str, float],
    fight_time_sec: float,
) -> None:
    """Update grappling-rate state after a completed fight."""

    state["td_landed"] = state.get("td_landed", 0) + own["td_landed"]
    state["td_attempted"] = state.get("td_attempted", 0) + own["td_attempted"]
    state["td_allowed"] = state.get("td_allowed", 0) + opp["td_landed"]
    state["td_attempted_against"] = state.get("td_attempted_against", 0) + opp["td_attempted"]
    state["sub_att"] = state.get("sub_att", 0) + own["sub_att"]
    state["ctrl"] = state.get("ctrl", 0) + own["ctrl"]
    state["ctrl_against"] = state.get("ctrl_against", 0) + opp["ctrl"]
    state["fight_time_sec"] = state.get("fight_time_sec", 0) + fight_time_sec
