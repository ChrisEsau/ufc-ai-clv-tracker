"""Striking-rate raw fighter feature plugin.

Mirrors the V5 prefight striking formulas and update behavior from
pipeline.features.base.prefight_features and fighter_state.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline.features.base.elo import safe_div


OUTPUT_COLUMNS = [
    "kd_avg",
    "kd_absorbed_avg",
    "splm",
    "sapm",
    "str_acc",
    "str_def",
]


def initial_state() -> dict[str, Any]:
    """Return striking-specific initial state for one fighter."""

    return {
        "fights": 0,
        "kd_for": 0,
        "kd_against": 0,
        "sig_str_landed": 0,
        "sig_str_attempted": 0,
        "sig_str_absorbed": 0,
        "sig_str_attempted_against": 0,
        "fight_time_sec": 0,
    }


def calculate(
    fighter_history: pd.DataFrame,
    fight_row: pd.Series,
    context: dict | None = None,
) -> dict[str, float]:
    """Return point-in-time striking-rate features for one fighter."""

    del fighter_history, fight_row
    context = context or {}
    state = context.get("state", initial_state())

    fights = state.get("fights", 0)
    minutes = state.get("fight_time_sec", 0) / 60

    return {
        "kd_avg": safe_div(state.get("kd_for", 0), fights),
        "kd_absorbed_avg": safe_div(state.get("kd_against", 0), fights),
        "splm": safe_div(state.get("sig_str_landed", 0), minutes),
        "sapm": safe_div(state.get("sig_str_absorbed", 0), minutes),
        "str_acc": safe_div(state.get("sig_str_landed", 0), state.get("sig_str_attempted", 0)),
        "str_def": 1 - safe_div(
            state.get("sig_str_absorbed", 0),
            state.get("sig_str_attempted_against", 0),
        ),
    }


def update_after_fight(
    *,
    state: dict[str, Any],
    own: dict[str, float],
    opp: dict[str, float],
    fight_time_sec: float,
) -> None:
    """Update striking-rate state after a completed fight."""

    state["fights"] = state.get("fights", 0) + 1
    state["kd_for"] = state.get("kd_for", 0) + own["kd"]
    state["kd_against"] = state.get("kd_against", 0) + opp["kd"]
    state["sig_str_landed"] = state.get("sig_str_landed", 0) + own["sig_str_landed"]
    state["sig_str_attempted"] = state.get("sig_str_attempted", 0) + own["sig_str_attempted"]
    state["sig_str_absorbed"] = state.get("sig_str_absorbed", 0) + opp["sig_str_landed"]
    state["sig_str_attempted_against"] = state.get("sig_str_attempted_against", 0) + opp["sig_str_attempted"]
    state["fight_time_sec"] = state.get("fight_time_sec", 0) + fight_time_sec
