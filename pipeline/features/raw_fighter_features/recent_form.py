"""Recent-form raw fighter feature plugin.

Mirrors the V5 recent-form deque behavior from pipeline.features.base.fighter_state
and prefight_features.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import pandas as pd

from pipeline.features.base.elo import RECENT_N, safe_div
from pipeline.features.base.fighter_state import FINISH_METHODS


OUTPUT_COLUMNS = [
    "recent_win_pct",
    "recent_splm",
    "recent_sapm",
    "recent_td_avg",
    "recent_finish_rate",
    "recent_avg_fight_time",
]


def initial_state() -> dict[str, Any]:
    """Return recent-form initial state for one fighter."""

    return {
        "recent_results": deque(maxlen=RECENT_N),
        "recent_sig_landed": deque(maxlen=RECENT_N),
        "recent_sig_absorbed": deque(maxlen=RECENT_N),
        "recent_td_landed": deque(maxlen=RECENT_N),
        "recent_finish_results": deque(maxlen=RECENT_N),
        "recent_fight_times": deque(maxlen=RECENT_N),
    }


def calculate(
    fighter_history: pd.DataFrame,
    fight_row: pd.Series,
    context: dict | None = None,
) -> dict[str, float]:
    """Return point-in-time recent-form features for one fighter."""

    del fighter_history, fight_row
    context = context or {}
    state = context.get("state", initial_state())

    return {
        "recent_win_pct": safe_div(sum(state.get("recent_results", [])), len(state.get("recent_results", []))),
        "recent_splm": safe_div(sum(state.get("recent_sig_landed", [])), len(state.get("recent_sig_landed", []))),
        "recent_sapm": safe_div(sum(state.get("recent_sig_absorbed", [])), len(state.get("recent_sig_absorbed", []))),
        "recent_td_avg": safe_div(sum(state.get("recent_td_landed", [])), len(state.get("recent_td_landed", []))),
        "recent_finish_rate": safe_div(sum(state.get("recent_finish_results", [])), len(state.get("recent_finish_results", []))),
        "recent_avg_fight_time": safe_div(sum(state.get("recent_fight_times", [])), len(state.get("recent_fight_times", []))),
    }


def update_after_fight(
    *,
    state: dict[str, Any],
    method: Any,
    won: bool,
    own: dict[str, float],
    opp: dict[str, float],
    fight_time_sec: float,
) -> None:
    """Update recent-form state after a completed fight."""

    fight_minutes = fight_time_sec / 60
    method = str(method)
    finish_flag = 1 if method in FINISH_METHODS else 0

    state.setdefault("recent_results", deque(maxlen=RECENT_N)).append(1 if won else 0)
    state.setdefault("recent_sig_landed", deque(maxlen=RECENT_N)).append(
        safe_div(own["sig_str_landed"], fight_minutes)
    )
    state.setdefault("recent_sig_absorbed", deque(maxlen=RECENT_N)).append(
        safe_div(opp["sig_str_landed"], fight_minutes)
    )
    state.setdefault("recent_td_landed", deque(maxlen=RECENT_N)).append(own["td_landed"])
    state.setdefault("recent_finish_results", deque(maxlen=RECENT_N)).append(finish_flag)
    state.setdefault("recent_fight_times", deque(maxlen=RECENT_N)).append(fight_time_sec)
