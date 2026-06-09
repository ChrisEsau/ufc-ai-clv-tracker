"""Elo raw fighter feature plugin.

This plugin mirrors the existing V5 Elo behavior:

- New fighters start at START_ELO.
- Expected score uses the shared Elo helper.
- Elo updates happen after both prefight snapshots are emitted.
- Best-win, worst-loss, and average opponent Elo are updated from opponent
  prefight Elo values.

The plugin is intentionally small and state-oriented so it can be validated in
shadow mode before replacing the Elo logic inside history_builder.py.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline.features.base.elo import K_FACTOR, START_ELO, expected_score, safe_div


OUTPUT_COLUMNS = [
    "elo",
    "avg_opponent_elo",
    "best_win_elo",
    "worst_loss_elo",
]


def initial_state() -> dict[str, Any]:
    """Return the Elo-specific initial state for one fighter."""

    return {
        "elo": START_ELO,
        "fights": 0,
        "opponent_elo_sum": 0,
        "best_win_elo": START_ELO,
        "worst_loss_elo": START_ELO,
    }


def calculate(
    fighter_history: pd.DataFrame,
    fight_row: pd.Series,
    context: dict | None = None,
) -> dict[str, float]:
    """Return point-in-time Elo features for one fighter.

    Parameters follow the raw fighter feature plugin contract. The current
    shadow implementation expects ``context`` to provide a ``state`` dictionary
    for the fighter being snapshotted.
    """

    del fighter_history, fight_row
    context = context or {}
    state = context.get("state", initial_state())

    fights = state.get("fights", 0)
    return {
        "elo": state.get("elo", START_ELO),
        "avg_opponent_elo": safe_div(state.get("opponent_elo_sum", 0), fights),
        "best_win_elo": state.get("best_win_elo", START_ELO),
        "worst_loss_elo": state.get("worst_loss_elo", START_ELO),
    }


def update_after_fight(
    *,
    red_state: dict[str, Any],
    blue_state: dict[str, Any],
    red_won: bool,
) -> None:
    """Update red/blue Elo states after a completed fight.

    This mirrors the existing history_builder.py Elo update order. Opponent
    quality stats use prefight Elo values, then Elo ratings are updated.
    """

    r_elo = red_state.get("elo", START_ELO)
    b_elo = blue_state.get("elo", START_ELO)

    red_state["opponent_elo_sum"] = red_state.get("opponent_elo_sum", 0) + b_elo
    blue_state["opponent_elo_sum"] = blue_state.get("opponent_elo_sum", 0) + r_elo

    red_state["fights"] = red_state.get("fights", 0) + 1
    blue_state["fights"] = blue_state.get("fights", 0) + 1

    if red_won:
        red_state["best_win_elo"] = max(red_state.get("best_win_elo", START_ELO), b_elo)
        blue_state["worst_loss_elo"] = min(blue_state.get("worst_loss_elo", START_ELO), r_elo)
    else:
        blue_state["best_win_elo"] = max(blue_state.get("best_win_elo", START_ELO), r_elo)
        red_state["worst_loss_elo"] = min(red_state.get("worst_loss_elo", START_ELO), b_elo)

    r_actual = 1 if red_won else 0
    b_actual = 1 - r_actual

    r_expected = expected_score(r_elo, b_elo)
    b_expected = expected_score(b_elo, r_elo)

    red_state["elo"] = r_elo + K_FACTOR * (r_actual - r_expected)
    blue_state["elo"] = b_elo + K_FACTOR * (b_actual - b_expected)
