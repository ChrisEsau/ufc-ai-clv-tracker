"""Victory-path profile raw fighter feature plugin.

Builds point-in-time fighter-level win-method concentration, finish dependency,
and specialist-path features. These features are calculated strictly before the
current fight and are intended to support upset-potential and path-to-victory
modeling.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from pipeline.features.base.elo import safe_div
from pipeline.features.base.fighter_state import KO_METHODS

OUTPUT_COLUMNS = [
    "win_method_entropy",
    "victory_concentration_index",
    "finish_dependency",
    "ko_dependency",
    "submission_dependency",
    "decision_dependency",
    "ko_win_concentration",
    "submission_win_concentration",
    "style_flexibility_score",
]


def initial_state() -> dict[str, Any]:
    """Return victory-path initial state for one fighter."""

    return {
        "wins": 0,
        "ko_wins": 0,
        "sub_wins": 0,
        "decision_wins": 0,
        "finish_wins": 0,
    }


def _entropy(probabilities: list[float]) -> float:
    """Return Shannon entropy for nonzero method probabilities."""

    return -sum(p * math.log(p) for p in probabilities if p > 0)


def calculate(
    fighter_history: pd.DataFrame,
    fight_row: pd.Series,
    context: dict | None = None,
) -> dict[str, float]:
    """Return point-in-time victory-path features for one fighter."""

    del fighter_history, fight_row
    context = context or {}
    state = context.get("state", initial_state())

    wins = float(state.get("wins", 0) or 0)
    ko_wins = float(state.get("ko_wins", 0) or 0)
    sub_wins = float(state.get("sub_wins", 0) or 0)
    decision_wins = float(state.get("decision_wins", 0) or 0)
    finish_wins = float(state.get("finish_wins", 0) or 0)

    ko_dependency = safe_div(ko_wins, wins)
    submission_dependency = safe_div(sub_wins, wins)
    decision_dependency = safe_div(decision_wins, wins)
    finish_dependency = safe_div(finish_wins, wins)

    method_entropy = _entropy([ko_dependency, submission_dependency, decision_dependency])
    max_entropy = math.log(3.0)
    normalized_entropy = safe_div(method_entropy, max_entropy)
    victory_concentration_index = 1.0 - normalized_entropy if wins > 0 else 0.0

    return {
        "win_method_entropy": method_entropy,
        "victory_concentration_index": victory_concentration_index,
        "finish_dependency": finish_dependency,
        "ko_dependency": ko_dependency,
        "submission_dependency": submission_dependency,
        "decision_dependency": decision_dependency,
        "ko_win_concentration": safe_div(ko_wins, finish_wins),
        "submission_win_concentration": safe_div(sub_wins, finish_wins),
        "style_flexibility_score": normalized_entropy,
    }


def update_after_fight(
    *,
    state: dict[str, Any],
    method: Any,
    won: bool,
) -> None:
    """Update victory-path state after a completed fight."""

    if not won:
        return

    method_text = str(method)
    state["wins"] = state.get("wins", 0) + 1

    if method_text in KO_METHODS:
        state["ko_wins"] = state.get("ko_wins", 0) + 1
        state["finish_wins"] = state.get("finish_wins", 0) + 1
    elif method_text == "Submission":
        state["sub_wins"] = state.get("sub_wins", 0) + 1
        state["finish_wins"] = state.get("finish_wins", 0) + 1
    elif "Decision" in method_text:
        state["decision_wins"] = state.get("decision_wins", 0) + 1
