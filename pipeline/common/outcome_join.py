# ============================================================
# pipeline/common/outcome_join.py
# ============================================================

"""Shared outcome join-key helpers.

The permanent outcome join contract is:

    fight_id + market_key + outcome_join_key

This supports both fighter-specific outcomes, such as moneyline, and fight-level
props, such as goes distance, method, or round totals.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _is_missing(value: Any) -> bool:
    """Return True when a scalar value should be treated as missing."""

    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_outcome_token(value: Any) -> str:
    """Normalize labels/sides into stable lowercase join-key tokens."""

    if _is_missing(value):
        return "unknown"

    token = str(value).strip().lower()
    token = token.replace(" ", "_").replace("-", "_").replace("/", "_")
    token = token.replace(".", "_")

    while "__" in token:
        token = token.replace("__", "_")

    return token.strip("_") or "unknown"


def build_outcome_join_key(
    *,
    market_key: Any,
    outcome_label: Any = None,
    outcome_fighter_id: Any = None,
    outcome_key: Any = None,
    side: Any = None,
    line: Any = None,
) -> str:
    """Build the permanent sportsbook/model outcome join key.

    Fighter-specific outcomes use the fighter ID. Fight-level props use a stable
    outcome token so both sides of the same fight-level market remain distinct.
    """

    market = normalize_outcome_token(market_key)

    if not _is_missing(outcome_fighter_id) and str(outcome_fighter_id).strip() not in {"", "nan", "None", "<NA>"}:
        return f"fighter:{str(outcome_fighter_id).strip()}"

    label_token = normalize_outcome_token(outcome_label)
    outcome_token = normalize_outcome_token(outcome_key)
    side_token = normalize_outcome_token(side)

    if market == "goes_distance":
        if label_token in {"goes_distance", "yes", "over"} or side_token == "yes":
            return "fight:goes_distance"
        if label_token in {"inside_distance", "does_not_go_distance", "no", "under"} or side_token == "no":
            return "fight:inside_distance"

    if market == "over_under_2_5":
        if label_token in {"over_2_5", "over", "yes"} or side_token == "over":
            return "fight:over_2_5"
        if label_token in {"under_2_5", "under", "no"} or side_token == "under":
            return "fight:under_2_5"

    if market == "method_of_victory":
        if label_token in {"ko_tko", "ko_tko_dq", "ko", "tko", "knockout"} or side_token in {"ko_tko", "ko", "tko"}:
            return "fight:ko_tko"
        if label_token in {"submission", "sub"} or side_token in {"submission", "sub"}:
            return "fight:submission"
        if label_token in {"decision", "dec"} or side_token in {"decision", "dec"}:
            return "fight:decision"

    if market in {"total_rounds", "totals"}:
        base = side_token if side_token in {"over", "under"} else label_token
        if not _is_missing(line):
            return f"fight:{base}_{normalize_outcome_token(line)}"
        return f"fight:{base}"

    if outcome_token != "unknown":
        return f"fight:{outcome_token}"

    return f"fight:{label_token}"
