"""Moneyline feature view builder.

This view intentionally emits V5-compatible column names first so the new
fighter-state architecture can be parity-tested against the existing rolling
feature artifact. Future cleanup should patch this file in place instead of
creating duplicate moneyline_v2-style modules.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


STATE_CONTEXT_COLUMNS = {
    "event_id",
    "event_name",
    "fight_id",
    "date",
    "division",
    "title_fight",
    "total_rounds",
    "fight_date",
    "source_row_index",
    "fighter_id",
    "fighter_name",
    "corner",
    "opponent_id",
    "opponent_name",
}
STATE_JOIN_COLUMNS = ["fight_id", "fighter_id"]
UPSET_POTENTIAL_ABS_DIFF_COLUMNS = [
    "win_method_entropy_diff",
    "finish_dependency_diff",
    "decision_dependency_diff",
    "ko_dependency_diff",
    "submission_dependency_diff",
    "style_flexibility_score_diff",
]


def get_state_feature_columns(fighter_state_history_df: pd.DataFrame) -> list[str]:
    """Return fighter-state feature columns, excluding snapshot context."""

    return [
        column
        for column in fighter_state_history_df.columns
        if column not in STATE_CONTEXT_COLUMNS
    ]


def _prefixed_state_frame(
    fighter_state_history_df: pd.DataFrame,
    side_prefix: str,
    state_columns: list[str],
) -> pd.DataFrame:
    """Return fighter-state history keyed for a red or blue side join."""

    if side_prefix not in {"r", "b"}:
        raise ValueError(f"Unsupported side prefix: {side_prefix}")

    join_id_column = f"{side_prefix}_id"
    rename_map = {"fight_id": "state_fight_id", "fighter_id": join_id_column}

    for column in state_columns:
        if column.startswith("ewm_"):
            rename_map[column] = f"{side_prefix}_{column}"
        elif column.startswith("form_delta_"):
            base_name = column.replace("form_delta_", "", 1)
            rename_map[column] = f"{side_prefix}_recent_form_{base_name}"
        else:
            rename_map[column] = f"{side_prefix}_pre_{column}"

    return fighter_state_history_df[["fight_id", "fighter_id", *state_columns]].rename(
        columns=rename_map
    )


def _add_diff_columns(
    view_df: pd.DataFrame,
    state_columns: list[str],
) -> pd.DataFrame:
    """Add red-minus-blue difference columns using V5-compatible names."""

    out = view_df.copy()
    new_columns: dict[str, Any] = {}

    for column in state_columns:
        if column.startswith("ewm_"):
            stat_name = column.replace("ewm_", "", 1)
            r_col = f"r_ewm_{stat_name}"
            b_col = f"b_ewm_{stat_name}"
            diff_col = f"ewm_{stat_name}_diff"
        elif column.startswith("form_delta_"):
            stat_name = column.replace("form_delta_", "", 1)
            r_col = f"r_recent_form_{stat_name}"
            b_col = f"b_recent_form_{stat_name}"
            diff_col = f"recent_form_{stat_name}_diff"
        else:
            r_col = f"r_pre_{column}"
            b_col = f"b_pre_{column}"
            diff_col = f"{column}_diff"

        if r_col in out.columns and b_col in out.columns:
            new_columns[diff_col] = out[r_col] - out[b_col]

    for diff_col in UPSET_POTENTIAL_ABS_DIFF_COLUMNS:
        if diff_col in new_columns:
            abs_col = diff_col.replace("_diff", "_abs_diff")
            new_columns[abs_col] = new_columns[diff_col].abs()

    if new_columns:
        out = pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1)

    return out


def build_moneyline_feature_view(
    prepared_fights_df: pd.DataFrame,
    fighter_state_history_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a V5-compatible moneyline feature view from fighter-state history."""

    required_fight_columns = {"fight_id", "r_id", "b_id"}
    missing_fight_columns = required_fight_columns - set(prepared_fights_df.columns)
    if missing_fight_columns:
        raise ValueError(f"Prepared fights missing required columns: {sorted(missing_fight_columns)}")

    missing_state_columns = set(STATE_JOIN_COLUMNS) - set(fighter_state_history_df.columns)
    if missing_state_columns:
        raise ValueError(f"Fighter state history missing required columns: {sorted(missing_state_columns)}")

    state_columns = get_state_feature_columns(fighter_state_history_df)
    red_state = _prefixed_state_frame(fighter_state_history_df, "r", state_columns)
    blue_state = _prefixed_state_frame(fighter_state_history_df, "b", state_columns)

    view_df = prepared_fights_df.copy()
    if "state_fight_id" not in view_df.columns:
        view_df["state_fight_id"] = view_df["fight_id"]

    view_df = view_df.merge(red_state, on=["state_fight_id", "r_id"], how="left")
    view_df = view_df.merge(blue_state, on=["state_fight_id", "b_id"], how="left")
    view_df = _add_diff_columns(view_df, state_columns)

    return view_df
