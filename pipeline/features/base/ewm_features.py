"""Exponentially weighted moving-average feature generation.

Notebook source sections:
- ADD EXPONENTIALLY WEIGHTED RECENT-FORM FEATURES
- CREATE LONG FIGHTER-LEVEL DATASET
- EWM merge-back logic
- EWM differential features
- Recent-form edge features

Migration status:
- Migrated EWM feature helpers from UFC_rolling_dataset_V4_refactored.ipynb.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EWM_SPAN = 3


def get_ewm_stat_names(rolling_df: pd.DataFrame) -> list[str]:
    """Return stat names that have matching r_pre_* and b_pre_* columns."""
    r_pre_cols = [col for col in rolling_df.columns if col.startswith("r_pre_")]
    return [
        col.replace("r_pre_", "")
        for col in r_pre_cols
        if f"b_pre_{col.replace('r_pre_', '')}" in rolling_df.columns
    ]


def create_fighter_long_df(rolling_df: pd.DataFrame, stat_names: list[str]) -> pd.DataFrame:
    """Convert fight-level rolling rows into fighter-level rows for EWM calculations."""
    fighter_rows: list[dict[str, object]] = []

    for idx, row in rolling_df.iterrows():
        red_row: dict[str, object] = {
            "fight_index": idx,
            "date": row["date"],
            "fighter_id": row["r_id"],
            "corner": "r",
        }
        blue_row: dict[str, object] = {
            "fight_index": idx,
            "date": row["date"],
            "fighter_id": row["b_id"],
            "corner": "b",
        }

        for stat in stat_names:
            red_row[stat] = row.get(f"r_pre_{stat}", np.nan)
            blue_row[stat] = row.get(f"b_pre_{stat}", np.nan)

        fighter_rows.append(red_row)
        fighter_rows.append(blue_row)

    fighter_long_df = pd.DataFrame(fighter_rows)
    return fighter_long_df.sort_values(["fighter_id", "date"]).reset_index(drop=True)


def add_fighter_ewm_columns(
    fighter_long_df: pd.DataFrame,
    stat_names: list[str],
    span: int = EWM_SPAN,
) -> pd.DataFrame:
    """Add ewm_* columns to the fighter-level dataframe."""
    fighter_long_df = fighter_long_df.copy()

    for stat in stat_names:
        fighter_long_df[f"ewm_{stat}"] = (
            fighter_long_df.groupby("fighter_id")[stat]
            .transform(lambda x: x.ewm(span=span, adjust=False).mean())
        )

    return fighter_long_df


def merge_ewm_features(
    rolling_df: pd.DataFrame,
    fighter_long_df: pd.DataFrame,
    stat_names: list[str],
) -> pd.DataFrame:
    """Merge r_ewm_* and b_ewm_* columns back into fight-level rolling rows."""
    rolling_df = rolling_df.copy()

    r_ewm = fighter_long_df[fighter_long_df["corner"] == "r"].copy()
    b_ewm = fighter_long_df[fighter_long_df["corner"] == "b"].copy()

    ewm_cols = ["fight_index"] + [f"ewm_{stat}" for stat in stat_names]

    r_ewm = r_ewm[ewm_cols].rename(
        columns={f"ewm_{stat}": f"r_ewm_{stat}" for stat in stat_names}
    )
    b_ewm = b_ewm[ewm_cols].rename(
        columns={f"ewm_{stat}": f"b_ewm_{stat}" for stat in stat_names}
    )

    rolling_df = rolling_df.merge(
        r_ewm,
        left_index=True,
        right_on="fight_index",
        how="left",
    ).drop(columns=["fight_index"])

    rolling_df = rolling_df.merge(
        b_ewm,
        left_index=True,
        right_on="fight_index",
        how="left",
    ).drop(columns=["fight_index"])

    return rolling_df


def add_ewm_diff_features(rolling_df: pd.DataFrame, stat_names: list[str]) -> pd.DataFrame:
    """Add ewm_*_diff columns for red-minus-blue EWM features."""
    rolling_df = rolling_df.copy()

    for stat in stat_names:
        r_col = f"r_ewm_{stat}"
        b_col = f"b_ewm_{stat}"
        diff_col = f"ewm_{stat}_diff"

        if r_col in rolling_df.columns and b_col in rolling_df.columns:
            rolling_df[diff_col] = rolling_df[r_col] - rolling_df[b_col]

    return rolling_df


def add_recent_form_features(rolling_df: pd.DataFrame, stat_names: list[str]) -> pd.DataFrame:
    """Add recent-form edge features comparing EWM form to career prefight form."""
    rolling_df = rolling_df.copy()

    for stat in stat_names:
        r_ewm = f"r_ewm_{stat}"
        b_ewm = f"b_ewm_{stat}"
        r_career = f"r_pre_{stat}"
        b_career = f"b_pre_{stat}"

        if r_ewm in rolling_df.columns and r_career in rolling_df.columns:
            rolling_df[f"r_recent_form_{stat}"] = rolling_df[r_ewm] - rolling_df[r_career]

        if b_ewm in rolling_df.columns and b_career in rolling_df.columns:
            rolling_df[f"b_recent_form_{stat}"] = rolling_df[b_ewm] - rolling_df[b_career]

        r_recent = f"r_recent_form_{stat}"
        b_recent = f"b_recent_form_{stat}"
        if r_recent in rolling_df.columns and b_recent in rolling_df.columns:
            rolling_df[f"recent_form_{stat}_diff"] = rolling_df[r_recent] - rolling_df[b_recent]

    return rolling_df


def add_ewm_feature_layer(
    rolling_df: pd.DataFrame,
    span: int = EWM_SPAN,
) -> pd.DataFrame:
    """Apply the full EWM feature layer to the intermediate rolling dataframe."""
    rolling_df = rolling_df.sort_values("date").reset_index(drop=True)
    stat_names = get_ewm_stat_names(rolling_df)
    fighter_long_df = create_fighter_long_df(rolling_df, stat_names)
    fighter_long_df = add_fighter_ewm_columns(fighter_long_df, stat_names, span=span)
    rolling_df = merge_ewm_features(rolling_df, fighter_long_df, stat_names)
    rolling_df = add_ewm_diff_features(rolling_df, stat_names)
    rolling_df = add_recent_form_features(rolling_df, stat_names)
    return rolling_df
