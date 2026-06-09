"""Trace one legacy-vs-new EWM mismatch through a fighter sequence.

This archived diagnostic finds the largest mismatch for a selected EWM diff
feature, identifies the red/blue side responsible, then prints the fighter's
base-state sequence alongside legacy and new EWM values.

Run from repo root:

    python archive/migration_validation/run_trace_ewm_sequence.py

Optionally set FEATURE_NAME, for example:

    FEATURE_NAME=ewm_elo_diff python archive/migration_validation/run_trace_ewm_sequence.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


LEGACY_ROLLING_PATH = Path("data/features/UFC_enhanced_rolling_features_EWM.parquet")
NEW_MONEYLINE_VIEW_PATH = Path("data/features/moneyline_feature_view.parquet")
DEFAULT_FEATURE_NAME = "ewm_avg_opponent_elo_diff"


def _feature_parts(feature_name: str) -> tuple[str, str, str, str, str]:
    """Return stat and side-specific column names for an ewm diff feature."""

    if not feature_name.startswith("ewm_") or not feature_name.endswith("_diff"):
        raise ValueError(f"Expected ewm_*_diff feature, got: {feature_name}")

    stat = feature_name.replace("ewm_", "", 1).replace("_diff", "", 1)
    return (
        stat,
        f"r_pre_{stat}",
        f"b_pre_{stat}",
        f"r_ewm_{stat}",
        f"b_ewm_{stat}",
    )


def _side_sequence(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    fighter_id: str,
    side: str,
    stat: str,
) -> pd.DataFrame:
    """Return all fights for one fighter/side with base and EWM values."""

    id_col = f"{side}_id"
    name_col = f"{side}_name"
    base_col = f"{side}_pre_{stat}"
    ewm_col = f"{side}_ewm_{stat}"

    old_side = old_df[old_df[id_col] == fighter_id][
        ["fight_id", "date", "event_name", "r_name", "b_name", id_col, name_col, base_col, ewm_col]
    ].copy()
    old_side = old_side.rename(
        columns={
            id_col: "fighter_id",
            name_col: "fighter_name",
            base_col: "base_old",
            ewm_col: "ewm_old",
        }
    )
    old_side["side"] = side

    new_side = new_df[new_df[id_col] == fighter_id][["fight_id", base_col, ewm_col]].copy()
    new_side = new_side.rename(columns={base_col: "base_new", ewm_col: "ewm_new"})

    seq = old_side.merge(new_side, on="fight_id", how="left")
    seq["base_abs_diff"] = (seq["base_old"] - seq["base_new"]).abs()
    seq["ewm_abs_diff"] = (seq["ewm_old"] - seq["ewm_new"]).abs()
    return seq.sort_values(["date", "fight_id"]).reset_index(drop=True)


def main() -> None:
    """Trace the largest EWM mismatch for a selected feature."""

    feature_name = os.getenv("FEATURE_NAME", DEFAULT_FEATURE_NAME)
    stat, r_base, b_base, r_ewm, b_ewm = _feature_parts(feature_name)

    print("=" * 80)
    print("TRACE EWM SEQUENCE")
    print("=" * 80)
    print(f"Legacy path: {LEGACY_ROLLING_PATH}")
    print(f"New path   : {NEW_MONEYLINE_VIEW_PATH}")
    print(f"Feature    : {feature_name}")
    print(f"Stat       : {stat}")

    old_df = pd.read_parquet(LEGACY_ROLLING_PATH)
    new_df = pd.read_parquet(NEW_MONEYLINE_VIEW_PATH)

    required_cols = ["fight_id", "date", "event_name", "r_id", "b_id", "r_name", "b_name", feature_name, r_base, b_base, r_ewm, b_ewm]
    missing_old = [col for col in required_cols if col not in old_df.columns]
    missing_new = [col for col in ["fight_id", feature_name, r_base, b_base, r_ewm, b_ewm] if col not in new_df.columns]
    if missing_old or missing_new:
        raise ValueError(f"Missing columns. old={missing_old}, new={missing_new}")

    merged = old_df[required_cols].merge(
        new_df[["fight_id", feature_name, r_base, b_base, r_ewm, b_ewm]],
        on="fight_id",
        how="inner",
        suffixes=("_old", "_new"),
    )
    merged["diff_abs_delta"] = (merged[f"{feature_name}_old"] - merged[f"{feature_name}_new"]).abs()

    worst = merged.sort_values("diff_abs_delta", ascending=False).iloc[0]
    print("\nWorst diff row:")
    print(worst[["fight_id", "date", "event_name", "r_name", "b_name", f"{feature_name}_old", f"{feature_name}_new", "diff_abs_delta"]].to_string())

    red_delta = abs(worst[f"{r_ewm}_old"] - worst[f"{r_ewm}_new"])
    blue_delta = abs(worst[f"{b_ewm}_old"] - worst[f"{b_ewm}_new"])
    side = "r" if red_delta >= blue_delta else "b"
    fighter_id = worst[f"{side}_id"]
    fighter_name = worst[f"{side}_name"]

    print("\nLargest side contributor:")
    print(f"side        : {side}")
    print(f"fighter_id  : {fighter_id}")
    print(f"fighter_name: {fighter_name}")
    print(f"red_delta   : {red_delta}")
    print(f"blue_delta  : {blue_delta}")

    seq = _side_sequence(old_df, new_df, fighter_id=fighter_id, side=side, stat=stat)
    print("\nFighter sequence:")
    print(
        seq[[
            "fight_id",
            "date",
            "event_name",
            "r_name",
            "b_name",
            "side",
            "base_old",
            "base_new",
            "base_abs_diff",
            "ewm_old",
            "ewm_new",
            "ewm_abs_diff",
        ]].to_string(index=False)
    )

    print("\nSummary:")
    print(f"sequence rows        : {len(seq)}")
    print(f"base max abs diff    : {seq['base_abs_diff'].max()}")
    print(f"ewm max abs diff     : {seq['ewm_abs_diff'].max()}")
    print(f"ewm nonzero rows     : {(seq['ewm_abs_diff'] > 1e-9).sum()}")
    print("DONE")


if __name__ == "__main__":
    main()
