"""Experimental join helpers for Round Fighter State features.

This module does not replace the production moneyline feature view.
It only provides reusable join functions so we can test RFS features safely.

Historical/training join:
    base rows with fight_id + r_id + b_id
    joined to data/features/round_fighter_state_history.parquet

Live/latest join:
    base rows with r_id + b_id, or configured fighter-id aliases
    joined to data/features/round_latest_fighter_state.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.common.paths import (
    ROUND_FIGHTER_STATE_HISTORY_PATH,
    ROUND_LATEST_FIGHTER_STATE_PATH,
)


RFS_METADATA_COLUMNS = {
    "event_id",
    "fight_id",
    "fighter_id",
    "opponent_id",
    "fighter_name",
    "opponent_name",
    "event_name",
    "date",
    "corner",
}


class RoundFighterStateJoinError(RuntimeError):
    """Raised when RFS features cannot be joined safely."""


def _read_parquet(path: str | Path, artifact_name: str) -> pd.DataFrame:
    """Read a parquet artifact with a clear error message."""
    path = Path(path)
    if not path.exists():
        raise RoundFighterStateJoinError(
            f"{artifact_name} artifact not found: {path}"
        )
    return pd.read_parquet(path)


def _rfs_feature_columns(
    rfs_df: pd.DataFrame,
    include_fight_observations: bool = False,
) -> list[str]:
    """Return model-safe RFS feature columns.

    By default, exclude rfs_traj_fight_* columns because they describe the
    current fight's round trajectory and are not available before prediction.
    """
    columns = [
        column
        for column in rfs_df.columns
        if column.startswith("rfs_traj_")
        and column not in RFS_METADATA_COLUMNS
    ]

    if include_fight_observations:
        return columns

    return [
        column
        for column in columns
        if not column.startswith("rfs_traj_fight_")
    ]


def _prefix_rfs_columns(
    rfs_df: pd.DataFrame,
    prefix: str,
    key_columns: list[str],
    include_fight_observations: bool = False,
) -> pd.DataFrame:
    """Prefix RFS feature columns while preserving join keys."""
    missing = [column for column in key_columns if column not in rfs_df.columns]
    if missing:
        raise RoundFighterStateJoinError(
            f"RFS artifact missing required join keys: {missing}"
        )

    feature_columns = _rfs_feature_columns(
        rfs_df,
        include_fight_observations=include_fight_observations,
    )
    keep_columns = key_columns + feature_columns

    out = rfs_df[keep_columns].copy()
    out = out.rename(
        columns={
            column: f"{prefix}{column}"
            for column in feature_columns
        }
    )

    return out


def add_rfs_diffs(
    feature_df: pd.DataFrame,
    red_prefix: str = "r_",
    blue_prefix: str = "b_",
    diff_prefix: str = "rfs_diff_",
) -> pd.DataFrame:
    """Add red-minus-blue diffs for matching numeric RFS feature columns."""
    out = feature_df.copy()

    red_cols = [
        column for column in out.columns
        if column.startswith(f"{red_prefix}rfs_traj_")
    ]

    new_columns: dict[str, pd.Series] = {}

    for red_col in red_cols:
        base = red_col.removeprefix(red_prefix)
        blue_col = f"{blue_prefix}{base}"

        if blue_col not in out.columns:
            continue

        red_values = pd.to_numeric(out[red_col], errors="coerce")
        blue_values = pd.to_numeric(out[blue_col], errors="coerce")

        diff_col = f"{diff_prefix}{base.removeprefix('rfs_traj_')}"
        new_columns[diff_col] = red_values - blue_values

    if new_columns:
        out = pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1)

    return out


def drop_side_specific_rfs_columns(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Drop red/blue RFS columns so model inputs use matchup diffs only."""
    side_columns = [
        column for column in feature_df.columns
        if column.startswith(("r_rfs_traj_", "b_rfs_traj_"))
    ]

    if not side_columns:
        return feature_df

    return feature_df.drop(columns=side_columns)


def join_round_fighter_state_history(
    base_df: pd.DataFrame,
    history_path: str | Path = ROUND_FIGHTER_STATE_HISTORY_PATH,
    red_fighter_id_col: str = "r_id",
    blue_fighter_id_col: str = "b_id",
    fight_id_col: str = "fight_id",
    add_diffs: bool = True,
    include_fight_observations: bool = False,
    keep_side_features: bool = False,
) -> pd.DataFrame:
    """Join point-in-time RFS history onto historical/training rows.

    Required base columns by default:
    - fight_id
    - r_id
    - b_id

    The history artifact is keyed by fight_id + fighter_id.

    By default, this returns model-safe matchup-diff RFS columns only:
    - no rfs_traj_fight_* current-fight observations
    - no r_rfs_traj_* or b_rfs_traj_* side-specific columns
    """
    required_base = [fight_id_col, red_fighter_id_col, blue_fighter_id_col]
    missing_base = [column for column in required_base if column not in base_df.columns]
    if missing_base:
        raise RoundFighterStateJoinError(
            f"Base dataframe missing required columns: {missing_base}"
        )

    history = _read_parquet(history_path, "Round Fighter State history")

    red_rfs = _prefix_rfs_columns(
        history,
        prefix="r_",
        key_columns=["fight_id", "fighter_id"],
        include_fight_observations=include_fight_observations,
    ).rename(columns={"fighter_id": red_fighter_id_col})

    blue_rfs = _prefix_rfs_columns(
        history,
        prefix="b_",
        key_columns=["fight_id", "fighter_id"],
        include_fight_observations=include_fight_observations,
    ).rename(columns={"fighter_id": blue_fighter_id_col})

    out = base_df.copy()

    out = out.merge(
        red_rfs,
        left_on=[fight_id_col, red_fighter_id_col],
        right_on=["fight_id", red_fighter_id_col],
        how="left",
        validate="many_to_one",
    )

    out = out.merge(
        blue_rfs,
        left_on=[fight_id_col, blue_fighter_id_col],
        right_on=["fight_id", blue_fighter_id_col],
        how="left",
        validate="many_to_one",
        suffixes=("", "_blue_rfs_duplicate"),
    )

    duplicate_join_cols = [
        column for column in out.columns
        if column.endswith("_blue_rfs_duplicate")
    ]
    if duplicate_join_cols:
        out = out.drop(columns=duplicate_join_cols)

    if add_diffs:
        out = add_rfs_diffs(out)

    if not keep_side_features:
        out = drop_side_specific_rfs_columns(out)

    return out


def join_round_latest_fighter_state(
    base_df: pd.DataFrame,
    latest_path: str | Path = ROUND_LATEST_FIGHTER_STATE_PATH,
    red_fighter_id_col: str = "r_id",
    blue_fighter_id_col: str = "b_id",
    add_diffs: bool = True,
    keep_side_features: bool = False,
) -> pd.DataFrame:
    """Join current/latest RFS state onto future/live rows.

    Required base columns by default:
    - r_id
    - b_id

    The latest artifact is keyed by fighter_id only and contains no
    current-fight observation columns.
    """
    required_base = [red_fighter_id_col, blue_fighter_id_col]
    missing_base = [column for column in required_base if column not in base_df.columns]
    if missing_base:
        raise RoundFighterStateJoinError(
            f"Base dataframe missing required columns: {missing_base}"
        )

    latest = _read_parquet(latest_path, "Round latest fighter state")

    current_fight_cols = [
        column for column in latest.columns
        if column.startswith("rfs_traj_fight_")
    ]
    if current_fight_cols:
        raise RoundFighterStateJoinError(
            "Latest RFS artifact must not contain fight-observation columns: "
            f"{current_fight_cols}"
        )

    red_rfs = _prefix_rfs_columns(
        latest,
        prefix="r_",
        key_columns=["fighter_id"],
    ).rename(columns={"fighter_id": red_fighter_id_col})

    blue_rfs = _prefix_rfs_columns(
        latest,
        prefix="b_",
        key_columns=["fighter_id"],
    ).rename(columns={"fighter_id": blue_fighter_id_col})

    out = base_df.copy()
    out = out.merge(
        red_rfs,
        on=red_fighter_id_col,
        how="left",
        validate="many_to_one",
    )
    out = out.merge(
        blue_rfs,
        on=blue_fighter_id_col,
        how="left",
        validate="many_to_one",
    )

    if add_diffs:
        out = add_rfs_diffs(out)

    if not keep_side_features:
        out = drop_side_specific_rfs_columns(out)

    return out


def summarize_rfs_join(feature_df: pd.DataFrame) -> dict[str, int | float]:
    """Return basic completeness summary for joined RFS columns."""
    rfs_cols = [
        column for column in feature_df.columns
        if column.startswith(("r_rfs_traj_", "b_rfs_traj_", "rfs_diff_"))
    ]

    if not rfs_cols:
        return {
            "rfs_column_count": 0,
            "rfs_non_null_cells": 0,
            "rfs_total_cells": 0,
            "rfs_completeness": 0.0,
        }

    non_null = int(feature_df[rfs_cols].notna().sum().sum())
    total = int(feature_df[rfs_cols].size)

    return {
        "rfs_column_count": len(rfs_cols),
        "rfs_non_null_cells": non_null,
        "rfs_total_cells": total,
        "rfs_completeness": non_null / total if total else 0.0,
    }
