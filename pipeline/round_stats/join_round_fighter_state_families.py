"""Join helpers for full-family Round Fighter State feature views.

This module is intentionally experimental. It does not replace the production
moneyline feature view or production prediction path.

It joins point-in-time RFS family history artifacts onto fight-level feature
views using:

    fight_id + red fighter id
    fight_id + blue fighter id

The model-safe default excludes all current-fight observation columns such as:

    rfs_traj_fight_*
    rfs_suppress_fight_*
    rfs_wrestle_fight_*
    rfs_def_fight_*

The output can keep side-specific columns for audit, but the default model view
keeps matchup diffs plus family availability flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from pipeline.common.paths import (
    ROUND_FIGHTER_STATE_HISTORY_PATH,
    ROUND_FIGHTER_SUPPRESSION_P0_2_HISTORY_PATH,
    ROUND_FIGHTER_WRESTLING_P0_3_HISTORY_PATH,
    ROUND_FIGHTER_DEFENSE_P1_4_HISTORY_PATH,
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


@dataclass(frozen=True)
class RfsFamilyConfig:
    """Configuration for one RFS family history artifact."""

    family_key: str
    feature_prefix: str
    has_state_column: str
    history_path: Path


DEFAULT_RFS_FAMILY_CONFIGS: tuple[RfsFamilyConfig, ...] = (
    RfsFamilyConfig(
        family_key="traj",
        feature_prefix="rfs_traj_",
        has_state_column="rfs_traj_has_state",
        history_path=ROUND_FIGHTER_STATE_HISTORY_PATH,
    ),
    RfsFamilyConfig(
        family_key="open",
        feature_prefix="rfs_open_",
        has_state_column="rfs_open_has_state",
        history_path=ROUND_FIGHTER_STATE_HISTORY_PATH,
    ),
    RfsFamilyConfig(
        family_key="suppress",
        feature_prefix="rfs_suppress_",
        has_state_column="rfs_suppress_has_state",
        history_path=ROUND_FIGHTER_SUPPRESSION_P0_2_HISTORY_PATH,
    ),
    RfsFamilyConfig(
        family_key="wrestle",
        feature_prefix="rfs_wrestle_",
        has_state_column="rfs_wrestle_has_state",
        history_path=ROUND_FIGHTER_WRESTLING_P0_3_HISTORY_PATH,
    ),
    RfsFamilyConfig(
        family_key="def",
        feature_prefix="rfs_def_",
        has_state_column="rfs_def_has_state",
        history_path=ROUND_FIGHTER_DEFENSE_P1_4_HISTORY_PATH,
    ),
)


class RoundFighterStateFamilyJoinError(RuntimeError):
    """Raised when full-family RFS features cannot be joined safely."""


def _read_parquet(path: str | Path, artifact_name: str) -> pd.DataFrame:
    """Read a parquet artifact with a clear error message."""

    path = Path(path)
    if not path.exists():
        raise RoundFighterStateFamilyJoinError(
            f"{artifact_name} artifact not found: {path}"
        )
    return pd.read_parquet(path)


def _validate_history_grain(
    history_df: pd.DataFrame,
    *,
    family_key: str,
) -> None:
    """Validate one row per fight_id + fighter_id for a family artifact."""

    required = {"fight_id", "fighter_id"}
    missing = required - set(history_df.columns)
    if missing:
        raise RoundFighterStateFamilyJoinError(
            f"RFS family '{family_key}' missing required keys: {sorted(missing)}"
        )

    duplicate_count = int(history_df.duplicated(["fight_id", "fighter_id"]).sum())
    if duplicate_count:
        raise RoundFighterStateFamilyJoinError(
            f"RFS family '{family_key}' has duplicate fight_id/fighter_id rows: "
            f"{duplicate_count}"
        )


def _family_feature_columns(
    history_df: pd.DataFrame,
    *,
    family: RfsFamilyConfig,
    include_fight_observations: bool = False,
) -> list[str]:
    """Return model-safe columns for one RFS family."""

    columns = [
        column
        for column in history_df.columns
        if column.startswith(family.feature_prefix)
        and column not in RFS_METADATA_COLUMNS
    ]

    if include_fight_observations:
        return columns

    # Current-fight observations describe the target fight itself and are not
    # model-safe for pre-fight prediction.
    return [column for column in columns if "_fight_" not in column]


def _prefixed_family_frame(
    history_df: pd.DataFrame,
    *,
    family: RfsFamilyConfig,
    side_prefix: str,
    key_columns: list[str],
    include_fight_observations: bool = False,
) -> pd.DataFrame:
    """Prefix one RFS family for red or blue side joins."""

    missing = [column for column in key_columns if column not in history_df.columns]
    if missing:
        raise RoundFighterStateFamilyJoinError(
            f"RFS family '{family.family_key}' missing join keys: {missing}"
        )

    feature_columns = _family_feature_columns(
        history_df,
        family=family,
        include_fight_observations=include_fight_observations,
    )

    keep_columns = [*key_columns, *feature_columns]
    out = history_df[keep_columns].copy()

    rename_map = {
        column: f"{side_prefix}{column}"
        for column in feature_columns
    }

    return out.rename(columns=rename_map)


def _add_family_availability_flags(
    feature_df: pd.DataFrame,
    *,
    family_configs: Iterable[RfsFamilyConfig],
    red_prefix: str = "r_",
    blue_prefix: str = "b_",
) -> pd.DataFrame:
    """Add RFS family availability flags before dropping side-specific features."""

    out = feature_df.copy()
    new_columns: dict[str, pd.Series] = {}

    for family in family_configs:
        red_col = f"{red_prefix}{family.has_state_column}"
        blue_col = f"{blue_prefix}{family.has_state_column}"

        red_has = (
            pd.to_numeric(out[red_col], errors="coerce").fillna(0).astype(int)
            if red_col in out.columns
            else pd.Series(0, index=out.index)
        )
        blue_has = (
            pd.to_numeric(out[blue_col], errors="coerce").fillna(0).astype(int)
            if blue_col in out.columns
            else pd.Series(0, index=out.index)
        )

        prefix = f"rfs_{family.family_key}"
        new_columns[f"{prefix}_r_has_state"] = red_has
        new_columns[f"{prefix}_b_has_state"] = blue_has
        new_columns[f"{prefix}_both_have_state"] = ((red_has == 1) & (blue_has == 1)).astype(int)
        new_columns[f"{prefix}_either_has_state"] = ((red_has == 1) | (blue_has == 1)).astype(int)

    if new_columns:
        out = pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1)

    return out


def add_rfs_family_diffs(
    feature_df: pd.DataFrame,
    *,
    family_configs: Iterable[RfsFamilyConfig] = DEFAULT_RFS_FAMILY_CONFIGS,
    red_prefix: str = "r_",
    blue_prefix: str = "b_",
) -> pd.DataFrame:
    """Add red-minus-blue RFS diffs for all matching numeric family columns.

    Diff naming follows the RFS docs:

        r_rfs_traj_ewm_late_output_ratio
        b_rfs_traj_ewm_late_output_ratio
        rfs_traj_ewm_late_output_ratio_diff
    """

    out = feature_df.copy()
    family_prefixes = tuple(family.feature_prefix for family in family_configs)

    red_cols = [
        column
        for column in out.columns
        if column.startswith(red_prefix)
        and column.removeprefix(red_prefix).startswith(family_prefixes)
    ]

    new_columns: dict[str, pd.Series] = {}

    for red_col in red_cols:
        base = red_col.removeprefix(red_prefix)
        blue_col = f"{blue_prefix}{base}"

        if blue_col not in out.columns:
            continue

        red_values = pd.to_numeric(out[red_col], errors="coerce")
        blue_values = pd.to_numeric(out[blue_col], errors="coerce")

        diff_col = f"{base}_diff"
        new_columns[diff_col] = red_values - blue_values

    if new_columns:
        out = pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1)

    return out


def drop_side_specific_rfs_family_columns(
    feature_df: pd.DataFrame,
    *,
    family_configs: Iterable[RfsFamilyConfig] = DEFAULT_RFS_FAMILY_CONFIGS,
    red_prefix: str = "r_",
    blue_prefix: str = "b_",
) -> pd.DataFrame:
    """Drop red/blue RFS columns so model input can use matchup diffs by default."""

    family_prefixes = tuple(family.feature_prefix for family in family_configs)

    side_columns = [
        column
        for column in feature_df.columns
        if (
            column.startswith(red_prefix)
            and column.removeprefix(red_prefix).startswith(family_prefixes)
        )
        or (
            column.startswith(blue_prefix)
            and column.removeprefix(blue_prefix).startswith(family_prefixes)
        )
    ]

    if not side_columns:
        return feature_df

    return feature_df.drop(columns=side_columns)


def join_round_fighter_state_families_history(
    base_df: pd.DataFrame,
    *,
    family_configs: Iterable[RfsFamilyConfig] = DEFAULT_RFS_FAMILY_CONFIGS,
    red_fighter_id_col: str = "r_id",
    blue_fighter_id_col: str = "b_id",
    fight_id_col: str = "fight_id",
    add_diffs: bool = True,
    include_fight_observations: bool = False,
    keep_side_features: bool = False,
) -> pd.DataFrame:
    """Join all configured point-in-time RFS family histories onto fight rows."""

    required_base = {fight_id_col, red_fighter_id_col, blue_fighter_id_col}
    missing_base = required_base - set(base_df.columns)
    if missing_base:
        raise RoundFighterStateFamilyJoinError(
            f"Base dataframe missing required columns: {sorted(missing_base)}"
        )

    configs = tuple(family_configs)
    out = base_df.copy()

    for family in configs:
        history = _read_parquet(
            family.history_path,
            f"Round Fighter State family '{family.family_key}' history",
        )
        _validate_history_grain(history, family_key=family.family_key)

        red_rfs = _prefixed_family_frame(
            history,
            family=family,
            side_prefix="r_",
            key_columns=["fight_id", "fighter_id"],
            include_fight_observations=include_fight_observations,
        ).rename(columns={"fighter_id": red_fighter_id_col})

        blue_rfs = _prefixed_family_frame(
            history,
            family=family,
            side_prefix="b_",
            key_columns=["fight_id", "fighter_id"],
            include_fight_observations=include_fight_observations,
        ).rename(columns={"fighter_id": blue_fighter_id_col})

        out = out.merge(
            red_rfs,
            left_on=[fight_id_col, red_fighter_id_col],
            right_on=["fight_id", red_fighter_id_col],
            how="left",
            validate="one_to_one",
            suffixes=("", f"_{family.family_key}_red_duplicate"),
        )

        duplicate_cols = [
            column
            for column in out.columns
            if column.endswith(f"_{family.family_key}_red_duplicate")
        ]
        if duplicate_cols:
            out = out.drop(columns=duplicate_cols)

        out = out.merge(
            blue_rfs,
            left_on=[fight_id_col, blue_fighter_id_col],
            right_on=["fight_id", blue_fighter_id_col],
            how="left",
            validate="one_to_one",
            suffixes=("", f"_{family.family_key}_blue_duplicate"),
        )

        duplicate_cols = [
            column
            for column in out.columns
            if column.endswith(f"_{family.family_key}_blue_duplicate")
        ]
        if duplicate_cols:
            out = out.drop(columns=duplicate_cols)

    out = _add_family_availability_flags(out, family_configs=configs)

    if add_diffs:
        out = add_rfs_family_diffs(out, family_configs=configs)

    if not keep_side_features:
        out = drop_side_specific_rfs_family_columns(out, family_configs=configs)

    return out


def summarize_rfs_family_join(
    feature_df: pd.DataFrame,
    *,
    family_configs: Iterable[RfsFamilyConfig] = DEFAULT_RFS_FAMILY_CONFIGS,
) -> dict[str, int | float]:
    """Return completeness summary for full-family joined RFS columns."""

    configs = tuple(family_configs)
    summary: dict[str, int | float] = {}

    all_rfs_cols = [
        column
        for column in feature_df.columns
        if column.startswith("rfs_")
    ]

    summary["rfs_column_count"] = len(all_rfs_cols)
    summary["rfs_non_null_cells"] = int(feature_df[all_rfs_cols].notna().sum().sum()) if all_rfs_cols else 0
    summary["rfs_total_cells"] = int(feature_df[all_rfs_cols].size) if all_rfs_cols else 0
    summary["rfs_completeness"] = (
        summary["rfs_non_null_cells"] / summary["rfs_total_cells"]
        if summary["rfs_total_cells"]
        else 0.0
    )

    for family in configs:
        family_cols = [
            column
            for column in feature_df.columns
            if column.startswith(family.feature_prefix)
            or column.startswith(f"rfs_{family.family_key}_")
        ]
        summary[f"{family.family_key}_column_count"] = len(family_cols)

        both_col = f"rfs_{family.family_key}_both_have_state"
        either_col = f"rfs_{family.family_key}_either_has_state"

        summary[f"{family.family_key}_both_side_rows"] = (
            int(pd.to_numeric(feature_df[both_col], errors="coerce").fillna(0).sum())
            if both_col in feature_df.columns
            else 0
        )
        summary[f"{family.family_key}_either_side_rows"] = (
            int(pd.to_numeric(feature_df[either_col], errors="coerce").fillna(0).sum())
            if either_col in feature_df.columns
            else 0
        )

    return summary
