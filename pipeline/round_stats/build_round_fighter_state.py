"""Build Round Fighter State artifacts from UFCStats round-level data.

P0.1 scope:
- Load data/fight_details/ufc_round_stats.parquet
- Standardize round-stats schema
- Normalize event_date/date into date
- Normalize ctrl_sec/control time into control_seconds
- Build standalone Round Fighter State artifact shell
- Do not touch production fighter-state artifacts
- Do not touch prediction or model feature views yet

Run from repo root:

    python -m pipeline.round_stats.build_round_fighter_state
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from pipeline.common.paths import (
    MASTER_PATH,
    ROUND_FIGHTER_STATE_HISTORY_PATH,
    ROUND_LATEST_FIGHTER_STATE_PATH,
    ROUND_STATS_PATH,
    ensure_data_dirs,
)


REQUIRED_ID_COLUMNS = [
    "event_id",
    "fight_id",
    "fighter_id",
    "opponent_id",
    "fighter_name",
    "opponent_name",
    "event_name",
    "round",
    "corner",
]

REQUIRED_NUMERIC_COLUMNS = [
    "sig_str_landed",
    "sig_str_attempted",
    "total_str_landed",
    "total_str_attempted",
    "td_landed",
    "td_attempted",
    "control_seconds",
    "kd",
]

DATE_CANDIDATES = [
    "date",
    "event_date",
    "fight_date",
]

CONTROL_SECONDS_CANDIDATES = [
    "control_seconds",
    "ctrl_seconds",
    "ctrl_sec",
    "control_time_seconds",
    "control_time_sec",
    "ctrl",
    "control",
    "control_time",
]


class RoundFighterStateBuildError(RuntimeError):
    """Raised when Round Fighter State artifacts cannot be built."""


@dataclass(frozen=True)
class RoundFighterStateBuildResult:
    """Container for Round Fighter State build outputs."""

    history_df: pd.DataFrame
    latest_df: pd.DataFrame


def read_round_stats(round_stats_path: str | Path = ROUND_STATS_PATH) -> pd.DataFrame:
    """Read the round-level UFCStats parquet file."""
    path = Path(round_stats_path)

    if not path.exists():
        raise RoundFighterStateBuildError(
            f"Round stats input not found: {path}. "
            "Run the historical round-stats backfill before building Round Fighter State."
        )

    return pd.read_parquet(path)


def _parse_time_to_seconds(value: object) -> float | None:
    """Parse numeric seconds or common M:SS/MM:SS control-time strings."""
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)

    match = re.fullmatch(r"(\d+):([0-5]?\d)", text)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        return float(minutes * 60 + seconds)

    return None


def _standardize_date(round_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a normalized date column exists."""
    out = round_stats_df.copy()

    source_column = None
    for candidate in DATE_CANDIDATES:
        if candidate in out.columns:
            source_column = candidate
            break

    if source_column is not None:
        out["date"] = pd.to_datetime(out[source_column], errors="coerce")
        if source_column != "date":
            print(f"Standardized date from column: {source_column}")
        return out

    # Fallback only if no usable date column exists in round stats.
    path = Path(MASTER_PATH)
    if not path.exists():
        raise RoundFighterStateBuildError(
            "Round stats input is missing date/event_date, and master parquet was not found: "
            f"{path}"
        )

    master = pd.read_parquet(path)
    required_master_columns = ["fight_id", "date"]
    missing_master = [column for column in required_master_columns if column not in master.columns]
    if missing_master:
        raise RoundFighterStateBuildError(
            "Cannot join date from master because master is missing columns: "
            f"{missing_master}"
        )

    join_keys = ["fight_id"]
    if "event_id" in master.columns and "event_id" in out.columns:
        join_keys = ["event_id", "fight_id"]

    date_lookup = master[[*join_keys, "date"]].drop_duplicates(subset=join_keys).copy()
    date_lookup["date"] = pd.to_datetime(date_lookup["date"], errors="coerce")
    out = out.merge(date_lookup, on=join_keys, how="left")

    missing_date_count = int(out["date"].isna().sum())
    if missing_date_count:
        raise RoundFighterStateBuildError(
            "Date join from master left missing dates in round stats. "
            f"Missing rows: {missing_date_count}. Join keys: {join_keys}"
        )

    print(f"Joined date from master using keys: {join_keys}")
    return out


def _standardize_control_seconds(round_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a numeric control_seconds column exists."""
    out = round_stats_df.copy()

    source_column = None
    for candidate in CONTROL_SECONDS_CANDIDATES:
        if candidate in out.columns:
            source_column = candidate
            break

    if source_column is None:
        raise RoundFighterStateBuildError(
            "Round stats input is missing control time. "
            f"Checked candidates: {CONTROL_SECONDS_CANDIDATES}"
        )

    out["control_seconds"] = out[source_column].map(_parse_time_to_seconds)

    if source_column != "control_seconds":
        print(f"Standardized control_seconds from column: {source_column}")

    return out


def standardize_round_stats_input(round_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize round-stats columns required by P0.1."""
    out = round_stats_df.copy()
    out = _standardize_date(out)
    out = _standardize_control_seconds(out)

    out["round"] = pd.to_numeric(out["round"], errors="coerce")
    out["corner"] = out["corner"].astype("string").str.strip().str.lower()

    for column in REQUIRED_NUMERIC_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    return out


def validate_round_stats_input(round_stats_df: pd.DataFrame) -> None:
    """Validate minimum columns required for P0.1."""
    required_columns = REQUIRED_ID_COLUMNS + ["date"] + REQUIRED_NUMERIC_COLUMNS
    missing = [column for column in required_columns if column not in round_stats_df.columns]

    if missing:
        raise RoundFighterStateBuildError(
            "Round stats input is missing required P0.1 columns after standardization: "
            f"{missing}"
        )

    key_columns = ["fight_id", "fighter_id", "round", "corner"]
    duplicate_count = int(round_stats_df.duplicated(subset=key_columns).sum())
    if duplicate_count:
        raise RoundFighterStateBuildError(
            "Round stats input has duplicate fighter-round keys. "
            f"Duplicate count: {duplicate_count}"
        )

    invalid_corner_mask = ~round_stats_df["corner"].isin(["red", "blue"])
    bad_corner_count = int(invalid_corner_mask.sum())
    if bad_corner_count:
        bad_values = sorted(round_stats_df.loc[invalid_corner_mask, "corner"].dropna().unique().tolist())
        raise RoundFighterStateBuildError(
            "Round stats input has invalid corner values. "
            f"Invalid rows: {bad_corner_count}. Values: {bad_values}"
        )

    missing_date_count = int(round_stats_df["date"].isna().sum())
    if missing_date_count:
        raise RoundFighterStateBuildError(
            f"Round stats input has missing date values after standardization: {missing_date_count}"
        )


def build_round_fighter_state_history(round_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Build the P0.1 Round Fighter State history shell.

    Full trajectory feature calculations will be added after the artifact shell
    and input contract are verified.
    """
    standardized_df = standardize_round_stats_input(round_stats_df)
    validate_round_stats_input(standardized_df)

    out = standardized_df[
        [
            "event_id",
            "fight_id",
            "fighter_id",
            "opponent_id",
            "fighter_name",
            "opponent_name",
            "event_name",
            "date",
            "corner",
        ]
    ].drop_duplicates(subset=["fight_id", "fighter_id"]).copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)

    # P0.1 shell metadata.
    # Actual trajectory state features will be added in the next step.
    out["rfs_traj_prior_fight_count"] = out.groupby("fighter_id").cumcount()
    out["rfs_traj_prior_valid_trajectory_count"] = 0
    out["rfs_traj_has_state"] = 0

    return out


def build_latest_round_fighter_state(history_df: pd.DataFrame) -> pd.DataFrame:
    """Return the latest Round Fighter State row for each fighter."""
    if history_df.empty:
        return history_df.copy()

    return (
        history_df
        .sort_values(["fighter_id", "date", "fight_id"])
        .groupby("fighter_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def build_round_fighter_state(
    round_stats_path: str | Path = ROUND_STATS_PATH,
) -> RoundFighterStateBuildResult:
    """Build history and latest Round Fighter State artifacts."""
    round_stats_df = read_round_stats(round_stats_path)
    history_df = build_round_fighter_state_history(round_stats_df)
    latest_df = build_latest_round_fighter_state(history_df)

    return RoundFighterStateBuildResult(
        history_df=history_df,
        latest_df=latest_df,
    )


def main() -> None:
    """Build and save Round Fighter State artifacts."""
    ensure_data_dirs()

    print("=" * 80)
    print("BUILD ROUND FIGHTER STATE ARTIFACTS")
    print("=" * 80)
    print(f"Round stats path : {ROUND_STATS_PATH}")
    print(f"History output   : {ROUND_FIGHTER_STATE_HISTORY_PATH}")
    print(f"Latest output    : {ROUND_LATEST_FIGHTER_STATE_PATH}")
    print("Scope            : P0.1 shell")

    result = build_round_fighter_state()

    ROUND_FIGHTER_STATE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROUND_LATEST_FIGHTER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    result.history_df.to_parquet(ROUND_FIGHTER_STATE_HISTORY_PATH, index=False)
    result.latest_df.to_parquet(ROUND_LATEST_FIGHTER_STATE_PATH, index=False)

    print(f"History shape    : {result.history_df.shape}")
    print(f"Latest shape     : {result.latest_df.shape}")
    print("Saved Round Fighter State artifacts successfully.")
    print("DONE")


if __name__ == "__main__":
    main()
