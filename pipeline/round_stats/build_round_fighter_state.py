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

from pipeline.round_stats.round_state_formulas import (
    late_diff,
    late_ratio,
    ols_slope,
    safe_div,
)

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


def _add_per_round_metrics(round_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Add safe per-round rates used by P0.1 trajectory formulas."""
    out = round_stats_df.copy()

    out["sig_accuracy"] = [
        safe_div(landed, attempted)
        for landed, attempted in zip(out["sig_str_landed"], out["sig_str_attempted"])
    ]
    out["total_accuracy"] = [
        safe_div(landed, attempted)
        for landed, attempted in zip(out["total_str_landed"], out["total_str_attempted"])
    ]
    out["td_accuracy"] = [
        safe_div(landed, attempted)
        for landed, attempted in zip(out["td_landed"], out["td_attempted"])
    ]

    return out


def _join_opponent_round_metrics(round_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Attach opponent same-round metrics for defensive trajectory proxies."""
    opponent_columns = [
        "fight_id",
        "round",
        "fighter_id",
        "sig_str_attempted",
        "sig_accuracy",
        "total_accuracy",
        "control_seconds",
    ]

    opponent = round_stats_df[opponent_columns].rename(
        columns={
            "fighter_id": "opponent_id",
            "sig_str_attempted": "opp_sig_str_attempted",
            "sig_accuracy": "opp_sig_accuracy",
            "total_accuracy": "opp_total_accuracy",
            "control_seconds": "opp_control_seconds",
        }
    )

    out = round_stats_df.merge(
        opponent,
        on=["fight_id", "round", "opponent_id"],
        how="left",
        validate="many_to_one",
    )

    return out


def _build_fight_observation_rows(round_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Build one P0.1 fight-observation row per fighter per fight."""
    rows: list[dict] = []

    group_keys = [
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

    metric_specs = {
        "sig_attempt": "sig_str_attempted",
        "total_attempt": "total_str_attempted",
        "sig_landed": "sig_str_landed",
        "total_landed": "total_str_landed",
        "sig_accuracy": "sig_accuracy",
        "total_accuracy": "total_accuracy",
        "td_attempt": "td_attempted",
        "td_accuracy": "td_accuracy",
        "control_seconds": "control_seconds",
        "opp_sig_accuracy_allowed": "opp_sig_accuracy",
        "opp_total_accuracy_allowed": "opp_total_accuracy",
        "opp_sig_attempt_allowed": "opp_sig_str_attempted",
        "opp_control_allowed": "opp_control_seconds",
    }

    late_ratio_metrics = {
        "sig_attempt": "sig_str_attempted",
        "total_attempt": "total_str_attempted",
        "sig_landed": "sig_str_landed",
        "td_attempt": "td_attempted",
        "control": "control_seconds",
    }

    late_diff_metrics = {
        "sig_accuracy": "sig_accuracy",
        "total_accuracy": "total_accuracy",
    }

    for keys, group in round_stats_df.groupby(group_keys, dropna=False, sort=False):
        group = group.sort_values("round").copy()
        row = dict(zip(group_keys, keys))
        row["rfs_traj_fight_rounds_observed"] = int(group["round"].nunique())

        rounds = group["round"]

        for feature_name, source_column in metric_specs.items():
            row[f"rfs_traj_fight_{feature_name}_slope"] = ols_slope(
                rounds,
                group[source_column],
            )

        for feature_name, source_column in late_ratio_metrics.items():
            row[f"rfs_traj_fight_{feature_name}_late_ratio"] = late_ratio(
                group[source_column],
            )

        for feature_name, source_column in late_diff_metrics.items():
            row[f"rfs_traj_fight_{feature_name}_late_diff"] = late_diff(
                group[source_column],
            )

        rows.append(row)

    return pd.DataFrame(rows)


def _add_prior_state_features(fight_observation_df: pd.DataFrame) -> pd.DataFrame:
    """Convert fight observations into point-in-time prior fighter-state features."""
    out = fight_observation_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)

    observation_columns = [
        column
        for column in out.columns
        if column.startswith("rfs_traj_fight_")
        and column != "rfs_traj_fight_rounds_observed"
    ]

    out["rfs_traj_prior_fight_count"] = out.groupby("fighter_id").cumcount()

    valid_observation_mask = out[observation_columns].notna().any(axis=1)
    out["_valid_trajectory_observation"] = valid_observation_mask.astype(int)
    out["rfs_traj_prior_valid_trajectory_count"] = (
        out.groupby("fighter_id")["_valid_trajectory_observation"]
        .transform(lambda series: series.cumsum().shift(1).fillna(0))
        .astype(int)
    )

    for column in observation_columns:
        base = column.replace("rfs_traj_fight_", "", 1)
        group = out.groupby("fighter_id")[column]

        out[f"rfs_traj_exp_{base}"] = group.transform(
            lambda series: series.shift(1).expanding(min_periods=1).mean()
        )
        out[f"rfs_traj_last3_{base}"] = group.transform(
            lambda series: series.shift(1).rolling(window=3, min_periods=1).mean()
        )
        out[f"rfs_traj_ewm_{base}"] = group.transform(
            lambda series: series.shift(1).ewm(alpha=0.35, adjust=False, ignore_na=True).mean()
        )

    out["rfs_traj_has_state"] = out["rfs_traj_prior_valid_trajectory_count"].gt(0).astype(int)
    out = out.drop(columns=["_valid_trajectory_observation"])

    return out


def build_round_fighter_state_history(round_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Build P0.1 Round Fighter State history with leakage-safe prior features."""
    standardized_df = standardize_round_stats_input(round_stats_df)
    validate_round_stats_input(standardized_df)

    metric_df = _add_per_round_metrics(standardized_df)
    metric_df = _join_opponent_round_metrics(metric_df)

    fight_observation_df = _build_fight_observation_rows(metric_df)
    history_df = _add_prior_state_features(fight_observation_df)

    return history_df


def _none_if_nan(value: object) -> float | None:
    """Return None when pandas/numpy produced NaN."""
    if pd.isna(value):
        return None
    return float(value)


def build_latest_round_fighter_state(history_df: pd.DataFrame) -> pd.DataFrame:
    """Return current post-fight Round Fighter State for each fighter.

    History rows are point-in-time rows entering each historical fight.
    Latest rows are different: they represent the state available after all
    completed fights in the round-stats artifact, for future/live joins.
    """
    if history_df.empty:
        return history_df.copy()

    metadata_columns = [
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

    observation_columns = [
        column
        for column in history_df.columns
        if column.startswith("rfs_traj_fight_")
        and column != "rfs_traj_fight_rounds_observed"
    ]

    rows: list[dict] = []

    sorted_history = history_df.copy()
    sorted_history["date"] = pd.to_datetime(sorted_history["date"], errors="coerce")
    sorted_history = sorted_history.sort_values(["fighter_id", "date", "fight_id"])

    for fighter_id, group in sorted_history.groupby("fighter_id", sort=False):
        group = group.sort_values(["date", "fight_id"]).copy()
        latest_meta = group.tail(1)[metadata_columns].iloc[0].to_dict()

        valid_observation_mask = group[observation_columns].notna().any(axis=1)
        valid_count = int(valid_observation_mask.sum())

        row = dict(latest_meta)
        row["rfs_traj_prior_fight_count"] = int(len(group))
        row["rfs_traj_prior_valid_trajectory_count"] = valid_count
        row["rfs_traj_has_state"] = int(valid_count > 0)

        for column in observation_columns:
            base = column.replace("rfs_traj_fight_", "", 1)
            series = pd.to_numeric(group[column], errors="coerce")

            if series.notna().any():
                row[f"rfs_traj_exp_{base}"] = _none_if_nan(
                    series.expanding(min_periods=1).mean().iloc[-1]
                )
                row[f"rfs_traj_last3_{base}"] = _none_if_nan(
                    series.tail(3).mean()
                )
                row[f"rfs_traj_ewm_{base}"] = _none_if_nan(
                    series.ewm(alpha=0.35, adjust=False, ignore_na=True).mean().iloc[-1]
                )
            else:
                row[f"rfs_traj_exp_{base}"] = None
                row[f"rfs_traj_last3_{base}"] = None
                row[f"rfs_traj_ewm_{base}"] = None

        rows.append(row)

    latest_df = pd.DataFrame(rows)
    latest_df = latest_df.sort_values(["fighter_name", "fighter_id"]).reset_index(drop=True)

    return latest_df

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
