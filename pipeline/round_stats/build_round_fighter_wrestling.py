"""Build Round Fighter State P0.3 wrestling control conversion features.

P0.3 answers:
    Does wrestling activity become meaningful control, damage,
    submission threat, or stable position?

This module creates standalone RFS wrestling artifacts and does not alter
production model contracts.

Run from repo root:

    python -m pipeline.round_stats.build_round_fighter_wrestling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline.common.paths import (
    ROUND_FIGHTER_WRESTLING_P0_3_HISTORY_PATH,
    ROUND_LATEST_FIGHTER_WRESTLING_P0_3_PATH,
    ROUND_STATS_PATH,
    ensure_data_dirs,
)


WRESTLING_FIGHT_OBSERVATION_COLUMNS = [
    "rfs_wrestle_fight_control_per_td_attempt",
    "rfs_wrestle_fight_control_per_td_landed",
    "rfs_wrestle_fight_td_to_control_conversion",
    "rfs_wrestle_fight_ground_strikes_per_control_min",
    "rfs_wrestle_fight_sig_ground_strikes_per_control_min",
    "rfs_wrestle_fight_control_to_damage_score",
    "rfs_wrestle_fight_sub_attempts_per_control_min",
    "rfs_wrestle_fight_submission_pressure_score",
    "rfs_wrestle_fight_reversal_allowed_per_control_min",
    "rfs_wrestle_fight_control_stability_score",
    "rfs_wrestle_fight_td_attempt_slope",
    "rfs_wrestle_fight_td_persistence_score",
    "rfs_wrestle_fight_failed_td_persistence",
]


REQUIRED_ROUND_COLUMNS = [
    "event_id",
    "event_name",
    "fight_id",
    "corner",
    "fighter_id",
    "fighter_name",
    "opponent_id",
    "opponent_name",
    "round",
    "td_landed",
    "td_attempted",
    "sub_att",
    "rev",
    "ground_landed",
    "ground_attempted",
]


@dataclass(frozen=True)
class WrestlingBuildResult:
    history: pd.DataFrame
    latest: pd.DataFrame


class RoundFighterWrestlingBuildError(RuntimeError):
    """Raised when P0.3 wrestling features cannot be built safely."""


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    out = numerator / denominator.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def safe_scalar_div(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    value = numerator / denominator
    if np.isfinite(value):
        return float(value)
    return np.nan


def ols_slope(values: pd.Series) -> float:
    y = pd.to_numeric(values, errors="coerce").dropna()
    if len(y) < 2:
        return np.nan

    x = np.arange(1, len(y) + 1, dtype=float)
    y_values = y.to_numpy(dtype=float)

    if np.allclose(x.var(), 0):
        return np.nan

    return float(np.polyfit(x, y_values, 1)[0])


def expanding_prior(series: pd.Series) -> pd.Series:
    return series.shift(1).expanding(min_periods=1).mean()


def last3_prior(series: pd.Series) -> pd.Series:
    return series.shift(1).rolling(3, min_periods=1).mean()


def ewm_prior(series: pd.Series, alpha: float = 0.35) -> pd.Series:
    return series.shift(1).ewm(alpha=alpha, adjust=False).mean()


def _require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RoundFighterWrestlingBuildError(
            f"{label} is missing required columns: {missing}"
        )


def standardize_round_stats(round_stats_df: pd.DataFrame) -> pd.DataFrame:
    df = round_stats_df.copy()

    if "date" not in df.columns:
        if "event_date" in df.columns:
            df["date"] = pd.to_datetime(df["event_date"], errors="coerce")
        else:
            raise RoundFighterWrestlingBuildError(
                "Round stats must include either date or event_date."
            )
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "control_seconds" not in df.columns:
        for candidate in ["ctrl_sec", "ctrl_seconds", "control_time_sec", "control_time_seconds"]:
            if candidate in df.columns:
                df["control_seconds"] = df[candidate]
                break

    _require_columns(
        df,
        [
            *REQUIRED_ROUND_COLUMNS,
            "date",
            "control_seconds",
        ],
        "round stats",
    )

    df["corner"] = df["corner"].astype(str).str.lower().str.strip()

    numeric_cols = [
        "round",
        "td_landed",
        "td_attempted",
        "sub_att",
        "rev",
        "ground_landed",
        "ground_attempted",
        "control_seconds",
    ]

    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    df["failed_td_attempts"] = (df["td_attempted"] - df["td_landed"]).clip(lower=0)

    return df


def build_round_level_wrestling_shape(rounds_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    group_cols = [
        "fight_id",
        "fighter_id",
    ]

    for (fight_id, fighter_id), group in rounds_df.groupby(group_cols, dropna=False):
        g = group.sort_values("round").copy()

        first_round = g.iloc[0]
        last_round = g.iloc[-1]

        first_td_attempts = float(first_round["td_attempted"])
        last_td_attempts = float(last_round["td_attempted"])

        # Use denominator floor of 1 so delayed wrestling pressure is visible:
        # 0 -> 3 attempts becomes score 3 instead of NaN.
        td_persistence_score = last_td_attempts / max(first_td_attempts, 1.0)

        rows.append(
            {
                "fight_id": fight_id,
                "fighter_id": fighter_id,
                "rfs_wrestle_fight_td_attempt_slope": ols_slope(g["td_attempted"]),
                "rfs_wrestle_fight_td_persistence_score": td_persistence_score,
                "rfs_wrestle_fight_failed_td_persistence": ols_slope(g["failed_td_attempts"]),
            }
        )

    return pd.DataFrame(rows)


def build_fight_level_observations(rounds_df: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = [
        "event_id",
        "event_name",
        "date",
        "fight_id",
        "corner",
        "fighter_id",
        "fighter_name",
        "opponent_id",
        "opponent_name",
    ]

    fight_df = (
        rounds_df.groupby(metadata_cols, dropna=False)
        .agg(
            rounds_observed=("round", "nunique"),
            td_landed=("td_landed", "sum"),
            td_attempted=("td_attempted", "sum"),
            failed_td_attempts=("failed_td_attempts", "sum"),
            control_seconds=("control_seconds", "sum"),
            ground_landed=("ground_landed", "sum"),
            ground_attempted=("ground_attempted", "sum"),
            sub_att=("sub_att", "sum"),
            reversals=("rev", "sum"),
        )
        .reset_index()
    )

    opponent_reversals = fight_df[
        ["fight_id", "fighter_id", "reversals"]
    ].rename(
        columns={
            "fighter_id": "opponent_id",
            "reversals": "opponent_reversals",
        }
    )

    fight_df = fight_df.merge(
        opponent_reversals,
        on=["fight_id", "opponent_id"],
        how="left",
        validate="one_to_one",
    )

    fight_df["control_minutes"] = fight_df["control_seconds"] / 60.0
    fight_df["ground_accuracy"] = safe_div(
        fight_df["ground_landed"],
        fight_df["ground_attempted"],
    )

    fight_df["rfs_wrestle_fight_control_per_td_attempt"] = safe_div(
        fight_df["control_seconds"],
        fight_df["td_attempted"],
    )
    fight_df["rfs_wrestle_fight_control_per_td_landed"] = safe_div(
        fight_df["control_seconds"],
        fight_df["td_landed"],
    )

    # Ratio of control minutes created per takedown attempt.
    # Values above 1 mean each attempt yielded more than one minute of control on average.
    fight_df["rfs_wrestle_fight_td_to_control_conversion"] = safe_div(
        fight_df["control_minutes"],
        fight_df["td_attempted"],
    )

    # Available UFCStats round feed has ground significant strikes.
    # Attempts = activity, landed = meaningful landed ground offense.
    fight_df["rfs_wrestle_fight_ground_strikes_per_control_min"] = safe_div(
        fight_df["ground_attempted"],
        fight_df["control_minutes"],
    )
    fight_df["rfs_wrestle_fight_sig_ground_strikes_per_control_min"] = safe_div(
        fight_df["ground_landed"],
        fight_df["control_minutes"],
    )

    fight_df["rfs_wrestle_fight_control_to_damage_score"] = (
        fight_df["rfs_wrestle_fight_sig_ground_strikes_per_control_min"]
        * fight_df["ground_accuracy"]
    )

    fight_df["rfs_wrestle_fight_sub_attempts_per_control_min"] = safe_div(
        fight_df["sub_att"],
        fight_df["control_minutes"],
    )

    # Submission pressure rewards submission activity but requires actual control time.
    fight_df["rfs_wrestle_fight_submission_pressure_score"] = (
        fight_df["rfs_wrestle_fight_sub_attempts_per_control_min"]
        * np.log1p(fight_df["control_minutes"].fillna(0))
    )

    fight_df["rfs_wrestle_fight_reversal_allowed_per_control_min"] = safe_div(
        fight_df["opponent_reversals"],
        fight_df["control_minutes"],
    )

    fight_df["rfs_wrestle_fight_control_stability_score"] = 1.0 / (
        1.0 + fight_df["rfs_wrestle_fight_reversal_allowed_per_control_min"]
    )

    shape_df = build_round_level_wrestling_shape(rounds_df)

    fight_df = fight_df.merge(
        shape_df,
        on=["fight_id", "fighter_id"],
        how="left",
        validate="one_to_one",
    )

    return fight_df.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)


def add_prior_wrestling_state(observations_df: pd.DataFrame) -> pd.DataFrame:
    df = observations_df.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True).copy()

    for fight_col in WRESTLING_FIGHT_OBSERVATION_COLUMNS:
        base_name = fight_col.replace("rfs_wrestle_fight_", "rfs_wrestle_")

        df[base_name.replace("rfs_wrestle_", "rfs_wrestle_exp_")] = (
            df.groupby("fighter_id", group_keys=False)[fight_col]
            .transform(expanding_prior)
        )
        df[base_name.replace("rfs_wrestle_", "rfs_wrestle_last3_")] = (
            df.groupby("fighter_id", group_keys=False)[fight_col]
            .transform(last3_prior)
        )
        df[base_name.replace("rfs_wrestle_", "rfs_wrestle_ewm_")] = (
            df.groupby("fighter_id", group_keys=False)[fight_col]
            .transform(ewm_prior)
        )

    df["rfs_wrestle_prior_fight_count"] = df.groupby("fighter_id").cumcount()

    valid_obs = df[WRESTLING_FIGHT_OBSERVATION_COLUMNS].notna().any(axis=1).astype(int)
    df["rfs_wrestle_prior_valid_wrestling_count"] = (
        valid_obs.groupby(df["fighter_id"])
        .transform(lambda series: series.cumsum().shift(1).fillna(0))
        .astype(int)
    )

    state_cols = [
        column
        for column in df.columns
        if column.startswith(("rfs_wrestle_exp_", "rfs_wrestle_last3_", "rfs_wrestle_ewm_"))
    ]

    df["rfs_wrestle_has_state"] = df[state_cols].notna().any(axis=1).astype(int)

    return df


def build_latest_wrestling_state(history_df: pd.DataFrame) -> pd.DataFrame:
    latest_rows = []

    for fighter_id, group in history_df.sort_values(["fighter_id", "date", "fight_id"]).groupby("fighter_id", sort=False):
        group = group.sort_values(["date", "fight_id"]).reset_index(drop=True)
        last = group.iloc[-1]

        state = {
            "fighter_id": last["fighter_id"],
            "fighter_name": last["fighter_name"],
            "latest_event_name": last["event_name"],
            "latest_date": last["date"],
            "rfs_wrestle_prior_fight_count": len(group),
            "rfs_wrestle_prior_valid_wrestling_count": int(
                group[WRESTLING_FIGHT_OBSERVATION_COLUMNS].notna().any(axis=1).sum()
            ),
        }

        for fight_col in WRESTLING_FIGHT_OBSERVATION_COLUMNS:
            base_name = fight_col.replace("rfs_wrestle_fight_", "rfs_wrestle_")
            values = pd.to_numeric(group[fight_col], errors="coerce")

            state[base_name.replace("rfs_wrestle_", "rfs_wrestle_exp_")] = (
                values.expanding(min_periods=1).mean().iloc[-1]
            )
            state[base_name.replace("rfs_wrestle_", "rfs_wrestle_last3_")] = (
                values.tail(3).mean()
            )
            state[base_name.replace("rfs_wrestle_", "rfs_wrestle_ewm_")] = (
                values.ewm(alpha=0.35, adjust=False).mean().iloc[-1]
            )

        state_cols = [
            key
            for key in state
            if key.startswith(("rfs_wrestle_exp_", "rfs_wrestle_last3_", "rfs_wrestle_ewm_"))
        ]
        state["rfs_wrestle_has_state"] = int(any(pd.notna(state[key]) for key in state_cols))

        latest_rows.append(state)

    return pd.DataFrame(latest_rows)


def build_round_fighter_wrestling(round_stats_df: pd.DataFrame) -> WrestlingBuildResult:
    rounds = standardize_round_stats(round_stats_df)
    observations = build_fight_level_observations(rounds)
    history = add_prior_wrestling_state(observations)
    latest = build_latest_wrestling_state(history)

    return WrestlingBuildResult(history=history, latest=latest)


def main() -> None:
    ensure_data_dirs()

    if not ROUND_STATS_PATH.exists():
        raise RoundFighterWrestlingBuildError(f"Round stats artifact not found: {ROUND_STATS_PATH}")

    round_stats_df = pd.read_parquet(ROUND_STATS_PATH)
    result = build_round_fighter_wrestling(round_stats_df)

    result.history.to_parquet(ROUND_FIGHTER_WRESTLING_P0_3_HISTORY_PATH, index=False)
    result.latest.to_parquet(ROUND_LATEST_FIGHTER_WRESTLING_P0_3_PATH, index=False)

    state_cols = [
        column
        for column in result.history.columns
        if column.startswith(("rfs_wrestle_exp_", "rfs_wrestle_last3_", "rfs_wrestle_ewm_"))
    ]

    print("=" * 80)
    print("ROUND FIGHTER WRESTLING P0.3 BUILD")
    print("=" * 80)
    print(f"Round stats path : {ROUND_STATS_PATH}")
    print(f"History path     : {ROUND_FIGHTER_WRESTLING_P0_3_HISTORY_PATH}")
    print(f"Latest path      : {ROUND_LATEST_FIGHTER_WRESTLING_P0_3_PATH}")
    print(f"History shape    : {result.history.shape}")
    print(f"Latest shape     : {result.latest.shape}")
    print(f"Fight obs cols   : {len(WRESTLING_FIGHT_OBSERVATION_COLUMNS)}")
    print(
        "Rows with obs    : "
        f"{int(result.history[WRESTLING_FIGHT_OBSERVATION_COLUMNS].notna().any(axis=1).sum())}"
    )
    print(
        "Rows with state  : "
        f"{int(result.history[state_cols].notna().any(axis=1).sum())}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
