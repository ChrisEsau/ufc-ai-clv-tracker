"""Build Round Fighter State P0.2 opponent suppression features.

P0.2 measures whether a fighter suppresses an opponent below that opponent's
own point-in-time historical baseline.

This module intentionally does not modify production model feature views.
It creates standalone RFS suppression artifacts:

    data/features/round_fighter_suppression_p0_2_history.parquet
    data/features/round_latest_fighter_suppression_p0_2.parquet

Run from repo root:

    python -m pipeline.round_stats.build_round_fighter_suppression
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline.common.paths import (
    ROUND_FIGHTER_SUPPRESSION_P0_2_HISTORY_PATH,
    ROUND_LATEST_FIGHTER_SUPPRESSION_P0_2_PATH,
    ROUND_STATS_PATH,
    ensure_data_dirs,
)


SUPPRESSION_FIGHT_OBSERVATION_COLUMNS = [
    "rfs_suppress_fight_opp_sig_attempt_delta",
    "rfs_suppress_fight_opp_total_attempt_delta",
    "rfs_suppress_fight_opp_late_output_delta",
    "rfs_suppress_fight_opp_sig_accuracy_delta",
    "rfs_suppress_fight_opp_total_accuracy_delta",
    "rfs_suppress_fight_opp_distance_accuracy_delta",
    "rfs_suppress_fight_opp_td_attempt_delta",
    "rfs_suppress_fight_opp_td_accuracy_delta",
    "rfs_suppress_fight_opp_control_delta",
    "rfs_suppress_fight_opp_distance_share_delta",
    "rfs_suppress_fight_opp_clinch_share_delta",
    "rfs_suppress_fight_opp_ground_share_delta",
    "rfs_suppress_fight_opp_phase_mix_disruption",
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
    "sig_str_landed",
    "sig_str_attempted",
    "total_str_landed",
    "total_str_attempted",
    "td_landed",
    "td_attempted",
    "distance_landed",
    "distance_attempted",
    "clinch_attempted",
    "ground_attempted",
]


@dataclass(frozen=True)
class SuppressionBuildResult:
    history: pd.DataFrame
    latest: pd.DataFrame


class RoundFighterSuppressionBuildError(RuntimeError):
    """Raised when P0.2 suppression features cannot be built safely."""


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    out = numerator / denominator.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def expanding_prior(series: pd.Series) -> pd.Series:
    return series.shift(1).expanding(min_periods=1).mean()


def last3_prior(series: pd.Series) -> pd.Series:
    return series.shift(1).rolling(3, min_periods=1).mean()


def ewm_prior(series: pd.Series, alpha: float = 0.35) -> pd.Series:
    return series.shift(1).ewm(alpha=alpha, adjust=False).mean()


def _require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RoundFighterSuppressionBuildError(
            f"{label} is missing required columns: {missing}"
        )


def standardize_round_stats(round_stats_df: pd.DataFrame) -> pd.DataFrame:
    df = round_stats_df.copy()

    if "date" not in df.columns:
        if "event_date" in df.columns:
            df["date"] = pd.to_datetime(df["event_date"], errors="coerce")
        else:
            raise RoundFighterSuppressionBuildError(
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
        "sig_str_landed",
        "sig_str_attempted",
        "total_str_landed",
        "total_str_attempted",
        "td_landed",
        "td_attempted",
        "distance_landed",
        "distance_attempted",
        "clinch_attempted",
        "ground_attempted",
        "control_seconds",
    ]

    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    return df


def build_fight_level_actuals(rounds_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
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
        rounds_df.groupby(group_cols, dropna=False)
        .agg(
            rounds_observed=("round", "nunique"),
            sig_landed=("sig_str_landed", "sum"),
            sig_attempted=("sig_str_attempted", "sum"),
            total_landed=("total_str_landed", "sum"),
            total_attempted=("total_str_attempted", "sum"),
            td_landed=("td_landed", "sum"),
            td_attempted=("td_attempted", "sum"),
            distance_landed=("distance_landed", "sum"),
            distance_attempted=("distance_attempted", "sum"),
            clinch_attempted=("clinch_attempted", "sum"),
            ground_attempted=("ground_attempted", "sum"),
            control_seconds=("control_seconds", "sum"),
        )
        .reset_index()
    )

    fight_df["actual_sig_attempt_per_round"] = safe_div(
        fight_df["sig_attempted"], fight_df["rounds_observed"]
    )
    fight_df["actual_total_attempt_per_round"] = safe_div(
        fight_df["total_attempted"], fight_df["rounds_observed"]
    )
    fight_df["actual_td_attempt_per_round"] = safe_div(
        fight_df["td_attempted"], fight_df["rounds_observed"]
    )
    fight_df["actual_control_seconds_per_round"] = safe_div(
        fight_df["control_seconds"], fight_df["rounds_observed"]
    )

    fight_df["actual_sig_accuracy"] = safe_div(
        fight_df["sig_landed"], fight_df["sig_attempted"]
    )
    fight_df["actual_total_accuracy"] = safe_div(
        fight_df["total_landed"], fight_df["total_attempted"]
    )
    fight_df["actual_distance_accuracy"] = safe_div(
        fight_df["distance_landed"], fight_df["distance_attempted"]
    )
    fight_df["actual_td_accuracy"] = safe_div(
        fight_df["td_landed"], fight_df["td_attempted"]
    )

    phase_total = (
        fight_df["distance_attempted"]
        + fight_df["clinch_attempted"]
        + fight_df["ground_attempted"]
    )

    fight_df["actual_distance_share"] = safe_div(fight_df["distance_attempted"], phase_total)
    fight_df["actual_clinch_share"] = safe_div(fight_df["clinch_attempted"], phase_total)
    fight_df["actual_ground_share"] = safe_div(fight_df["ground_attempted"], phase_total)

    # Practical round-aggregate proxy for late opponent output.
    # This is intentionally fight-level and point-in-time safe only after it is shifted into state.
    late_output = (
        rounds_df.sort_values(["fight_id", "fighter_id", "round"])
        .groupby(["fight_id", "fighter_id"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "actual_late_output_per_round": float(
                        g.loc[g["round"].ge(g["round"].max()), "total_str_attempted"].sum()
                    )
                    / max(float(g.loc[g["round"].ge(g["round"].max()), "round"].nunique()), 1.0)
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    fight_df = fight_df.merge(
        late_output,
        on=["fight_id", "fighter_id"],
        how="left",
        validate="one_to_one",
    )

    return fight_df.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)


def add_point_in_time_baselines(fight_actuals_df: pd.DataFrame) -> pd.DataFrame:
    df = fight_actuals_df.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True).copy()

    baseline_sources = [
        "actual_sig_attempt_per_round",
        "actual_total_attempt_per_round",
        "actual_late_output_per_round",
        "actual_sig_accuracy",
        "actual_total_accuracy",
        "actual_distance_accuracy",
        "actual_td_attempt_per_round",
        "actual_td_accuracy",
        "actual_control_seconds_per_round",
        "actual_distance_share",
        "actual_clinch_share",
        "actual_ground_share",
    ]

    for source_col in baseline_sources:
        baseline_col = source_col.replace("actual_", "baseline_")
        df[baseline_col] = (
            df.groupby("fighter_id", group_keys=False)[source_col]
            .transform(expanding_prior)
        )

    df["baseline_prior_fight_count"] = df.groupby("fighter_id").cumcount()

    return df


def build_suppression_observations(actuals_with_baselines_df: pd.DataFrame) -> pd.DataFrame:
    df = actuals_with_baselines_df.copy()

    opponent_source_cols = [
        "fight_id",
        "fighter_id",
        "actual_sig_attempt_per_round",
        "actual_total_attempt_per_round",
        "actual_late_output_per_round",
        "actual_sig_accuracy",
        "actual_total_accuracy",
        "actual_distance_accuracy",
        "actual_td_attempt_per_round",
        "actual_td_accuracy",
        "actual_control_seconds_per_round",
        "actual_distance_share",
        "actual_clinch_share",
        "actual_ground_share",
        "baseline_sig_attempt_per_round",
        "baseline_total_attempt_per_round",
        "baseline_late_output_per_round",
        "baseline_sig_accuracy",
        "baseline_total_accuracy",
        "baseline_distance_accuracy",
        "baseline_td_attempt_per_round",
        "baseline_td_accuracy",
        "baseline_control_seconds_per_round",
        "baseline_distance_share",
        "baseline_clinch_share",
        "baseline_ground_share",
        "baseline_prior_fight_count",
    ]

    opponent_df = df[opponent_source_cols].rename(
        columns={
            "fighter_id": "opponent_id",
            **{
                column: f"opp_{column}"
                for column in opponent_source_cols
                if column not in {"fight_id", "fighter_id"}
            },
        }
    )

    obs = df.merge(
        opponent_df,
        on=["fight_id", "opponent_id"],
        how="left",
        validate="one_to_one",
    )

    obs["rfs_suppress_fight_has_opp_baseline"] = (
        pd.to_numeric(obs["opp_baseline_prior_fight_count"], errors="coerce")
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    delta_pairs = {
        "rfs_suppress_fight_opp_sig_attempt_delta": (
            "opp_baseline_sig_attempt_per_round",
            "opp_actual_sig_attempt_per_round",
        ),
        "rfs_suppress_fight_opp_total_attempt_delta": (
            "opp_baseline_total_attempt_per_round",
            "opp_actual_total_attempt_per_round",
        ),
        "rfs_suppress_fight_opp_late_output_delta": (
            "opp_baseline_late_output_per_round",
            "opp_actual_late_output_per_round",
        ),
        "rfs_suppress_fight_opp_sig_accuracy_delta": (
            "opp_baseline_sig_accuracy",
            "opp_actual_sig_accuracy",
        ),
        "rfs_suppress_fight_opp_total_accuracy_delta": (
            "opp_baseline_total_accuracy",
            "opp_actual_total_accuracy",
        ),
        "rfs_suppress_fight_opp_distance_accuracy_delta": (
            "opp_baseline_distance_accuracy",
            "opp_actual_distance_accuracy",
        ),
        "rfs_suppress_fight_opp_td_attempt_delta": (
            "opp_baseline_td_attempt_per_round",
            "opp_actual_td_attempt_per_round",
        ),
        "rfs_suppress_fight_opp_td_accuracy_delta": (
            "opp_baseline_td_accuracy",
            "opp_actual_td_accuracy",
        ),
        "rfs_suppress_fight_opp_control_delta": (
            "opp_baseline_control_seconds_per_round",
            "opp_actual_control_seconds_per_round",
        ),
        "rfs_suppress_fight_opp_distance_share_delta": (
            "opp_baseline_distance_share",
            "opp_actual_distance_share",
        ),
        "rfs_suppress_fight_opp_clinch_share_delta": (
            "opp_baseline_clinch_share",
            "opp_actual_clinch_share",
        ),
        "rfs_suppress_fight_opp_ground_share_delta": (
            "opp_baseline_ground_share",
            "opp_actual_ground_share",
        ),
    }

    for output_col, (baseline_col, actual_col) in delta_pairs.items():
        obs[output_col] = obs[baseline_col] - obs[actual_col]

    obs["rfs_suppress_fight_opp_phase_mix_disruption"] = 0.5 * (
        obs["rfs_suppress_fight_opp_distance_share_delta"].abs()
        + obs["rfs_suppress_fight_opp_clinch_share_delta"].abs()
        + obs["rfs_suppress_fight_opp_ground_share_delta"].abs()
    )

    for column in SUPPRESSION_FIGHT_OBSERVATION_COLUMNS:
        obs.loc[obs["rfs_suppress_fight_has_opp_baseline"].eq(0), column] = np.nan

    return obs


def add_prior_suppression_state(observations_df: pd.DataFrame) -> pd.DataFrame:
    df = observations_df.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True).copy()

    for fight_col in SUPPRESSION_FIGHT_OBSERVATION_COLUMNS:
        base_name = fight_col.replace("rfs_suppress_fight_", "rfs_suppress_")

        df[base_name.replace("rfs_suppress_", "rfs_suppress_exp_")] = (
            df.groupby("fighter_id", group_keys=False)[fight_col]
            .transform(expanding_prior)
        )
        df[base_name.replace("rfs_suppress_", "rfs_suppress_last3_")] = (
            df.groupby("fighter_id", group_keys=False)[fight_col]
            .transform(last3_prior)
        )
        df[base_name.replace("rfs_suppress_", "rfs_suppress_ewm_")] = (
            df.groupby("fighter_id", group_keys=False)[fight_col]
            .transform(ewm_prior)
        )

    df["rfs_suppress_prior_fight_count"] = df.groupby("fighter_id").cumcount()

    valid_obs = df[SUPPRESSION_FIGHT_OBSERVATION_COLUMNS].notna().any(axis=1).astype(int)
    df["rfs_suppress_prior_valid_suppression_count"] = (
        valid_obs.groupby(df["fighter_id"])
        .transform(lambda series: series.cumsum().shift(1).fillna(0))
        .astype(int)
    )

    state_cols = [
        column
        for column in df.columns
        if column.startswith(("rfs_suppress_exp_", "rfs_suppress_last3_", "rfs_suppress_ewm_"))
    ]

    df["rfs_suppress_has_state"] = df[state_cols].notna().any(axis=1).astype(int)

    return df


def build_latest_suppression_state(history_df: pd.DataFrame) -> pd.DataFrame:
    latest_rows = []
    state_source_cols = SUPPRESSION_FIGHT_OBSERVATION_COLUMNS

    for fighter_id, group in history_df.sort_values(["fighter_id", "date", "fight_id"]).groupby("fighter_id", sort=False):
        group = group.sort_values(["date", "fight_id"]).reset_index(drop=True)
        last = group.iloc[-1]

        state = {
            "fighter_id": last["fighter_id"],
            "fighter_name": last["fighter_name"],
            "latest_event_name": last["event_name"],
            "latest_date": last["date"],
            "rfs_suppress_prior_fight_count": len(group),
            "rfs_suppress_prior_valid_suppression_count": int(
                group[state_source_cols].notna().any(axis=1).sum()
            ),
        }

        for fight_col in state_source_cols:
            base_name = fight_col.replace("rfs_suppress_fight_", "rfs_suppress_")
            values = pd.to_numeric(group[fight_col], errors="coerce")

            state[base_name.replace("rfs_suppress_", "rfs_suppress_exp_")] = (
                values.expanding(min_periods=1).mean().iloc[-1]
            )
            state[base_name.replace("rfs_suppress_", "rfs_suppress_last3_")] = (
                values.tail(3).mean()
            )
            state[base_name.replace("rfs_suppress_", "rfs_suppress_ewm_")] = (
                values.ewm(alpha=0.35, adjust=False).mean().iloc[-1]
            )

        state_cols = [
            key
            for key in state
            if key.startswith(("rfs_suppress_exp_", "rfs_suppress_last3_", "rfs_suppress_ewm_"))
        ]
        state["rfs_suppress_has_state"] = int(any(pd.notna(state[key]) for key in state_cols))

        latest_rows.append(state)

    return pd.DataFrame(latest_rows)


def build_round_fighter_suppression(round_stats_df: pd.DataFrame) -> SuppressionBuildResult:
    rounds = standardize_round_stats(round_stats_df)
    fight_actuals = build_fight_level_actuals(rounds)
    with_baselines = add_point_in_time_baselines(fight_actuals)
    observations = build_suppression_observations(with_baselines)
    history = add_prior_suppression_state(observations)
    latest = build_latest_suppression_state(history)

    return SuppressionBuildResult(history=history, latest=latest)


def main() -> None:
    ensure_data_dirs()

    if not ROUND_STATS_PATH.exists():
        raise RoundFighterSuppressionBuildError(f"Round stats artifact not found: {ROUND_STATS_PATH}")

    round_stats_df = pd.read_parquet(ROUND_STATS_PATH)
    result = build_round_fighter_suppression(round_stats_df)

    result.history.to_parquet(ROUND_FIGHTER_SUPPRESSION_P0_2_HISTORY_PATH, index=False)
    result.latest.to_parquet(ROUND_LATEST_FIGHTER_SUPPRESSION_P0_2_PATH, index=False)

    print("=" * 80)
    print("ROUND FIGHTER SUPPRESSION P0.2 BUILD")
    print("=" * 80)
    print(f"Round stats path : {ROUND_STATS_PATH}")
    print(f"History path     : {ROUND_FIGHTER_SUPPRESSION_P0_2_HISTORY_PATH}")
    print(f"Latest path      : {ROUND_LATEST_FIGHTER_SUPPRESSION_P0_2_PATH}")
    print(f"History shape    : {result.history.shape}")
    print(f"Latest shape     : {result.latest.shape}")
    print(f"Fight obs cols   : {len(SUPPRESSION_FIGHT_OBSERVATION_COLUMNS)}")
    print(
        "Rows with obs    : "
        f"{int(result.history[SUPPRESSION_FIGHT_OBSERVATION_COLUMNS].notna().any(axis=1).sum())}"
    )
    print(
        "Rows with state  : "
        f"{int(result.history.filter(regex='^rfs_suppress_(exp|last3|ewm)_').notna().any(axis=1).sum())}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
