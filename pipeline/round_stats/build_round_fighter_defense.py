"""Build Round Fighter State P1.4 defensive degradation features.

P1.4 answers:
    Does a fighter become easier to hit, control, damage, or take down
    as the fight progresses?

This module creates standalone RFS defense artifacts and does not alter
production model contracts.

Run from repo root:

    python -m pipeline.round_stats.build_round_fighter_defense
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline.common.paths import (
    ROUND_FIGHTER_DEFENSE_P1_4_HISTORY_PATH,
    ROUND_LATEST_FIGHTER_DEFENSE_P1_4_PATH,
    ROUND_STATS_PATH,
    ensure_data_dirs,
)


DEFENSE_FIGHT_OBSERVATION_COLUMNS = [
    "rfs_def_fight_opp_sig_accuracy_allowed_slope",
    "rfs_def_fight_opp_total_accuracy_allowed_slope",
    "rfs_def_fight_sig_absorbed_slope",
    "rfs_def_fight_total_absorbed_slope",
    "rfs_def_fight_head_absorbed_slope",
    "rfs_def_fight_opp_output_acceleration",
    "rfs_def_fight_opp_control_allowed_slope",
    "rfs_def_fight_late_sig_damage_allowed_delta",
    "rfs_def_fight_late_total_damage_allowed_delta",
    "rfs_def_fight_late_head_damage_allowed_delta",
    "rfs_def_fight_late_control_allowed_delta",
    "rfs_def_fight_late_td_defense_decay",
    "rfs_def_fight_kd_absorbed",
    "rfs_def_fight_late_kd_absorbed_delta",
    "rfs_def_fight_defensive_deterioration_score",
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
    "kd",
]

HEAD_LANDED_CANDIDATES = [
    "head_landed",
    "head_str_landed",
    "head_strikes_landed",
    "sig_head_landed",
    "head_sig_str_landed",
]

HEAD_ATTEMPTED_CANDIDATES = [
    "head_attempted",
    "head_str_attempted",
    "head_strikes_attempted",
    "sig_head_attempted",
    "head_sig_str_attempted",
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


@dataclass(frozen=True)
class DefenseBuildResult:
    history: pd.DataFrame
    latest: pd.DataFrame


class RoundFighterDefenseBuildError(RuntimeError):
    """Raised when P1.4 defensive degradation features cannot be built safely."""


def _require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RoundFighterDefenseBuildError(
            f"{label} is missing required columns: {missing}"
        )


def _parse_time_to_seconds(value: object) -> float | None:
    """Parse numeric seconds or common M:SS/MM:SS control-time strings."""

    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    if text.replace(".", "", 1).isdigit():
        return float(text)

    if ":" in text:
        pieces = text.split(":")
        if len(pieces) == 2:
            try:
                minutes = int(pieces[0])
                seconds = int(pieces[1])
                return float(minutes * 60 + seconds)
            except ValueError:
                return None

    return None


def _first_present_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


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


def late_delta(values: pd.Series) -> float:
    y = pd.to_numeric(values, errors="coerce").dropna()
    if len(y) < 2:
        return np.nan
    return float(y.iloc[-1] - y.iloc[0])


def positive_or_zero(value: float) -> float:
    if pd.isna(value) or not np.isfinite(value):
        return np.nan
    return float(max(value, 0.0))


def expanding_prior(series: pd.Series) -> pd.Series:
    return series.shift(1).expanding(min_periods=1).mean()


def last3_prior(series: pd.Series) -> pd.Series:
    return series.shift(1).rolling(3, min_periods=1).mean()


def ewm_prior(series: pd.Series, alpha: float = 0.35) -> pd.Series:
    return series.shift(1).ewm(alpha=alpha, adjust=False, ignore_na=True).mean()


def standardize_round_stats(round_stats_df: pd.DataFrame) -> pd.DataFrame:
    df = round_stats_df.copy()

    date_source = _first_present_column(df, DATE_CANDIDATES)
    if date_source is None:
        raise RoundFighterDefenseBuildError(
            f"Round stats must include one date column from: {DATE_CANDIDATES}"
        )
    df["date"] = pd.to_datetime(df[date_source], errors="coerce")

    control_source = _first_present_column(df, CONTROL_SECONDS_CANDIDATES)
    if control_source is None:
        raise RoundFighterDefenseBuildError(
            f"Round stats must include one control-time column from: {CONTROL_SECONDS_CANDIDATES}"
        )
    df["control_seconds"] = df[control_source].map(_parse_time_to_seconds)

    head_landed_source = _first_present_column(df, HEAD_LANDED_CANDIDATES)
    head_attempted_source = _first_present_column(df, HEAD_ATTEMPTED_CANDIDATES)

    if head_landed_source is not None:
        df["head_landed"] = df[head_landed_source]
    else:
        # Keep P1.4 buildable even if the current round feed does not expose head splits.
        df["head_landed"] = np.nan

    if head_attempted_source is not None:
        df["head_attempted"] = df[head_attempted_source]
    else:
        df["head_attempted"] = np.nan

    _require_columns(
        df,
        [
            *REQUIRED_ROUND_COLUMNS,
            "date",
            "control_seconds",
            "head_landed",
            "head_attempted",
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
        "kd",
        "control_seconds",
        "head_landed",
        "head_attempted",
    ]

    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    required_numeric = [
        "round",
        "sig_str_landed",
        "sig_str_attempted",
        "total_str_landed",
        "total_str_attempted",
        "td_landed",
        "td_attempted",
        "kd",
        "control_seconds",
    ]

    for column in required_numeric:
        df[column] = df[column].fillna(0)

    df["sig_accuracy"] = safe_div(df["sig_str_landed"], df["sig_str_attempted"])
    df["total_accuracy"] = safe_div(df["total_str_landed"], df["total_str_attempted"])
    df["td_accuracy"] = safe_div(df["td_landed"], df["td_attempted"])
    df["head_accuracy"] = safe_div(df["head_landed"], df["head_attempted"])

    return df


def join_opponent_round_metrics(rounds_df: pd.DataFrame) -> pd.DataFrame:
    opponent_cols = [
        "fight_id",
        "round",
        "fighter_id",
        "sig_str_landed",
        "sig_str_attempted",
        "sig_accuracy",
        "total_str_landed",
        "total_str_attempted",
        "total_accuracy",
        "td_landed",
        "td_attempted",
        "td_accuracy",
        "control_seconds",
        "kd",
        "head_landed",
        "head_attempted",
        "head_accuracy",
    ]

    opponent = rounds_df[opponent_cols].rename(
        columns={
            "fighter_id": "opponent_id",
            "sig_str_landed": "opp_sig_landed",
            "sig_str_attempted": "opp_sig_attempted",
            "sig_accuracy": "opp_sig_accuracy",
            "total_str_landed": "opp_total_landed",
            "total_str_attempted": "opp_total_attempted",
            "total_accuracy": "opp_total_accuracy",
            "td_landed": "opp_td_landed",
            "td_attempted": "opp_td_attempted",
            "td_accuracy": "opp_td_accuracy",
            "control_seconds": "opp_control_seconds",
            "kd": "opp_kd",
            "head_landed": "opp_head_landed",
            "head_attempted": "opp_head_attempted",
            "head_accuracy": "opp_head_accuracy",
        }
    )

    return rounds_df.merge(
        opponent,
        on=["fight_id", "round", "opponent_id"],
        how="left",
        validate="many_to_one",
    )


def _normalized_positive_slope(group: pd.DataFrame, source_col: str) -> float:
    slope = ols_slope(group[source_col])
    mean_value = pd.to_numeric(group[source_col], errors="coerce").mean()
    if pd.isna(slope):
        return np.nan
    return positive_or_zero(safe_scalar_div(slope, max(float(mean_value), 1.0)))


def _build_deterioration_score(row: dict, group: pd.DataFrame) -> float:
    """Create a bounded directional score where higher means late defensive decay.

    The score intentionally rewards deterioration only. Defensive improvement
    does not offset other deterioration signals; it simply contributes zero.
    """

    components = [
        positive_or_zero(row["rfs_def_fight_opp_sig_accuracy_allowed_slope"]),
        positive_or_zero(row["rfs_def_fight_opp_total_accuracy_allowed_slope"]),
        _normalized_positive_slope(group, "opp_sig_landed"),
        _normalized_positive_slope(group, "opp_total_landed"),
        _normalized_positive_slope(group, "opp_head_landed"),
        _normalized_positive_slope(group, "opp_sig_attempted"),
        positive_or_zero(
            safe_scalar_div(
                row["rfs_def_fight_opp_control_allowed_slope"],
                60.0,
            )
        ),
        positive_or_zero(row["rfs_def_fight_late_td_defense_decay"]),
    ]

    valid_components = [value for value in components if pd.notna(value)]
    if not valid_components:
        return np.nan

    return float(np.mean(valid_components))


def build_fight_level_observations(rounds_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    group_keys = [
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

    for keys, group in rounds_df.groupby(group_keys, dropna=False, sort=False):
        group = group.sort_values("round").copy()
        row = dict(zip(group_keys, keys))

        row["rounds_observed"] = int(group["round"].nunique())

        # If these increase round-to-round, the fighter is becoming easier to hit/control.
        row["rfs_def_fight_opp_sig_accuracy_allowed_slope"] = ols_slope(group["opp_sig_accuracy"])
        row["rfs_def_fight_opp_total_accuracy_allowed_slope"] = ols_slope(group["opp_total_accuracy"])
        row["rfs_def_fight_sig_absorbed_slope"] = ols_slope(group["opp_sig_landed"])
        row["rfs_def_fight_total_absorbed_slope"] = ols_slope(group["opp_total_landed"])
        row["rfs_def_fight_head_absorbed_slope"] = ols_slope(group["opp_head_landed"])
        row["rfs_def_fight_opp_output_acceleration"] = ols_slope(group["opp_sig_attempted"])
        row["rfs_def_fight_opp_control_allowed_slope"] = ols_slope(group["opp_control_seconds"])

        row["rfs_def_fight_late_sig_damage_allowed_delta"] = late_delta(group["opp_sig_landed"])
        row["rfs_def_fight_late_total_damage_allowed_delta"] = late_delta(group["opp_total_landed"])
        row["rfs_def_fight_late_head_damage_allowed_delta"] = late_delta(group["opp_head_landed"])
        row["rfs_def_fight_late_control_allowed_delta"] = late_delta(group["opp_control_seconds"])

        # Positive value means opponent takedown success improved late.
        row["rfs_def_fight_late_td_defense_decay"] = late_delta(group["opp_td_accuracy"])

        row["rfs_def_fight_kd_absorbed"] = float(
            pd.to_numeric(group["opp_kd"], errors="coerce").fillna(0).sum()
        )
        row["rfs_def_fight_late_kd_absorbed_delta"] = late_delta(group["opp_kd"])

        row["rfs_def_fight_defensive_deterioration_score"] = _build_deterioration_score(
            row,
            group,
        )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["fighter_id", "date", "fight_id"])
        .reset_index(drop=True)
    )


def add_prior_defense_state(observations_df: pd.DataFrame) -> pd.DataFrame:
    df = observations_df.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True).copy()

    for fight_col in DEFENSE_FIGHT_OBSERVATION_COLUMNS:
        base_name = fight_col.replace("rfs_def_fight_", "rfs_def_")

        df[base_name.replace("rfs_def_", "rfs_def_exp_")] = (
            df.groupby("fighter_id", group_keys=False)[fight_col]
            .transform(expanding_prior)
        )
        df[base_name.replace("rfs_def_", "rfs_def_last3_")] = (
            df.groupby("fighter_id", group_keys=False)[fight_col]
            .transform(last3_prior)
        )
        df[base_name.replace("rfs_def_", "rfs_def_ewm_")] = (
            df.groupby("fighter_id", group_keys=False)[fight_col]
            .transform(ewm_prior)
        )

    df["rfs_def_prior_fight_count"] = df.groupby("fighter_id").cumcount()

    valid_obs = df[DEFENSE_FIGHT_OBSERVATION_COLUMNS].notna().any(axis=1).astype(int)
    df["rfs_def_prior_valid_defense_count"] = (
        valid_obs.groupby(df["fighter_id"])
        .transform(lambda series: series.cumsum().shift(1).fillna(0))
        .astype(int)
    )

    state_cols = [
        column
        for column in df.columns
        if column.startswith(("rfs_def_exp_", "rfs_def_last3_", "rfs_def_ewm_"))
    ]

    df["rfs_def_has_state"] = df[state_cols].notna().any(axis=1).astype(int)

    return df


def build_latest_defense_state(history_df: pd.DataFrame) -> pd.DataFrame:
    latest_rows = []

    for fighter_id, group in history_df.sort_values(["fighter_id", "date", "fight_id"]).groupby("fighter_id", sort=False):
        group = group.sort_values(["date", "fight_id"]).reset_index(drop=True)
        last = group.iloc[-1]

        state = {
            "fighter_id": last["fighter_id"],
            "fighter_name": last["fighter_name"],
            "latest_event_name": last["event_name"],
            "latest_date": last["date"],
            "rfs_def_prior_fight_count": len(group),
            "rfs_def_prior_valid_defense_count": int(
                group[DEFENSE_FIGHT_OBSERVATION_COLUMNS].notna().any(axis=1).sum()
            ),
        }

        for fight_col in DEFENSE_FIGHT_OBSERVATION_COLUMNS:
            base_name = fight_col.replace("rfs_def_fight_", "rfs_def_")
            values = pd.to_numeric(group[fight_col], errors="coerce")

            state[base_name.replace("rfs_def_", "rfs_def_exp_")] = (
                values.expanding(min_periods=1).mean().iloc[-1]
            )
            state[base_name.replace("rfs_def_", "rfs_def_last3_")] = values.tail(3).mean()
            state[base_name.replace("rfs_def_", "rfs_def_ewm_")] = (
                values.ewm(alpha=0.35, adjust=False, ignore_na=True).mean().iloc[-1]
            )

        state_cols = [
            key
            for key in state
            if key.startswith(("rfs_def_exp_", "rfs_def_last3_", "rfs_def_ewm_"))
        ]
        state["rfs_def_has_state"] = int(any(pd.notna(state[key]) for key in state_cols))

        latest_rows.append(state)

    return pd.DataFrame(latest_rows)


def build_round_fighter_defense(round_stats_df: pd.DataFrame) -> DefenseBuildResult:
    rounds = standardize_round_stats(round_stats_df)
    rounds = join_opponent_round_metrics(rounds)
    observations = build_fight_level_observations(rounds)
    history = add_prior_defense_state(observations)
    latest = build_latest_defense_state(history)

    return DefenseBuildResult(history=history, latest=latest)


def main() -> None:
    ensure_data_dirs()

    if not ROUND_STATS_PATH.exists():
        raise RoundFighterDefenseBuildError(f"Round stats artifact not found: {ROUND_STATS_PATH}")

    round_stats_df = pd.read_parquet(ROUND_STATS_PATH)
    result = build_round_fighter_defense(round_stats_df)

    result.history.to_parquet(ROUND_FIGHTER_DEFENSE_P1_4_HISTORY_PATH, index=False)
    result.latest.to_parquet(ROUND_LATEST_FIGHTER_DEFENSE_P1_4_PATH, index=False)

    state_cols = [
        column
        for column in result.history.columns
        if column.startswith(("rfs_def_exp_", "rfs_def_last3_", "rfs_def_ewm_"))
    ]

    print("=" * 80)
    print("ROUND FIGHTER DEFENSE P1.4 BUILD")
    print("=" * 80)
    print(f"Round stats path : {ROUND_STATS_PATH}")
    print(f"History path     : {ROUND_FIGHTER_DEFENSE_P1_4_HISTORY_PATH}")
    print(f"Latest path      : {ROUND_LATEST_FIGHTER_DEFENSE_P1_4_PATH}")
    print(f"History shape    : {result.history.shape}")
    print(f"Latest shape     : {result.latest.shape}")
    print(f"Fight obs cols   : {len(DEFENSE_FIGHT_OBSERVATION_COLUMNS)}")
    print(
        "Rows with obs    : "
        f"{int(result.history[DEFENSE_FIGHT_OBSERVATION_COLUMNS].notna().any(axis=1).sum())}"
    )
    print(
        "Rows with state  : "
        f"{int(result.history[state_cols].notna().any(axis=1).sum())}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
