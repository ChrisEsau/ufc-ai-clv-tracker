"""Leakage-safe historical replay for the heuristic fight simulator.

The current mechanics engine is run against completed holdout fights. Fighter
states are reconstructed only from fights completed before each target matchup.
Target-fight outcomes and statistics are used only after simulation for scoring.

This is a shadow diagnostic, not a production or wagering-grade backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from pipeline.simulation.contracts import (
    FighterSimulationState,
    MatchupSimulationInput,
    SimulatorConfig,
)
from pipeline.simulation.engine import SIMULATOR_VERSION, run_simulation


class HistoricalSimulatorReplayError(RuntimeError):
    """Raised when historical replay inputs or outputs are invalid."""


@dataclass(frozen=True)
class HistoricalSimulatorReplayResult:
    fight_predictions: pd.DataFrame
    metrics: pd.DataFrame
    calibration: pd.DataFrame
    aggregate_comparison: pd.DataFrame
    population_priors: Mapping[str, float]


REQUIRED_COLUMNS = (
    "fight_id",
    "fighter_id",
    "opponent_id",
    "corner",
    "date",
    "round",
    "total_rounds",
    "winner_id",
    "method_family",
    "match_time_sec",
    "target_finish_time_in_round_seconds",
    "target_sig_attempted",
    "target_sig_landed",
    "target_td_attempted",
    "target_td_landed",
    "target_control_seconds",
    "target_knockdowns",
    "target_submission_attempts",
)

STAT_TARGETS = {
    "sig_attempted": "target_sig_attempted",
    "sig_landed": "target_sig_landed",
    "td_attempted": "target_td_attempted",
    "td_landed": "target_td_landed",
    "control_seconds": "target_control_seconds",
    "knockdowns": "target_knockdowns",
    "submission_attempts": "target_submission_attempts",
}

PRIOR_BASE_COLUMNS = (
    "fight_sig_attempted",
    "fight_sig_landed",
    "fight_td_attempted",
    "fight_td_landed",
    "fight_control_seconds",
    "fight_knockdowns",
    "fight_submission_attempts",
    "fight_exposure_seconds",
    "fight_round1_sig_attempted",
    "fight_round1_exposure_seconds",
    "fight_late_sig_attempted",
    "fight_late_exposure_seconds",
    "fight_ko_win",
    "fight_ko_loss",
    "fight_sub_win",
    "fight_sub_loss",
)

METHODS = ("decision", "ko_tko", "submission")


def _require_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise HistoricalSimulatorReplayError(
            f"{label} is missing required columns: {missing}"
        )


def _clip_probability(value: float, low: float = 0.02, high: float = 0.98) -> float:
    return float(np.clip(float(value), low, high))


def _safe_divide(numerator: float, denominator: float, fallback: float) -> float:
    if denominator <= 0 or not np.isfinite(denominator):
        return float(fallback)
    value = float(numerator) / float(denominator)
    return float(value) if np.isfinite(value) else float(fallback)


def _smoothed_rate(
    numerator: float,
    exposure: float,
    prior_rate: float,
    prior_exposure: float,
) -> float:
    return _safe_divide(
        float(numerator) + float(prior_rate) * float(prior_exposure),
        float(exposure) + float(prior_exposure),
        fallback=float(prior_rate),
    )


def _smoothed_probability(
    successes: float,
    trials: float,
    prior_probability: float,
    prior_trials: float,
) -> float:
    return _clip_probability(
        _safe_divide(
            float(successes) + float(prior_probability) * float(prior_trials),
            float(trials) + float(prior_trials),
            fallback=float(prior_probability),
        )
    )


def _scaled_ratio(value: float, reference: float, sensitivity: float = 0.75) -> float:
    ratio = max(1e-6, float(value)) / max(1e-6, float(reference))
    return _clip_probability(
        0.5 + 0.34 * np.tanh(log(ratio) * sensitivity),
        low=0.10,
        high=0.90,
    )


def _coerce_training_frame(training_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(training_df, REQUIRED_COLUMNS, "Simulator training table")
    df = training_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise HistoricalSimulatorReplayError("Replay rows require valid dates")

    df["corner"] = df["corner"].astype("string").str.strip().str.lower()
    if (~df["corner"].isin(["red", "blue"])).any():
        raise HistoricalSimulatorReplayError("Replay rows contain invalid corners")

    numeric_columns = (
        "round",
        "total_rounds",
        "match_time_sec",
        "target_finish_time_in_round_seconds",
        *STAT_TARGETS.values(),
    )
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[list(numeric_columns)].isna().any().any():
        raise HistoricalSimulatorReplayError(
            "Replay rows contain missing numeric values"
        )
    if df.duplicated(["fight_id", "fighter_id", "round"]).any():
        raise HistoricalSimulatorReplayError(
            "Replay rows contain duplicate fighter-round keys"
        )
    return df


def build_fighter_fight_history(training_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse fighter-round rows and add shifted career histories."""
    df = _coerce_training_frame(training_df)
    keys = ["fight_id", "fighter_id"]

    first_columns = [
        "opponent_id",
        "corner",
        "date",
        "total_rounds",
        "winner_id",
        "method_family",
        "match_time_sec",
    ]
    for optional in (
        "event_id",
        "event_name",
        "division",
        "title_fight",
        "fighter_name",
        "opponent_name",
    ):
        if optional in df.columns:
            first_columns.append(optional)

    aggregate_spec: dict[str, tuple[str, str]] = {
        column: (column, "first") for column in first_columns
    }
    aggregate_spec.update(
        {
            f"fight_{name}": (source, "sum")
            for name, source in STAT_TARGETS.items()
        }
    )
    aggregate_spec["fight_exposure_seconds"] = (
        "target_finish_time_in_round_seconds",
        "sum",
    )
    aggregate_spec["fight_rounds_observed"] = ("round", "nunique")

    fights = df.groupby(keys, dropna=False).agg(**aggregate_spec).reset_index()

    round_one = (
        df.loc[df["round"].eq(1)]
        .groupby(keys, dropna=False)
        .agg(
            fight_round1_sig_attempted=("target_sig_attempted", "sum"),
            fight_round1_exposure_seconds=(
                "target_finish_time_in_round_seconds",
                "sum",
            ),
        )
        .reset_index()
    )
    late = (
        df.loc[df["round"].ge(2)]
        .groupby(keys, dropna=False)
        .agg(
            fight_late_sig_attempted=("target_sig_attempted", "sum"),
            fight_late_exposure_seconds=(
                "target_finish_time_in_round_seconds",
                "sum",
            ),
        )
        .reset_index()
    )
    fights = fights.merge(round_one, on=keys, how="left", validate="one_to_one")
    fights = fights.merge(late, on=keys, how="left", validate="one_to_one")
    for column in (
        "fight_round1_sig_attempted",
        "fight_round1_exposure_seconds",
        "fight_late_sig_attempted",
        "fight_late_exposure_seconds",
    ):
        fights[column] = pd.to_numeric(fights[column], errors="coerce").fillna(0.0)

    fighter_won = fights["winner_id"].astype("string").eq(
        fights["fighter_id"].astype("string")
    )
    fighter_lost = fights["winner_id"].astype("string").eq(
        fights["opponent_id"].astype("string")
    )
    fights["fight_ko_win"] = (
        fighter_won & fights["method_family"].eq("ko_tko")
    ).astype(float)
    fights["fight_ko_loss"] = (
        fighter_lost & fights["method_family"].eq("ko_tko")
    ).astype(float)
    fights["fight_sub_win"] = (
        fighter_won & fights["method_family"].eq("submission")
    ).astype(float)
    fights["fight_sub_loss"] = (
        fighter_lost & fights["method_family"].eq("submission")
    ).astype(float)

    own_stat_columns = [f"fight_{name}" for name in STAT_TARGETS]
    opponent = fights[["fight_id", "fighter_id", *own_stat_columns]].rename(
        columns={
            "fighter_id": "opponent_id",
            **{
                column: column.replace("fight_", "fight_allowed_", 1)
                for column in own_stat_columns
            },
        }
    )
    fights = fights.merge(
        opponent,
        on=["fight_id", "opponent_id"],
        how="left",
        validate="one_to_one",
    )
    allowed_columns = [
        column for column in fights.columns if column.startswith("fight_allowed_")
    ]
    if not allowed_columns or fights[allowed_columns].isna().any().any():
        raise HistoricalSimulatorReplayError(
            "Historical replay requires complete paired fighter rows"
        )

    fights = fights.sort_values(["fighter_id", "date", "fight_id"]).reset_index(
        drop=True
    )
    group = fights.groupby("fighter_id", sort=False)
    fights["prior_fights"] = group.cumcount().astype(float)

    for column in (*PRIOR_BASE_COLUMNS, *allowed_columns):
        fights[column] = pd.to_numeric(fights[column], errors="coerce").fillna(0.0)
        fights[f"prior_{column}"] = group[column].cumsum() - fights[column]

    return fights.sort_values(["date", "fight_id", "corner"]).reset_index(drop=True)


def population_priors(history_df: pd.DataFrame, test_year: int) -> dict[str, float]:
    """Estimate fixed cold-start priors using only pre-holdout fights."""
    history = history_df.loc[history_df["date"].dt.year.lt(int(test_year))].copy()
    if history.empty:
        raise HistoricalSimulatorReplayError(
            f"No pre-{test_year} fights are available for population priors"
        )

    exposure_minutes = float(history["fight_exposure_seconds"].sum() / 60.0)
    round1_minutes = float(history["fight_round1_exposure_seconds"].sum() / 60.0)
    late_minutes = float(history["fight_late_exposure_seconds"].sum() / 60.0)
    fighter_fights = float(len(history))

    sig_attempted = float(history["fight_sig_attempted"].sum())
    sig_landed = float(history["fight_sig_landed"].sum())
    allowed_sig_attempted = float(history["fight_allowed_sig_attempted"].sum())
    allowed_sig_landed = float(history["fight_allowed_sig_landed"].sum())
    td_attempted = float(history["fight_td_attempted"].sum())
    td_landed = float(history["fight_td_landed"].sum())
    allowed_td_attempted = float(history["fight_allowed_td_attempted"].sum())
    allowed_td_landed = float(history["fight_allowed_td_landed"].sum())
    sig_rate = _safe_divide(sig_attempted, exposure_minutes, 8.0)

    return {
        "sig_rate_per_min": sig_rate,
        "sig_accuracy": _safe_divide(sig_landed, sig_attempted, 0.45),
        "sig_allowed_accuracy": _safe_divide(
            allowed_sig_landed,
            allowed_sig_attempted,
            0.45,
        ),
        "td_rate_per_15": _safe_divide(td_attempted, exposure_minutes, 0.30) * 15.0,
        "td_accuracy": _safe_divide(td_landed, td_attempted, 0.35),
        "td_allowed_accuracy": _safe_divide(
            allowed_td_landed,
            allowed_td_attempted,
            0.35,
        ),
        "control_seconds_per_td": _safe_divide(
            float(history["fight_control_seconds"].sum()),
            td_landed,
            45.0,
        ),
        "kd_per_sig_landed": _safe_divide(
            float(history["fight_knockdowns"].sum()),
            sig_landed,
            0.015,
        ),
        "allowed_kd_per_sig_landed": _safe_divide(
            float(history["fight_allowed_knockdowns"].sum()),
            allowed_sig_landed,
            0.015,
        ),
        "ko_win_rate": _safe_divide(
            float(history["fight_ko_win"].sum()),
            fighter_fights,
            0.12,
        ),
        "ko_loss_rate": _safe_divide(
            float(history["fight_ko_loss"].sum()),
            fighter_fights,
            0.12,
        ),
        "sub_win_rate": _safe_divide(
            float(history["fight_sub_win"].sum()),
            fighter_fights,
            0.08,
        ),
        "sub_loss_rate": _safe_divide(
            float(history["fight_sub_loss"].sum()),
            fighter_fights,
            0.08,
        ),
        "submission_attempt_rate_per_15": _safe_divide(
            float(history["fight_submission_attempts"].sum()),
            exposure_minutes,
            0.05,
        )
        * 15.0,
        "round1_sig_rate_per_min": _safe_divide(
            float(history["fight_round1_sig_attempted"].sum()),
            round1_minutes,
            sig_rate,
        ),
        "late_sig_rate_per_min": _safe_divide(
            float(history["fight_late_sig_attempted"].sum()),
            late_minutes,
            sig_rate,
        ),
    }


def fighter_state_from_history(
    row: pd.Series,
    priors: Mapping[str, float],
) -> FighterSimulationState:
    """Translate one shifted fighter history into the public state contract."""
    exposure_minutes = float(row["prior_fight_exposure_seconds"]) / 60.0
    pace = _smoothed_rate(
        float(row["prior_fight_sig_attempted"]),
        exposure_minutes,
        priors["sig_rate_per_min"],
        prior_exposure=30.0,
    )
    sig_accuracy = _smoothed_probability(
        float(row["prior_fight_sig_landed"]),
        float(row["prior_fight_sig_attempted"]),
        priors["sig_accuracy"],
        prior_trials=60.0,
    )
    allowed_accuracy = _smoothed_probability(
        float(row["prior_fight_allowed_sig_landed"]),
        float(row["prior_fight_allowed_sig_attempted"]),
        priors["sig_allowed_accuracy"],
        prior_trials=60.0,
    )

    td_rate_per_min = _smoothed_rate(
        float(row["prior_fight_td_attempted"]),
        exposure_minutes,
        priors["td_rate_per_15"] / 15.0,
        prior_exposure=45.0,
    )
    td_accuracy = _smoothed_probability(
        float(row["prior_fight_td_landed"]),
        float(row["prior_fight_td_attempted"]),
        priors["td_accuracy"],
        prior_trials=15.0,
    )
    allowed_td_accuracy = _smoothed_probability(
        float(row["prior_fight_allowed_td_landed"]),
        float(row["prior_fight_allowed_td_attempted"]),
        priors["td_allowed_accuracy"],
        prior_trials=15.0,
    )

    control_per_td = _safe_divide(
        float(row["prior_fight_control_seconds"])
        + priors["control_seconds_per_td"] * 3.0,
        float(row["prior_fight_td_landed"]) + 3.0,
        priors["control_seconds_per_td"],
    )
    control_per_td = float(np.clip(control_per_td, 5.0, 180.0))

    prior_fights = float(row["prior_fights"])
    ko_win_rate = _smoothed_probability(
        float(row["prior_fight_ko_win"]),
        prior_fights,
        priors["ko_win_rate"],
        prior_trials=5.0,
    )
    ko_loss_rate = _smoothed_probability(
        float(row["prior_fight_ko_loss"]),
        prior_fights,
        priors["ko_loss_rate"],
        prior_trials=5.0,
    )
    sub_win_rate = _smoothed_probability(
        float(row["prior_fight_sub_win"]),
        prior_fights,
        priors["sub_win_rate"],
        prior_trials=5.0,
    )
    sub_loss_rate = _smoothed_probability(
        float(row["prior_fight_sub_loss"]),
        prior_fights,
        priors["sub_loss_rate"],
        prior_trials=5.0,
    )

    kd_rate = _safe_divide(
        float(row["prior_fight_knockdowns"])
        + priors["kd_per_sig_landed"] * 75.0,
        float(row["prior_fight_sig_landed"]) + 75.0,
        priors["kd_per_sig_landed"],
    )
    allowed_kd_rate = _safe_divide(
        float(row["prior_fight_allowed_knockdowns"])
        + priors["allowed_kd_per_sig_landed"] * 75.0,
        float(row["prior_fight_allowed_sig_landed"]) + 75.0,
        priors["allowed_kd_per_sig_landed"],
    )
    power = _clip_probability(
        0.65 * _scaled_ratio(kd_rate, priors["kd_per_sig_landed"])
        + 0.35 * _scaled_ratio(ko_win_rate, priors["ko_win_rate"])
    )
    vulnerability = _clip_probability(
        0.65
        * _scaled_ratio(
            allowed_kd_rate,
            priors["allowed_kd_per_sig_landed"],
        )
        + 0.35 * _scaled_ratio(ko_loss_rate, priors["ko_loss_rate"])
    )
    durability = _clip_probability(1.0 - vulnerability)

    sub_attempt_rate = _smoothed_rate(
        float(row["prior_fight_submission_attempts"]),
        exposure_minutes,
        priors["submission_attempt_rate_per_15"] / 15.0,
        prior_exposure=45.0,
    ) * 15.0
    submission_threat = _clip_probability(
        0.55 * _scaled_ratio(sub_win_rate, priors["sub_win_rate"])
        + 0.45
        * _scaled_ratio(
            sub_attempt_rate,
            priors["submission_attempt_rate_per_15"],
        )
    )
    submission_defense = _clip_probability(
        1.0 - _scaled_ratio(sub_loss_rate, priors["sub_loss_rate"])
    )

    round1_minutes = float(row["prior_fight_round1_exposure_seconds"]) / 60.0
    late_minutes = float(row["prior_fight_late_exposure_seconds"]) / 60.0
    early_pace = _smoothed_rate(
        float(row["prior_fight_round1_sig_attempted"]),
        round1_minutes,
        priors["round1_sig_rate_per_min"],
        prior_exposure=15.0,
    )
    late_pace = _smoothed_rate(
        float(row["prior_fight_late_sig_attempted"]),
        late_minutes,
        priors["late_sig_rate_per_min"],
        prior_exposure=20.0,
    )
    pace_ratio = late_pace / max(early_pace, 1e-6)
    pace_sustainability = _clip_probability(
        0.5 + 0.30 * np.tanh(log(max(1e-6, pace_ratio))),
        low=0.12,
        high=0.88,
    )
    cardio = _clip_probability(0.65 * pace_sustainability + 0.35 * durability)
    recovery = _clip_probability(0.65 * durability + 0.35 * pace_sustainability)

    own_attempts = float(row["prior_fight_sig_attempted"])
    allowed_attempts = float(row["prior_fight_allowed_sig_attempted"])
    initiative = _clip_probability(
        _safe_divide(
            own_attempts + 50.0,
            own_attempts + allowed_attempts + 100.0,
            0.5,
        ),
        low=0.18,
        high=0.82,
    )
    own_control = float(row["prior_fight_control_seconds"])
    allowed_control = float(row["prior_fight_allowed_control_seconds"])
    control_share = _safe_divide(
        own_control + 90.0,
        own_control + allowed_control + 180.0,
        0.5,
    )
    own_td = float(row["prior_fight_td_attempted"])
    allowed_td = float(row["prior_fight_allowed_td_attempted"])
    td_share = _safe_divide(own_td + 4.0, own_td + allowed_td + 8.0, 0.5)
    phase_imposition = _clip_probability(
        0.55 * control_share + 0.45 * td_share,
        low=0.15,
        high=0.85,
    )

    fighter_name = row.get("fighter_name", row["fighter_id"])
    if pd.isna(fighter_name) or not str(fighter_name).strip():
        fighter_name = row["fighter_id"]

    return FighterSimulationState(
        fighter_id=str(row["fighter_id"]),
        fighter_name=str(fighter_name),
        sig_attempts_per_minute=float(np.clip(pace, 1.0, 30.0)),
        sig_accuracy=sig_accuracy,
        sig_defense=_clip_probability(1.0 - allowed_accuracy),
        power=power,
        durability=durability,
        td_attempts_per_15=float(np.clip(td_rate_per_min * 15.0, 0.0, 20.0)),
        td_accuracy=td_accuracy,
        td_defense=_clip_probability(1.0 - allowed_td_accuracy),
        control_seconds_per_takedown=control_per_td,
        submission_threat=submission_threat,
        submission_defense=submission_defense,
        cardio=cardio,
        recovery=recovery,
        pace_sustainability=pace_sustainability,
        adaptability=0.5,
        initiative=initiative,
        phase_imposition=phase_imposition,
        metadata={
            "prior_fights": int(prior_fights),
            "state_source": "shifted_historical_fight_aggregates",
        },
    )


def build_holdout_matchups(
    history_df: pd.DataFrame,
    test_year: int,
    priors: Mapping[str, float],
    max_fights: int | None = None,
) -> list[dict[str, object]]:
    """Build paired simulator inputs and actual scoring records."""
    candidates = history_df.loc[
        history_df["date"].dt.year.eq(int(test_year))
        & history_df["method_family"].isin(METHODS)
    ].copy()
    if candidates.empty:
        raise HistoricalSimulatorReplayError(
            f"No eligible fights were found for holdout year {test_year}"
        )

    records: list[dict[str, object]] = []
    for fight_id, group in candidates.groupby("fight_id", sort=False):
        if len(group) != 2 or set(group["corner"]) != {"red", "blue"}:
            continue
        red = group.loc[group["corner"].eq("red")].iloc[0]
        blue = group.loc[group["corner"].eq("blue")].iloc[0]
        winner_id = str(red["winner_id"])
        fighter_ids = {str(red["fighter_id"]), str(blue["fighter_id"])}
        if winner_id not in fighter_ids:
            continue

        matchup = MatchupSimulationInput(
            fight_id=str(fight_id),
            event_id=(
                None
                if "event_id" not in red.index or pd.isna(red["event_id"])
                else str(red["event_id"])
            ),
            red=fighter_state_from_history(red, priors),
            blue=fighter_state_from_history(blue, priors),
            scheduled_rounds=int(red["total_rounds"]),
            round_seconds=300,
            source_snapshot_id=f"historical_holdout_{test_year}",
            metadata={"date": str(pd.Timestamp(red["date"]).date())},
        )
        records.append(
            {
                "matchup": matchup,
                "date": pd.Timestamp(red["date"]),
                "actual_winner_corner": (
                    "red" if winner_id == str(red["fighter_id"]) else "blue"
                ),
                "actual_method": str(red["method_family"]),
                "actual_fight_time_seconds": float(red["match_time_sec"]),
                "actual_red_sig_attempted": float(red["fight_sig_attempted"]),
                "actual_blue_sig_attempted": float(blue["fight_sig_attempted"]),
                "red_prior_fights": int(red["prior_fights"]),
                "blue_prior_fights": int(blue["prior_fights"]),
            }
        )

    records.sort(key=lambda value: (value["date"], value["matchup"].fight_id))
    if max_fights is not None:
        records = records[-int(max_fights) :]
    if not records:
        raise HistoricalSimulatorReplayError("No complete holdout matchups were built")
    return records


def _fight_level_baselines(
    history_df: pd.DataFrame,
    test_year: int,
) -> dict[int, dict[str, float]]:
    pretest = history_df.loc[history_df["date"].dt.year.lt(int(test_year))].copy()
    rows: list[dict[str, object]] = []
    for _, group in pretest.groupby("fight_id", sort=False):
        if len(group) != 2 or set(group["corner"]) != {"red", "blue"}:
            continue
        red = group.loc[group["corner"].eq("red")].iloc[0]
        blue = group.loc[group["corner"].eq("blue")].iloc[0]
        winner_id = str(red["winner_id"])
        method = str(red["method_family"])
        if winner_id not in {str(red["fighter_id"]), str(blue["fighter_id"])}:
            continue
        if method not in METHODS:
            continue
        rows.append(
            {
                "total_rounds": int(red["total_rounds"]),
                "red_win": float(winner_id == str(red["fighter_id"])),
                "method": method,
                "fight_time_seconds": float(red["match_time_sec"]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise HistoricalSimulatorReplayError("No pretest fights were available")

    global_method = frame["method"].value_counts(normalize=True)
    global_values = {
        "red_win_probability": float(frame["red_win"].mean()),
        "fight_time_seconds": float(frame["fight_time_seconds"].mean()),
        **{
            f"method_{method}": float(global_method.get(method, 0.0))
            for method in METHODS
        },
    }
    result: dict[int, dict[str, float]] = {}
    for rounds in (3, 5):
        subset = frame.loc[frame["total_rounds"].eq(rounds)]
        if len(subset) < 50:
            result[rounds] = dict(global_values)
            continue
        method_counts = subset["method"].value_counts(normalize=True)
        result[rounds] = {
            "red_win_probability": float(subset["red_win"].mean()),
            "fight_time_seconds": float(subset["fight_time_seconds"].mean()),
            **{
                f"method_{method}": float(method_counts.get(method, 0.0))
                for method in METHODS
            },
        }
    return result


def run_historical_simulator_replay(
    training_df: pd.DataFrame,
    test_year: int = 2026,
    simulations_per_fight: int = 750,
    seed: int = 91,
    max_fights: int | None = None,
) -> HistoricalSimulatorReplayResult:
    """Run the current simulator on completed holdout matchups."""
    if simulations_per_fight <= 0:
        raise HistoricalSimulatorReplayError("simulations_per_fight must be positive")

    history = build_fighter_fight_history(training_df)
    priors = population_priors(history, test_year=test_year)
    matchups = build_holdout_matchups(
        history,
        test_year=test_year,
        priors=priors,
        max_fights=max_fights,
    )
    baselines = _fight_level_baselines(history, test_year=test_year)

    rows: list[dict[str, object]] = []
    for index, record in enumerate(matchups):
        matchup = record["matchup"]
        summary, _ = run_simulation(
            matchup,
            SimulatorConfig(
                simulations=int(simulations_per_fight),
                seed=int(seed + index * 9973),
                retain_outcomes=False,
            ),
        )
        probabilities = summary.probabilities
        expectations = summary.expectations
        method_probabilities = {
            "decision": float(probabilities["goes_distance"]),
            "ko_tko": float(
                probabilities["red_by_ko_tko"] + probabilities["blue_by_ko_tko"]
            ),
            "submission": float(
                probabilities["red_by_submission"]
                + probabilities["blue_by_submission"]
            ),
        }
        method_total = sum(method_probabilities.values())
        if method_total <= 0:
            raise HistoricalSimulatorReplayError(
                f"Simulator returned zero method mass for {matchup.fight_id}"
            )
        method_probabilities = {
            key: value / method_total for key, value in method_probabilities.items()
        }
        baseline = baselines[int(matchup.scheduled_rounds)]
        baseline_time = float(baseline["fight_time_seconds"])

        rows.append(
            {
                "fight_id": matchup.fight_id,
                "event_id": matchup.event_id,
                "date": record["date"],
                "scheduled_rounds": matchup.scheduled_rounds,
                "red_fighter_id": matchup.red.fighter_id,
                "red_fighter_name": matchup.red.fighter_name,
                "blue_fighter_id": matchup.blue.fighter_id,
                "blue_fighter_name": matchup.blue.fighter_name,
                "red_prior_fights": record["red_prior_fights"],
                "blue_prior_fights": record["blue_prior_fights"],
                "actual_winner_corner": record["actual_winner_corner"],
                "actual_method": record["actual_method"],
                "actual_fight_time_seconds": record["actual_fight_time_seconds"],
                "actual_red_sig_attempted": record["actual_red_sig_attempted"],
                "actual_blue_sig_attempted": record["actual_blue_sig_attempted"],
                "sim_red_win_probability": float(probabilities["red_win"]),
                "sim_decision_probability": method_probabilities["decision"],
                "sim_ko_tko_probability": method_probabilities["ko_tko"],
                "sim_submission_probability": method_probabilities["submission"],
                "sim_fight_time_seconds": float(expectations["fight_time_seconds"]),
                "sim_red_sig_attempted": float(expectations["red_sig_attempted"]),
                "sim_blue_sig_attempted": float(expectations["blue_sig_attempted"]),
                "baseline_red_win_probability": float(
                    baseline["red_win_probability"]
                ),
                "baseline_decision_probability": float(baseline["method_decision"]),
                "baseline_ko_tko_probability": float(baseline["method_ko_tko"]),
                "baseline_submission_probability": float(
                    baseline["method_submission"]
                ),
                "baseline_fight_time_seconds": baseline_time,
                "baseline_red_sig_attempted": float(
                    matchup.red.sig_attempts_per_minute * baseline_time / 60.0
                ),
                "baseline_blue_sig_attempted": float(
                    matchup.blue.sig_attempts_per_minute * baseline_time / 60.0
                ),
                "simulations": int(simulations_per_fight),
                "simulator_version": SIMULATOR_VERSION,
            }
        )

    predictions = pd.DataFrame(rows)
    return HistoricalSimulatorReplayResult(
        fight_predictions=predictions,
        metrics=score_historical_replay(predictions),
        calibration=calibration_tables(predictions),
        aggregate_comparison=aggregate_comparison(predictions),
        population_priors=priors,
    )


def _binary_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return {
        "accuracy": float(np.mean((p >= 0.5).astype(float) == y)),
        "brier": float(np.mean(np.square(p - y))),
        "log_loss": float(
            -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        ),
    }


def _multiclass_metrics(
    actual_method: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, float]:
    method_index = {method: index for index, method in enumerate(METHODS)}
    actual = np.asarray(
        [method_index[str(value)] for value in actual_method],
        dtype=int,
    )
    probability = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0)
    probability = probability / probability.sum(axis=1, keepdims=True)
    return {
        "accuracy": float(np.mean(np.argmax(probability, axis=1) == actual)),
        "log_loss": float(
            -np.mean(np.log(probability[np.arange(len(actual)), actual]))
        ),
    }


def _continuous_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(sqrt(np.mean(np.square(residual)))),
        "bias": float(np.mean(residual)),
    }


def score_historical_replay(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return long-form simulator-versus-baseline metrics."""
    if predictions.empty:
        raise HistoricalSimulatorReplayError("Replay predictions are empty")

    rows: list[dict[str, object]] = []
    actual_red = predictions["actual_winner_corner"].eq("red").astype(float).to_numpy()
    actual_distance = predictions["actual_method"].eq("decision").astype(float).to_numpy()

    for model, column in (
        ("simulator", "sim_red_win_probability"),
        ("historical_baseline", "baseline_red_win_probability"),
    ):
        for metric, value in _binary_metrics(actual_red, predictions[column]).items():
            rows.append(
                {
                    "task": "winner",
                    "model": model,
                    "metric": metric,
                    "value": value,
                    "rows": len(predictions),
                }
            )

    for model, prefix in (
        ("simulator", "sim"),
        ("historical_baseline", "baseline"),
    ):
        probability = predictions[
            [
                f"{prefix}_decision_probability",
                f"{prefix}_ko_tko_probability",
                f"{prefix}_submission_probability",
            ]
        ].to_numpy(dtype=float)
        for metric, value in _multiclass_metrics(
            predictions["actual_method"],
            probability,
        ).items():
            rows.append(
                {
                    "task": "method",
                    "model": model,
                    "metric": metric,
                    "value": value,
                    "rows": len(predictions),
                }
            )
        for metric, value in _binary_metrics(
            actual_distance,
            predictions[f"{prefix}_decision_probability"],
        ).items():
            rows.append(
                {
                    "task": "goes_distance",
                    "model": model,
                    "metric": metric,
                    "value": value,
                    "rows": len(predictions),
                }
            )

    for model, column in (
        ("simulator", "sim_fight_time_seconds"),
        ("historical_baseline", "baseline_fight_time_seconds"),
    ):
        for metric, value in _continuous_metrics(
            predictions["actual_fight_time_seconds"],
            predictions[column],
        ).items():
            rows.append(
                {
                    "task": "fight_time_seconds",
                    "model": model,
                    "metric": metric,
                    "value": value,
                    "rows": len(predictions),
                }
            )

    actual_strikes = np.concatenate(
        [
            predictions["actual_red_sig_attempted"].to_numpy(dtype=float),
            predictions["actual_blue_sig_attempted"].to_numpy(dtype=float),
        ]
    )
    for model, red_column, blue_column in (
        ("simulator", "sim_red_sig_attempted", "sim_blue_sig_attempted"),
        (
            "historical_baseline",
            "baseline_red_sig_attempted",
            "baseline_blue_sig_attempted",
        ),
    ):
        predicted_strikes = np.concatenate(
            [
                predictions[red_column].to_numpy(dtype=float),
                predictions[blue_column].to_numpy(dtype=float),
            ]
        )
        for metric, value in _continuous_metrics(
            actual_strikes,
            predicted_strikes,
        ).items():
            rows.append(
                {
                    "task": "fighter_sig_attempted",
                    "model": model,
                    "metric": metric,
                    "value": value,
                    "rows": len(actual_strikes),
                }
            )

    return pd.DataFrame(rows).sort_values(["task", "metric", "model"]).reset_index(
        drop=True
    )


def calibration_tables(predictions: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Return equal-frequency winner and distance calibration bins."""
    rows: list[pd.DataFrame] = []
    for task, actual_column, probability_column in (
        ("winner_red", "actual_winner_corner", "sim_red_win_probability"),
        ("goes_distance", "actual_method", "sim_decision_probability"),
    ):
        frame = predictions[[actual_column, probability_column]].copy()
        frame["actual"] = (
            frame[actual_column].eq("red").astype(float)
            if task == "winner_red"
            else frame[actual_column].eq("decision").astype(float)
        )
        rank = frame[probability_column].rank(method="first", pct=True)
        frame["bin"] = np.ceil(rank * int(bins)).clip(1, bins).astype(int)
        grouped = (
            frame.groupby("bin")
            .agg(
                rows=("actual", "size"),
                predicted_mean=(probability_column, "mean"),
                actual_rate=("actual", "mean"),
                predicted_min=(probability_column, "min"),
                predicted_max=(probability_column, "max"),
            )
            .reset_index()
        )
        grouped.insert(0, "task", task)
        grouped["calibration_error"] = (
            grouped["predicted_mean"] - grouped["actual_rate"]
        )
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def aggregate_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare observed and simulated aggregate holdout behavior."""
    actual_method = predictions["actual_method"].value_counts(normalize=True)
    actual_strikes = pd.concat(
        [
            predictions["actual_red_sig_attempted"],
            predictions["actual_blue_sig_attempted"],
        ],
        ignore_index=True,
    )
    simulated_strikes = pd.concat(
        [
            predictions["sim_red_sig_attempted"],
            predictions["sim_blue_sig_attempted"],
        ],
        ignore_index=True,
    )
    result = pd.DataFrame(
        [
            {
                "quantity": "red_win_rate",
                "actual": float(
                    predictions["actual_winner_corner"].eq("red").mean()
                ),
                "simulator": float(predictions["sim_red_win_probability"].mean()),
            },
            {
                "quantity": "decision_rate",
                "actual": float(actual_method.get("decision", 0.0)),
                "simulator": float(predictions["sim_decision_probability"].mean()),
            },
            {
                "quantity": "ko_tko_rate",
                "actual": float(actual_method.get("ko_tko", 0.0)),
                "simulator": float(predictions["sim_ko_tko_probability"].mean()),
            },
            {
                "quantity": "submission_rate",
                "actual": float(actual_method.get("submission", 0.0)),
                "simulator": float(
                    predictions["sim_submission_probability"].mean()
                ),
            },
            {
                "quantity": "fight_time_seconds",
                "actual": float(predictions["actual_fight_time_seconds"].mean()),
                "simulator": float(predictions["sim_fight_time_seconds"].mean()),
            },
            {
                "quantity": "fighter_sig_attempted",
                "actual": float(actual_strikes.mean()),
                "simulator": float(simulated_strikes.mean()),
            },
        ]
    )
    result["error"] = result["simulator"] - result["actual"]
    result["relative_error"] = np.where(
        result["actual"].abs().gt(1e-9),
        result["error"] / result["actual"],
        np.nan,
    )
    return result


def metric_lookup(
    metrics: pd.DataFrame,
    task: str,
    model: str,
    metric: str,
) -> float:
    match = metrics.loc[
        metrics["task"].eq(task)
        & metrics["model"].eq(model)
        & metrics["metric"].eq(metric),
        "value",
    ]
    if len(match) != 1:
        raise HistoricalSimulatorReplayError(
            f"Metric lookup failed: {task}/{model}/{metric}"
        )
    return float(match.iloc[0])
