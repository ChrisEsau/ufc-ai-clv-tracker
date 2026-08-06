"""Leakage-safe offensive power features for RFS Monte Carlo V1.

These features are derived from prior UFC round statistics and represent
offensive impact rather than defensive vulnerability.

No target-date or future fights may enter the profile.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from pipeline.simulation.rfs_mc_v1.contracts import (
    FighterSimulationProfile,
    ParameterEstimate,
    ProfileSource,
)


class OffensivePowerError(ValueError):
    """Raised when offensive power inputs are invalid."""


POWER_PARAMETER_NAMES = (
    "offensive_kd_per_sig_landed",
    "offensive_kd_per_fight",
    "round1_kd_per_fight",
    "head_strike_accuracy",
    "ground_strike_accuracy",
    "round1_sig_landed_per_fight",
)


def _normalize_date(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)

    return timestamp.normalize()


def _safe_ratio(
    numerator: float,
    denominator: float,
    *,
    default: float = 0.0,
) -> float:
    if denominator <= 0:
        return default

    return float(numerator / denominator)


def _historical_population_median(
    prior_history: pd.DataFrame,
    *,
    value_builder,
) -> float:
    """Calculate a fighter-level historical median before the target date."""

    values: list[float] = []

    for _, fighter_rows in prior_history.groupby("fighter_id"):
        value = value_builder(fighter_rows)

        if np.isfinite(value):
            values.append(float(value))

    if not values:
        return 0.0

    return float(np.median(values))


def _shrink(
    fighter_value: float,
    population_value: float,
    *,
    sample_size: int,
    prior_strength: float,
) -> float:
    """Empirical-Bayes style shrinkage toward a historical median."""

    weight = sample_size / (sample_size + prior_strength)

    return float(
        weight * fighter_value
        + (1.0 - weight) * population_value
    )


def build_offensive_power_estimates(
    round_stats: pd.DataFrame,
    *,
    fighter_id: str,
    target_date: Any,
    prior_strength: float = 5.0,
) -> dict[str, ParameterEstimate]:
    """Build offensive power estimates using strictly prior UFC fights."""

    required = {
        "fight_id",
        "event_date",
        "fighter_id",
        "round",
        "kd",
        "sig_str_landed",
        "head_landed",
        "head_attempted",
        "ground_landed",
        "ground_attempted",
    }

    missing = required - set(round_stats.columns)
    if missing:
        raise OffensivePowerError(
            f"Round stats missing required columns: {sorted(missing)}"
        )

    if prior_strength <= 0:
        raise OffensivePowerError(
            "prior_strength must be positive"
        )

    target_ts = _normalize_date(target_date)

    history = round_stats.copy()
    history["event_date"] = pd.to_datetime(
        history["event_date"],
        errors="coerce",
    ).dt.tz_localize(None)

    prior_history = history.loc[
        history["event_date"] < target_ts
    ].copy()

    fighter_rows = prior_history.loc[
        prior_history["fighter_id"].astype(str) == str(fighter_id)
    ].copy()

    if fighter_rows.empty:
        raise OffensivePowerError(
            f"No prior round stats for fighter_id={fighter_id!r}"
        )

    numeric_columns = [
        "round",
        "kd",
        "sig_str_landed",
        "head_landed",
        "head_attempted",
        "ground_landed",
        "ground_attempted",
    ]

    for column in numeric_columns:
        prior_history[column] = pd.to_numeric(
            prior_history[column],
            errors="coerce",
        ).fillna(0.0)

        fighter_rows[column] = pd.to_numeric(
            fighter_rows[column],
            errors="coerce",
        ).fillna(0.0)

    prior_fight_count = int(
        fighter_rows["fight_id"].nunique()
    )

    if prior_fight_count <= 0:
        raise OffensivePowerError(
            "At least one prior fight is required"
        )

    def kd_per_sig(rows: pd.DataFrame) -> float:
        return _safe_ratio(
            rows["kd"].sum(),
            rows["sig_str_landed"].sum(),
        )

    def kd_per_fight(rows: pd.DataFrame) -> float:
        return _safe_ratio(
            rows["kd"].sum(),
            rows["fight_id"].nunique(),
        )

    def round1_kd_per_fight(rows: pd.DataFrame) -> float:
        round1 = rows.loc[rows["round"] == 1]
        return _safe_ratio(
            round1["kd"].sum(),
            rows["fight_id"].nunique(),
        )

    def head_accuracy(rows: pd.DataFrame) -> float:
        return _safe_ratio(
            rows["head_landed"].sum(),
            rows["head_attempted"].sum(),
        )

    def ground_accuracy(rows: pd.DataFrame) -> float:
        return _safe_ratio(
            rows["ground_landed"].sum(),
            rows["ground_attempted"].sum(),
        )

    def round1_sig_landed(rows: pd.DataFrame) -> float:
        round1 = rows.loc[rows["round"] == 1]
        return _safe_ratio(
            round1["sig_str_landed"].sum(),
            rows["fight_id"].nunique(),
        )

    builders = {
        "offensive_kd_per_sig_landed": kd_per_sig,
        "offensive_kd_per_fight": kd_per_fight,
        "round1_kd_per_fight": round1_kd_per_fight,
        "head_strike_accuracy": head_accuracy,
        "ground_strike_accuracy": ground_accuracy,
        "round1_sig_landed_per_fight": round1_sig_landed,
    }

    estimates: dict[str, ParameterEstimate] = {}

    for name, builder in builders.items():
        fighter_value = float(builder(fighter_rows))
        population_value = _historical_population_median(
            prior_history,
            value_builder=builder,
        )

        shrunk_value = _shrink(
            fighter_value,
            population_value,
            sample_size=prior_fight_count,
            prior_strength=prior_strength,
        )

        estimates[name] = ParameterEstimate(
            value=shrunk_value,
            source=(
                ProfileSource.FIGHTER
                if prior_fight_count >= 3
                else ProfileSource.GLOBAL
            ),
            effective_sample_size=float(prior_fight_count),
            uncertainty=float(
                1.0 / np.sqrt(prior_fight_count)
            ),
        )

    return estimates


def augment_profile_with_offensive_power(
    profile: FighterSimulationProfile,
    round_stats: pd.DataFrame,
    *,
    prior_strength: float = 5.0,
) -> FighterSimulationProfile:
    """Return a profile containing RFS and offensive power parameters."""

    power_estimates = build_offensive_power_estimates(
        round_stats,
        fighter_id=profile.fighter_id,
        target_date=profile.target_date,
        prior_strength=prior_strength,
    )

    parameters = dict(profile.parameters)
    parameters.update(power_estimates)

    return replace(
        profile,
        parameters=parameters,
    )
