"""Leakage-safe fighter profile construction for RFS Monte Carlo V1.

Phase 2 scope only:
- read pre-fight RFS state rows
- map approved RFS fields into profile parameters
- preserve provenance and uncertainty
- reject target-date leakage

No simulation mechanics are implemented here.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping

import pandas as pd

from pipeline.simulation.rfs_mc_v1.contracts import (
    FighterSimulationProfile,
    ParameterEstimate,
    ProfileSource,
)


class ProfileBuilderError(ValueError):
    """Raised when a fighter profile cannot be built safely."""


def _normalize_date(value: Any) -> pd.Timestamp:
    """Convert a supported date-like value into a normalized timestamp."""

    if isinstance(value, (str, date, datetime, pd.Timestamp)):
        timestamp = pd.Timestamp(value)
    else:
        raise ProfileBuilderError(f"Unsupported date value: {value!r}")

    if pd.isna(timestamp):
        raise ProfileBuilderError("Date cannot be null")

    return timestamp.normalize()


def select_latest_prior_row(
    history: pd.DataFrame,
    *,
    fighter_id: str,
    target_date: Any,
) -> pd.Series:
    """Select the latest fighter-state row strictly before the target date."""

    required = {"fighter_id", "date"}
    missing = required - set(history.columns)
    if missing:
        raise ProfileBuilderError(
            f"History is missing required columns: {sorted(missing)}"
        )

    target_ts = _normalize_date(target_date)

    fighter_rows = history.loc[history["fighter_id"] == fighter_id].copy()
    if fighter_rows.empty:
        raise ProfileBuilderError(
            f"No history rows found for fighter_id={fighter_id!r}"
        )

    fighter_rows["date"] = pd.to_datetime(
        fighter_rows["date"],
        errors="coerce",
    ).dt.normalize()

    fighter_rows = fighter_rows.loc[
        fighter_rows["date"].notna()
        & (fighter_rows["date"] < target_ts)
    ]

    if fighter_rows.empty:
        raise ProfileBuilderError(
            f"No prior state exists for fighter_id={fighter_id!r} "
            f"before target_date={target_ts.date()}"
        )

    fighter_rows = fighter_rows.sort_values(
        ["date", "fight_id"],
        kind="stable",
    )

    return fighter_rows.iloc[-1]


def make_parameter_estimate(
    *,
    value: Any,
    prior_valid_count: int,
    fighter_minimum: int = 3,
) -> ParameterEstimate:
    """Create a parameter estimate with simple V1 provenance rules.

    This is an initial contract-level implementation. Hierarchical subgroup
    fallback values will be added after the fallback datasets are approved.
    """

    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        raise ProfileBuilderError("Parameter value is missing or non-numeric")

    if prior_valid_count < 0:
        raise ProfileBuilderError("prior_valid_count cannot be negative")

    if prior_valid_count >= fighter_minimum:
        source = ProfileSource.FIGHTER
    else:
        source = ProfileSource.GLOBAL

    uncertainty = 1.0 / max(float(prior_valid_count), 1.0) ** 0.5

    return ParameterEstimate(
        value=float(numeric_value),
        source=source,
        effective_sample_size=float(prior_valid_count),
        uncertainty=uncertainty,
    )


def build_profile_from_row(
    row: Mapping[str, Any],
    *,
    target_date: Any,
    scheduled_rounds: int,
    weight_class: str | None,
    gender: str | None,
    parameter_map: Mapping[str, str],
    prior_fight_count_column: str,
    prior_valid_count_column: str,
) -> FighterSimulationProfile:
    """Build one immutable fighter profile from a point-in-time state row."""

    required = {
        "fighter_id",
        "fighter_name",
        "date",
        prior_fight_count_column,
        prior_valid_count_column,
        *parameter_map.values(),
    }

    missing = required - set(row.keys())
    if missing:
        raise ProfileBuilderError(
            f"State row is missing required fields: {sorted(missing)}"
        )

    target_ts = _normalize_date(target_date)
    state_date = _normalize_date(row["date"])

    if state_date >= target_ts:
        raise ProfileBuilderError(
            "State row must be strictly earlier than target_date"
        )

    prior_fight_count = int(row[prior_fight_count_column])
    prior_valid_count = int(row[prior_valid_count_column])

    parameters = {
        parameter_name: make_parameter_estimate(
            value=row[source_column],
            prior_valid_count=prior_valid_count,
        )
        for parameter_name, source_column in parameter_map.items()
    }

    return FighterSimulationProfile(
        fighter_id=str(row["fighter_id"]),
        fighter_name=str(row["fighter_name"]),
        target_date=str(target_ts.date()),
        weight_class=weight_class,
        gender=gender,
        scheduled_rounds=scheduled_rounds,
        prior_fight_count=prior_fight_count,
        valid_round_fight_count=prior_valid_count,
        parameters=parameters,
        is_low_experience=prior_fight_count < 3,
    )


def build_profile_from_history(
    history: pd.DataFrame,
    *,
    fighter_id: str,
    target_date: Any,
    scheduled_rounds: int,
    weight_class: str | None,
    gender: str | None,
    parameter_map: Mapping[str, str],
    prior_fight_count_column: str,
    prior_valid_count_column: str,
) -> FighterSimulationProfile:
    """Select the latest prior row and build a fighter profile."""

    row = select_latest_prior_row(
        history,
        fighter_id=fighter_id,
        target_date=target_date,
    )

    return build_profile_from_row(
        row,
        target_date=target_date,
        scheduled_rounds=scheduled_rounds,
        weight_class=weight_class,
        gender=gender,
        parameter_map=parameter_map,
        prior_fight_count_column=prior_fight_count_column,
        prior_valid_count_column=prior_valid_count_column,
    )
