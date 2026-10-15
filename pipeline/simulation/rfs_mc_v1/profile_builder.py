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

import numpy as np
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


def select_latest_prior_rows_by_family(
    histories: Mapping[str, pd.DataFrame],
    *,
    fighter_id: str,
    target_date: Any,
) -> dict[str, pd.Series]:
    """Select one strictly prior state row from each required RFS family."""

    if not histories:
        raise ProfileBuilderError("At least one RFS history is required")

    selected: dict[str, pd.Series] = {}

    for family_name, history in histories.items():
        try:
            selected[family_name] = select_latest_prior_row(
                history,
                fighter_id=fighter_id,
                target_date=target_date,
            )
        except ProfileBuilderError as exc:
            raise ProfileBuilderError(
                f"Unable to select {family_name!r} state: {exc}"
            ) from exc

    return selected


def make_historical_fallback_estimate(
    history: pd.DataFrame,
    *,
    source_column: str,
    target_date: Any,
) -> ParameterEstimate:
    """Build a leakage-safe fallback from prior historical family values.

    This is used only when an individual fighter parameter is missing or
    nonnumeric despite an otherwise valid prior RFS row.
    """

    if source_column not in history.columns:
        raise ProfileBuilderError(
            f"Fallback source column {source_column!r} is missing"
        )

    if "date" not in history.columns:
        raise ProfileBuilderError(
            "Fallback history requires a date column"
        )

    target_ts = _normalize_date(target_date)

    prior = history.loc[
        pd.to_datetime(history["date"]) < target_ts,
        source_column,
    ]

    numeric = pd.to_numeric(
        prior,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        raise ProfileBuilderError(
            f"No prior fallback values available for "
            f"{source_column!r}"
        )

    value = float(numeric.median())
    sample_size = float(len(numeric))

    return ParameterEstimate(
        value=value,
        source=ProfileSource.GLOBAL,
        effective_sample_size=sample_size,
        uncertainty=float(
            1.0 / np.sqrt(max(sample_size, 1.0))
        ),
    )



def build_composite_profile_from_histories(
    histories: Mapping[str, pd.DataFrame],
    *,
    fighter_id: str,
    target_date: Any,
    scheduled_rounds: int,
    weight_class: str | None,
    gender: str | None,
    parameter_definitions: Any,
) -> FighterSimulationProfile:
    """Build one profile by combining approved parameters across RFS families.

    Each family independently selects its latest row strictly before the target
    date. This prevents a future or target-fight row from entering the profile.
    """

    selected_rows = select_latest_prior_rows_by_family(
        histories,
        fighter_id=fighter_id,
        target_date=target_date,
    )

    parameters: dict[str, ParameterEstimate] = {}
    prior_fight_counts: list[int] = []
    valid_counts: list[int] = []
    fighter_names: set[str] = set()

    for definition in parameter_definitions:
        family_key = definition.family.value

        if family_key not in selected_rows:
            raise ProfileBuilderError(
                f"Missing history for RFS family {family_key!r}"
            )

        row = selected_rows[family_key]

        required = {
            "fighter_id",
            "fighter_name",
            "date",
            definition.source_column,
            definition.prior_fight_count_column,
            definition.prior_valid_count_column,
        }

        if definition.fallback_source_column is not None:
            required.add(definition.fallback_source_column)

        missing = required - set(row.index)
        if missing:
            raise ProfileBuilderError(
                f"Family {family_key!r} is missing fields: "
                f"{sorted(missing)}"
            )

        if str(row["fighter_id"]) != fighter_id:
            raise ProfileBuilderError(
                f"Family {family_key!r} returned the wrong fighter"
            )

        state_date = _normalize_date(row["date"])
        target_ts = _normalize_date(target_date)

        if state_date >= target_ts:
            raise ProfileBuilderError(
                f"Family {family_key!r} contains target-date leakage"
            )

        prior_fight_count = int(row[definition.prior_fight_count_column])
        prior_valid_count = int(row[definition.prior_valid_count_column])

        prior_fight_counts.append(prior_fight_count)
        valid_counts.append(prior_valid_count)
        fighter_names.add(str(row["fighter_name"]))

        raw_value = row[definition.source_column]
        numeric_value = pd.to_numeric(
            pd.Series([raw_value]),
            errors="coerce",
        ).iloc[0]

        # Last-3 variants first fall back to this fighter's EWM value.
        if (
            pd.isna(numeric_value)
            and definition.fallback_source_column is not None
        ):
            fallback_raw_value = row[
                definition.fallback_source_column
            ]
            numeric_value = pd.to_numeric(
                pd.Series([fallback_raw_value]),
                errors="coerce",
            ).iloc[0]

        # If both fighter-level values are missing, use the existing
        # leakage-safe historical median.
        if pd.isna(numeric_value):
            historical_source_column = (
                definition.fallback_source_column
                or definition.source_column
            )

            parameters[definition.name] = (
                make_historical_fallback_estimate(
                    histories[family_key],
                    source_column=historical_source_column,
                    target_date=target_date,
                )
            )
        else:
            parameters[definition.name] = make_parameter_estimate(
                value=numeric_value,
                prior_valid_count=prior_valid_count,
            )

    if len(fighter_names) != 1:
        raise ProfileBuilderError(
            f"RFS families disagree on fighter name: {sorted(fighter_names)}"
        )

    # Use the minimum observed count across families so profile experience never
    # overstates the weakest required family.
    prior_fight_count = min(prior_fight_counts)
    valid_round_fight_count = min(valid_counts)

    return FighterSimulationProfile(
        fighter_id=fighter_id,
        fighter_name=next(iter(fighter_names)),
        target_date=str(_normalize_date(target_date).date()),
        weight_class=weight_class,
        gender=gender,
        scheduled_rounds=scheduled_rounds,
        prior_fight_count=prior_fight_count,
        valid_round_fight_count=valid_round_fight_count,
        parameters=parameters,
        is_low_experience=prior_fight_count < 3,
    )


def load_default_rfs_histories(
    *,
    feature_root: str = "data/features",
) -> dict[str, pd.DataFrame]:
    """Load the currently approved RFS history families."""

    root = pd.io.common.stringify_path(feature_root)

    paths = {
        "trajectory": (
            f"{root}/round_fighter_state_history.parquet"
        ),
        "opening_offense": (
            f"{root}/round_fighter_state_history.parquet"
        ),
        "suppression": (
            f"{root}/round_fighter_suppression_p0_2_history.parquet"
        ),
        "wrestling": (
            f"{root}/round_fighter_wrestling_p0_3_history.parquet"
        ),
        "defense": (
            f"{root}/round_fighter_defense_p1_4_history.parquet"
        ),
        "submission_results": (
            f"{root}/rfs_mc_v1_submission_history.parquet"
        ),
    }

    histories: dict[str, pd.DataFrame] = {}

    for family_name, path in paths.items():
        try:
            histories[family_name] = pd.read_parquet(path)
        except Exception as exc:
            raise ProfileBuilderError(
                f"Unable to load {family_name!r} history from {path}: {exc}"
            ) from exc

    return histories
