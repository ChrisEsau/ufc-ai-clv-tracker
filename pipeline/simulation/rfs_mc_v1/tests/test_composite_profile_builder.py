"""Tests for multi-family RFS profile construction."""

import pandas as pd
import pytest

from pipeline.simulation.rfs_mc_v1.parameter_registry import (
    PROFILE_PARAMETER_DEFINITIONS,
)
from pipeline.simulation.rfs_mc_v1.profile_builder import (
    ProfileBuilderError,
    build_composite_profile_from_histories,
)


def make_family_history(
    *,
    family: str,
    prior_fights: int = 4,
    valid_fights: int = 4,
) -> pd.DataFrame:
    common = {
        "fight_id": f"{family}-fight",
        "fighter_id": "fighter-a",
        "fighter_name": "Fighter A",
        "date": "2025-06-01",
    }

    row = dict(common)

    for definition in PROFILE_PARAMETER_DEFINITIONS:
        if definition.family.value != family:
            continue

        row[definition.source_column] = 0.25
        row[definition.prior_fight_count_column] = prior_fights
        row[definition.prior_valid_count_column] = valid_fights

    return pd.DataFrame([row])


def make_histories() -> dict[str, pd.DataFrame]:
    return {
        family: make_family_history(family=family)
        for family in (
            "trajectory",
            "opening_offense",
            "suppression",
            "wrestling",
            "defense",
            "submission_results",
        )
    }


def test_builds_all_registered_parameters() -> None:
    profile = build_composite_profile_from_histories(
        make_histories(),
        fighter_id="fighter-a",
        target_date="2026-01-01",
        scheduled_rounds=3,
        weight_class="Lightweight",
        gender="male",
        parameter_definitions=PROFILE_PARAMETER_DEFINITIONS,
    )

    assert len(profile.parameters) == len(PROFILE_PARAMETER_DEFINITIONS)
    assert profile.prior_fight_count == 4
    assert profile.valid_round_fight_count == 4
    assert profile.is_low_experience is False


def test_uses_minimum_family_experience_count() -> None:
    histories = make_histories()
    histories["suppression"] = make_family_history(
        family="suppression",
        prior_fights=2,
        valid_fights=1,
    )

    profile = build_composite_profile_from_histories(
        histories,
        fighter_id="fighter-a",
        target_date="2026-01-01",
        scheduled_rounds=3,
        weight_class="Lightweight",
        gender="male",
        parameter_definitions=PROFILE_PARAMETER_DEFINITIONS,
    )

    assert profile.prior_fight_count == 2
    assert profile.valid_round_fight_count == 1
    assert profile.is_low_experience is True


def test_missing_family_raises() -> None:
    histories = make_histories()
    histories.pop("defense")

    with pytest.raises(ProfileBuilderError, match="Missing history"):
        build_composite_profile_from_histories(
            histories,
            fighter_id="fighter-a",
            target_date="2026-01-01",
            scheduled_rounds=3,
            weight_class="Lightweight",
            gender="male",
            parameter_definitions=PROFILE_PARAMETER_DEFINITIONS,
        )


def test_target_date_row_is_rejected() -> None:
    histories = make_histories()
    histories["trajectory"]["date"] = "2026-01-01"

    with pytest.raises(ProfileBuilderError, match="No prior state"):
        build_composite_profile_from_histories(
            histories,
            fighter_id="fighter-a",
            target_date="2026-01-01",
            scheduled_rounds=3,
            weight_class="Lightweight",
            gender="male",
            parameter_definitions=PROFILE_PARAMETER_DEFINITIONS,
        )


def test_missing_parameter_uses_prior_historical_fallback() -> None:
    histories = make_histories()

    trajectory_definition = next(
        definition
        for definition in PROFILE_PARAMETER_DEFINITIONS
        if definition.family.value == "trajectory"
    )

    fighter_row = histories["trajectory"].copy()
    fighter_row.loc[
        0,
        trajectory_definition.source_column,
    ] = float("nan")

    fallback_row = fighter_row.copy()
    fallback_row.loc[0, "fighter_id"] = "fighter-b"
    fallback_row.loc[0, "fighter_name"] = "Fighter B"
    fallback_row.loc[0, "fight_id"] = "older-fallback"
    fallback_row.loc[0, "date"] = "2025-01-01"
    fallback_row.loc[
        0,
        trajectory_definition.source_column,
    ] = 0.75

    histories["trajectory"] = pd.concat(
        [fighter_row, fallback_row],
        ignore_index=True,
    )

    profile = build_composite_profile_from_histories(
        histories,
        fighter_id="fighter-a",
        target_date="2026-01-01",
        scheduled_rounds=3,
        weight_class="Lightweight",
        gender="male",
        parameter_definitions=PROFILE_PARAMETER_DEFINITIONS,
    )

    estimate = profile.parameters[
        trajectory_definition.name
    ]

    assert estimate.value == pytest.approx(0.75)
    assert estimate.source.value == "global"


def test_fallback_excludes_target_date_values() -> None:
    histories = make_histories()

    trajectory_definition = next(
        definition
        for definition in PROFILE_PARAMETER_DEFINITIONS
        if definition.family.value == "trajectory"
    )

    histories["trajectory"].loc[
        0,
        trajectory_definition.source_column,
    ] = float("nan")

    future_row = histories["trajectory"].copy()
    future_row.loc[0, "fighter_id"] = "fighter-b"
    future_row.loc[0, "fighter_name"] = "Fighter B"
    future_row.loc[0, "fight_id"] = "target-date-row"
    future_row.loc[0, "date"] = "2026-01-01"
    future_row.loc[
        0,
        trajectory_definition.source_column,
    ] = 999.0

    prior_row = future_row.copy()
    prior_row.loc[0, "fight_id"] = "prior-row"
    prior_row.loc[0, "date"] = "2025-01-01"
    prior_row.loc[
        0,
        trajectory_definition.source_column,
    ] = 0.50

    histories["trajectory"] = pd.concat(
        [
            histories["trajectory"],
            future_row,
            prior_row,
        ],
        ignore_index=True,
    )

    profile = build_composite_profile_from_histories(
        histories,
        fighter_id="fighter-a",
        target_date="2026-01-01",
        scheduled_rounds=3,
        weight_class="Lightweight",
        gender="male",
        parameter_definitions=PROFILE_PARAMETER_DEFINITIONS,
    )

    estimate = profile.parameters[
        trajectory_definition.name
    ]

    assert estimate.value == pytest.approx(0.50)
