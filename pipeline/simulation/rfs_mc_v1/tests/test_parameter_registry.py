"""Tests for the RFS Monte Carlo V1 parameter registry."""

import pyarrow.parquet as pq

from pipeline.simulation.rfs_mc_v1.parameter_registry import (
    PARAMETER_DEFINITIONS_BY_NAME,
    PROFILE_PARAMETER_DEFINITIONS,
    RFSFamily,
    validate_parameter_registry,
)


ARTIFACTS = {
    RFSFamily.TRAJECTORY: (
        "data/features/round_fighter_state_history.parquet"
    ),
    RFSFamily.OPENING_OFFENSE: (
        "data/features/round_fighter_state_history.parquet"
    ),
    RFSFamily.SUPPRESSION: (
        "data/features/round_fighter_suppression_p0_2_history.parquet"
    ),
    RFSFamily.WRESTLING: (
        "data/features/round_fighter_wrestling_p0_3_history.parquet"
    ),
    RFSFamily.DEFENSE: (
        "data/features/round_fighter_defense_p1_4_history.parquet"
    ),
    RFSFamily.SUBMISSION_RESULTS: (
        "data/features/rfs_mc_v1_submission_history.parquet"
    ),
}


def test_parameter_registry_is_valid() -> None:
    validate_parameter_registry()


def test_parameter_lookup_contains_every_definition() -> None:
    assert len(PARAMETER_DEFINITIONS_BY_NAME) == len(
        PROFILE_PARAMETER_DEFINITIONS
    )


def test_registered_columns_exist_in_family_artifacts() -> None:
    schemas = {
        family: set(pq.read_schema(path).names)
        for family, path in ARTIFACTS.items()
    }

    missing = []

    for definition in PROFILE_PARAMETER_DEFINITIONS:
        required = {
            definition.source_column,
            definition.prior_fight_count_column,
            definition.prior_valid_count_column,
        }

        absent = required - schemas[definition.family]
        if absent:
            missing.append(
                (
                    definition.name,
                    definition.family.value,
                    sorted(absent),
                )
            )

    assert not missing, f"Missing registered RFS columns: {missing}"
