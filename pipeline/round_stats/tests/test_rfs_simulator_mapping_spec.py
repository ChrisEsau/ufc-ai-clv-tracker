"""Tests for the authoritative RFS simulator mapping specification."""

from collections import Counter
from dataclasses import FrozenInstanceError, replace

import pytest

from pipeline.round_stats.rfs_simulator_feature_contracts import (
    MappingSupportLevel,
    SIMULATOR_TARGET_BY_NAME,
)
from pipeline.round_stats.rfs_simulator_mapping_spec import (
    AUTHORITATIVE_ROUND_SOURCE_COLUMNS,
    AUTHORITATIVE_ROUND_STATS_PATH,
    DIVISION_GLOBAL,
    MappingFallbackLevel,
    MappingSampleBasis,
    MappingShrinkageBasis,
    MappingTransformation,
    OpponentInteractionMode,
    SIMULATOR_TARGET_MAPPING_BY_NAME,
    SIMULATOR_TARGET_MAPPING_SPECS,
    validate_simulator_mapping_registry,
)


def valid_spec():
    """Return one known-valid latent mapping specification."""

    return SIMULATOR_TARGET_MAPPING_SPECS[0]


def test_authoritative_source_path_is_locked():
    assert (
        AUTHORITATIVE_ROUND_STATS_PATH
        == "data/fight_details/ufc_round_stats.parquet"
    )


def test_mapping_registry_contains_exactly_37_targets():
    assert len(SIMULATOR_TARGET_MAPPING_SPECS) == 37
    assert len(SIMULATOR_TARGET_MAPPING_BY_NAME) == 37


def test_mapping_registry_exactly_matches_target_registry():
    assert {
        spec.target_parameter
        for spec in SIMULATOR_TARGET_MAPPING_SPECS
    } == set(SIMULATOR_TARGET_BY_NAME)


def test_mapping_lookup_references_registry_objects():
    for spec in SIMULATOR_TARGET_MAPPING_SPECS:
        assert (
            SIMULATOR_TARGET_MAPPING_BY_NAME[
                spec.target_parameter
            ]
            is spec
        )


def test_mapping_registry_validator_passes():
    validate_simulator_mapping_registry()


def test_mapping_support_counts_match_locked_contract():
    counts = Counter(
        spec.support_level
        for spec in SIMULATOR_TARGET_MAPPING_SPECS
    )

    assert counts == {
        MappingSupportLevel.DIRECT_OBSERVED: 6,
        MappingSupportLevel.DERIVED_OBSERVED: 10,
        MappingSupportLevel.LATENT_CALIBRATED: 21,
    }


def test_mapping_support_matches_target_registry():
    for spec in SIMULATOR_TARGET_MAPPING_SPECS:
        assert (
            spec.support_level
            is SIMULATOR_TARGET_BY_NAME[
                spec.target_parameter
            ].support_level
        )


def test_all_declared_columns_are_authoritative():
    for spec in SIMULATOR_TARGET_MAPPING_SPECS:
        declared_columns = (
            spec.source_columns
            + spec.opponent_source_columns
        )

        assert declared_columns

        for column in declared_columns:
            assert column in AUTHORITATIVE_ROUND_SOURCE_COLUMNS


def test_source_column_tuples_do_not_contain_duplicates():
    for spec in SIMULATOR_TARGET_MAPPING_SPECS:
        assert len(spec.source_columns) == len(
            set(spec.source_columns)
        )
        assert len(spec.opponent_source_columns) == len(
            set(spec.opponent_source_columns)
        )


def test_every_mapping_uses_locked_fallback_hierarchy():
    for spec in SIMULATOR_TARGET_MAPPING_SPECS:
        assert spec.fallback_hierarchy == DIVISION_GLOBAL


def test_all_latent_mappings_require_calibration():
    for spec in SIMULATOR_TARGET_MAPPING_SPECS:
        if (
            spec.support_level
            is MappingSupportLevel.LATENT_CALIBRATED
        ):
            assert spec.requires_calibration is True


def test_noncalibrated_mappings_are_directly_observed():
    for spec in SIMULATOR_TARGET_MAPPING_SPECS:
        if not spec.requires_calibration:
            assert (
                spec.support_level
                is MappingSupportLevel.DIRECT_OBSERVED
            )


def test_outcome_join_targets_are_explicitly_locked():
    outcome_join_targets = {
        spec.target_parameter
        for spec in SIMULATOR_TARGET_MAPPING_SPECS
        if spec.requires_outcome_join
    }

    assert outcome_join_targets == {
        "phase.ground_defender.submission_defense",
        "dynamic.damage_resistance",
    }


def test_outcome_join_targets_require_calibration():
    for spec in SIMULATOR_TARGET_MAPPING_SPECS:
        if spec.requires_outcome_join:
            assert spec.requires_calibration is True


def test_all_transformation_categories_are_used():
    assert {
        spec.transformation
        for spec in SIMULATOR_TARGET_MAPPING_SPECS
    } == set(MappingTransformation)


def test_all_opponent_interaction_modes_are_used():
    assert {
        spec.opponent_interaction
        for spec in SIMULATOR_TARGET_MAPPING_SPECS
    } == set(OpponentInteractionMode)


def test_all_sample_bases_are_used():
    assert {
        spec.sample_basis
        for spec in SIMULATOR_TARGET_MAPPING_SPECS
    } == set(MappingSampleBasis)


def test_required_shrinkage_bases_are_used():
    used = {
        spec.shrinkage_basis
        for spec in SIMULATOR_TARGET_MAPPING_SPECS
    }

    assert MappingShrinkageBasis.OPPORTUNITY_COUNT in used
    assert MappingShrinkageBasis.EFFECTIVE_EXPOSURE in used
    assert MappingShrinkageBasis.HIERARCHICAL_COMPOSITE in used


def test_mapping_spec_is_immutable():
    with pytest.raises(FrozenInstanceError):
        valid_spec().target_parameter = "changed"


def test_unknown_target_is_rejected():
    with pytest.raises(
        ValueError,
        match="target_parameter is not registered",
    ):
        replace(
            valid_spec(),
            target_parameter="transition.unknown",
        )


def test_support_mismatch_is_rejected():
    with pytest.raises(
        ValueError,
        match="support_level does not match",
    ):
        replace(
            valid_spec(),
            support_level=(
                MappingSupportLevel.DIRECT_OBSERVED
            ),
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "source_columns",
        "opponent_source_columns",
    ],
)
def test_source_columns_must_be_tuples(field_name):
    with pytest.raises(TypeError, match="must be a tuple"):
        replace(
            valid_spec(),
            **{field_name: ["round"]},
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "source_columns",
        "opponent_source_columns",
    ],
)
def test_duplicate_source_columns_are_rejected(field_name):
    with pytest.raises(
        ValueError,
        match="cannot contain duplicates",
    ):
        replace(
            valid_spec(),
            **{field_name: ("round", "round")},
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "source_columns",
        "opponent_source_columns",
    ],
)
def test_unknown_source_columns_are_rejected(field_name):
    with pytest.raises(
        ValueError,
        match="unknown round source column",
    ):
        replace(
            valid_spec(),
            **{field_name: ("not_a_real_column",)},
        )


def test_mapping_requires_at_least_one_source_column():
    with pytest.raises(
        ValueError,
        match="at least one source column",
    ):
        replace(
            valid_spec(),
            source_columns=(),
            opponent_source_columns=(),
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("transformation", "latent_phase_process"),
        ("opponent_interaction", "none"),
        ("sample_basis", "prior_rounds"),
        ("shrinkage_basis", "effective_exposure"),
    ],
)
def test_mapping_enum_fields_require_enum_values(
    field_name,
    invalid_value,
):
    with pytest.raises(TypeError):
        replace(
            valid_spec(),
            **{field_name: invalid_value},
        )


def test_fallback_hierarchy_must_be_tuple():
    with pytest.raises(
        TypeError,
        match="fallback_hierarchy must be a tuple",
    ):
        replace(
            valid_spec(),
            fallback_hierarchy=[
                MappingFallbackLevel.GLOBAL
            ],
        )


def test_fallback_hierarchy_cannot_be_empty():
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        replace(
            valid_spec(),
            fallback_hierarchy=(),
        )


def test_fallback_hierarchy_rejects_duplicates():
    with pytest.raises(
        ValueError,
        match="cannot contain duplicates",
    ):
        replace(
            valid_spec(),
            fallback_hierarchy=(
                MappingFallbackLevel.GLOBAL,
                MappingFallbackLevel.GLOBAL,
            ),
        )


def test_fallback_hierarchy_must_end_with_global():
    with pytest.raises(
        ValueError,
        match="must end with GLOBAL",
    ):
        replace(
            valid_spec(),
            fallback_hierarchy=(
                MappingFallbackLevel.DIVISION,
            ),
        )


def test_fallback_hierarchy_requires_enum_values():
    with pytest.raises(
        TypeError,
        match="MappingFallbackLevel values",
    ):
        replace(
            valid_spec(),
            fallback_hierarchy=("global",),
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("requires_calibration", 1),
        ("requires_outcome_join", 0),
    ],
)
def test_boolean_fields_require_exact_booleans(
    field_name,
    invalid_value,
):
    with pytest.raises(TypeError, match="must be boolean"):
        replace(
            valid_spec(),
            **{field_name: invalid_value},
        )


def test_latent_mapping_cannot_disable_calibration():
    with pytest.raises(
        ValueError,
        match="latent mappings must require calibration",
    ):
        replace(
            valid_spec(),
            requires_calibration=False,
        )


def test_rationale_must_be_string():
    with pytest.raises(
        TypeError,
        match="rationale must be a string",
    ):
        replace(
            valid_spec(),
            rationale=None,
        )


@pytest.mark.parametrize(
    "invalid_rationale",
    ["", "   "],
)
def test_rationale_cannot_be_blank(invalid_rationale):
    with pytest.raises(
        ValueError,
        match="rationale cannot be empty",
    ):
        replace(
            valid_spec(),
            rationale=invalid_rationale,
        )
