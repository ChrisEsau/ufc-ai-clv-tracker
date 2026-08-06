"""Tests for RFS simulator Phase Baseline feature contracts."""

from collections import Counter
from dataclasses import FrozenInstanceError, replace

import pytest

from pipeline.round_stats.rfs_phase_baseline_feature_contracts import (
    PHASE_BASELINE_AGGREGATE_NAMES,
    PHASE_BASELINE_AGGREGATE_SPECS,
    PHASE_BASELINE_EVIDENCE_BY_NAME,
    PHASE_BASELINE_EVIDENCE_SPECS,
    PHASE_BASELINE_FIGHT_OBSERVATION_COLUMNS,
    PHASE_BASELINE_PREFIX,
    PHASE_BASELINE_TARGET_EVIDENCE,
    FightAggregateRule,
    PhaseBaselineFormula,
    validate_phase_baseline_feature_contracts,
)
from pipeline.round_stats.rfs_simulator_feature_contracts import (
    SIMULATOR_TARGET_BY_NAME,
    SimulatorFeatureFamily,
)
from pipeline.round_stats.rfs_simulator_mapping_spec import (
    AUTHORITATIVE_ROUND_SOURCE_COLUMNS,
)


def valid_aggregate():
    """Return one known-valid aggregate contract."""

    return PHASE_BASELINE_AGGREGATE_SPECS[0]


def valid_evidence():
    """Return one known-valid derived evidence contract."""

    return PHASE_BASELINE_EVIDENCE_SPECS[0]


def test_contract_validator_passes():
    validate_phase_baseline_feature_contracts()


def test_locked_contract_counts():
    assert len(PHASE_BASELINE_AGGREGATE_SPECS) == 9
    assert len(PHASE_BASELINE_EVIDENCE_SPECS) == 19
    assert len(PHASE_BASELINE_FIGHT_OBSERVATION_COLUMNS) == 28
    assert len(PHASE_BASELINE_TARGET_EVIDENCE) == 10


def test_aggregate_names_are_unique():
    names = [
        spec.feature_name
        for spec in PHASE_BASELINE_AGGREGATE_SPECS
    ]

    assert len(names) == len(set(names))
    assert PHASE_BASELINE_AGGREGATE_NAMES == frozenset(names)


def test_evidence_names_are_unique():
    names = [
        spec.feature_name
        for spec in PHASE_BASELINE_EVIDENCE_SPECS
    ]

    assert len(names) == len(set(names))
    assert len(PHASE_BASELINE_EVIDENCE_BY_NAME) == len(names)


def test_evidence_lookup_references_registry_objects():
    for spec in PHASE_BASELINE_EVIDENCE_SPECS:
        assert (
            PHASE_BASELINE_EVIDENCE_BY_NAME[
                spec.feature_name
            ]
            is spec
        )


def test_aggregate_and_evidence_names_are_disjoint():
    aggregate_names = {
        spec.feature_name
        for spec in PHASE_BASELINE_AGGREGATE_SPECS
    }
    evidence_names = {
        spec.feature_name
        for spec in PHASE_BASELINE_EVIDENCE_SPECS
    }

    assert aggregate_names.isdisjoint(evidence_names)


def test_fight_observation_columns_are_unique():
    assert len(PHASE_BASELINE_FIGHT_OBSERVATION_COLUMNS) == len(
        set(PHASE_BASELINE_FIGHT_OBSERVATION_COLUMNS)
    )


def test_fight_observation_columns_match_contracts():
    expected = tuple(
        [
            spec.feature_name
            for spec in PHASE_BASELINE_AGGREGATE_SPECS
        ]
        + [
            spec.feature_name
            for spec in PHASE_BASELINE_EVIDENCE_SPECS
        ]
    )

    assert PHASE_BASELINE_FIGHT_OBSERVATION_COLUMNS == expected


def test_all_feature_names_use_locked_prefix():
    for feature_name in PHASE_BASELINE_FIGHT_OBSERVATION_COLUMNS:
        assert feature_name.startswith(PHASE_BASELINE_PREFIX)


def test_observed_features_do_not_claim_30_second_exposure():
    for feature_name in PHASE_BASELINE_FIGHT_OBSERVATION_COLUMNS:
        assert "30_seconds" not in feature_name


def test_locked_aggregate_names():
    assert PHASE_BASELINE_AGGREGATE_NAMES == {
        f"{PHASE_BASELINE_PREFIX}rounds_observed",
        f"{PHASE_BASELINE_PREFIX}sig_strike_attempts",
        f"{PHASE_BASELINE_PREFIX}distance_attempts",
        f"{PHASE_BASELINE_PREFIX}clinch_attempts",
        f"{PHASE_BASELINE_PREFIX}ground_attempts",
        f"{PHASE_BASELINE_PREFIX}td_attempts",
        f"{PHASE_BASELINE_PREFIX}td_landed",
        f"{PHASE_BASELINE_PREFIX}failed_td_attempts",
        f"{PHASE_BASELINE_PREFIX}control_seconds",
    }


def test_aggregate_sources_are_authoritative():
    for spec in PHASE_BASELINE_AGGREGATE_SPECS:
        assert spec.source_columns

        for column in spec.source_columns:
            assert column in AUTHORITATIVE_ROUND_SOURCE_COLUMNS


def test_evidence_inputs_are_known_features_or_raw_columns():
    known_features = set(
        PHASE_BASELINE_FIGHT_OBSERVATION_COLUMNS
    )

    for spec in PHASE_BASELINE_EVIDENCE_SPECS:
        for input_name in spec.input_features:
            assert (
                input_name in known_features
                or input_name
                in AUTHORITATIVE_ROUND_SOURCE_COLUMNS
            )


def test_reliability_inputs_are_fight_aggregates():
    for spec in PHASE_BASELINE_EVIDENCE_SPECS:
        assert spec.reliability_features

        for feature_name in spec.reliability_features:
            assert feature_name in PHASE_BASELINE_AGGREGATE_NAMES


def test_formula_counts_are_locked():
    counts = Counter(
        spec.formula
        for spec in PHASE_BASELINE_EVIDENCE_SPECS
    )

    assert counts == {
        PhaseBaselineFormula.PER_OBSERVED_ROUND: 6,
        PhaseBaselineFormula.SAFE_RATIO: 3,
        PhaseBaselineFormula.PHASE_ATTEMPT_SHARE: 3,
        PhaseBaselineFormula.NON_DISTANCE_PHASE_SHARE: 2,
        PhaseBaselineFormula.PER_CONTROL_MINUTE: 2,
        PhaseBaselineFormula.OLS_ROUND_SLOPE: 2,
        PhaseBaselineFormula.FIRST_LAST_PERSISTENCE_RATIO: 1,
    }


def test_every_formula_category_is_used():
    assert {
        spec.formula
        for spec in PHASE_BASELINE_EVIDENCE_SPECS
    } == set(PhaseBaselineFormula)


def test_unit_interval_features_are_locked():
    unit_interval_features = {
        spec.feature_name
        for spec in PHASE_BASELINE_EVIDENCE_SPECS
        if spec.unit_interval
    }

    assert unit_interval_features == {
        f"{PHASE_BASELINE_PREFIX}distance_attempt_share",
        f"{PHASE_BASELINE_PREFIX}clinch_attempt_share",
        f"{PHASE_BASELINE_PREFIX}ground_attempt_share",
        f"{PHASE_BASELINE_PREFIX}td_completion_rate",
        f"{PHASE_BASELINE_PREFIX}non_distance_clinch_share",
        f"{PHASE_BASELINE_PREFIX}non_distance_ground_share",
    }


def test_target_evidence_matches_phase_baseline_targets():
    expected_targets = {
        spec.target_parameter
        for spec in SIMULATOR_TARGET_BY_NAME.values()
        if (
            spec.primary_family
            is SimulatorFeatureFamily.PHASE_BASELINE
        )
    }

    assert set(PHASE_BASELINE_TARGET_EVIDENCE) == expected_targets


def test_every_target_has_known_evidence():
    known_features = set(
        PHASE_BASELINE_FIGHT_OBSERVATION_COLUMNS
    )

    for target, evidence_features in (
        PHASE_BASELINE_TARGET_EVIDENCE.items()
    ):
        assert target in SIMULATOR_TARGET_BY_NAME
        assert evidence_features

        for feature_name in evidence_features:
            assert feature_name in known_features


def test_contract_descriptions_are_nonblank():
    for spec in (
        *PHASE_BASELINE_AGGREGATE_SPECS,
        *PHASE_BASELINE_EVIDENCE_SPECS,
    ):
        assert spec.description.strip()


def test_aggregate_contract_is_immutable():
    with pytest.raises(FrozenInstanceError):
        valid_aggregate().feature_name = "changed"


def test_evidence_contract_is_immutable():
    with pytest.raises(FrozenInstanceError):
        valid_evidence().feature_name = "changed"


def test_aggregate_feature_name_must_be_string():
    with pytest.raises(
        TypeError,
        match="feature_name must be a string",
    ):
        replace(
            valid_aggregate(),
            feature_name=None,
        )


def test_aggregate_feature_name_requires_prefix():
    with pytest.raises(
        ValueError,
        match="must use Phase Baseline prefix",
    ):
        replace(
            valid_aggregate(),
            feature_name="rounds_observed",
        )


def test_aggregate_rule_requires_enum():
    with pytest.raises(
        TypeError,
        match="rule must be FightAggregateRule",
    ):
        replace(
            valid_aggregate(),
            rule="sum",
        )


def test_aggregate_source_columns_must_be_tuple():
    with pytest.raises(
        TypeError,
        match="source_columns must be a tuple",
    ):
        replace(
            valid_aggregate(),
            source_columns=["round"],
        )


def test_aggregate_source_columns_reject_duplicates():
    with pytest.raises(
        ValueError,
        match="cannot contain duplicates",
    ):
        replace(
            valid_aggregate(),
            source_columns=("round", "round"),
        )


def test_aggregate_source_columns_reject_unknown_column():
    with pytest.raises(
        ValueError,
        match="unknown round source column",
    ):
        replace(
            valid_aggregate(),
            source_columns=("not_a_real_column",),
        )


def test_aggregate_source_columns_cannot_be_empty():
    with pytest.raises(
        ValueError,
        match="source_columns cannot be empty",
    ):
        replace(
            valid_aggregate(),
            source_columns=(),
        )


@pytest.mark.parametrize(
    "rule",
    [
        FightAggregateRule.UNIQUE_COUNT,
        FightAggregateRule.SUM,
    ],
)
def test_single_column_aggregate_rules_require_one_column(
    rule,
):
    with pytest.raises(
        ValueError,
        match="requires exactly one source column",
    ):
        replace(
            valid_aggregate(),
            rule=rule,
            source_columns=("round", "td_attempted"),
        )


def test_difference_aggregate_requires_two_columns():
    with pytest.raises(
        ValueError,
        match="requires exactly two source columns",
    ):
        replace(
            valid_aggregate(),
            rule=(
                FightAggregateRule
                .SUM_DIFFERENCE_CLIPPED_ZERO
            ),
            source_columns=("td_attempted",),
        )


def test_evidence_feature_name_must_be_string():
    with pytest.raises(
        TypeError,
        match="feature_name must be a string",
    ):
        replace(
            valid_evidence(),
            feature_name=None,
        )


def test_evidence_feature_name_requires_prefix():
    with pytest.raises(
        ValueError,
        match="must use Phase Baseline prefix",
    ):
        replace(
            valid_evidence(),
            feature_name="distance_attempts_per_round",
        )


def test_evidence_name_rejects_false_30_second_claim():
    with pytest.raises(
        ValueError,
        match="cannot claim exact 30-second",
    ):
        replace(
            valid_evidence(),
            feature_name=(
                f"{PHASE_BASELINE_PREFIX}"
                "distance_rate_per_30_seconds"
            ),
        )


def test_evidence_formula_requires_enum():
    with pytest.raises(
        TypeError,
        match="formula must be PhaseBaselineFormula",
    ):
        replace(
            valid_evidence(),
            formula="per_observed_round",
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "input_features",
        "reliability_features",
    ],
)
def test_evidence_feature_collections_must_be_tuples(
    field_name,
):
    with pytest.raises(
        TypeError,
        match=f"{field_name} must be a tuple",
    ):
        replace(
            valid_evidence(),
            **{field_name: ["round"]},
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "input_features",
        "reliability_features",
    ],
)
def test_evidence_feature_collections_reject_duplicates(
    field_name,
):
    with pytest.raises(
        ValueError,
        match="cannot contain duplicates",
    ):
        replace(
            valid_evidence(),
            **{field_name: ("round", "round")},
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "input_features",
        "reliability_features",
    ],
)
def test_evidence_feature_collections_require_strings(
    field_name,
):
    with pytest.raises(
        TypeError,
        match="must contain strings",
    ):
        replace(
            valid_evidence(),
            **{field_name: (1,)},
        )


def test_evidence_requires_input_features():
    with pytest.raises(
        ValueError,
        match="input_features cannot be empty",
    ):
        replace(
            valid_evidence(),
            input_features=(),
        )


def test_evidence_requires_reliability_features():
    with pytest.raises(
        ValueError,
        match="reliability_features cannot be empty",
    ):
        replace(
            valid_evidence(),
            reliability_features=(),
        )


def test_unit_interval_requires_exact_boolean():
    with pytest.raises(
        TypeError,
        match="unit_interval must be boolean",
    ):
        replace(
            valid_evidence(),
            unit_interval=1,
        )


@pytest.mark.parametrize(
    "spec_factory",
    [
        valid_aggregate,
        valid_evidence,
    ],
)
def test_description_must_be_string(spec_factory):
    with pytest.raises(
        TypeError,
        match="description must be a string",
    ):
        replace(
            spec_factory(),
            description=None,
        )


@pytest.mark.parametrize(
    "spec_factory",
    [
        valid_aggregate,
        valid_evidence,
    ],
)
@pytest.mark.parametrize(
    "invalid_description",
    ["", "   "],
)
def test_description_cannot_be_blank(
    spec_factory,
    invalid_description,
):
    with pytest.raises(
        ValueError,
        match="description cannot be empty",
    ):
        replace(
            spec_factory(),
            description=invalid_description,
        )
