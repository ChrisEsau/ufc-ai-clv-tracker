"""Tests for RFS simulator Phase Interaction feature contracts."""

from collections import Counter
from dataclasses import FrozenInstanceError, replace

import pytest

from pipeline.round_stats.rfs_phase_interaction_feature_contracts import (
    PHASE_INTERACTION_AGGREGATE_NAMES,
    PHASE_INTERACTION_AGGREGATE_SPECS,
    PHASE_INTERACTION_EVIDENCE_BY_NAME,
    PHASE_INTERACTION_EVIDENCE_SPECS,
    PHASE_INTERACTION_FIGHT_OBSERVATION_COLUMNS,
    PHASE_INTERACTION_PREFIX,
    PHASE_INTERACTION_TARGET_EVIDENCE,
    InteractionAggregateRule,
    PhaseInteractionFormula,
    validate_phase_interaction_feature_contracts,
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

    return PHASE_INTERACTION_AGGREGATE_SPECS[0]


def valid_evidence():
    """Return one known-valid evidence contract."""

    return PHASE_INTERACTION_EVIDENCE_SPECS[0]


def test_contract_validator_passes():
    validate_phase_interaction_feature_contracts()


def test_locked_registry_counts():
    assert len(PHASE_INTERACTION_AGGREGATE_SPECS) == 13
    assert len(PHASE_INTERACTION_AGGREGATE_NAMES) == 26
    assert len(PHASE_INTERACTION_EVIDENCE_SPECS) == 36
    assert len(PHASE_INTERACTION_FIGHT_OBSERVATION_COLUMNS) == 62
    assert len(PHASE_INTERACTION_TARGET_EVIDENCE) == 17


def test_aggregate_names_match_mirrored_contracts():
    expected = {
        feature_name
        for spec in PHASE_INTERACTION_AGGREGATE_SPECS
        for feature_name in (
            spec.feature_name,
            spec.opponent_feature_name,
        )
    }

    assert PHASE_INTERACTION_AGGREGATE_NAMES == expected


def test_aggregate_names_are_unique():
    names = [
        feature_name
        for spec in PHASE_INTERACTION_AGGREGATE_SPECS
        for feature_name in (
            spec.feature_name,
            spec.opponent_feature_name,
        )
    ]

    assert len(names) == len(set(names))


def test_opponent_aggregate_names_use_opp_marker():
    for spec in PHASE_INTERACTION_AGGREGATE_SPECS:
        assert spec.opponent_feature_name.startswith(
            f"{PHASE_INTERACTION_PREFIX}opp_"
        )


def test_evidence_names_are_unique():
    names = [
        spec.feature_name
        for spec in PHASE_INTERACTION_EVIDENCE_SPECS
    ]

    assert len(names) == len(set(names))
    assert len(PHASE_INTERACTION_EVIDENCE_BY_NAME) == len(names)


def test_evidence_lookup_references_registry_objects():
    for spec in PHASE_INTERACTION_EVIDENCE_SPECS:
        assert (
            PHASE_INTERACTION_EVIDENCE_BY_NAME[
                spec.feature_name
            ]
            is spec
        )


def test_aggregate_and_evidence_names_are_disjoint():
    evidence_names = {
        spec.feature_name
        for spec in PHASE_INTERACTION_EVIDENCE_SPECS
    }

    assert PHASE_INTERACTION_AGGREGATE_NAMES.isdisjoint(
        evidence_names
    )


def test_fight_observation_columns_match_contract_order():
    expected = tuple(
        [
            feature_name
            for spec in PHASE_INTERACTION_AGGREGATE_SPECS
            for feature_name in (
                spec.feature_name,
                spec.opponent_feature_name,
            )
        ]
        + [
            spec.feature_name
            for spec in PHASE_INTERACTION_EVIDENCE_SPECS
        ]
    )

    assert PHASE_INTERACTION_FIGHT_OBSERVATION_COLUMNS == expected


def test_fight_observation_columns_are_unique():
    assert len(PHASE_INTERACTION_FIGHT_OBSERVATION_COLUMNS) == len(
        set(PHASE_INTERACTION_FIGHT_OBSERVATION_COLUMNS)
    )


def test_all_feature_names_use_locked_prefix():
    for feature_name in PHASE_INTERACTION_FIGHT_OBSERVATION_COLUMNS:
        assert feature_name.startswith(
            PHASE_INTERACTION_PREFIX
        )


def test_observed_features_do_not_claim_30_second_exposure():
    for feature_name in PHASE_INTERACTION_FIGHT_OBSERVATION_COLUMNS:
        assert "30_seconds" not in feature_name


def test_aggregate_sources_are_authoritative():
    for spec in PHASE_INTERACTION_AGGREGATE_SPECS:
        assert (
            spec.source_column
            in AUTHORITATIVE_ROUND_SOURCE_COLUMNS
        )


def test_evidence_inputs_are_fight_aggregates():
    for spec in PHASE_INTERACTION_EVIDENCE_SPECS:
        assert spec.input_features

        for feature_name in spec.input_features:
            assert feature_name in PHASE_INTERACTION_AGGREGATE_NAMES


def test_reliability_inputs_are_fight_aggregates():
    for spec in PHASE_INTERACTION_EVIDENCE_SPECS:
        assert spec.reliability_features

        for feature_name in spec.reliability_features:
            assert feature_name in PHASE_INTERACTION_AGGREGATE_NAMES


def test_formula_counts_are_locked():
    counts = Counter(
        spec.formula
        for spec in PHASE_INTERACTION_EVIDENCE_SPECS
    )

    assert counts == {
        PhaseInteractionFormula.SAFE_RATIO: 7,
        PhaseInteractionFormula.COMPLEMENT_RATIO: 1,
        PhaseInteractionFormula.PHASE_ATTEMPT_SHARE: 6,
        PhaseInteractionFormula.NON_DISTANCE_ATTEMPT_SHARE: 2,
        PhaseInteractionFormula.SHARE_OF_COMBINED: 5,
        PhaseInteractionFormula.BALANCE_INDEX: 1,
        PhaseInteractionFormula.DIFFERENCE_PER_ROUND: 1,
        PhaseInteractionFormula.PER_OBSERVED_ROUND: 4,
        PhaseInteractionFormula.PER_CONTROL_MINUTE: 8,
        PhaseInteractionFormula.COMBINED_PER_CONTROL_MINUTE: 1,
    }


def test_every_formula_category_is_used():
    assert {
        spec.formula
        for spec in PHASE_INTERACTION_EVIDENCE_SPECS
    } == set(PhaseInteractionFormula)


def test_unit_interval_semantics_match_formula_categories():
    unit_interval_formulas = {
        PhaseInteractionFormula.SAFE_RATIO,
        PhaseInteractionFormula.COMPLEMENT_RATIO,
        PhaseInteractionFormula.PHASE_ATTEMPT_SHARE,
        PhaseInteractionFormula.NON_DISTANCE_ATTEMPT_SHARE,
        PhaseInteractionFormula.SHARE_OF_COMBINED,
        PhaseInteractionFormula.BALANCE_INDEX,
    }

    for spec in PHASE_INTERACTION_EVIDENCE_SPECS:
        assert spec.unit_interval is (
            spec.formula in unit_interval_formulas
        )


def test_unit_interval_feature_count_is_locked():
    assert sum(
        spec.unit_interval
        for spec in PHASE_INTERACTION_EVIDENCE_SPECS
    ) == 22


def test_target_evidence_matches_phase_interaction_targets():
    expected_targets = {
        spec.target_parameter
        for spec in SIMULATOR_TARGET_BY_NAME.values()
        if (
            spec.primary_family
            is SimulatorFeatureFamily.PHASE_INTERACTION
        )
    }

    assert set(PHASE_INTERACTION_TARGET_EVIDENCE) == expected_targets


def test_every_target_has_known_evidence():
    known_features = set(
        PHASE_INTERACTION_FIGHT_OBSERVATION_COLUMNS
    )

    for target, evidence_features in (
        PHASE_INTERACTION_TARGET_EVIDENCE.items()
    ):
        assert target in SIMULATOR_TARGET_BY_NAME
        assert evidence_features

        for feature_name in evidence_features:
            assert feature_name in known_features


def test_contract_descriptions_are_nonblank():
    for spec in (
        *PHASE_INTERACTION_AGGREGATE_SPECS,
        *PHASE_INTERACTION_EVIDENCE_SPECS,
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
        match="must use the Phase Interaction prefix",
    ):
        replace(
            valid_aggregate(),
            feature_name="rounds_observed",
        )


def test_opponent_feature_name_must_be_string():
    with pytest.raises(
        TypeError,
        match="opponent_feature_name must be a string",
    ):
        replace(
            valid_aggregate(),
            opponent_feature_name=None,
        )


def test_opponent_feature_name_requires_prefix():
    with pytest.raises(
        ValueError,
        match="must use the Phase Interaction prefix",
    ):
        replace(
            valid_aggregate(),
            opponent_feature_name="opp_rounds_observed",
        )


def test_opponent_feature_name_requires_opp_marker():
    with pytest.raises(
        ValueError,
        match="must use the opp_ marker",
    ):
        replace(
            valid_aggregate(),
            opponent_feature_name=(
                f"{PHASE_INTERACTION_PREFIX}"
                "other_rounds_observed"
            ),
        )


def test_aggregate_names_must_differ():
    with pytest.raises(
        ValueError,
        match="must differ",
    ):
        replace(
            valid_aggregate(),
            opponent_feature_name=(
                valid_aggregate().feature_name
            ),
        )


def test_aggregate_rule_requires_enum():
    with pytest.raises(
        TypeError,
        match="rule must be InteractionAggregateRule",
    ):
        replace(
            valid_aggregate(),
            rule="sum",
        )


def test_aggregate_source_column_must_be_string():
    with pytest.raises(
        TypeError,
        match="source_column must be a string",
    ):
        replace(
            valid_aggregate(),
            source_column=None,
        )


def test_aggregate_source_column_must_be_authoritative():
    with pytest.raises(
        ValueError,
        match="source_column is not authoritative",
    ):
        replace(
            valid_aggregate(),
            source_column="not_a_real_column",
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
        match="must use the Phase Interaction prefix",
    ):
        replace(
            valid_evidence(),
            feature_name="distance_accuracy",
        )


def test_evidence_name_rejects_false_30_second_claim():
    with pytest.raises(
        ValueError,
        match="cannot claim exact 30-second",
    ):
        replace(
            valid_evidence(),
            feature_name=(
                f"{PHASE_INTERACTION_PREFIX}"
                "distance_rate_per_30_seconds"
            ),
        )


def test_evidence_formula_requires_enum():
    with pytest.raises(
        TypeError,
        match="formula must be PhaseInteractionFormula",
    ):
        replace(
            valid_evidence(),
            formula="safe_ratio",
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
