"""Focused tests for Dynamic Response feature contracts."""

from __future__ import annotations

import pytest

from pipeline.round_stats.rfs_dynamic_response_feature_contracts import (
    DYNAMIC_RESPONSE_AGGREGATE_NAMES,
    DYNAMIC_RESPONSE_AGGREGATE_SPECS,
    DYNAMIC_RESPONSE_EVIDENCE_BY_NAME,
    DYNAMIC_RESPONSE_EVIDENCE_SPECS,
    DYNAMIC_RESPONSE_FIGHT_OBSERVATION_COLUMNS,
    DYNAMIC_RESPONSE_PREFIX,
    DYNAMIC_RESPONSE_TARGET_EVIDENCE,
    DYNAMIC_RESPONSE_TARGETS,
    DynamicAggregatePerspective,
    DynamicAggregateRule,
    DynamicResponseAggregateSpec,
    DynamicResponseEvidenceSpec,
    DynamicResponseFormula,
    validate_dynamic_response_contracts,
)
from pipeline.round_stats.rfs_simulator_feature_contracts import (
    SIMULATOR_TARGET_BY_NAME,
    SimulatorFeatureFamily,
)


def test_dynamic_response_target_coverage_is_exact() -> None:
    """The family must cover exactly its four registered targets."""

    expected_targets = {
        target_name
        for target_name, target_spec
        in SIMULATOR_TARGET_BY_NAME.items()
        if (
            target_spec.primary_family
            is SimulatorFeatureFamily.DYNAMIC_RESPONSE
        )
    }

    assert DYNAMIC_RESPONSE_TARGETS == expected_targets
    assert len(DYNAMIC_RESPONSE_TARGETS) == 4


def test_dynamic_response_contract_registry_validates() -> None:
    """The committed contract registry should validate cleanly."""

    validate_dynamic_response_contracts()


def test_observation_names_are_unique_and_current_fight_scoped() -> None:
    """All current observations must be unique and contain _fight_."""

    assert (
        len(DYNAMIC_RESPONSE_FIGHT_OBSERVATION_COLUMNS)
        == len(set(DYNAMIC_RESPONSE_FIGHT_OBSERVATION_COLUMNS))
    )

    assert all(
        name.startswith(DYNAMIC_RESPONSE_PREFIX)
        for name in DYNAMIC_RESPONSE_FIGHT_OBSERVATION_COLUMNS
    )

    assert all(
        "_fight_" in name
        for name in DYNAMIC_RESPONSE_FIGHT_OBSERVATION_COLUMNS
    )


def test_aggregate_and_evidence_counts_are_stable() -> None:
    """Lock the initial Dynamic Response contract surface."""

    assert len(DYNAMIC_RESPONSE_AGGREGATE_SPECS) == 12
    assert len(DYNAMIC_RESPONSE_AGGREGATE_NAMES) == 12
    assert len(DYNAMIC_RESPONSE_EVIDENCE_SPECS) == 24
    assert len(DYNAMIC_RESPONSE_EVIDENCE_BY_NAME) == 24

    assert len(
        DYNAMIC_RESPONSE_FIGHT_OBSERVATION_COLUMNS
    ) == 36


def test_all_target_evidence_names_exist() -> None:
    """Every mapped target feature must exist in the observation registry."""

    observation_names = set(
        DYNAMIC_RESPONSE_FIGHT_OBSERVATION_COLUMNS
    )

    for evidence_names in DYNAMIC_RESPONSE_TARGET_EVIDENCE.values():
        assert evidence_names
        assert set(evidence_names).issubset(
            observation_names
        )


def test_multiple_round_features_are_marked_explicitly() -> None:
    """Trajectory and rebound formulas require multiple rounds."""

    multi_round_formulas = {
        DynamicResponseFormula.FIRST_LAST_RATIO,
        DynamicResponseFormula.FIRST_LAST_DIFFERENCE,
        DynamicResponseFormula.OLS_ROUND_SLOPE,
        DynamicResponseFormula.LATE_EARLY_RATIO,
        DynamicResponseFormula.LATE_EARLY_DIFFERENCE,
        DynamicResponseFormula.EFFICIENCY_CHANGE,
        DynamicResponseFormula.POST_ADVERSITY_REBOUND,
        DynamicResponseFormula.POST_ADVERSITY_PRESERVATION,
    }

    for spec in DYNAMIC_RESPONSE_EVIDENCE_SPECS:
        if spec.formula in multi_round_formulas:
            assert spec.requires_multiple_rounds is True


def test_contracts_do_not_claim_exact_segment_exposure() -> None:
    """Observed round evidence must not claim exact 30-second rates."""

    assert all(
        "30_seconds" not in name
        for name in DYNAMIC_RESPONSE_FIGHT_OBSERVATION_COLUMNS
    )


def test_invalid_aggregate_contract_rejects_unknown_source() -> None:
    """Aggregate contracts must use authoritative source columns."""

    with pytest.raises(
        ValueError,
        match="not authoritative",
    ):
        DynamicResponseAggregateSpec(
            feature_name=(
                "rfs_dynamic_response_fight_invalid"
            ),
            rule=DynamicAggregateRule.SUM,
            source_column="not_a_real_round_column",
            perspective=DynamicAggregatePerspective.FIGHTER,
            description="Invalid source test.",
        )


def test_invalid_evidence_contract_rejects_wrong_prefix() -> None:
    """Evidence names must use the locked Dynamic Response prefix."""

    with pytest.raises(
        ValueError,
        match="must use the Dynamic Response prefix",
    ):
        DynamicResponseEvidenceSpec(
            feature_name="invalid_dynamic_response_feature",
            formula=DynamicResponseFormula.SAFE_RATIO,
            input_features=("a", "b"),
            reliability_features=("b",),
            unit_interval=True,
            requires_multiple_rounds=False,
            description="Invalid naming test.",
        )


def test_unit_interval_evidence_is_limited_to_ratios() -> None:
    """Only bounded ratio evidence should claim unit-interval semantics."""

    allowed_formulas = {
        DynamicResponseFormula.SAFE_RATIO,
    }

    for spec in DYNAMIC_RESPONSE_EVIDENCE_SPECS:
        if spec.unit_interval:
            assert spec.formula in allowed_formulas
