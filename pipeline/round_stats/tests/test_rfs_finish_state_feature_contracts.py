"""Focused tests for Finish State feature contracts."""

from __future__ import annotations

import pytest

from pipeline.round_stats.rfs_finish_state_feature_contracts import (
    FINISH_STATE_AGGREGATE_NAMES,
    FINISH_STATE_AGGREGATE_SPECS,
    FINISH_STATE_EVIDENCE_BY_NAME,
    FINISH_STATE_EVIDENCE_SPECS,
    FINISH_STATE_FIGHT_OBSERVATION_COLUMNS,
    FINISH_STATE_PREFIX,
    FINISH_STATE_TARGET_EVIDENCE,
    FINISH_STATE_TARGETS,
    FinishAggregatePerspective,
    FinishAggregateRule,
    FinishSourceKind,
    FinishStateAggregateSpec,
    FinishStateEvidenceSpec,
    FinishStateFormula,
    validate_finish_state_contracts,
)
from pipeline.round_stats.rfs_simulator_feature_contracts import (
    SIMULATOR_TARGET_BY_NAME,
    SimulatorFeatureFamily,
)


def test_finish_state_target_coverage_is_exact() -> None:
    """The family must cover exactly its six registered targets."""

    expected_targets = {
        target_name
        for target_name, target_spec
        in SIMULATOR_TARGET_BY_NAME.items()
        if (
            target_spec.primary_family
            is SimulatorFeatureFamily.FINISH_STATE
        )
    }

    assert FINISH_STATE_TARGETS == expected_targets
    assert len(FINISH_STATE_TARGETS) == 6


def test_finish_state_contract_registry_validates() -> None:
    """The committed contract registry should validate cleanly."""

    validate_finish_state_contracts()


def test_observation_names_are_unique_and_current_fight_scoped() -> None:
    """All current observations must be unique and fight-scoped."""

    assert (
        len(FINISH_STATE_FIGHT_OBSERVATION_COLUMNS)
        == len(set(FINISH_STATE_FIGHT_OBSERVATION_COLUMNS))
    )

    assert all(
        name.startswith(FINISH_STATE_PREFIX)
        for name in FINISH_STATE_FIGHT_OBSERVATION_COLUMNS
    )

    assert all(
        "_fight_" in name
        for name in FINISH_STATE_FIGHT_OBSERVATION_COLUMNS
    )


def test_aggregate_and_evidence_counts_are_stable() -> None:
    """Lock the initial Finish State contract surface."""

    assert len(FINISH_STATE_AGGREGATE_SPECS) == 20
    assert len(FINISH_STATE_AGGREGATE_NAMES) == 20
    assert len(FINISH_STATE_EVIDENCE_SPECS) == 14
    assert len(FINISH_STATE_EVIDENCE_BY_NAME) == 14

    assert len(
        FINISH_STATE_FIGHT_OBSERVATION_COLUMNS
    ) == 34


def test_all_target_evidence_names_exist() -> None:
    """Every mapped target feature must exist in the registry."""

    observation_names = set(
        FINISH_STATE_FIGHT_OBSERVATION_COLUMNS
    )

    for evidence_names in FINISH_STATE_TARGET_EVIDENCE.values():
        assert evidence_names
        assert set(evidence_names).issubset(
            observation_names
        )


def test_outcome_aggregates_are_explicitly_typed() -> None:
    """Outcome observations must use the outcome source contract."""

    outcome_specs = [
        spec
        for spec in FINISH_STATE_AGGREGATE_SPECS
        if spec.source_kind is FinishSourceKind.OUTCOME
    ]

    assert len(outcome_specs) == 1

    for spec in outcome_specs:
        assert spec.rule is FinishAggregateRule.OUTCOME_FLAG
        assert (
            spec.perspective
            is FinishAggregatePerspective.FIGHT
        )


def test_outcome_evidence_requires_valid_outcome() -> None:
    """Outcome-derived evidence cannot include invalid outcomes."""

    outcome_formulas = {
        FinishStateFormula.OUTCOME_INDICATOR,
        FinishStateFormula.SURVIVAL_INDICATOR,
    }

    for spec in FINISH_STATE_EVIDENCE_SPECS:
        if spec.formula in outcome_formulas:
            assert spec.requires_valid_outcome is True


def test_preservation_evidence_requires_adversity() -> None:
    """Immediate resistance evidence needs adversity exposure."""

    preservation_formulas = {
        FinishStateFormula.SAME_ROUND_OUTPUT_PRESERVATION,
        FinishStateFormula.SAME_ROUND_EFFICIENCY_PRESERVATION,
    }

    for spec in FINISH_STATE_EVIDENCE_SPECS:
        if spec.formula in preservation_formulas:
            assert spec.requires_adversity is True


def test_contracts_do_not_claim_exact_segment_exposure() -> None:
    """Observed evidence must not claim exact 30-second exposure."""

    assert all(
        "30_seconds" not in name
        for name in FINISH_STATE_FIGHT_OBSERVATION_COLUMNS
    )


def test_invalid_round_aggregate_rejects_unknown_source() -> None:
    """Round aggregates must use authoritative source columns."""

    with pytest.raises(
        ValueError,
        match="round source_column is not authoritative",
    ):
        FinishStateAggregateSpec(
            feature_name="rfs_finish_state_fight_invalid",
            rule=FinishAggregateRule.SUM,
            source_kind=FinishSourceKind.ROUND,
            source_column="not_a_real_round_column",
            perspective=FinishAggregatePerspective.FIGHTER,
            description="Invalid source test.",
        )


def test_invalid_outcome_aggregate_rejects_wrong_perspective() -> None:
    """Outcome aggregates must be fight-scoped."""

    with pytest.raises(
        ValueError,
        match="outcome observations require fight perspective",
    ):
        FinishStateAggregateSpec(
            feature_name=(
                "rfs_finish_state_fight_invalid_outcome"
            ),
            rule=FinishAggregateRule.OUTCOME_FLAG,
            source_kind=FinishSourceKind.OUTCOME,
            source_column="winner_id",
            perspective=FinishAggregatePerspective.FIGHTER,
            description="Invalid outcome perspective test.",
        )


def test_invalid_evidence_rejects_wrong_prefix() -> None:
    """Evidence names must use the Finish State prefix."""

    with pytest.raises(
        ValueError,
        match="must use the Finish State prefix",
    ):
        FinishStateEvidenceSpec(
            feature_name="invalid_finish_state_feature",
            formula=FinishStateFormula.SAFE_RATIO,
            input_features=("a", "b"),
            reliability_features=("b",),
            unit_interval=True,
            requires_adversity=False,
            requires_valid_outcome=False,
            description="Invalid naming test.",
        )


def test_outcome_formula_rejects_missing_valid_outcome_flag() -> None:
    """Outcome evidence must explicitly require valid outcomes."""

    with pytest.raises(
        ValueError,
        match="must require a valid outcome",
    ):
        FinishStateEvidenceSpec(
            feature_name=(
                "rfs_finish_state_fight_invalid_outcome_evidence"
            ),
            formula=FinishStateFormula.OUTCOME_INDICATOR,
            input_features=("winner_id", "method"),
            reliability_features=(
                "rfs_finish_state_fight_valid_outcome",
            ),
            unit_interval=True,
            requires_adversity=False,
            requires_valid_outcome=False,
            description="Invalid outcome evidence test.",
        )
