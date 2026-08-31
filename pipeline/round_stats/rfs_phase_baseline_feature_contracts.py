"""Observed feature contracts for the RFS simulator Phase Baseline family.

This module defines the observable fight-level evidence used later to estimate
the ten simulator targets assigned to the Phase Baseline family.

Important boundary
------------------
The UFCStats round source does not provide exact seconds spent at distance,
in the clinch, or on the ground. Therefore:

* observed rates are expressed per observed round or per recorded opportunity;
* phase shares are evidence about phase tendency, not exact phase exposure;
* no observed feature is presented as a final per-30-second simulator rate;
* clinch-versus-ground control allocation remains latent and calibrated;
* opportunity counts are preserved for reliability shrinkage.

This module contains contracts only. It does not build parquet artifacts,
calculate rolling states, or fit calibration coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pipeline.round_stats.rfs_simulator_feature_contracts import (
    SIMULATOR_TARGET_BY_NAME,
    SimulatorFeatureFamily,
)
from pipeline.round_stats.rfs_simulator_mapping_spec import (
    AUTHORITATIVE_ROUND_SOURCE_COLUMNS,
)


PHASE_BASELINE_PREFIX = "rfs_phase_base_fight_"


class FightAggregateRule(str, Enum):
    """How one fight-level opportunity value is aggregated from rounds."""

    UNIQUE_COUNT = "unique_count"
    SUM = "sum"
    SUM_DIFFERENCE_CLIPPED_ZERO = (
        "sum_difference_clipped_zero"
    )


class PhaseBaselineFormula(str, Enum):
    """Locked formula category for one derived evidence feature."""

    PER_OBSERVED_ROUND = "per_observed_round"
    SAFE_RATIO = "safe_ratio"
    PHASE_ATTEMPT_SHARE = "phase_attempt_share"
    NON_DISTANCE_PHASE_SHARE = "non_distance_phase_share"
    PER_CONTROL_MINUTE = "per_control_minute"
    OLS_ROUND_SLOPE = "ols_round_slope"
    FIRST_LAST_PERSISTENCE_RATIO = (
        "first_last_persistence_ratio"
    )


@dataclass(frozen=True)
class FightAggregateSpec:
    """One fight-level count or exposure aggregated from round rows."""

    feature_name: str
    rule: FightAggregateRule
    source_columns: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.feature_name, str):
            raise TypeError("feature_name must be a string")

        if not self.feature_name.startswith(
            PHASE_BASELINE_PREFIX
        ):
            raise ValueError(
                "feature_name must use Phase Baseline prefix"
            )

        if not isinstance(self.rule, FightAggregateRule):
            raise TypeError(
                "rule must be FightAggregateRule"
            )

        _validate_raw_source_columns(
            self.source_columns,
            field_name="source_columns",
        )

        if not self.source_columns:
            raise ValueError(
                "source_columns cannot be empty"
            )

        if (
            self.rule is FightAggregateRule.UNIQUE_COUNT
            and len(self.source_columns) != 1
        ):
            raise ValueError(
                "UNIQUE_COUNT requires exactly one source column"
            )

        if (
            self.rule is FightAggregateRule.SUM
            and len(self.source_columns) != 1
        ):
            raise ValueError(
                "SUM requires exactly one source column"
            )

        if (
            self.rule
            is FightAggregateRule.SUM_DIFFERENCE_CLIPPED_ZERO
            and len(self.source_columns) != 2
        ):
            raise ValueError(
                "SUM_DIFFERENCE_CLIPPED_ZERO requires "
                "exactly two source columns"
            )

        _validate_description(self.description)


@dataclass(frozen=True)
class PhaseBaselineEvidenceSpec:
    """One derived fight-level Phase Baseline evidence feature."""

    feature_name: str
    formula: PhaseBaselineFormula
    input_features: tuple[str, ...]
    reliability_features: tuple[str, ...]
    unit_interval: bool
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.feature_name, str):
            raise TypeError("feature_name must be a string")

        if not self.feature_name.startswith(
            PHASE_BASELINE_PREFIX
        ):
            raise ValueError(
                "feature_name must use Phase Baseline prefix"
            )

        if "30_seconds" in self.feature_name:
            raise ValueError(
                "observed evidence cannot claim exact "
                "30-second phase exposure"
            )

        if not isinstance(
            self.formula,
            PhaseBaselineFormula,
        ):
            raise TypeError(
                "formula must be PhaseBaselineFormula"
            )

        _validate_feature_tuple(
            self.input_features,
            field_name="input_features",
        )
        _validate_feature_tuple(
            self.reliability_features,
            field_name="reliability_features",
        )

        if not self.input_features:
            raise ValueError(
                "input_features cannot be empty"
            )

        if not self.reliability_features:
            raise ValueError(
                "reliability_features cannot be empty"
            )

        if type(self.unit_interval) is not bool:
            raise TypeError(
                "unit_interval must be boolean"
            )

        _validate_description(self.description)


def _validate_raw_source_columns(
    values: object,
    *,
    field_name: str,
) -> None:
    """Validate a tuple of authoritative round-source columns."""

    if not isinstance(values, tuple):
        raise TypeError(
            f"{field_name} must be a tuple"
        )

    if len(values) != len(set(values)):
        raise ValueError(
            f"{field_name} cannot contain duplicates"
        )

    for column in values:
        if not isinstance(column, str):
            raise TypeError(
                f"{field_name} must contain strings"
            )

        if column not in AUTHORITATIVE_ROUND_SOURCE_COLUMNS:
            raise ValueError(
                f"{field_name} contains unknown round "
                f"source column: {column}"
            )


def _validate_feature_tuple(
    values: object,
    *,
    field_name: str,
) -> None:
    """Validate one tuple of Phase Baseline feature names."""

    if not isinstance(values, tuple):
        raise TypeError(
            f"{field_name} must be a tuple"
        )

    if len(values) != len(set(values)):
        raise ValueError(
            f"{field_name} cannot contain duplicates"
        )

    for feature_name in values:
        if not isinstance(feature_name, str):
            raise TypeError(
                f"{field_name} must contain strings"
            )


def _validate_description(value: object) -> None:
    """Validate one human-readable contract description."""

    if not isinstance(value, str):
        raise TypeError(
            "description must be a string"
        )

    if not value.strip():
        raise ValueError(
            "description cannot be empty"
        )


def _name(suffix: str) -> str:
    """Return one locked current-fight Phase Baseline name."""

    return f"{PHASE_BASELINE_PREFIX}{suffix}"


# ---------------------------------------------------------------------
# Fight-level opportunity and exposure values
# ---------------------------------------------------------------------

PHASE_BASELINE_AGGREGATE_SPECS: tuple[
    FightAggregateSpec,
    ...,
] = (
    FightAggregateSpec(
        feature_name=_name("rounds_observed"),
        rule=FightAggregateRule.UNIQUE_COUNT,
        source_columns=("round",),
        description=(
            "Number of distinct recorded rounds for the fighter "
            "in the fight."
        ),
    ),
    FightAggregateSpec(
        feature_name=_name("sig_strike_attempts"),
        rule=FightAggregateRule.SUM,
        source_columns=("sig_str_attempted",),
        description=(
            "Total recorded significant-strike attempts."
        ),
    ),
    FightAggregateSpec(
        feature_name=_name("distance_attempts"),
        rule=FightAggregateRule.SUM,
        source_columns=("distance_attempted",),
        description=(
            "Total recorded distance significant-strike attempts."
        ),
    ),
    FightAggregateSpec(
        feature_name=_name("clinch_attempts"),
        rule=FightAggregateRule.SUM,
        source_columns=("clinch_attempted",),
        description=(
            "Total recorded clinch significant-strike attempts."
        ),
    ),
    FightAggregateSpec(
        feature_name=_name("ground_attempts"),
        rule=FightAggregateRule.SUM,
        source_columns=("ground_attempted",),
        description=(
            "Total recorded ground significant-strike attempts."
        ),
    ),
    FightAggregateSpec(
        feature_name=_name("td_attempts"),
        rule=FightAggregateRule.SUM,
        source_columns=("td_attempted",),
        description=(
            "Total recorded takedown attempts."
        ),
    ),
    FightAggregateSpec(
        feature_name=_name("td_landed"),
        rule=FightAggregateRule.SUM,
        source_columns=("td_landed",),
        description=(
            "Total recorded completed takedowns."
        ),
    ),
    FightAggregateSpec(
        feature_name=_name("failed_td_attempts"),
        rule=(
            FightAggregateRule.SUM_DIFFERENCE_CLIPPED_ZERO
        ),
        source_columns=(
            "td_attempted",
            "td_landed",
        ),
        description=(
            "Sum of max(takedown attempts minus takedowns "
            "landed, zero) across rounds."
        ),
    ),
    FightAggregateSpec(
        feature_name=_name("control_seconds"),
        rule=FightAggregateRule.SUM,
        source_columns=("ctrl_sec",),
        description=(
            "Total recorded control seconds without assigning "
            "those seconds to clinch or ground."
        ),
    ),
)


PHASE_BASELINE_AGGREGATE_NAMES = frozenset(
    spec.feature_name
    for spec in PHASE_BASELINE_AGGREGATE_SPECS
)


# ---------------------------------------------------------------------
# Derived fight-level evidence formulas
# ---------------------------------------------------------------------

PHASE_BASELINE_EVIDENCE_SPECS: tuple[
    PhaseBaselineEvidenceSpec,
    ...,
] = (
    PhaseBaselineEvidenceSpec(
        feature_name=_name(
            "distance_attempts_per_round"
        ),
        formula=(
            PhaseBaselineFormula.PER_OBSERVED_ROUND
        ),
        input_features=(
            _name("distance_attempts"),
            _name("rounds_observed"),
        ),
        reliability_features=(
            _name("rounds_observed"),
            _name("distance_attempts"),
        ),
        unit_interval=False,
        description=(
            "Distance attempts divided by observed rounds. "
            "This is not a phase-conditional 30-second rate."
        ),
    ),
    PhaseBaselineEvidenceSpec(
        feature_name=_name(
            "distance_attempt_share"
        ),
        formula=(
            PhaseBaselineFormula.PHASE_ATTEMPT_SHARE
        ),
        input_features=(
            _name("distance_attempts"),
            _name("sig_strike_attempts"),
        ),
        reliability_features=(
            _name("sig_strike_attempts"),
        ),
        unit_interval=True,
        description=(
            "Distance attempts divided by total significant-"
            "strike attempts."
        ),
    ),
    PhaseBaselineEvidenceSpec(
        feature_name=_name(
            "clinch_attempts_per_round"
        ),
        formula=(
            PhaseBaselineFormula.PER_OBSERVED_ROUND
        ),
        input_features=(
            _name("clinch_attempts"),
            _name("rounds_observed"),
        ),
        reliability_features=(
            _name("rounds_observed"),
            _name("clinch_attempts"),
        ),
        unit_interval=False,
        description=(
            "Clinch attempts divided by observed rounds. "
            "This is not a phase-conditional 30-second rate."
        ),
    ),
    PhaseBaselineEvidenceSpec(
        feature_name=_name(
            "clinch_attempt_share"
        ),
        formula=(
            PhaseBaselineFormula.PHASE_ATTEMPT_SHARE
        ),
        input_features=(
            _name("clinch_attempts"),
            _name("sig_strike_attempts"),
        ),
        reliability_features=(
            _name("sig_strike_attempts"),
        ),
        unit_interval=True,
        description=(
            "Clinch attempts divided by total significant-"
            "strike attempts."
        ),
    ),
    PhaseBaselineEvidenceSpec(
        feature_name=_name(
            "ground_attempts_per_round"
        ),
        formula=(
            PhaseBaselineFormula.PER_OBSERVED_ROUND
        ),
        input_features=(
            _name("ground_attempts"),
            _name("rounds_observed"),
        ),
        reliability_features=(
            _name("rounds_observed"),
            _name("ground_attempts"),
        ),
        unit_interval=False,
        description=(
            "Ground attempts divided by observed rounds. "
            "This is not a phase-conditional 30-second rate."
        ),
    ),
    PhaseBaselineEvidenceSpec(
        feature_name=_name(
            "ground_attempt_share"
        ),
        formula=(
            PhaseBaselineFormula.PHASE_ATTEMPT_SHARE
        ),
        input_features=(
            _name("ground_attempts"),
            _name("sig_strike_attempts"),
        ),
        reliability_features=(
            _name("sig_strike_attempts"),
        ),
        unit_interval=True,
        description=(
            "Ground attempts divided by total significant-"
            "strike attempts."
        ),
    ),
    PhaseBaselineEvidenceSpec(
        feature_name=_name(
            "td_attempts_per_round"
        ),
        formula=(
            PhaseBaselineFormula.PER_OBSERVED_ROUND
        ),
        input_features=(
            _name("td_attempts"),
            _name("rounds_observed"),
        ),
        reliability_features=(
            _name("rounds_observed"),
            _name("td_attempts"),
        ),
        unit_interval=False,
        description=(
            "Takedown attempts divided by observed rounds."
        ),
    ),
    PhaseBaselineEvidenceSpec(
        feature_name=_name("td_completion_rate"),
        formula=PhaseBaselineFormula.SAFE_RATIO,
        input_features=(
            _name("td_landed"),
            _name("td_attempts"),
        ),
        reliability_features=(
            _name("td_attempts"),
        ),
        unit_interval=True,
        description=(
            "Takedowns landed divided by takedowns attempted."
        ),
    ),
    PhaseBaselineEvidenceSpec(
        feature_name=_name(
            "failed_td_attempts_per_round"
        ),
        formula=(
            PhaseBaselineFormula.PER_OBSERVED_ROUND
        ),
        input_features=(
            _name("failed_td_attempts"),
            _name("rounds_observed"),
        ),
        reliability_features=(
            _name("rounds_observed"),
            _name("td_attempts"),
        ),
        unit_interval=False,
        description=(
            "Failed takedown attempts divided by observed "
            "rounds."
        ),
    ),
    PhaseBaselineEvidenceSpec(
        feature_name=_name(
            "control_seconds_per_round"
        ),
        formula=(
            PhaseBaselineFormula.PER_OBSERVED_ROUND
        ),
        input_features=(
            _name("control_seconds"),
            _name("rounds_observed"),
        ),
        reliability_features=(
            _name("rounds_observed"),
            _name("control_seconds"),
        ),
        unit_interval=False,
        description=(
            "Recorded control seconds divided by observed "
            "rounds without phase allocation."
        ),
    ),
    PhaseBaselineEvidenceSpec(
        feature_name=_name(
            "non_distance_clinch_share"
        ),
        formula=(
            PhaseBaselineFormula.NON_DISTANCE_PHASE_SHARE
        ),
        input_features=(
            _name("clinch_attempts"),
            _name("ground_attempts"),
        ),
        reliability_features=(
            _name("clinch_attempts"),
            _name("ground_attempts"),
        ),
        unit_interval=True,
        description=(
            "Clinch attempts divided by clinch plus ground "
            "attempts."
        ),
    ),
    PhaseBaselineEvidenceSpec(
        feature_name=_name(
            "non_distance_ground_share"
        ),
        formula=(
            PhaseBaselineFormula.NON_DISTANCE_PHASE_SHARE
        ),
        input_features=(
            _name("ground_attempts"),
            _name("clinch_attempts"),
        ),
        reliability_features=(
            _name("clinch_attempts"),
            _name("ground_attempts"),
        ),
        unit_interval=True,
        description=(
            "Ground attempts divided by ground plus clinch "
            "attempts."
        ),
    ),
    PhaseBaselineEvidenceSpec(
        feature_name=_name(
            "control_seconds_per_td_attempt"
        ),
        formula=PhaseBaselineFormula.SAFE_RATIO,
        input_features=(
            _name("control_seconds"),
            _name("td_attempts"),
        ),
        reliability_features=(
            _name("td_attempts"),
            _name("control_seconds"),
        ),
        unit_interval=False,
        description=(
            "Recorded control seconds divided by takedown "
            "attempts."
        ),
    ),
    PhaseBaselineEvidenceSpec(
        feature_name=_name(
            "control_seconds_per_td_landed"
        ),
        formula=PhaseBaselineFormula.SAFE_RATIO,
        input_features=(
            _name("control_seconds"),
            _name("td_landed"),
        ),
        reliability_features=(
            _name("td_landed"),
            _name("control_seconds"),
        ),
        unit_interval=False,
        description=(
            "Recorded control seconds divided by completed "
            "takedowns."
        ),
    ),
    PhaseBaselineEvidenceSpec(
        feature_name=_name(
            "clinch_attempts_per_control_min"
        ),
        formula=(
            PhaseBaselineFormula.PER_CONTROL_MINUTE
        ),
        input_features=(
            _name("clinch_attempts"),
            _name("control_seconds"),
        ),
        reliability_features=(
            _name("clinch_attempts"),
            _name("control_seconds"),
        ),
        unit_interval=False,
        description=(
            "Clinch attempts divided by recorded control "
            "minutes; evidence only, not allocated clinch time."
        ),
    ),
    PhaseBaselineEvidenceSpec(
        feature_name=_name(
            "ground_attempts_per_control_min"
        ),
        formula=(
            PhaseBaselineFormula.PER_CONTROL_MINUTE
        ),
        input_features=(
            _name("ground_attempts"),
            _name("control_seconds"),
        ),
        reliability_features=(
            _name("ground_attempts"),
            _name("control_seconds"),
        ),
        unit_interval=False,
        description=(
            "Ground attempts divided by recorded control "
            "minutes; evidence only, not exact ground exposure."
        ),
    ),
    PhaseBaselineEvidenceSpec(
        feature_name=_name("td_attempt_slope"),
        formula=PhaseBaselineFormula.OLS_ROUND_SLOPE,
        input_features=(
            "round",
            "td_attempted",
        ),
        reliability_features=(
            _name("rounds_observed"),
            _name("td_attempts"),
        ),
        unit_interval=False,
        description=(
            "OLS slope of takedown attempts across recorded "
            "rounds within the fight."
        ),
    ),
    PhaseBaselineEvidenceSpec(
        feature_name=_name(
            "td_persistence_ratio"
        ),
        formula=(
            PhaseBaselineFormula.FIRST_LAST_PERSISTENCE_RATIO
        ),
        input_features=(
            "round",
            "td_attempted",
        ),
        reliability_features=(
            _name("rounds_observed"),
            _name("td_attempts"),
        ),
        unit_interval=False,
        description=(
            "Last-round takedown attempts divided by "
            "max(first-round attempts, one)."
        ),
    ),
    PhaseBaselineEvidenceSpec(
        feature_name=_name(
            "failed_td_attempt_slope"
        ),
        formula=PhaseBaselineFormula.OLS_ROUND_SLOPE,
        input_features=(
            "round",
            "td_attempted",
            "td_landed",
        ),
        reliability_features=(
            _name("rounds_observed"),
            _name("td_attempts"),
        ),
        unit_interval=False,
        description=(
            "OLS slope of max(takedown attempts minus "
            "takedowns landed, zero) across rounds."
        ),
    ),
)


PHASE_BASELINE_EVIDENCE_BY_NAME = {
    spec.feature_name: spec
    for spec in PHASE_BASELINE_EVIDENCE_SPECS
}


PHASE_BASELINE_FIGHT_OBSERVATION_COLUMNS = tuple(
    [
        spec.feature_name
        for spec in PHASE_BASELINE_AGGREGATE_SPECS
    ]
    + [
        spec.feature_name
        for spec in PHASE_BASELINE_EVIDENCE_SPECS
    ]
)


# ---------------------------------------------------------------------
# Final simulator target to observed evidence mapping
# ---------------------------------------------------------------------

PHASE_BASELINE_TARGET_EVIDENCE: dict[
    str,
    tuple[str, ...],
] = {
    "transition.clinch_entry_tendency": (
        _name("clinch_attempts_per_round"),
        _name("clinch_attempt_share"),
        _name("non_distance_clinch_share"),
        _name("rounds_observed"),
    ),
    "transition.takedown_entry_tendency": (
        _name("td_attempts_per_round"),
        _name("failed_td_attempts_per_round"),
        _name("td_attempts"),
        _name("rounds_observed"),
    ),
    "transition.takedown_completion_ability": (
        _name("td_completion_rate"),
        _name("td_attempts"),
        _name("td_landed"),
    ),
    "transition.takedown_persistence": (
        _name("td_attempt_slope"),
        _name("td_persistence_ratio"),
        _name("td_attempts_per_round"),
        _name("rounds_observed"),
    ),
    "transition.failed_takedown_persistence": (
        _name("failed_td_attempt_slope"),
        _name("failed_td_attempts_per_round"),
        _name("td_attempts"),
        _name("rounds_observed"),
    ),
    "phase.distance.sig_strike_attempt_rate": (
        _name("distance_attempts_per_round"),
        _name("distance_attempt_share"),
        _name("distance_attempts"),
        _name("rounds_observed"),
    ),
    "phase.clinch.clinch_strike_attempt_rate": (
        _name("clinch_attempts_per_round"),
        _name("clinch_attempt_share"),
        _name("clinch_attempts_per_control_min"),
        _name("clinch_attempts"),
        _name("rounds_observed"),
    ),
    "phase.clinch.control_seconds_mean": (
        _name("control_seconds_per_round"),
        _name("non_distance_clinch_share"),
        _name("clinch_attempts_per_control_min"),
        _name("control_seconds"),
        _name("rounds_observed"),
    ),
    "phase.ground_owner.ground_strike_attempt_rate": (
        _name("ground_attempts_per_round"),
        _name("ground_attempt_share"),
        _name("ground_attempts_per_control_min"),
        _name("ground_attempts"),
        _name("rounds_observed"),
    ),
    "phase.ground_owner.control_seconds_mean": (
        _name("control_seconds_per_round"),
        _name("non_distance_ground_share"),
        _name("control_seconds_per_td_attempt"),
        _name("control_seconds_per_td_landed"),
        _name("ground_attempts_per_control_min"),
        _name("control_seconds"),
        _name("rounds_observed"),
    ),
}


def validate_phase_baseline_feature_contracts() -> None:
    """Validate exact targets, names, inputs, and evidence coverage."""

    aggregate_names = [
        spec.feature_name
        for spec in PHASE_BASELINE_AGGREGATE_SPECS
    ]
    evidence_names = [
        spec.feature_name
        for spec in PHASE_BASELINE_EVIDENCE_SPECS
    ]

    if len(aggregate_names) != len(set(aggregate_names)):
        raise RuntimeError(
            "Phase Baseline aggregate names must be unique"
        )

    if len(evidence_names) != len(set(evidence_names)):
        raise RuntimeError(
            "Phase Baseline evidence names must be unique"
        )

    if set(aggregate_names) & set(evidence_names):
        raise RuntimeError(
            "aggregate and derived evidence names cannot overlap"
        )

    expected_targets = {
        spec.target_parameter
        for spec in SIMULATOR_TARGET_BY_NAME.values()
        if (
            spec.primary_family
            is SimulatorFeatureFamily.PHASE_BASELINE
        )
    }

    mapped_targets = set(
        PHASE_BASELINE_TARGET_EVIDENCE
    )

    if mapped_targets != expected_targets:
        missing = sorted(
            expected_targets - mapped_targets
        )
        extra = sorted(
            mapped_targets - expected_targets
        )

        raise RuntimeError(
            "Phase Baseline target evidence does not match "
            f"the target registry; missing={missing}, "
            f"extra={extra}"
        )

    valid_feature_names = (
        set(aggregate_names)
        | set(evidence_names)
    )

    for spec in PHASE_BASELINE_EVIDENCE_SPECS:
        for input_name in spec.input_features:
            if (
                input_name not in valid_feature_names
                and input_name
                not in AUTHORITATIVE_ROUND_SOURCE_COLUMNS
            ):
                raise RuntimeError(
                    "unknown Phase Baseline evidence input: "
                    f"{input_name}"
                )

        for reliability_name in (
            spec.reliability_features
        ):
            if reliability_name not in aggregate_names:
                raise RuntimeError(
                    "reliability feature must be a fight "
                    f"aggregate: {reliability_name}"
                )

    for target, feature_names in (
        PHASE_BASELINE_TARGET_EVIDENCE.items()
    ):
        if not feature_names:
            raise RuntimeError(
                f"target has no evidence features: {target}"
            )

        for feature_name in feature_names:
            if feature_name not in valid_feature_names:
                raise RuntimeError(
                    f"target {target} references unknown "
                    f"evidence feature: {feature_name}"
                )


validate_phase_baseline_feature_contracts()
