"""Observed feature contracts for the RFS Dynamic Response family.

Dynamic Response describes how fighter performance changes as a completed
fight progresses:

- workload accumulation
- pace sustainability
- output and efficiency preservation
- between-round rebound
- response after observable adversity
- late-round deterioration or recovery

Important boundary
------------------
UFCStats round totals do not expose exact segment-level fatigue, recovery
during round breaks, or the exact timing of adversity within a round.

These contracts therefore describe observable completed-fight evidence.
They do not directly encode the final simulator Dynamic Response parameters.

Current-fight observation names contain ``_fight_``. Leakage-safe prior state
created by the future builder must not contain ``_fight_``.
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


DYNAMIC_RESPONSE_PREFIX = "rfs_dynamic_response_fight_"


class DynamicAggregateRule(str, Enum):
    """How one completed-fight aggregate is built."""

    UNIQUE_COUNT = "unique_count"
    SUM = "sum"


class DynamicAggregatePerspective(str, Enum):
    """Which reciprocal fighter row supplies an aggregate."""

    FIGHTER = "fighter"
    OPPONENT = "opponent"


class DynamicResponseFormula(str, Enum):
    """Locked formula category for one Dynamic Response feature."""

    SAFE_RATIO = "safe_ratio"
    PER_OBSERVED_ROUND = "per_observed_round"
    FIRST_LAST_RATIO = "first_last_ratio"
    FIRST_LAST_DIFFERENCE = "first_last_difference"
    OLS_ROUND_SLOPE = "ols_round_slope"
    LATE_EARLY_RATIO = "late_early_ratio"
    LATE_EARLY_DIFFERENCE = "late_early_difference"
    EFFICIENCY_CHANGE = "efficiency_change"
    ADVERSITY_OPPORTUNITY = "adversity_opportunity"
    POST_ADVERSITY_REBOUND = "post_adversity_rebound"
    POST_ADVERSITY_PRESERVATION = "post_adversity_preservation"


@dataclass(frozen=True)
class DynamicResponseAggregateSpec:
    """One completed-fight aggregate contract."""

    feature_name: str
    rule: DynamicAggregateRule
    source_column: str
    perspective: DynamicAggregatePerspective
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.feature_name, str):
            raise TypeError("feature_name must be a string")

        if not self.feature_name.startswith(
            DYNAMIC_RESPONSE_PREFIX
        ):
            raise ValueError(
                "feature_name must use the Dynamic Response prefix"
            )

        if "_fight_" not in self.feature_name:
            raise ValueError(
                "current-fight observation names must contain _fight_"
            )

        if not isinstance(
            self.rule,
            DynamicAggregateRule,
        ):
            raise TypeError(
                "rule must be DynamicAggregateRule"
            )

        if not isinstance(self.source_column, str):
            raise TypeError(
                "source_column must be a string"
            )

        if not isinstance(
            self.perspective,
            DynamicAggregatePerspective,
        ):
            raise TypeError(
                "perspective must be "
                "DynamicAggregatePerspective"
            )

        if (
            self.source_column
            not in AUTHORITATIVE_ROUND_SOURCE_COLUMNS
        ):
            raise ValueError(
                "source_column is not authoritative: "
                f"{self.source_column}"
            )

        _validate_description(self.description)


@dataclass(frozen=True)
class DynamicResponseEvidenceSpec:
    """One derived completed-fight trajectory or adversity feature."""

    feature_name: str
    formula: DynamicResponseFormula
    input_features: tuple[str, ...]
    reliability_features: tuple[str, ...]
    unit_interval: bool
    requires_multiple_rounds: bool
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.feature_name, str):
            raise TypeError("feature_name must be a string")

        if not self.feature_name.startswith(
            DYNAMIC_RESPONSE_PREFIX
        ):
            raise ValueError(
                "feature_name must use the Dynamic Response prefix"
            )

        if "_fight_" not in self.feature_name:
            raise ValueError(
                "current-fight observation names must contain _fight_"
            )

        if "30_seconds" in self.feature_name:
            raise ValueError(
                "Dynamic Response evidence cannot claim exact "
                "30-second exposure"
            )

        if not isinstance(
            self.formula,
            DynamicResponseFormula,
        ):
            raise TypeError(
                "formula must be DynamicResponseFormula"
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

        if type(self.requires_multiple_rounds) is not bool:
            raise TypeError(
                "requires_multiple_rounds must be boolean"
            )

        _validate_description(self.description)


def _validate_feature_tuple(
    values: object,
    *,
    field_name: str,
) -> None:
    """Validate one tuple of feature names."""

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
    """Validate one nonblank description."""

    if not isinstance(value, str):
        raise TypeError(
            "description must be a string"
        )

    if not value.strip():
        raise ValueError(
            "description cannot be empty"
        )


def _name(suffix: str) -> str:
    """Return one current-fight Dynamic Response name."""

    return f"{DYNAMIC_RESPONSE_PREFIX}{suffix}"


def _aggregate(
    suffix: str,
    rule: DynamicAggregateRule,
    source_column: str,
    description: str,
    *,
    perspective: DynamicAggregatePerspective = (
        DynamicAggregatePerspective.FIGHTER
    ),
) -> DynamicResponseAggregateSpec:
    """Build one aggregate declaration."""

    return DynamicResponseAggregateSpec(
        feature_name=_name(suffix),
        rule=rule,
        source_column=source_column,
        perspective=perspective,
        description=description,
    )


def _evidence(
    suffix: str,
    formula: DynamicResponseFormula,
    input_features: tuple[str, ...],
    reliability_features: tuple[str, ...],
    *,
    unit_interval: bool,
    requires_multiple_rounds: bool,
    description: str,
) -> DynamicResponseEvidenceSpec:
    """Build one evidence declaration."""

    return DynamicResponseEvidenceSpec(
        feature_name=_name(suffix),
        formula=formula,
        input_features=input_features,
        reliability_features=reliability_features,
        unit_interval=unit_interval,
        requires_multiple_rounds=requires_multiple_rounds,
        description=description,
    )


UNIQUE = DynamicAggregateRule.UNIQUE_COUNT
SUM = DynamicAggregateRule.SUM

FIGHTER = DynamicAggregatePerspective.FIGHTER
OPPONENT = DynamicAggregatePerspective.OPPONENT

SAFE_RATIO = DynamicResponseFormula.SAFE_RATIO
PER_ROUND = DynamicResponseFormula.PER_OBSERVED_ROUND
FIRST_LAST_RATIO = DynamicResponseFormula.FIRST_LAST_RATIO
FIRST_LAST_DIFF = DynamicResponseFormula.FIRST_LAST_DIFFERENCE
OLS_SLOPE = DynamicResponseFormula.OLS_ROUND_SLOPE
LATE_EARLY_RATIO = DynamicResponseFormula.LATE_EARLY_RATIO
LATE_EARLY_DIFF = DynamicResponseFormula.LATE_EARLY_DIFFERENCE
EFFICIENCY_CHANGE = DynamicResponseFormula.EFFICIENCY_CHANGE
ADVERSITY_OPPORTUNITY = (
    DynamicResponseFormula.ADVERSITY_OPPORTUNITY
)
POST_ADVERSITY_REBOUND = (
    DynamicResponseFormula.POST_ADVERSITY_REBOUND
)
POST_ADVERSITY_PRESERVATION = (
    DynamicResponseFormula.POST_ADVERSITY_PRESERVATION
)


# ---------------------------------------------------------------------
# Completed-fight aggregate observations
# ---------------------------------------------------------------------

DYNAMIC_RESPONSE_AGGREGATE_SPECS: tuple[
    DynamicResponseAggregateSpec,
    ...,
] = (
    _aggregate(
        "rounds_observed",
        UNIQUE,
        "round",
        "Distinct recorded rounds for the fighter in the fight.",
    ),
    _aggregate(
        "sig_strikes_landed",
        SUM,
        "sig_str_landed",
        "Total significant strikes landed.",
    ),
    _aggregate(
        "sig_strike_attempts",
        SUM,
        "sig_str_attempted",
        "Total significant-strike attempts.",
    ),
    _aggregate(
        "total_strikes_landed",
        SUM,
        "total_str_landed",
        "Total strikes landed.",
    ),
    _aggregate(
        "total_strike_attempts",
        SUM,
        "total_str_attempted",
        "Total strike attempts.",
    ),
    _aggregate(
        "td_landed",
        SUM,
        "td_landed",
        "Completed takedowns.",
    ),
    _aggregate(
        "td_attempts",
        SUM,
        "td_attempted",
        "Attempted takedowns.",
    ),
    _aggregate(
        "control_seconds",
        SUM,
        "ctrl_sec",
        "Recorded control seconds.",
    ),
    _aggregate(
        "knockdowns_absorbed",
        SUM,
        "kd",
        "Opponent knockdowns absorbed by the fighter.",
        perspective=OPPONENT,
    ),
    _aggregate(
        "head_strikes_absorbed",
        SUM,
        "head_landed",
        "Opponent landed head strikes absorbed by the fighter.",
        perspective=OPPONENT,
    ),
    _aggregate(
        "ground_strikes_absorbed",
        SUM,
        "ground_landed",
        "Opponent landed ground strikes absorbed by the fighter.",
        perspective=OPPONENT,
    ),
    _aggregate(
        "opponent_control_seconds",
        SUM,
        "ctrl_sec",
        "Opponent control seconds recorded against the fighter.",
        perspective=OPPONENT,
    ),
)


DYNAMIC_RESPONSE_AGGREGATE_NAMES = frozenset(
    spec.feature_name
    for spec in DYNAMIC_RESPONSE_AGGREGATE_SPECS
)


# ---------------------------------------------------------------------
# Fight-level workload and efficiency evidence
# ---------------------------------------------------------------------

DYNAMIC_RESPONSE_EVIDENCE_SPECS: tuple[
    DynamicResponseEvidenceSpec,
    ...,
] = (
    _evidence(
        "sig_strike_accuracy",
        SAFE_RATIO,
        (
            _name("sig_strikes_landed"),
            _name("sig_strike_attempts"),
        ),
        (_name("sig_strike_attempts"),),
        unit_interval=True,
        requires_multiple_rounds=False,
        description=(
            "Completed-fight significant-strike accuracy."
        ),
    ),
    _evidence(
        "total_strike_accuracy",
        SAFE_RATIO,
        (
            _name("total_strikes_landed"),
            _name("total_strike_attempts"),
        ),
        (_name("total_strike_attempts"),),
        unit_interval=True,
        requires_multiple_rounds=False,
        description=(
            "Completed-fight total-strike accuracy."
        ),
    ),
    _evidence(
        "td_completion_rate",
        SAFE_RATIO,
        (
            _name("td_landed"),
            _name("td_attempts"),
        ),
        (_name("td_attempts"),),
        unit_interval=True,
        requires_multiple_rounds=False,
        description=(
            "Completed-fight takedown completion rate."
        ),
    ),
    _evidence(
        "sig_strike_attempts_per_round",
        PER_ROUND,
        (
            _name("sig_strike_attempts"),
            _name("rounds_observed"),
        ),
        (_name("rounds_observed"),),
        unit_interval=False,
        requires_multiple_rounds=False,
        description=(
            "Significant-strike attempts per observed round."
        ),
    ),
    _evidence(
        "total_strike_attempts_per_round",
        PER_ROUND,
        (
            _name("total_strike_attempts"),
            _name("rounds_observed"),
        ),
        (_name("rounds_observed"),),
        unit_interval=False,
        requires_multiple_rounds=False,
        description=(
            "Total-strike attempts per observed round."
        ),
    ),
    _evidence(
        "td_attempts_per_round",
        PER_ROUND,
        (
            _name("td_attempts"),
            _name("rounds_observed"),
        ),
        (_name("rounds_observed"),),
        unit_interval=False,
        requires_multiple_rounds=False,
        description=(
            "Takedown attempts per observed round."
        ),
    ),
    _evidence(
        "control_seconds_per_round",
        PER_ROUND,
        (
            _name("control_seconds"),
            _name("rounds_observed"),
        ),
        (_name("rounds_observed"),),
        unit_interval=False,
        requires_multiple_rounds=False,
        description=(
            "Recorded control seconds per observed round."
        ),
    ),

    # Pace trajectories.
    _evidence(
        "sig_strike_attempt_slope",
        OLS_SLOPE,
        (
            "round",
            "sig_str_attempted",
        ),
        (_name("rounds_observed"),),
        unit_interval=False,
        requires_multiple_rounds=True,
        description=(
            "OLS slope of significant-strike attempts across rounds."
        ),
    ),
    _evidence(
        "total_strike_attempt_slope",
        OLS_SLOPE,
        (
            "round",
            "total_str_attempted",
        ),
        (_name("rounds_observed"),),
        unit_interval=False,
        requires_multiple_rounds=True,
        description=(
            "OLS slope of total-strike attempts across rounds."
        ),
    ),
    _evidence(
        "td_attempt_slope",
        OLS_SLOPE,
        (
            "round",
            "td_attempted",
        ),
        (_name("rounds_observed"),),
        unit_interval=False,
        requires_multiple_rounds=True,
        description=(
            "OLS slope of takedown attempts across rounds."
        ),
    ),
    _evidence(
        "control_seconds_slope",
        OLS_SLOPE,
        (
            "round",
            "ctrl_sec",
        ),
        (_name("rounds_observed"),),
        unit_interval=False,
        requires_multiple_rounds=True,
        description=(
            "OLS slope of recorded control seconds across rounds."
        ),
    ),
    _evidence(
        "sig_strike_attempt_first_last_ratio",
        FIRST_LAST_RATIO,
        (
            "round",
            "sig_str_attempted",
        ),
        (_name("rounds_observed"),),
        unit_interval=False,
        requires_multiple_rounds=True,
        description=(
            "Last-round significant-strike attempts divided by "
            "first-round attempts."
        ),
    ),
    _evidence(
        "total_strike_attempt_first_last_ratio",
        FIRST_LAST_RATIO,
        (
            "round",
            "total_str_attempted",
        ),
        (_name("rounds_observed"),),
        unit_interval=False,
        requires_multiple_rounds=True,
        description=(
            "Last-round total-strike attempts divided by "
            "first-round attempts."
        ),
    ),
    _evidence(
        "late_early_workload_ratio",
        LATE_EARLY_RATIO,
        (
            "sig_str_attempted",
            "total_str_attempted",
            "td_attempted",
            "ctrl_sec",
        ),
        (_name("rounds_observed"),),
        unit_interval=False,
        requires_multiple_rounds=True,
        description=(
            "Late-round composite workload divided by early-round "
            "composite workload."
        ),
    ),
    _evidence(
        "late_early_workload_difference",
        LATE_EARLY_DIFF,
        (
            "sig_str_attempted",
            "total_str_attempted",
            "td_attempted",
            "ctrl_sec",
        ),
        (_name("rounds_observed"),),
        unit_interval=False,
        requires_multiple_rounds=True,
        description=(
            "Late-round minus early-round composite workload."
        ),
    ),

    # Output and efficiency preservation.
    _evidence(
        "sig_strike_landed_slope",
        OLS_SLOPE,
        (
            "round",
            "sig_str_landed",
        ),
        (_name("rounds_observed"),),
        unit_interval=False,
        requires_multiple_rounds=True,
        description=(
            "OLS slope of significant strikes landed across rounds."
        ),
    ),
    _evidence(
        "total_strike_landed_slope",
        OLS_SLOPE,
        (
            "round",
            "total_str_landed",
        ),
        (_name("rounds_observed"),),
        unit_interval=False,
        requires_multiple_rounds=True,
        description=(
            "OLS slope of total strikes landed across rounds."
        ),
    ),
    _evidence(
        "sig_strike_accuracy_change",
        EFFICIENCY_CHANGE,
        (
            "round",
            "sig_str_landed",
            "sig_str_attempted",
        ),
        (_name("sig_strike_attempts"),),
        unit_interval=False,
        requires_multiple_rounds=True,
        description=(
            "Late-round minus early-round significant-strike accuracy."
        ),
    ),
    _evidence(
        "total_strike_accuracy_change",
        EFFICIENCY_CHANGE,
        (
            "round",
            "total_str_landed",
            "total_str_attempted",
        ),
        (_name("total_strike_attempts"),),
        unit_interval=False,
        requires_multiple_rounds=True,
        description=(
            "Late-round minus early-round total-strike accuracy."
        ),
    ),
    _evidence(
        "late_early_output_ratio",
        LATE_EARLY_RATIO,
        (
            "sig_str_landed",
            "total_str_landed",
            "td_landed",
            "ctrl_sec",
        ),
        (_name("rounds_observed"),),
        unit_interval=False,
        requires_multiple_rounds=True,
        description=(
            "Late-round composite output divided by early-round "
            "composite output."
        ),
    ),

    # Adversity opportunity and response.
    _evidence(
        "adversity_round_count",
        ADVERSITY_OPPORTUNITY,
        (
            "opponent_kd",
            "opponent_head_landed",
            "opponent_ground_landed",
            "opponent_ctrl_sec",
        ),
        (_name("rounds_observed"),),
        unit_interval=False,
        requires_multiple_rounds=False,
        description=(
            "Number of rounds containing observable opponent adversity "
            "signals."
        ),
    ),
    _evidence(
        "post_adversity_sig_strike_rebound",
        POST_ADVERSITY_REBOUND,
        (
            "round",
            "sig_str_attempted",
            "opponent_kd",
            "opponent_head_landed",
            "opponent_ground_landed",
            "opponent_ctrl_sec",
        ),
        (_name("adversity_round_count"),),
        unit_interval=False,
        requires_multiple_rounds=True,
        description=(
            "Change in significant-strike attempts in the round after "
            "observable adversity."
        ),
    ),
    _evidence(
        "post_adversity_output_rebound",
        POST_ADVERSITY_REBOUND,
        (
            "round",
            "sig_str_landed",
            "td_landed",
            "ctrl_sec",
            "opponent_kd",
            "opponent_head_landed",
            "opponent_ground_landed",
            "opponent_ctrl_sec",
        ),
        (_name("adversity_round_count"),),
        unit_interval=False,
        requires_multiple_rounds=True,
        description=(
            "Change in composite output in the round after observable "
            "adversity."
        ),
    ),
    _evidence(
        "post_adversity_efficiency_preservation",
        POST_ADVERSITY_PRESERVATION,
        (
            "round",
            "sig_str_landed",
            "sig_str_attempted",
            "opponent_kd",
            "opponent_head_landed",
            "opponent_ground_landed",
            "opponent_ctrl_sec",
        ),
        (_name("adversity_round_count"),),
        unit_interval=False,
        requires_multiple_rounds=True,
        description=(
            "Significant-strike efficiency preservation after "
            "observable adversity."
        ),
    ),
)


DYNAMIC_RESPONSE_EVIDENCE_BY_NAME = {
    spec.feature_name: spec
    for spec in DYNAMIC_RESPONSE_EVIDENCE_SPECS
}


DYNAMIC_RESPONSE_FIGHT_OBSERVATION_COLUMNS = tuple(
    [
        spec.feature_name
        for spec in DYNAMIC_RESPONSE_AGGREGATE_SPECS
    ]
    + [
        spec.feature_name
        for spec in DYNAMIC_RESPONSE_EVIDENCE_SPECS
    ]
)


# ---------------------------------------------------------------------
# Simulator-target evidence coverage
# ---------------------------------------------------------------------

DYNAMIC_RESPONSE_TARGET_EVIDENCE: dict[
    str,
    tuple[str, ...],
] = {
    "dynamic.fatigue_accumulation_resistance": (
        _name("sig_strike_attempt_slope"),
        _name("total_strike_attempt_slope"),
        _name("td_attempt_slope"),
        _name("control_seconds_slope"),
        _name("sig_strike_attempt_first_last_ratio"),
        _name("total_strike_attempt_first_last_ratio"),
        _name("late_early_workload_ratio"),
        _name("late_early_workload_difference"),
    ),
    "dynamic.fatigue_performance_resilience": (
        _name("sig_strike_landed_slope"),
        _name("total_strike_landed_slope"),
        _name("sig_strike_accuracy_change"),
        _name("total_strike_accuracy_change"),
        _name("late_early_output_ratio"),
        _name("late_early_workload_ratio"),
    ),
    "dynamic.recovery_ability": (
        _name("adversity_round_count"),
        _name("post_adversity_sig_strike_rebound"),
        _name("post_adversity_output_rebound"),
        _name("post_adversity_efficiency_preservation"),
        _name("late_early_output_ratio"),
    ),
    "dynamic.acute_stress_recovery": (
        _name("adversity_round_count"),
        _name("post_adversity_sig_strike_rebound"),
        _name("post_adversity_output_rebound"),
        _name("post_adversity_efficiency_preservation"),
    ),
}


DYNAMIC_RESPONSE_TARGETS = frozenset(
    DYNAMIC_RESPONSE_TARGET_EVIDENCE
)


def validate_dynamic_response_contracts() -> None:
    """Validate exact Dynamic Response target and feature coverage."""

    expected_targets = {
        target_name
        for target_name, target_spec
        in SIMULATOR_TARGET_BY_NAME.items()
        if (
            target_spec.primary_family
            is SimulatorFeatureFamily.DYNAMIC_RESPONSE
        )
    }

    if DYNAMIC_RESPONSE_TARGETS != expected_targets:
        missing = sorted(
            expected_targets - DYNAMIC_RESPONSE_TARGETS
        )
        extra = sorted(
            DYNAMIC_RESPONSE_TARGETS - expected_targets
        )

        raise RuntimeError(
            "Dynamic Response target coverage mismatch: "
            f"missing={missing}, extra={extra}"
        )

    observation_names = set(
        DYNAMIC_RESPONSE_FIGHT_OBSERVATION_COLUMNS
    )

    if (
        len(observation_names)
        != len(DYNAMIC_RESPONSE_FIGHT_OBSERVATION_COLUMNS)
    ):
        raise RuntimeError(
            "Dynamic Response observation names must be unique"
        )

    for target_name, evidence_names in (
        DYNAMIC_RESPONSE_TARGET_EVIDENCE.items()
    ):
        if target_name not in SIMULATOR_TARGET_BY_NAME:
            raise RuntimeError(
                "Unknown Dynamic Response target: "
                f"{target_name}"
            )

        if not evidence_names:
            raise RuntimeError(
                "Dynamic Response target has no evidence: "
                f"{target_name}"
            )

        unknown_evidence = sorted(
            set(evidence_names) - observation_names
        )

        if unknown_evidence:
            raise RuntimeError(
                "Dynamic Response target references unknown "
                f"evidence: target={target_name}, "
                f"unknown={unknown_evidence}"
            )


validate_dynamic_response_contracts()
