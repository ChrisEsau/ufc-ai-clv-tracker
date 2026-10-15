"""Observed feature contracts for the RFS Finish State family.

Finish State represents completed-fight evidence related to:

- knockdown creation
- damaging clinch activity
- submission pressure
- submission survival
- damage survival
- immediate performance preservation under adversity

Important boundary
------------------
UFCStats does not identify the exact strike that caused a knockdown, exact
owned-clinch or owned-ground segment exposure, or precise within-round timing.

These contracts therefore define observable fight evidence and defensible
proxies. They do not directly encode final simulator probabilities.

Current-fight observations contain ``_fight_``. Leakage-safe historical state
created by the future builder must exclude the current fight.
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


FINISH_STATE_PREFIX = "rfs_finish_state_fight_"

AUTHORITATIVE_OUTCOME_SOURCE_COLUMNS = frozenset(
    {
        "fight_id",
        "winner",
        "winner_id",
        "method",
        "finish_round",
    }
)


class FinishSourceKind(str, Enum):
    """Authoritative source supplying an observation."""

    ROUND = "round"
    OUTCOME = "outcome"


class FinishAggregateRule(str, Enum):
    """How a completed-fight aggregate is constructed."""

    UNIQUE_COUNT = "unique_count"
    SUM = "sum"
    OUTCOME_FLAG = "outcome_flag"


class FinishAggregatePerspective(str, Enum):
    """Which reciprocal fight perspective supplies an observation."""

    FIGHTER = "fighter"
    OPPONENT = "opponent"
    FIGHT = "fight"


class FinishStateFormula(str, Enum):
    """Locked formula category for derived Finish State evidence."""

    SAFE_RATIO = "safe_ratio"
    PER_OBSERVED_ROUND = "per_observed_round"
    DAMAGE_EXPOSURE_COMPOSITE = "damage_exposure_composite"
    ADVERSITY_OPPORTUNITY = "adversity_opportunity"
    SAME_ROUND_OUTPUT_PRESERVATION = (
        "same_round_output_preservation"
    )
    SAME_ROUND_EFFICIENCY_PRESERVATION = (
        "same_round_efficiency_preservation"
    )
    OUTCOME_INDICATOR = "outcome_indicator"
    SURVIVAL_INDICATOR = "survival_indicator"


@dataclass(frozen=True)
class FinishStateAggregateSpec:
    """One completed-fight aggregate observation contract."""

    feature_name: str
    rule: FinishAggregateRule
    source_kind: FinishSourceKind
    source_column: str
    perspective: FinishAggregatePerspective
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.feature_name, str):
            raise TypeError("feature_name must be a string")

        if not self.feature_name.startswith(FINISH_STATE_PREFIX):
            raise ValueError(
                "feature_name must use the Finish State prefix"
            )

        if "_fight_" not in self.feature_name:
            raise ValueError(
                "current-fight observation names must contain _fight_"
            )

        if not isinstance(self.rule, FinishAggregateRule):
            raise TypeError(
                "rule must be FinishAggregateRule"
            )

        if not isinstance(self.source_kind, FinishSourceKind):
            raise TypeError(
                "source_kind must be FinishSourceKind"
            )

        if not isinstance(
            self.perspective,
            FinishAggregatePerspective,
        ):
            raise TypeError(
                "perspective must be FinishAggregatePerspective"
            )

        if not isinstance(self.source_column, str):
            raise TypeError("source_column must be a string")

        if self.source_kind is FinishSourceKind.ROUND:
            if (
                self.source_column
                not in AUTHORITATIVE_ROUND_SOURCE_COLUMNS
            ):
                raise ValueError(
                    "round source_column is not authoritative: "
                    f"{self.source_column}"
                )

            if self.rule is FinishAggregateRule.OUTCOME_FLAG:
                raise ValueError(
                    "round observations cannot use OUTCOME_FLAG"
                )

            if self.perspective is (
                FinishAggregatePerspective.FIGHT
            ):
                raise ValueError(
                    "round observations require fighter or opponent "
                    "perspective"
                )

        if self.source_kind is FinishSourceKind.OUTCOME:
            if (
                self.source_column
                not in AUTHORITATIVE_OUTCOME_SOURCE_COLUMNS
            ):
                raise ValueError(
                    "outcome source_column is not authoritative: "
                    f"{self.source_column}"
                )

            if self.rule is not FinishAggregateRule.OUTCOME_FLAG:
                raise ValueError(
                    "outcome observations must use OUTCOME_FLAG"
                )

            if self.perspective is not (
                FinishAggregatePerspective.FIGHT
            ):
                raise ValueError(
                    "outcome observations require fight perspective"
                )

        _validate_description(self.description)


@dataclass(frozen=True)
class FinishStateEvidenceSpec:
    """One derived completed-fight Finish State feature."""

    feature_name: str
    formula: FinishStateFormula
    input_features: tuple[str, ...]
    reliability_features: tuple[str, ...]
    unit_interval: bool
    requires_adversity: bool
    requires_valid_outcome: bool
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.feature_name, str):
            raise TypeError("feature_name must be a string")

        if not self.feature_name.startswith(FINISH_STATE_PREFIX):
            raise ValueError(
                "feature_name must use the Finish State prefix"
            )

        if "_fight_" not in self.feature_name:
            raise ValueError(
                "current-fight observation names must contain _fight_"
            )

        if "30_seconds" in self.feature_name:
            raise ValueError(
                "Finish State evidence cannot claim exact "
                "30-second segment exposure"
            )

        if not isinstance(self.formula, FinishStateFormula):
            raise TypeError(
                "formula must be FinishStateFormula"
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
            raise ValueError("input_features cannot be empty")

        if not self.reliability_features:
            raise ValueError(
                "reliability_features cannot be empty"
            )

        if type(self.unit_interval) is not bool:
            raise TypeError(
                "unit_interval must be boolean"
            )

        if type(self.requires_adversity) is not bool:
            raise TypeError(
                "requires_adversity must be boolean"
            )

        if type(self.requires_valid_outcome) is not bool:
            raise TypeError(
                "requires_valid_outcome must be boolean"
            )

        outcome_formulas = {
            FinishStateFormula.OUTCOME_INDICATOR,
            FinishStateFormula.SURVIVAL_INDICATOR,
        }

        if (
            self.formula in outcome_formulas
            and not self.requires_valid_outcome
        ):
            raise ValueError(
                "outcome-derived evidence must require a valid outcome"
            )

        adversity_formulas = {
            FinishStateFormula.ADVERSITY_OPPORTUNITY,
            FinishStateFormula.SAME_ROUND_OUTPUT_PRESERVATION,
            FinishStateFormula.SAME_ROUND_EFFICIENCY_PRESERVATION,
        }

        if (
            self.formula
            in {
                FinishStateFormula.SAME_ROUND_OUTPUT_PRESERVATION,
                FinishStateFormula.SAME_ROUND_EFFICIENCY_PRESERVATION,
            }
            and not self.requires_adversity
        ):
            raise ValueError(
                "preservation evidence must require adversity"
            )

        if (
            self.formula
            is FinishStateFormula.ADVERSITY_OPPORTUNITY
            and self.requires_adversity
        ):
            raise ValueError(
                "adversity opportunity itself cannot require adversity"
            )

        _validate_description(self.description)


def _validate_feature_tuple(
    values: object,
    *,
    field_name: str,
) -> None:
    """Validate one tuple of feature names."""

    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be a tuple")

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
        raise TypeError("description must be a string")

    if not value.strip():
        raise ValueError("description cannot be empty")


def _name(suffix: str) -> str:
    """Return one current-fight Finish State feature name."""

    return f"{FINISH_STATE_PREFIX}{suffix}"


def _aggregate(
    suffix: str,
    rule: FinishAggregateRule,
    source_column: str,
    description: str,
    *,
    source_kind: FinishSourceKind = FinishSourceKind.ROUND,
    perspective: FinishAggregatePerspective = (
        FinishAggregatePerspective.FIGHTER
    ),
) -> FinishStateAggregateSpec:
    """Build one aggregate declaration."""

    return FinishStateAggregateSpec(
        feature_name=_name(suffix),
        rule=rule,
        source_kind=source_kind,
        source_column=source_column,
        perspective=perspective,
        description=description,
    )


def _evidence(
    suffix: str,
    formula: FinishStateFormula,
    input_features: tuple[str, ...],
    reliability_features: tuple[str, ...],
    *,
    unit_interval: bool,
    requires_adversity: bool = False,
    requires_valid_outcome: bool = False,
    description: str,
) -> FinishStateEvidenceSpec:
    """Build one evidence declaration."""

    return FinishStateEvidenceSpec(
        feature_name=_name(suffix),
        formula=formula,
        input_features=input_features,
        reliability_features=reliability_features,
        unit_interval=unit_interval,
        requires_adversity=requires_adversity,
        requires_valid_outcome=requires_valid_outcome,
        description=description,
    )


UNIQUE = FinishAggregateRule.UNIQUE_COUNT
SUM = FinishAggregateRule.SUM
OUTCOME_FLAG = FinishAggregateRule.OUTCOME_FLAG

ROUND = FinishSourceKind.ROUND
OUTCOME = FinishSourceKind.OUTCOME

FIGHTER = FinishAggregatePerspective.FIGHTER
OPPONENT = FinishAggregatePerspective.OPPONENT
FIGHT = FinishAggregatePerspective.FIGHT

SAFE_RATIO = FinishStateFormula.SAFE_RATIO
PER_ROUND = FinishStateFormula.PER_OBSERVED_ROUND
DAMAGE_EXPOSURE = FinishStateFormula.DAMAGE_EXPOSURE_COMPOSITE
ADVERSITY_OPPORTUNITY = FinishStateFormula.ADVERSITY_OPPORTUNITY
OUTPUT_PRESERVATION = (
    FinishStateFormula.SAME_ROUND_OUTPUT_PRESERVATION
)
EFFICIENCY_PRESERVATION = (
    FinishStateFormula.SAME_ROUND_EFFICIENCY_PRESERVATION
)
OUTCOME_INDICATOR = FinishStateFormula.OUTCOME_INDICATOR
SURVIVAL_INDICATOR = FinishStateFormula.SURVIVAL_INDICATOR


# ---------------------------------------------------------------------
# Completed-fight aggregate observations
# ---------------------------------------------------------------------

FINISH_STATE_AGGREGATE_SPECS: tuple[
    FinishStateAggregateSpec,
    ...,
] = (
    _aggregate(
        "rounds_observed",
        UNIQUE,
        "round",
        "Distinct recorded rounds for the fighter.",
    ),
    _aggregate(
        "sig_strikes_landed",
        SUM,
        "sig_str_landed",
        "Significant strikes landed by the fighter.",
    ),
    _aggregate(
        "sig_strike_attempts",
        SUM,
        "sig_str_attempted",
        "Significant strikes attempted by the fighter.",
    ),
    _aggregate(
        "distance_strikes_landed",
        SUM,
        "distance_landed",
        "Distance significant strikes landed.",
    ),
    _aggregate(
        "clinch_strikes_landed",
        SUM,
        "clinch_landed",
        "Clinch significant strikes landed.",
    ),
    _aggregate(
        "clinch_strike_attempts",
        SUM,
        "clinch_attempted",
        "Clinch significant strikes attempted.",
    ),
    _aggregate(
        "ground_strikes_landed",
        SUM,
        "ground_landed",
        "Ground significant strikes landed.",
    ),
    _aggregate(
        "ground_strike_attempts",
        SUM,
        "ground_attempted",
        "Ground significant strikes attempted.",
    ),
    _aggregate(
        "knockdowns_scored",
        SUM,
        "kd",
        "Knockdowns credited to the fighter.",
    ),
    _aggregate(
        "submission_attempts",
        SUM,
        "sub_att",
        "Submission attempts credited to the fighter.",
    ),
    _aggregate(
        "takedowns_landed",
        SUM,
        "td_landed",
        "Takedowns landed by the fighter.",
    ),
    _aggregate(
        "control_seconds",
        SUM,
        "ctrl_sec",
        "Control seconds credited to the fighter.",
    ),
    _aggregate(
        "sig_strikes_absorbed",
        SUM,
        "sig_str_landed",
        "Opponent significant strikes absorbed.",
        perspective=OPPONENT,
    ),
    _aggregate(
        "head_strikes_absorbed",
        SUM,
        "head_landed",
        "Opponent landed head strikes absorbed.",
        perspective=OPPONENT,
    ),
    _aggregate(
        "clinch_strikes_absorbed",
        SUM,
        "clinch_landed",
        "Opponent landed clinch strikes absorbed.",
        perspective=OPPONENT,
    ),
    _aggregate(
        "ground_strikes_absorbed",
        SUM,
        "ground_landed",
        "Opponent landed ground strikes absorbed.",
        perspective=OPPONENT,
    ),
    _aggregate(
        "knockdowns_absorbed",
        SUM,
        "kd",
        "Opponent knockdowns absorbed.",
        perspective=OPPONENT,
    ),
    _aggregate(
        "opponent_submission_attempts",
        SUM,
        "sub_att",
        "Opponent submission attempts faced.",
        perspective=OPPONENT,
    ),
    _aggregate(
        "opponent_control_seconds",
        SUM,
        "ctrl_sec",
        "Opponent control seconds recorded against the fighter.",
        perspective=OPPONENT,
    ),
    _aggregate(
        "valid_outcome",
        OUTCOME_FLAG,
        "winner_id",
        "Whether the fight has a usable recorded winner outcome.",
        source_kind=OUTCOME,
        perspective=FIGHT,
    ),
)


FINISH_STATE_AGGREGATE_NAMES = frozenset(
    spec.feature_name
    for spec in FINISH_STATE_AGGREGATE_SPECS
)


# ---------------------------------------------------------------------
# Derived completed-fight evidence
# ---------------------------------------------------------------------

FINISH_STATE_EVIDENCE_SPECS: tuple[
    FinishStateEvidenceSpec,
    ...,
] = (
    _evidence(
        "knockdowns_per_sig_strike_landed",
        SAFE_RATIO,
        (
            _name("knockdowns_scored"),
            _name("sig_strikes_landed"),
        ),
        (_name("sig_strikes_landed"),),
        unit_interval=True,
        description=(
            "Knockdowns divided by landed significant strikes."
        ),
    ),
    _evidence(
        "knockdowns_per_distance_strike_landed_proxy",
        SAFE_RATIO,
        (
            _name("knockdowns_scored"),
            _name("distance_strikes_landed"),
        ),
        (_name("distance_strikes_landed"),),
        unit_interval=False,
        description=(
            "Knockdowns divided by landed distance strikes; this is "
            "a proxy because UFCStats does not identify knockdown phase."
        ),
    ),
    _evidence(
        "clinch_strike_accuracy",
        SAFE_RATIO,
        (
            _name("clinch_strikes_landed"),
            _name("clinch_strike_attempts"),
        ),
        (_name("clinch_strike_attempts"),),
        unit_interval=True,
        description="Completed-fight clinch strike accuracy.",
    ),
    _evidence(
        "clinch_damage_output_per_round",
        PER_ROUND,
        (
            _name("clinch_strikes_landed"),
            _name("rounds_observed"),
        ),
        (_name("rounds_observed"),),
        unit_interval=False,
        description=(
            "Landed clinch strikes per observed round as a damaging "
            "clinch activity proxy."
        ),
    ),
    _evidence(
        "submission_attempts_per_ground_opportunity_proxy",
        SAFE_RATIO,
        (
            _name("submission_attempts"),
            _name("takedowns_landed"),
            _name("ground_strike_attempts"),
        ),
        (
            _name("takedowns_landed"),
            _name("ground_strike_attempts"),
        ),
        unit_interval=False,
        description=(
            "Submission attempts divided by observable ground-entry "
            "and ground-activity opportunities."
        ),
    ),
    _evidence(
        "submission_attempts_per_control_minute",
        SAFE_RATIO,
        (
            _name("submission_attempts"),
            _name("control_seconds"),
        ),
        (_name("control_seconds"),),
        unit_interval=False,
        description=(
            "Submission attempts per recorded control minute."
        ),
    ),
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
        description=(
            "Rounds containing observable opponent adversity signals."
        ),
    ),
    _evidence(
        "damage_exposure_composite",
        DAMAGE_EXPOSURE,
        (
            _name("knockdowns_absorbed"),
            _name("head_strikes_absorbed"),
            _name("ground_strikes_absorbed"),
            _name("opponent_control_seconds"),
        ),
        (_name("rounds_observed"),),
        unit_interval=False,
        description=(
            "Uncalibrated composite of observable damaging exposure."
        ),
    ),
    _evidence(
        "same_round_output_preservation",
        OUTPUT_PRESERVATION,
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
        requires_adversity=True,
        description=(
            "Fighter output preserved within rounds containing "
            "observable adversity."
        ),
    ),
    _evidence(
        "same_round_efficiency_preservation",
        EFFICIENCY_PRESERVATION,
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
        requires_adversity=True,
        description=(
            "Significant-strike efficiency preserved within adversity "
            "rounds."
        ),
    ),
    _evidence(
        "ko_tko_loss_indicator",
        OUTCOME_INDICATOR,
        (
            "winner_id",
            "method",
        ),
        (_name("valid_outcome"),),
        unit_interval=True,
        requires_valid_outcome=True,
        description=(
            "Whether the fighter lost by KO/TKO or doctor stoppage."
        ),
    ),
    _evidence(
        "ko_tko_survival_indicator",
        SURVIVAL_INDICATOR,
        (
            "winner_id",
            "method",
        ),
        (
            _name("valid_outcome"),
            _name("damage_exposure_composite"),
        ),
        unit_interval=True,
        requires_valid_outcome=True,
        description=(
            "Whether the fighter avoided a KO/TKO loss in a valid "
            "completed outcome."
        ),
    ),
    _evidence(
        "submission_loss_indicator",
        OUTCOME_INDICATOR,
        (
            "winner_id",
            "method",
        ),
        (_name("valid_outcome"),),
        unit_interval=True,
        requires_valid_outcome=True,
        description="Whether the fighter lost by submission.",
    ),
    _evidence(
        "submission_survival_indicator",
        SURVIVAL_INDICATOR,
        (
            "winner_id",
            "method",
        ),
        (
            _name("valid_outcome"),
            _name("opponent_submission_attempts"),
        ),
        unit_interval=True,
        requires_valid_outcome=True,
        description=(
            "Whether the fighter avoided a submission loss in a valid "
            "completed outcome."
        ),
    ),
)


FINISH_STATE_EVIDENCE_BY_NAME = {
    spec.feature_name: spec
    for spec in FINISH_STATE_EVIDENCE_SPECS
}


FINISH_STATE_FIGHT_OBSERVATION_COLUMNS = tuple(
    [
        spec.feature_name
        for spec in FINISH_STATE_AGGREGATE_SPECS
    ]
    + [
        spec.feature_name
        for spec in FINISH_STATE_EVIDENCE_SPECS
    ]
)


# ---------------------------------------------------------------------
# Simulator-target evidence coverage
# ---------------------------------------------------------------------

FINISH_STATE_TARGET_EVIDENCE: dict[
    str,
    tuple[str, ...],
] = {
    "phase.distance.knockdown_probability_per_landed": (
        _name("knockdowns_per_sig_strike_landed"),
        _name("knockdowns_per_distance_strike_landed_proxy"),
        _name("sig_strikes_landed"),
        _name("distance_strikes_landed"),
    ),
    "phase.clinch.damaging_clinch_probability": (
        _name("clinch_strike_accuracy"),
        _name("clinch_damage_output_per_round"),
        _name("clinch_strikes_landed"),
        _name("clinch_strike_attempts"),
    ),
    "phase.ground_owner.submission_attempt_rate": (
        _name("submission_attempts"),
        _name("submission_attempts_per_ground_opportunity_proxy"),
        _name("submission_attempts_per_control_minute"),
        _name("takedowns_landed"),
        _name("ground_strike_attempts"),
        _name("control_seconds"),
    ),
    "phase.ground_defender.submission_defense": (
        _name("opponent_submission_attempts"),
        _name("submission_loss_indicator"),
        _name("submission_survival_indicator"),
        _name("opponent_control_seconds"),
    ),
    "dynamic.damage_resistance": (
        _name("damage_exposure_composite"),
        _name("knockdowns_absorbed"),
        _name("head_strikes_absorbed"),
        _name("ground_strikes_absorbed"),
        _name("ko_tko_loss_indicator"),
        _name("ko_tko_survival_indicator"),
    ),
    "dynamic.acute_stress_resistance": (
        _name("adversity_round_count"),
        _name("same_round_output_preservation"),
        _name("same_round_efficiency_preservation"),
        _name("knockdowns_absorbed"),
    ),
}


FINISH_STATE_TARGETS = frozenset(
    FINISH_STATE_TARGET_EVIDENCE
)


def validate_finish_state_contracts() -> None:
    """Validate exact Finish State target and feature coverage."""

    expected_targets = {
        target_name
        for target_name, target_spec
        in SIMULATOR_TARGET_BY_NAME.items()
        if (
            target_spec.primary_family
            is SimulatorFeatureFamily.FINISH_STATE
        )
    }

    if FINISH_STATE_TARGETS != expected_targets:
        missing = sorted(
            expected_targets - FINISH_STATE_TARGETS
        )
        extra = sorted(
            FINISH_STATE_TARGETS - expected_targets
        )

        raise RuntimeError(
            "Finish State target coverage mismatch: "
            f"missing={missing}, extra={extra}"
        )

    observation_names = set(
        FINISH_STATE_FIGHT_OBSERVATION_COLUMNS
    )

    if (
        len(observation_names)
        != len(FINISH_STATE_FIGHT_OBSERVATION_COLUMNS)
    ):
        raise RuntimeError(
            "Finish State observation names must be unique"
        )

    for target_name, evidence_names in (
        FINISH_STATE_TARGET_EVIDENCE.items()
    ):
        if target_name not in SIMULATOR_TARGET_BY_NAME:
            raise RuntimeError(
                "Unknown Finish State target: "
                f"{target_name}"
            )

        if not evidence_names:
            raise RuntimeError(
                "Finish State target has no evidence: "
                f"{target_name}"
            )

        unknown_evidence = sorted(
            set(evidence_names) - observation_names
        )

        if unknown_evidence:
            raise RuntimeError(
                "Finish State target references unknown evidence: "
                f"target={target_name}, "
                f"unknown={unknown_evidence}"
            )


validate_finish_state_contracts()
