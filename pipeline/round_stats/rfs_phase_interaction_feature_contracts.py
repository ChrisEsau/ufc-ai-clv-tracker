"""Observed feature contracts for the RFS simulator Phase Interaction family.

Phase Interaction describes how one fighter's behavior meets an opponent's
behavior. The authoritative round source provides reciprocal fighter rows, so
we can observe achieved offense, offense allowed, takedown resistance, control
competition, reversal activity, and phase-pressure balance.

Important boundary
------------------
UFCStats does not record exact phase entries, exits, escapes, scrambles, or
position advancements. Those simulator parameters remain calibrated latent
outputs. The features declared here are observable evidence and opportunity
counts; they are not final simulator transition probabilities.

This module contains contracts only. It does not build DataFrames, write
artifacts, or fit calibration coefficients.
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


PHASE_INTERACTION_PREFIX = "rfs_phase_interact_fight_"


class InteractionAggregateRule(str, Enum):
    """How one fighter-fight aggregate is built from round rows."""

    UNIQUE_COUNT = "unique_count"
    SUM = "sum"


class PhaseInteractionFormula(str, Enum):
    """Locked formula category for one interaction evidence feature."""

    SAFE_RATIO = "safe_ratio"
    COMPLEMENT_RATIO = "complement_ratio"
    PHASE_ATTEMPT_SHARE = "phase_attempt_share"
    NON_DISTANCE_ATTEMPT_SHARE = "non_distance_attempt_share"
    SHARE_OF_COMBINED = "share_of_combined"
    PER_OBSERVED_ROUND = "per_observed_round"
    DIFFERENCE_PER_ROUND = "difference_per_round"
    PER_CONTROL_MINUTE = "per_control_minute"
    COMBINED_PER_CONTROL_MINUTE = "combined_per_control_minute"
    BALANCE_INDEX = "balance_index"


@dataclass(frozen=True)
class PhaseInteractionAggregateSpec:
    """One mirrored fighter and opponent fight-level aggregate."""

    feature_name: str
    opponent_feature_name: str
    rule: InteractionAggregateRule
    source_column: str
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.feature_name, str):
            raise TypeError("feature_name must be a string")

        if not isinstance(self.opponent_feature_name, str):
            raise TypeError(
                "opponent_feature_name must be a string"
            )

        for feature_name in (
            self.feature_name,
            self.opponent_feature_name,
        ):
            if not feature_name.startswith(
                PHASE_INTERACTION_PREFIX
            ):
                raise ValueError(
                    "aggregate names must use the "
                    "Phase Interaction prefix"
                )

        if self.feature_name == self.opponent_feature_name:
            raise ValueError(
                "fighter and opponent feature names must differ"
            )

        if not self.opponent_feature_name.startswith(
            f"{PHASE_INTERACTION_PREFIX}opp_"
        ):
            raise ValueError(
                "opponent_feature_name must use the opp_ marker"
            )

        if not isinstance(
            self.rule,
            InteractionAggregateRule,
        ):
            raise TypeError(
                "rule must be InteractionAggregateRule"
            )

        if not isinstance(self.source_column, str):
            raise TypeError(
                "source_column must be a string"
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
class PhaseInteractionEvidenceSpec:
    """One derived fighter-versus-opponent evidence feature."""

    feature_name: str
    formula: PhaseInteractionFormula
    input_features: tuple[str, ...]
    reliability_features: tuple[str, ...]
    unit_interval: bool
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.feature_name, str):
            raise TypeError("feature_name must be a string")

        if not self.feature_name.startswith(
            PHASE_INTERACTION_PREFIX
        ):
            raise ValueError(
                "feature_name must use the "
                "Phase Interaction prefix"
            )

        if "30_seconds" in self.feature_name:
            raise ValueError(
                "observed interaction evidence cannot claim "
                "exact 30-second phase exposure"
            )

        if not isinstance(
            self.formula,
            PhaseInteractionFormula,
        ):
            raise TypeError(
                "formula must be PhaseInteractionFormula"
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
    """Validate one nonblank contract description."""

    if not isinstance(value, str):
        raise TypeError(
            "description must be a string"
        )

    if not value.strip():
        raise ValueError(
            "description cannot be empty"
        )


def _name(suffix: str) -> str:
    """Return one fighter-perspective observation name."""

    return f"{PHASE_INTERACTION_PREFIX}{suffix}"


def _opp_name(suffix: str) -> str:
    """Return the reciprocal opponent aggregate name."""

    return f"{PHASE_INTERACTION_PREFIX}opp_{suffix}"


def _aggregate(
    suffix: str,
    rule: InteractionAggregateRule,
    source_column: str,
    description: str,
) -> PhaseInteractionAggregateSpec:
    """Build one mirrored aggregate declaration."""

    return PhaseInteractionAggregateSpec(
        feature_name=_name(suffix),
        opponent_feature_name=_opp_name(suffix),
        rule=rule,
        source_column=source_column,
        description=description,
    )


def _evidence(
    suffix: str,
    formula: PhaseInteractionFormula,
    input_features: tuple[str, ...],
    reliability_features: tuple[str, ...],
    *,
    unit_interval: bool,
    description: str,
) -> PhaseInteractionEvidenceSpec:
    """Build one derived interaction evidence declaration."""

    return PhaseInteractionEvidenceSpec(
        feature_name=_name(suffix),
        formula=formula,
        input_features=input_features,
        reliability_features=reliability_features,
        unit_interval=unit_interval,
        description=description,
    )


UNIQUE = InteractionAggregateRule.UNIQUE_COUNT
SUM = InteractionAggregateRule.SUM

SAFE_RATIO = PhaseInteractionFormula.SAFE_RATIO
COMPLEMENT_RATIO = (
    PhaseInteractionFormula.COMPLEMENT_RATIO
)
PHASE_SHARE = (
    PhaseInteractionFormula.PHASE_ATTEMPT_SHARE
)
NON_DISTANCE_SHARE = (
    PhaseInteractionFormula.NON_DISTANCE_ATTEMPT_SHARE
)
COMBINED_SHARE = (
    PhaseInteractionFormula.SHARE_OF_COMBINED
)
PER_ROUND = (
    PhaseInteractionFormula.PER_OBSERVED_ROUND
)
DIFF_PER_ROUND = (
    PhaseInteractionFormula.DIFFERENCE_PER_ROUND
)
PER_CONTROL_MIN = (
    PhaseInteractionFormula.PER_CONTROL_MINUTE
)
COMBINED_PER_CONTROL_MIN = (
    PhaseInteractionFormula.COMBINED_PER_CONTROL_MINUTE
)
BALANCE = PhaseInteractionFormula.BALANCE_INDEX


# ---------------------------------------------------------------------
# Mirrored fighter and opponent fight aggregates
# ---------------------------------------------------------------------

PHASE_INTERACTION_AGGREGATE_SPECS: tuple[
    PhaseInteractionAggregateSpec,
    ...,
] = (
    _aggregate(
        "rounds_observed",
        UNIQUE,
        "round",
        "Distinct recorded rounds for each fighter in the fight.",
    ),
    _aggregate(
        "sig_strike_attempts",
        SUM,
        "sig_str_attempted",
        "Total significant-strike attempts.",
    ),
    _aggregate(
        "distance_landed",
        SUM,
        "distance_landed",
        "Landed distance significant strikes.",
    ),
    _aggregate(
        "distance_attempts",
        SUM,
        "distance_attempted",
        "Attempted distance significant strikes.",
    ),
    _aggregate(
        "clinch_landed",
        SUM,
        "clinch_landed",
        "Landed clinch significant strikes.",
    ),
    _aggregate(
        "clinch_attempts",
        SUM,
        "clinch_attempted",
        "Attempted clinch significant strikes.",
    ),
    _aggregate(
        "ground_landed",
        SUM,
        "ground_landed",
        "Landed ground significant strikes.",
    ),
    _aggregate(
        "ground_attempts",
        SUM,
        "ground_attempted",
        "Attempted ground significant strikes.",
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
        "sub_attempts",
        SUM,
        "sub_att",
        "Recorded submission attempts.",
    ),
    _aggregate(
        "reversals",
        SUM,
        "rev",
        "Recorded reversals.",
    ),
    _aggregate(
        "control_seconds",
        SUM,
        "ctrl_sec",
        (
            "Recorded control seconds without assigning exact "
            "clinch or ground exposure."
        ),
    ),
)


PHASE_INTERACTION_AGGREGATE_NAMES = frozenset(
    feature_name
    for spec in PHASE_INTERACTION_AGGREGATE_SPECS
    for feature_name in (
        spec.feature_name,
        spec.opponent_feature_name,
    )
)


# ---------------------------------------------------------------------
# Derived interaction evidence
# ---------------------------------------------------------------------

PHASE_INTERACTION_EVIDENCE_SPECS: tuple[
    PhaseInteractionEvidenceSpec,
    ...,
] = (
    # Achieved and allowed phase accuracy.
    _evidence(
        "distance_accuracy",
        SAFE_RATIO,
        (
            _name("distance_landed"),
            _name("distance_attempts"),
        ),
        (_name("distance_attempts"),),
        unit_interval=True,
        description=(
            "Fighter distance accuracy achieved against the opponent."
        ),
    ),
    _evidence(
        "distance_accuracy_allowed",
        SAFE_RATIO,
        (
            _opp_name("distance_landed"),
            _opp_name("distance_attempts"),
        ),
        (_opp_name("distance_attempts"),),
        unit_interval=True,
        description=(
            "Opponent distance accuracy achieved against the fighter."
        ),
    ),
    _evidence(
        "clinch_accuracy",
        SAFE_RATIO,
        (
            _name("clinch_landed"),
            _name("clinch_attempts"),
        ),
        (_name("clinch_attempts"),),
        unit_interval=True,
        description=(
            "Fighter clinch accuracy achieved against the opponent."
        ),
    ),
    _evidence(
        "clinch_accuracy_allowed",
        SAFE_RATIO,
        (
            _opp_name("clinch_landed"),
            _opp_name("clinch_attempts"),
        ),
        (_opp_name("clinch_attempts"),),
        unit_interval=True,
        description=(
            "Opponent clinch accuracy achieved against the fighter."
        ),
    ),
    _evidence(
        "ground_accuracy",
        SAFE_RATIO,
        (
            _name("ground_landed"),
            _name("ground_attempts"),
        ),
        (_name("ground_attempts"),),
        unit_interval=True,
        description=(
            "Fighter ground accuracy achieved against the opponent."
        ),
    ),
    _evidence(
        "ground_accuracy_allowed",
        SAFE_RATIO,
        (
            _opp_name("ground_landed"),
            _opp_name("ground_attempts"),
        ),
        (_opp_name("ground_attempts"),),
        unit_interval=True,
        description=(
            "Opponent ground accuracy achieved against the fighter."
        ),
    ),
    _evidence(
        "td_completion_allowed",
        SAFE_RATIO,
        (
            _opp_name("td_landed"),
            _opp_name("td_attempts"),
        ),
        (_opp_name("td_attempts"),),
        unit_interval=True,
        description=(
            "Opponent takedown completion rate against the fighter."
        ),
    ),
    _evidence(
        "td_defense_rate",
        COMPLEMENT_RATIO,
        (
            _opp_name("td_landed"),
            _opp_name("td_attempts"),
        ),
        (_opp_name("td_attempts"),),
        unit_interval=True,
        description=(
            "Complement of opponent takedown completion against "
            "the fighter."
        ),
    ),

    # Fighter and allowed phase mix.
    _evidence(
        "distance_attempt_share",
        PHASE_SHARE,
        (
            _name("distance_attempts"),
            _name("sig_strike_attempts"),
        ),
        (_name("sig_strike_attempts"),),
        unit_interval=True,
        description=(
            "Fighter distance attempts divided by total "
            "significant-strike attempts."
        ),
    ),
    _evidence(
        "distance_attempt_share_allowed",
        PHASE_SHARE,
        (
            _opp_name("distance_attempts"),
            _opp_name("sig_strike_attempts"),
        ),
        (_opp_name("sig_strike_attempts"),),
        unit_interval=True,
        description=(
            "Opponent distance-attempt share against the fighter."
        ),
    ),
    _evidence(
        "clinch_attempt_share",
        PHASE_SHARE,
        (
            _name("clinch_attempts"),
            _name("sig_strike_attempts"),
        ),
        (_name("sig_strike_attempts"),),
        unit_interval=True,
        description=(
            "Fighter clinch attempts divided by total "
            "significant-strike attempts."
        ),
    ),
    _evidence(
        "clinch_attempt_share_allowed",
        PHASE_SHARE,
        (
            _opp_name("clinch_attempts"),
            _opp_name("sig_strike_attempts"),
        ),
        (_opp_name("sig_strike_attempts"),),
        unit_interval=True,
        description=(
            "Opponent clinch-attempt share against the fighter."
        ),
    ),
    _evidence(
        "ground_attempt_share",
        PHASE_SHARE,
        (
            _name("ground_attempts"),
            _name("sig_strike_attempts"),
        ),
        (_name("sig_strike_attempts"),),
        unit_interval=True,
        description=(
            "Fighter ground attempts divided by total "
            "significant-strike attempts."
        ),
    ),
    _evidence(
        "ground_attempt_share_allowed",
        PHASE_SHARE,
        (
            _opp_name("ground_attempts"),
            _opp_name("sig_strike_attempts"),
        ),
        (_opp_name("sig_strike_attempts"),),
        unit_interval=True,
        description=(
            "Opponent ground-attempt share against the fighter."
        ),
    ),
    _evidence(
        "non_distance_attempt_share",
        NON_DISTANCE_SHARE,
        (
            _name("clinch_attempts"),
            _name("ground_attempts"),
            _name("sig_strike_attempts"),
        ),
        (_name("sig_strike_attempts"),),
        unit_interval=True,
        description=(
            "Fighter clinch plus ground attempts divided by total "
            "significant-strike attempts."
        ),
    ),
    _evidence(
        "non_distance_attempt_share_allowed",
        NON_DISTANCE_SHARE,
        (
            _opp_name("clinch_attempts"),
            _opp_name("ground_attempts"),
            _opp_name("sig_strike_attempts"),
        ),
        (_opp_name("sig_strike_attempts"),),
        unit_interval=True,
        description=(
            "Opponent non-distance attempt share against the fighter."
        ),
    ),

    # Competition shares between the two fighters.
    _evidence(
        "distance_pressure_share",
        COMBINED_SHARE,
        (
            _name("distance_attempts"),
            _opp_name("distance_attempts"),
        ),
        (
            _name("distance_attempts"),
            _opp_name("distance_attempts"),
        ),
        unit_interval=True,
        description=(
            "Fighter share of both fighters' distance attempts."
        ),
    ),
    _evidence(
        "clinch_pressure_share",
        COMBINED_SHARE,
        (
            _name("clinch_attempts"),
            _opp_name("clinch_attempts"),
        ),
        (
            _name("clinch_attempts"),
            _opp_name("clinch_attempts"),
        ),
        unit_interval=True,
        description=(
            "Fighter share of both fighters' clinch attempts."
        ),
    ),
    _evidence(
        "ground_pressure_share",
        COMBINED_SHARE,
        (
            _name("ground_attempts"),
            _opp_name("ground_attempts"),
        ),
        (
            _name("ground_attempts"),
            _opp_name("ground_attempts"),
        ),
        unit_interval=True,
        description=(
            "Fighter share of both fighters' ground attempts."
        ),
    ),
    _evidence(
        "td_pressure_share",
        COMBINED_SHARE,
        (
            _name("td_attempts"),
            _opp_name("td_attempts"),
        ),
        (
            _name("td_attempts"),
            _opp_name("td_attempts"),
        ),
        unit_interval=True,
        description=(
            "Fighter share of both fighters' takedown attempts."
        ),
    ),
    _evidence(
        "control_share",
        COMBINED_SHARE,
        (
            _name("control_seconds"),
            _opp_name("control_seconds"),
        ),
        (
            _name("control_seconds"),
            _opp_name("control_seconds"),
        ),
        unit_interval=True,
        description=(
            "Fighter share of combined recorded control time."
        ),
    ),
    _evidence(
        "control_exchange_balance",
        BALANCE,
        (
            _name("control_seconds"),
            _opp_name("control_seconds"),
        ),
        (
            _name("control_seconds"),
            _opp_name("control_seconds"),
        ),
        unit_interval=True,
        description=(
            "Symmetric balance of fighter and opponent control time; "
            "high values indicate both recorded meaningful control."
        ),
    ),

    # Opponent pressure allowed and control competition.
    _evidence(
        "control_differential_per_round",
        DIFF_PER_ROUND,
        (
            _name("control_seconds"),
            _opp_name("control_seconds"),
            _name("rounds_observed"),
        ),
        (
            _name("rounds_observed"),
            _name("control_seconds"),
            _opp_name("control_seconds"),
        ),
        unit_interval=False,
        description=(
            "Fighter minus opponent control seconds per observed round."
        ),
    ),
    _evidence(
        "control_seconds_allowed_per_round",
        PER_ROUND,
        (
            _opp_name("control_seconds"),
            _name("rounds_observed"),
        ),
        (
            _name("rounds_observed"),
            _opp_name("control_seconds"),
        ),
        unit_interval=False,
        description=(
            "Opponent control seconds against the fighter per "
            "observed round."
        ),
    ),
    _evidence(
        "clinch_attempts_allowed_per_round",
        PER_ROUND,
        (
            _opp_name("clinch_attempts"),
            _name("rounds_observed"),
        ),
        (
            _name("rounds_observed"),
            _opp_name("clinch_attempts"),
        ),
        unit_interval=False,
        description=(
            "Opponent clinch attempts against the fighter per round."
        ),
    ),
    _evidence(
        "ground_attempts_allowed_per_round",
        PER_ROUND,
        (
            _opp_name("ground_attempts"),
            _name("rounds_observed"),
        ),
        (
            _name("rounds_observed"),
            _opp_name("ground_attempts"),
        ),
        unit_interval=False,
        description=(
            "Opponent ground attempts against the fighter per round."
        ),
    ),
    _evidence(
        "td_attempts_allowed_per_round",
        PER_ROUND,
        (
            _opp_name("td_attempts"),
            _name("rounds_observed"),
        ),
        (
            _name("rounds_observed"),
            _opp_name("td_attempts"),
        ),
        unit_interval=False,
        description=(
            "Opponent takedown attempts against the fighter per round."
        ),
    ),

    # Control-normalized ground and reversal evidence.
    _evidence(
        "reversal_rate_per_opponent_control_min",
        PER_CONTROL_MIN,
        (
            _name("reversals"),
            _opp_name("control_seconds"),
        ),
        (
            _name("reversals"),
            _opp_name("control_seconds"),
        ),
        unit_interval=False,
        description=(
            "Fighter reversals per minute of opponent control."
        ),
    ),
    _evidence(
        "reversal_allowed_per_control_min",
        PER_CONTROL_MIN,
        (
            _opp_name("reversals"),
            _name("control_seconds"),
        ),
        (
            _opp_name("reversals"),
            _name("control_seconds"),
        ),
        unit_interval=False,
        description=(
            "Opponent reversals per minute of fighter control."
        ),
    ),
    _evidence(
        "ground_attempts_per_control_min",
        PER_CONTROL_MIN,
        (
            _name("ground_attempts"),
            _name("control_seconds"),
        ),
        (
            _name("ground_attempts"),
            _name("control_seconds"),
        ),
        unit_interval=False,
        description=(
            "Fighter ground attempts per minute of fighter control; "
            "evidence only because exact ground exposure is unavailable."
        ),
    ),
    _evidence(
        "ground_landed_per_control_min",
        PER_CONTROL_MIN,
        (
            _name("ground_landed"),
            _name("control_seconds"),
        ),
        (
            _name("ground_landed"),
            _name("control_seconds"),
        ),
        unit_interval=False,
        description=(
            "Fighter landed ground strikes per minute of control."
        ),
    ),
    _evidence(
        "sub_attempts_per_control_min",
        PER_CONTROL_MIN,
        (
            _name("sub_attempts"),
            _name("control_seconds"),
        ),
        (
            _name("sub_attempts"),
            _name("control_seconds"),
        ),
        unit_interval=False,
        description=(
            "Fighter submission attempts per minute of control."
        ),
    ),
    _evidence(
        "ground_attempts_allowed_per_control_min",
        PER_CONTROL_MIN,
        (
            _opp_name("ground_attempts"),
            _opp_name("control_seconds"),
        ),
        (
            _opp_name("ground_attempts"),
            _opp_name("control_seconds"),
        ),
        unit_interval=False,
        description=(
            "Opponent ground attempts per minute of opponent control."
        ),
    ),
    _evidence(
        "ground_landed_allowed_per_control_min",
        PER_CONTROL_MIN,
        (
            _opp_name("ground_landed"),
            _opp_name("control_seconds"),
        ),
        (
            _opp_name("ground_landed"),
            _opp_name("control_seconds"),
        ),
        unit_interval=False,
        description=(
            "Opponent landed ground strikes per minute of "
            "opponent control."
        ),
    ),
    _evidence(
        "sub_attempts_allowed_per_control_min",
        PER_CONTROL_MIN,
        (
            _opp_name("sub_attempts"),
            _opp_name("control_seconds"),
        ),
        (
            _opp_name("sub_attempts"),
            _opp_name("control_seconds"),
        ),
        unit_interval=False,
        description=(
            "Opponent submission attempts per minute of "
            "opponent control."
        ),
    ),
    _evidence(
        "combined_reversals_per_control_min",
        COMBINED_PER_CONTROL_MIN,
        (
            _name("reversals"),
            _opp_name("reversals"),
            _name("control_seconds"),
            _opp_name("control_seconds"),
        ),
        (
            _name("reversals"),
            _opp_name("reversals"),
            _name("control_seconds"),
            _opp_name("control_seconds"),
        ),
        unit_interval=False,
        description=(
            "Both fighters' reversals per combined control minute; "
            "a fight-level indicator of scramble-rich interaction."
        ),
    ),
)


PHASE_INTERACTION_EVIDENCE_BY_NAME = {
    spec.feature_name: spec
    for spec in PHASE_INTERACTION_EVIDENCE_SPECS
}


PHASE_INTERACTION_FIGHT_OBSERVATION_COLUMNS = tuple(
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


# ---------------------------------------------------------------------
# Final simulator targets linked to observable interaction evidence
# ---------------------------------------------------------------------

PHASE_INTERACTION_TARGET_EVIDENCE: dict[
    str,
    tuple[str, ...],
] = {
    "transition.distance_retention": (
        _name("distance_attempt_share"),
        _name("distance_pressure_share"),
        _name("non_distance_attempt_share_allowed"),
        _name("td_attempts_allowed_per_round"),
        _name("control_seconds_allowed_per_round"),
        _name("control_share"),
    ),
    "transition.clinch_entry_resistance": (
        _name("clinch_attempt_share_allowed"),
        _name("clinch_attempts_allowed_per_round"),
        _name("control_seconds_allowed_per_round"),
        _name("distance_pressure_share"),
    ),
    "transition.takedown_resistance": (
        _name("td_defense_rate"),
        _name("td_completion_allowed"),
        _name("td_attempts_allowed_per_round"),
        _opp_name("td_attempts"),
    ),
    "transition.clinch_retention": (
        _name("clinch_pressure_share"),
        _name("control_share"),
        _name("control_exchange_balance"),
        _name("reversal_allowed_per_control_min"),
    ),
    "transition.clinch_escape_ability": (
        _name("clinch_attempt_share_allowed"),
        _name("clinch_attempts_allowed_per_round"),
        _name("control_seconds_allowed_per_round"),
        _name("reversal_rate_per_opponent_control_min"),
        _name("distance_pressure_share"),
    ),
    "transition.ground_retention": (
        _name("ground_pressure_share"),
        _name("control_share"),
        _name("ground_attempts_per_control_min"),
        _name("ground_landed_per_control_min"),
        _name("sub_attempts_per_control_min"),
        _name("reversal_allowed_per_control_min"),
    ),
    "transition.ground_escape_ability": (
        _name("control_seconds_allowed_per_round"),
        _name("ground_attempts_allowed_per_control_min"),
        _name("ground_landed_allowed_per_control_min"),
        _name("sub_attempts_allowed_per_control_min"),
        _name("reversal_rate_per_opponent_control_min"),
        _name("distance_pressure_share"),
    ),
    "transition.reversal_ability": (
        _name("reversal_rate_per_opponent_control_min"),
        _name("reversals"),
        _opp_name("control_seconds"),
    ),
    "transition.phase_imposition": (
        _name("non_distance_attempt_share"),
        _name("clinch_pressure_share"),
        _name("ground_pressure_share"),
        _name("td_pressure_share"),
        _name("control_share"),
    ),
    "transition.phase_resistance": (
        _name("non_distance_attempt_share_allowed"),
        _name("td_defense_rate"),
        _name("distance_pressure_share"),
        _name("control_seconds_allowed_per_round"),
        _name("control_exchange_balance"),
    ),
    "phase.distance.sig_strike_accuracy": (
        _name("distance_accuracy"),
        _name("distance_accuracy_allowed"),
        _name("distance_attempts"),
        _opp_name("distance_attempts"),
    ),
    "phase.clinch.clinch_strike_accuracy": (
        _name("clinch_accuracy"),
        _name("clinch_accuracy_allowed"),
        _name("clinch_attempts"),
        _opp_name("clinch_attempts"),
    ),
    "phase.ground_owner.ground_strike_accuracy": (
        _name("ground_accuracy"),
        _name("ground_accuracy_allowed"),
        _name("ground_attempts"),
        _opp_name("ground_attempts"),
    ),
    "phase.ground_owner.position_advancement_probability": (
        _name("ground_pressure_share"),
        _name("control_share"),
        _name("ground_landed_per_control_min"),
        _name("sub_attempts_per_control_min"),
        _name("reversal_allowed_per_control_min"),
    ),
    "phase.ground_defender.escape_attempt_rate": (
        _name("control_seconds_allowed_per_round"),
        _name("reversal_rate_per_opponent_control_min"),
        _name("distance_pressure_share"),
        _name("ground_attempts_allowed_per_control_min"),
    ),
    "phase.ground_defender.reversal_attempt_rate": (
        _name("reversal_rate_per_opponent_control_min"),
        _name("reversals"),
        _opp_name("control_seconds"),
    ),
    "phase.ground_defender.scramble_attempt_rate": (
        _name("combined_reversals_per_control_min"),
        _name("control_exchange_balance"),
        _name("reversal_rate_per_opponent_control_min"),
        _name("control_seconds_allowed_per_round"),
    ),
}


def validate_phase_interaction_feature_contracts() -> None:
    """Validate names, formulas, target coverage, and evidence inputs."""

    aggregate_names = [
        feature_name
        for spec in PHASE_INTERACTION_AGGREGATE_SPECS
        for feature_name in (
            spec.feature_name,
            spec.opponent_feature_name,
        )
    ]
    evidence_names = [
        spec.feature_name
        for spec in PHASE_INTERACTION_EVIDENCE_SPECS
    ]

    if len(aggregate_names) != len(set(aggregate_names)):
        raise RuntimeError(
            "Phase Interaction aggregate names must be unique"
        )

    if len(evidence_names) != len(set(evidence_names)):
        raise RuntimeError(
            "Phase Interaction evidence names must be unique"
        )

    if set(aggregate_names) & set(evidence_names):
        raise RuntimeError(
            "aggregate and evidence names cannot overlap"
        )

    expected_targets = {
        spec.target_parameter
        for spec in SIMULATOR_TARGET_BY_NAME.values()
        if (
            spec.primary_family
            is SimulatorFeatureFamily.PHASE_INTERACTION
        )
    }
    mapped_targets = set(
        PHASE_INTERACTION_TARGET_EVIDENCE
    )

    if mapped_targets != expected_targets:
        missing = sorted(
            expected_targets - mapped_targets
        )
        extra = sorted(
            mapped_targets - expected_targets
        )

        raise RuntimeError(
            "Phase Interaction target evidence does not match "
            f"the target registry; missing={missing}, extra={extra}"
        )

    valid_feature_names = (
        set(aggregate_names)
        | set(evidence_names)
    )

    for spec in PHASE_INTERACTION_EVIDENCE_SPECS:
        for input_name in spec.input_features:
            if input_name not in aggregate_names:
                raise RuntimeError(
                    "interaction formula inputs must be "
                    f"fight aggregates: {input_name}"
                )

        for reliability_name in (
            spec.reliability_features
        ):
            if reliability_name not in aggregate_names:
                raise RuntimeError(
                    "interaction reliability inputs must be "
                    f"fight aggregates: {reliability_name}"
                )

    for target, feature_names in (
        PHASE_INTERACTION_TARGET_EVIDENCE.items()
    ):
        if not feature_names:
            raise RuntimeError(
                f"target has no interaction evidence: {target}"
            )

        for feature_name in feature_names:
            if feature_name not in valid_feature_names:
                raise RuntimeError(
                    f"target {target} references unknown "
                    f"interaction evidence: {feature_name}"
                )


validate_phase_interaction_feature_contracts()
