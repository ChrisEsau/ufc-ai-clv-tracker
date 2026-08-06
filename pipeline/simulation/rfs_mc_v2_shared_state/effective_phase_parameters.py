"""Temporary effective phase parameters for RFS Monte Carlo V2.

Current fatigue, persistent damage, and acute stress reduce a fighter's
baseline phase capabilities for the next segment.

Baseline ``FighterPhaseParameters`` are never mutated. This module returns a
new temporary parameter bundle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_effect_calibration import (
    DynamicEffectCalibration,
    StatePenaltyWeights,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_parameters import (
    FighterDynamicParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_state import (
    FighterDynamicState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    ClinchRateParameters,
    DistanceRateParameters,
    FighterPhaseParameters,
    GroundDefenderRateParameters,
    GroundOwnerRateParameters,
)


@dataclass(frozen=True)
class CapabilityMultipliers:
    """Temporary multipliers applied to phase-capability families."""

    output: float
    accuracy: float
    power: float
    control: float
    grappling: float
    defense: float

    def __post_init__(self) -> None:
        """Validate normalized capability multipliers."""

        for name, value in vars(self).items():
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name} must be numeric"
                )

            selected = float(value)

            if not math.isfinite(selected):
                raise ValueError(
                    f"{name} must be finite"
                )

            if not 0.0 <= selected <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )


def calculate_fatigue_effect_multiplier(
    parameters: FighterDynamicParameters,
    calibration: DynamicEffectCalibration,
) -> float:
    """Return how strongly accumulated fatigue affects this fighter.

    Performance resilience zero receives the full fatigue effect.

    Performance resilience one receives the configured minimum fatigue-effect
    multiplier rather than complete immunity.
    """

    if not isinstance(
        parameters,
        FighterDynamicParameters,
    ):
        raise TypeError(
            "parameters must be FighterDynamicParameters"
        )

    if not isinstance(
        calibration,
        DynamicEffectCalibration,
    ):
        raise TypeError(
            "calibration must be DynamicEffectCalibration"
        )

    minimum_effect = (
        calibration.minimum_fatigue_effect_multiplier
    )

    return (
        1.0
        - parameters.fatigue_performance_resilience
        * (
            1.0
            - minimum_effect
        )
    )


def calculate_capability_multiplier(
    state: FighterDynamicState,
    parameters: FighterDynamicParameters,
    weights: StatePenaltyWeights,
    calibration: DynamicEffectCalibration,
) -> float:
    """Calculate one temporary capability multiplier.

    Fatigue is first adjusted by the fighter's fatigue-performance
    resilience. Damage and acute stress apply directly through their
    capability-specific weights.
    """

    if not isinstance(
        state,
        FighterDynamicState,
    ):
        raise TypeError(
            "state must be FighterDynamicState"
        )

    if not isinstance(
        parameters,
        FighterDynamicParameters,
    ):
        raise TypeError(
            "parameters must be FighterDynamicParameters"
        )

    if not isinstance(
        weights,
        StatePenaltyWeights,
    ):
        raise TypeError(
            "weights must be StatePenaltyWeights"
        )

    if not isinstance(
        calibration,
        DynamicEffectCalibration,
    ):
        raise TypeError(
            "calibration must be DynamicEffectCalibration"
        )

    fatigue_effect_multiplier = (
        calculate_fatigue_effect_multiplier(
            parameters,
            calibration,
        )
    )

    fatigue_penalty = (
        state.fatigue
        * fatigue_effect_multiplier
        * weights.fatigue
    )
    damage_penalty = (
        state.damage
        * weights.damage
    )
    acute_stress_penalty = (
        state.acute_stress
        * weights.acute_stress
    )

    total_penalty = (
        fatigue_penalty
        + damage_penalty
        + acute_stress_penalty
    )

    return min(
        1.0,
        max(
            calibration.minimum_effective_capability_multiplier,
            1.0 - total_penalty,
        ),
    )


def calculate_capability_multipliers(
    state: FighterDynamicState,
    parameters: FighterDynamicParameters,
    calibration: DynamicEffectCalibration,
) -> CapabilityMultipliers:
    """Calculate every temporary capability-family multiplier."""

    if not isinstance(
        state,
        FighterDynamicState,
    ):
        raise TypeError(
            "state must be FighterDynamicState"
        )

    if not isinstance(
        parameters,
        FighterDynamicParameters,
    ):
        raise TypeError(
            "parameters must be FighterDynamicParameters"
        )

    if not isinstance(
        calibration,
        DynamicEffectCalibration,
    ):
        raise TypeError(
            "calibration must be DynamicEffectCalibration"
        )

    return CapabilityMultipliers(
        output=calculate_capability_multiplier(
            state,
            parameters,
            calibration.output,
            calibration,
        ),
        accuracy=calculate_capability_multiplier(
            state,
            parameters,
            calibration.accuracy,
            calibration,
        ),
        power=calculate_capability_multiplier(
            state,
            parameters,
            calibration.power,
            calibration,
        ),
        control=calculate_capability_multiplier(
            state,
            parameters,
            calibration.control,
            calibration,
        ),
        grappling=calculate_capability_multiplier(
            state,
            parameters,
            calibration.grappling,
            calibration,
        ),
        defense=calculate_capability_multiplier(
            state,
            parameters,
            calibration.defense,
            calibration,
        ),
    )


def build_effective_phase_parameters(
    baseline: FighterPhaseParameters,
    state: FighterDynamicState,
    dynamic_parameters: FighterDynamicParameters,
    calibration: DynamicEffectCalibration,
) -> FighterPhaseParameters:
    """Build temporary phase parameters for the current dynamic state."""

    if not isinstance(
        baseline,
        FighterPhaseParameters,
    ):
        raise TypeError(
            "baseline must be FighterPhaseParameters"
        )

    multipliers = calculate_capability_multipliers(
        state,
        dynamic_parameters,
        calibration,
    )

    return FighterPhaseParameters(
        distance=DistanceRateParameters(
            sig_strike_attempt_rate=(
                baseline.distance.sig_strike_attempt_rate
                * multipliers.output
            ),
            sig_strike_accuracy=(
                baseline.distance.sig_strike_accuracy
                * multipliers.accuracy
            ),
            knockdown_probability_per_landed=(
                baseline.distance
                .knockdown_probability_per_landed
                * multipliers.power
            ),
        ),
        clinch=ClinchRateParameters(
            clinch_strike_attempt_rate=(
                baseline.clinch.clinch_strike_attempt_rate
                * multipliers.output
            ),
            clinch_strike_accuracy=(
                baseline.clinch.clinch_strike_accuracy
                * multipliers.accuracy
            ),
            control_seconds_mean=(
                baseline.clinch.control_seconds_mean
                * multipliers.control
            ),
            damaging_clinch_probability=(
                baseline.clinch.damaging_clinch_probability
                * multipliers.power
            ),
        ),
        ground_owner=GroundOwnerRateParameters(
            ground_strike_attempt_rate=(
                baseline.ground_owner.ground_strike_attempt_rate
                * multipliers.output
            ),
            ground_strike_accuracy=(
                baseline.ground_owner.ground_strike_accuracy
                * multipliers.accuracy
            ),
            control_seconds_mean=(
                baseline.ground_owner.control_seconds_mean
                * multipliers.control
            ),
            submission_attempt_rate=(
                baseline.ground_owner.submission_attempt_rate
                * multipliers.output
            ),
            position_advancement_probability=(
                baseline.ground_owner
                .position_advancement_probability
                * multipliers.grappling
            ),
        ),
        ground_defender=GroundDefenderRateParameters(
            escape_attempt_rate=(
                baseline.ground_defender.escape_attempt_rate
                * multipliers.grappling
            ),
            reversal_attempt_rate=(
                baseline.ground_defender.reversal_attempt_rate
                * multipliers.grappling
            ),
            scramble_attempt_rate=(
                baseline.ground_defender.scramble_attempt_rate
                * multipliers.grappling
            ),
            submission_defense=(
                baseline.ground_defender.submission_defense
                * multipliers.defense
            ),
        ),
    )
