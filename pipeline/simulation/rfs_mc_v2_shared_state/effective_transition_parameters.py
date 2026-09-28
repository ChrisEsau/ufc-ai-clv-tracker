"""Temporary effective transition parameters for RFS Monte Carlo V2.

Current fatigue, persistent damage, and acute stress reduce a fighter's
baseline transition capabilities for the next segment.

Baseline ``FighterTransitionParameters`` are never mutated. This module
returns a new temporary parameter bundle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_effect_calibration import (
    StatePenaltyWeights,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_parameters import (
    FighterDynamicParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_state import (
    FighterDynamicState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_transition_effect_calibration import (
    DynamicTransitionEffectCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)


@dataclass(frozen=True)
class TransitionCapabilityMultipliers:
    """Temporary multipliers for transition capability families."""

    entry: float
    completion: float
    retention: float
    escape: float
    reversal: float
    persistence: float
    imposition: float
    resistance: float

    def __post_init__(self) -> None:
        """Validate normalized transition multipliers."""

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


def calculate_transition_fatigue_effect_multiplier(
    parameters: FighterDynamicParameters,
    calibration: DynamicTransitionEffectCalibration,
) -> float:
    """Return how strongly accumulated fatigue affects transitions."""

    if not isinstance(
        parameters,
        FighterDynamicParameters,
    ):
        raise TypeError(
            "parameters must be FighterDynamicParameters"
        )

    if not isinstance(
        calibration,
        DynamicTransitionEffectCalibration,
    ):
        raise TypeError(
            "calibration must be "
            "DynamicTransitionEffectCalibration"
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


def calculate_transition_capability_multiplier(
    state: FighterDynamicState,
    parameters: FighterDynamicParameters,
    weights: StatePenaltyWeights,
    calibration: DynamicTransitionEffectCalibration,
) -> float:
    """Calculate one temporary transition-capability multiplier."""

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
        DynamicTransitionEffectCalibration,
    ):
        raise TypeError(
            "calibration must be "
            "DynamicTransitionEffectCalibration"
        )

    fatigue_effect = (
        calculate_transition_fatigue_effect_multiplier(
            parameters,
            calibration,
        )
    )

    fatigue_penalty = (
        state.fatigue
        * fatigue_effect
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
            calibration.minimum_effective_transition_multiplier,
            1.0 - total_penalty,
        ),
    )


def calculate_transition_capability_multipliers(
    state: FighterDynamicState,
    parameters: FighterDynamicParameters,
    calibration: DynamicTransitionEffectCalibration,
) -> TransitionCapabilityMultipliers:
    """Calculate every temporary transition multiplier."""

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
        DynamicTransitionEffectCalibration,
    ):
        raise TypeError(
            "calibration must be "
            "DynamicTransitionEffectCalibration"
        )

    return TransitionCapabilityMultipliers(
        entry=calculate_transition_capability_multiplier(
            state,
            parameters,
            calibration.entry,
            calibration,
        ),
        completion=calculate_transition_capability_multiplier(
            state,
            parameters,
            calibration.completion,
            calibration,
        ),
        retention=calculate_transition_capability_multiplier(
            state,
            parameters,
            calibration.retention,
            calibration,
        ),
        escape=calculate_transition_capability_multiplier(
            state,
            parameters,
            calibration.escape,
            calibration,
        ),
        reversal=calculate_transition_capability_multiplier(
            state,
            parameters,
            calibration.reversal,
            calibration,
        ),
        persistence=calculate_transition_capability_multiplier(
            state,
            parameters,
            calibration.persistence,
            calibration,
        ),
        imposition=calculate_transition_capability_multiplier(
            state,
            parameters,
            calibration.imposition,
            calibration,
        ),
        resistance=calculate_transition_capability_multiplier(
            state,
            parameters,
            calibration.resistance,
            calibration,
        ),
    )


def build_effective_transition_parameters(
    baseline: FighterTransitionParameters,
    state: FighterDynamicState,
    dynamic_parameters: FighterDynamicParameters,
    calibration: DynamicTransitionEffectCalibration,
) -> FighterTransitionParameters:
    """Build temporary transition parameters for the current state."""

    if not isinstance(
        baseline,
        FighterTransitionParameters,
    ):
        raise TypeError(
            "baseline must be FighterTransitionParameters"
        )

    multipliers = (
        calculate_transition_capability_multipliers(
            state,
            dynamic_parameters,
            calibration,
        )
    )

    return FighterTransitionParameters(
        distance_retention=(
            baseline.distance_retention
            * multipliers.retention
        ),
        clinch_entry_tendency=(
            baseline.clinch_entry_tendency
            * multipliers.entry
        ),
        clinch_entry_resistance=(
            baseline.clinch_entry_resistance
            * multipliers.resistance
        ),
        takedown_entry_tendency=(
            baseline.takedown_entry_tendency
            * multipliers.entry
        ),
        takedown_completion_ability=(
            baseline.takedown_completion_ability
            * multipliers.completion
        ),
        takedown_resistance=(
            baseline.takedown_resistance
            * multipliers.resistance
        ),
        takedown_persistence=(
            baseline.takedown_persistence
            * multipliers.persistence
        ),
        failed_takedown_persistence=(
            baseline.failed_takedown_persistence
            * multipliers.persistence
        ),
        clinch_retention=(
            baseline.clinch_retention
            * multipliers.retention
        ),
        clinch_escape_ability=(
            baseline.clinch_escape_ability
            * multipliers.escape
        ),
        ground_retention=(
            baseline.ground_retention
            * multipliers.retention
        ),
        ground_escape_ability=(
            baseline.ground_escape_ability
            * multipliers.escape
        ),
        reversal_ability=(
            baseline.reversal_ability
            * multipliers.reversal
        ),
        phase_imposition=(
            baseline.phase_imposition
            * multipliers.imposition
        ),
        phase_resistance=(
            baseline.phase_resistance
            * multipliers.resistance
        ),
    )
