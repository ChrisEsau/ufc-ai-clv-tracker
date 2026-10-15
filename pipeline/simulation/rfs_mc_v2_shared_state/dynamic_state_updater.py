"""Dynamic-state update logic for RFS Monte Carlo V2.

This module combines:

- previous mutable fighter condition
- raw segment workload and adversity exposure
- immutable fighter response traits
- universal dynamic-state calibration

Segment timing is:

1. recover eligible pre-existing fatigue and acute stress
2. apply resistance-scaled exposure from the completed segment
3. clip the next dynamic state to the normalized [0, 1] range

Persistent damage does not automatically recover.

``fatigue_performance_resilience`` is intentionally not consumed here. It
will later determine how strongly accumulated fatigue modifies temporary
effective phase and transition parameters.
"""

from __future__ import annotations

import math

from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_calibration import (
    DynamicStateCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_exposure import (
    FighterSegmentExposure,
    SegmentDynamicExposure,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_parameters import (
    FighterDynamicParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_state import (
    FightDynamicState,
    FighterDynamicState,
)


def _validate_unit_value(
    name: str,
    value: float,
) -> float:
    """Validate and return one finite normalized value."""

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

    return selected


def _clip_unit(value: float) -> float:
    """Clip a numeric result into the normalized state range."""

    return min(
        1.0,
        max(
            0.0,
            float(value),
        ),
    )


def calculate_resistance_multiplier(
    resistance: float,
    minimum_multiplier: float,
) -> float:
    """Convert normalized resistance into an accumulation multiplier.

    Resistance zero receives the full raw exposure multiplier of 1.0.

    Resistance one receives the configured minimum multiplier rather than
    automatic immunity.

    Values between zero and one are linearly interpolated.
    """

    selected_resistance = _validate_unit_value(
        "resistance",
        resistance,
    )
    selected_minimum = _validate_unit_value(
        "minimum_multiplier",
        minimum_multiplier,
    )

    return (
        1.0
        - selected_resistance
        * (
            1.0
            - selected_minimum
        )
    )


def update_fighter_dynamic_state(
    previous_state: FighterDynamicState,
    exposure: FighterSegmentExposure,
    parameters: FighterDynamicParameters,
    calibration: DynamicStateCalibration,
) -> FighterDynamicState:
    """Apply one completed segment to one fighter's dynamic state."""

    if not isinstance(
        previous_state,
        FighterDynamicState,
    ):
        raise TypeError(
            "previous_state must be FighterDynamicState"
        )

    if not isinstance(
        exposure,
        FighterSegmentExposure,
    ):
        raise TypeError(
            "exposure must be FighterSegmentExposure"
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
        DynamicStateCalibration,
    ):
        raise TypeError(
            "calibration must be DynamicStateCalibration"
        )

    resistance = calibration.resistance_scaling
    recovery = calibration.recovery

    fatigue_multiplier = calculate_resistance_multiplier(
        parameters.fatigue_accumulation_resistance,
        resistance.minimum_fatigue_accumulation_multiplier,
    )
    damage_multiplier = calculate_resistance_multiplier(
        parameters.damage_resistance,
        resistance.minimum_damage_accumulation_multiplier,
    )
    acute_stress_multiplier = calculate_resistance_multiplier(
        parameters.acute_stress_resistance,
        (
            resistance
            .minimum_acute_stress_accumulation_multiplier
        ),
    )

    fatigue_gain = (
        exposure.fatigue_workload
        * fatigue_multiplier
    )
    damage_gain = (
        exposure.persistent_damage_exposure
        * damage_multiplier
    )
    acute_stress_gain = (
        exposure.acute_stress_exposure
        * acute_stress_multiplier
    )

    # Fatigue recovers during a segment only when the completed segment's
    # raw workload qualifies as low workload.
    if (
        exposure.fatigue_workload
        <= recovery.low_workload_threshold
    ):
        fatigue_recovery = min(
            previous_state.fatigue,
            (
                recovery.segment_fatigue_recovery
                * parameters.recovery_ability
            ),
        )
    else:
        fatigue_recovery = 0.0

    # Acute stress decays every segment. Recovery applies only to stress
    # already present before the current segment's adversity.
    acute_stress_recovery = min(
        previous_state.acute_stress,
        (
            recovery.segment_acute_stress_recovery
            * parameters.acute_stress_recovery
        ),
    )

    next_fatigue = _clip_unit(
        previous_state.fatigue
        - fatigue_recovery
        + fatigue_gain
    )
    next_damage = _clip_unit(
        previous_state.damage
        + damage_gain
    )
    next_acute_stress = _clip_unit(
        previous_state.acute_stress
        - acute_stress_recovery
        + acute_stress_gain
    )

    return FighterDynamicState(
        fatigue=next_fatigue,
        damage=next_damage,
        acute_stress=next_acute_stress,
    )


def update_fight_dynamic_state(
    previous_state: FightDynamicState,
    exposure: SegmentDynamicExposure,
    red_parameters: FighterDynamicParameters,
    blue_parameters: FighterDynamicParameters,
    calibration: DynamicStateCalibration,
) -> FightDynamicState:
    """Apply one completed segment to both fighters."""

    if not isinstance(
        previous_state,
        FightDynamicState,
    ):
        raise TypeError(
            "previous_state must be FightDynamicState"
        )

    if not isinstance(
        exposure,
        SegmentDynamicExposure,
    ):
        raise TypeError(
            "exposure must be SegmentDynamicExposure"
        )

    return FightDynamicState(
        red=update_fighter_dynamic_state(
            previous_state.red,
            exposure.red,
            red_parameters,
            calibration,
        ),
        blue=update_fighter_dynamic_state(
            previous_state.blue,
            exposure.blue,
            blue_parameters,
            calibration,
        ),
    )


def apply_fighter_round_break_recovery(
    previous_state: FighterDynamicState,
    parameters: FighterDynamicParameters,
    calibration: DynamicStateCalibration,
) -> FighterDynamicState:
    """Apply one between-round recovery period to one fighter."""

    if not isinstance(
        previous_state,
        FighterDynamicState,
    ):
        raise TypeError(
            "previous_state must be FighterDynamicState"
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
        DynamicStateCalibration,
    ):
        raise TypeError(
            "calibration must be DynamicStateCalibration"
        )

    recovery = calibration.recovery

    fatigue_recovery = min(
        previous_state.fatigue,
        (
            recovery.round_break_fatigue_recovery
            * parameters.recovery_ability
        ),
    )
    acute_stress_recovery = min(
        previous_state.acute_stress,
        (
            recovery.round_break_acute_stress_recovery
            * parameters.acute_stress_recovery
        ),
    )

    return FighterDynamicState(
        fatigue=_clip_unit(
            previous_state.fatigue
            - fatigue_recovery
        ),
        damage=previous_state.damage,
        acute_stress=_clip_unit(
            previous_state.acute_stress
            - acute_stress_recovery
        ),
    )


def apply_round_break_recovery(
    previous_state: FightDynamicState,
    red_parameters: FighterDynamicParameters,
    blue_parameters: FighterDynamicParameters,
    calibration: DynamicStateCalibration,
) -> FightDynamicState:
    """Apply between-round recovery to both fighters."""

    if not isinstance(
        previous_state,
        FightDynamicState,
    ):
        raise TypeError(
            "previous_state must be FightDynamicState"
        )

    return FightDynamicState(
        red=apply_fighter_round_break_recovery(
            previous_state.red,
            red_parameters,
            calibration,
        ),
        blue=apply_fighter_round_break_recovery(
            previous_state.blue,
            blue_parameters,
            calibration,
        ),
    )
