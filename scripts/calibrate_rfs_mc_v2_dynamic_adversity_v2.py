"""V2 dynamic-adversity calibration for RFS Monte Carlo shared-state engine.

This checkpoint preserves the frozen V1 fighter bridge, KO/submission
calibration, fatigue accumulation, and fatigue-to-power decay.

It additionally activates:

- persistent damage accumulation
- acute stress accumulation
- acute stress recovery
- damage/stress effects on phase capabilities

Dynamic effects on transition parameters remain disabled so this checkpoint
isolates adversity response from phase-transition deterioration.

The adversity magnitudes and phase-effect weights originate from the existing
structural dynamic-path audit. They are structurally tested starting values,
not production-calibrated constants.
"""

from __future__ import annotations

from dataclasses import replace

from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_calibration import (
    AdversityCalibration,
    DynamicStateCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_effect_calibration import (
    DynamicEffectCalibration,
    StatePenaltyWeights,
)

# Preserve/re-export the frozen V1 bridge and finish calibration.
from scripts.calibrate_rfs_mc_v2_power_decay_v1 import (
    Candidate,
    MIN_PRIOR_FIGHTS,
    V1_KNOCKDOWN_BONUS_HAZARD,
    V1_LANDED_KO_HAZARD,
    build_matchup_inputs,
    finish_calibration,
    phase_effect_calibration as v1_phase_effect_calibration,
    state_calibration as v1_state_calibration,
    zero_transition_effect_calibration,
)


# ---------------------------------------------------------------------------
# V2 adversity-state calibration
# ---------------------------------------------------------------------------

def state_calibration(
    candidate: Candidate,
) -> DynamicStateCalibration:
    """Preserve V1 fatigue while activating damage and acute stress."""

    base = v1_state_calibration(candidate)

    adversity = AdversityCalibration(
        # Persistent damage.
        distance_landed_damage=0.0015,
        clinch_landed_damage=0.0015,
        damaging_clinch_bonus_damage=0.0080,
        ground_landed_damage=0.0020,
        knockdown_damage=0.0800,

        # Acute stress.
        distance_landed_stress=0.0020,
        clinch_landed_stress=0.0020,
        damaging_clinch_bonus_stress=0.0120,
        ground_landed_stress=0.0030,
        knockdown_stress=0.1200,

        # Positional adversity.
        control_second_received_stress=0.0005,
        submission_attempt_received_stress=0.0100,
        position_advancement_received_stress=0.0080,
    )

    # Preserve V1 fatigue recovery exactly.
    # Activate only acute-stress recovery.
    recovery = replace(
        base.recovery,
        segment_acute_stress_recovery=0.060,
        round_break_acute_stress_recovery=0.250,
    )

    return replace(
        base,
        adversity=adversity,
        recovery=recovery,
    )


# ---------------------------------------------------------------------------
# V2 adversity -> capability effects
# ---------------------------------------------------------------------------

def phase_effect_calibration(
    candidate: Candidate,
) -> DynamicEffectCalibration:
    """Add damage/stress effects while preserving V1 fatigue behavior."""

    base = v1_phase_effect_calibration(candidate)

    # Fatigue values intentionally preserve the frozen V1 behavior.
    # Only damage and acute-stress components are added here.
    return replace(
        base,

        output=StatePenaltyWeights(
            fatigue=base.output.fatigue,
            damage=0.15,
            acute_stress=0.20,
        ),

        accuracy=StatePenaltyWeights(
            fatigue=base.accuracy.fatigue,
            damage=0.25,
            acute_stress=0.30,
        ),

        power=StatePenaltyWeights(
            fatigue=base.power.fatigue,
            damage=0.30,
            acute_stress=0.15,
        ),

        control=StatePenaltyWeights(
            fatigue=base.control.fatigue,
            damage=0.15,
            acute_stress=0.15,
        ),

        grappling=StatePenaltyWeights(
            fatigue=base.grappling.fatigue,
            damage=0.20,
            acute_stress=0.20,
        ),

        defense=StatePenaltyWeights(
            fatigue=base.defense.fatigue,
            damage=0.35,
            acute_stress=0.25,
        ),
    )
