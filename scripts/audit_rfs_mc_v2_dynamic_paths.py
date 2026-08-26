"""Population audit for RFS Monte Carlo V2 dynamic fight paths.

The audit verifies that the integrated dynamic system behaves in the expected
direction across controlled simulation populations:

- fatigue accumulates under sustained workload
- fatigue resistance reduces accumulation
- performance resilience preserves late-fight capability
- round-break recovery reduces fatigue
- persistent damage does not recover
- acute-stress recovery reduces temporary impairment
- fatigue-sensitive retention shortens phase persistence
- all shared-state and activity legality contracts remain intact

These scenarios validate architecture and directionality. They do not establish
production-calibrated UFC values.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from statistics import mean

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.clinch_activity_engine import (
    ClinchSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    SEGMENTS_PER_ROUND,
    FighterSide,
)
from pipeline.simulation.rfs_mc_v2_shared_state.distance_activity_engine import (
    DistanceSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_calibration import (
    ActivityWorkloadCalibration,
    AdversityCalibration,
    DynamicStateCalibration,
    PhaseWorkloadCalibration,
    RecoveryCalibration,
    ResistanceScalingCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_effect_calibration import (
    DynamicEffectCalibration,
    StatePenaltyWeights,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_parameters import (
    FighterDynamicParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_path_runner import (
    DynamicActivityPath,
    run_dynamic_activity_path,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_transition_effect_calibration import (
    DynamicTransitionEffectCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.ground_activity_engine import (
    GroundSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    ClinchRateParameters,
    DistanceRateParameters,
    FighterPhaseParameters,
    GroundDefenderRateParameters,
    GroundOwnerRateParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)


@dataclass(frozen=True)
class AuditCheck:
    """One directional dynamic-system audit check."""

    name: str
    passed: bool
    detail: str


def zero_weights() -> StatePenaltyWeights:
    """Return a capability family with no dynamic penalty."""

    return StatePenaltyWeights(
        fatigue=0.0,
        damage=0.0,
        acute_stress=0.0,
    )


def fatigue_weights(
    strength: float = 1.0,
) -> StatePenaltyWeights:
    """Return a capability family affected only by fatigue."""

    return StatePenaltyWeights(
        fatigue=strength,
        damage=0.0,
        acute_stress=0.0,
    )


def neutral_transition_parameters() -> FighterTransitionParameters:
    """Return a neutral shared-state transition profile."""

    return FighterTransitionParameters(
        distance_retention=0.50,
        clinch_entry_tendency=0.50,
        clinch_entry_resistance=0.50,
        takedown_entry_tendency=0.50,
        takedown_completion_ability=0.50,
        takedown_resistance=0.50,
        takedown_persistence=0.50,
        failed_takedown_persistence=0.50,
        clinch_retention=0.50,
        clinch_escape_ability=0.50,
        ground_retention=0.50,
        ground_escape_ability=0.50,
        reversal_ability=0.50,
        phase_imposition=0.50,
        phase_resistance=0.50,
    )


def distance_only_transition_parameters() -> FighterTransitionParameters:
    """Return a profile that remains at distance."""

    return FighterTransitionParameters(
        distance_retention=1.0,
        clinch_entry_tendency=0.0,
        clinch_entry_resistance=0.0,
        takedown_entry_tendency=0.0,
        takedown_completion_ability=0.0,
        takedown_resistance=0.0,
        takedown_persistence=0.0,
        failed_takedown_persistence=0.0,
        clinch_retention=0.0,
        clinch_escape_ability=0.0,
        ground_retention=0.0,
        ground_escape_ability=0.0,
        reversal_ability=0.0,
        phase_imposition=0.0,
        phase_resistance=0.0,
    )


def phase_parameters(
    *,
    distance_attempt_rate: float = 4.0,
    distance_accuracy: float = 0.50,
) -> FighterPhaseParameters:
    """Return a complete fighter phase profile."""

    return FighterPhaseParameters(
        distance=DistanceRateParameters(
            sig_strike_attempt_rate=distance_attempt_rate,
            sig_strike_accuracy=distance_accuracy,
            knockdown_probability_per_landed=0.02,
        ),
        clinch=ClinchRateParameters(
            clinch_strike_attempt_rate=1.50,
            clinch_strike_accuracy=0.50,
            control_seconds_mean=8.0,
            damaging_clinch_probability=0.06,
        ),
        ground_owner=GroundOwnerRateParameters(
            ground_strike_attempt_rate=2.0,
            ground_strike_accuracy=0.50,
            control_seconds_mean=15.0,
            submission_attempt_rate=0.20,
            position_advancement_probability=0.25,
        ),
        ground_defender=GroundDefenderRateParameters(
            escape_attempt_rate=0.20,
            reversal_attempt_rate=0.08,
            scramble_attempt_rate=0.15,
            submission_defense=0.70,
        ),
    )


def dynamic_parameters(
    **overrides: float,
) -> FighterDynamicParameters:
    """Return a controlled fighter response profile."""

    baseline = FighterDynamicParameters(
        fatigue_accumulation_resistance=0.0,
        fatigue_performance_resilience=0.0,
        recovery_ability=0.0,
        damage_resistance=0.0,
        acute_stress_resistance=0.0,
        acute_stress_recovery=0.0,
    )

    return replace(
        baseline,
        **overrides,
    )


def controlled_state_calibration(
    *,
    phase_workload: float = 0.0,
    strike_attempt_cost: float = 0.0,
    distance_damage: float = 0.0,
    distance_stress: float = 0.0,
    segment_stress_recovery: float = 0.0,
    round_break_fatigue_recovery: float = 0.0,
    round_break_stress_recovery: float = 0.0,
) -> DynamicStateCalibration:
    """Build controlled universal workload and adversity costs."""

    return DynamicStateCalibration(
        phase_workload=PhaseWorkloadCalibration(
            distance=phase_workload,
            clinch_owner=phase_workload,
            clinch_defender=phase_workload,
            ground_owner=phase_workload,
            ground_defender=phase_workload,
        ),
        activity_workload=ActivityWorkloadCalibration(
            strike_attempt=strike_attempt_cost,
            control_second=0.0,
            submission_attempt=0.0,
            position_advancement=0.0,
            escape_attempt=0.0,
            reversal_attempt=0.0,
            scramble_attempt=0.0,
        ),
        adversity=AdversityCalibration(
            distance_landed_damage=distance_damage,
            clinch_landed_damage=0.0,
            damaging_clinch_bonus_damage=0.0,
            ground_landed_damage=0.0,
            knockdown_damage=0.0,
            distance_landed_stress=distance_stress,
            clinch_landed_stress=0.0,
            damaging_clinch_bonus_stress=0.0,
            ground_landed_stress=0.0,
            knockdown_stress=0.0,
            control_second_received_stress=0.0,
            submission_attempt_received_stress=0.0,
            position_advancement_received_stress=0.0,
        ),
        resistance_scaling=ResistanceScalingCalibration(
            minimum_fatigue_accumulation_multiplier=0.25,
            minimum_damage_accumulation_multiplier=0.20,
            minimum_acute_stress_accumulation_multiplier=0.15,
        ),
        recovery=RecoveryCalibration(
            low_workload_threshold=0.0,
            segment_fatigue_recovery=0.0,
            round_break_fatigue_recovery=(
                round_break_fatigue_recovery
            ),
            segment_acute_stress_recovery=(
                segment_stress_recovery
            ),
            round_break_acute_stress_recovery=(
                round_break_stress_recovery
            ),
        ),
    )


def structural_state_calibration() -> DynamicStateCalibration:
    """Return nonzero costs for the full structural audit."""

    return DynamicStateCalibration(
        phase_workload=PhaseWorkloadCalibration(
            distance=0.010,
            clinch_owner=0.018,
            clinch_defender=0.022,
            ground_owner=0.022,
            ground_defender=0.026,
        ),
        activity_workload=ActivityWorkloadCalibration(
            strike_attempt=0.0015,
            control_second=0.0004,
            submission_attempt=0.008,
            position_advancement=0.006,
            escape_attempt=0.004,
            reversal_attempt=0.006,
            scramble_attempt=0.005,
        ),
        adversity=AdversityCalibration(
            distance_landed_damage=0.0015,
            clinch_landed_damage=0.0015,
            damaging_clinch_bonus_damage=0.008,
            ground_landed_damage=0.002,
            knockdown_damage=0.080,
            distance_landed_stress=0.002,
            clinch_landed_stress=0.002,
            damaging_clinch_bonus_stress=0.012,
            ground_landed_stress=0.003,
            knockdown_stress=0.120,
            control_second_received_stress=0.0005,
            submission_attempt_received_stress=0.010,
            position_advancement_received_stress=0.008,
        ),
        resistance_scaling=ResistanceScalingCalibration(
            minimum_fatigue_accumulation_multiplier=0.25,
            minimum_damage_accumulation_multiplier=0.20,
            minimum_acute_stress_accumulation_multiplier=0.15,
        ),
        recovery=RecoveryCalibration(
            low_workload_threshold=0.040,
            segment_fatigue_recovery=0.004,
            round_break_fatigue_recovery=0.080,
            segment_acute_stress_recovery=0.060,
            round_break_acute_stress_recovery=0.250,
        ),
    )


def zero_phase_effect_calibration() -> DynamicEffectCalibration:
    """Return phase-effect calibration that preserves baselines."""

    return DynamicEffectCalibration(
        minimum_fatigue_effect_multiplier=0.20,
        minimum_effective_capability_multiplier=0.10,
        output=zero_weights(),
        accuracy=zero_weights(),
        power=zero_weights(),
        control=zero_weights(),
        grappling=zero_weights(),
        defense=zero_weights(),
    )


def fatigue_output_effect_calibration() -> DynamicEffectCalibration:
    """Return calibration where fatigue reduces activity output."""

    return DynamicEffectCalibration(
        minimum_fatigue_effect_multiplier=0.20,
        minimum_effective_capability_multiplier=0.10,
        output=fatigue_weights(0.80),
        accuracy=zero_weights(),
        power=zero_weights(),
        control=zero_weights(),
        grappling=zero_weights(),
        defense=zero_weights(),
    )


def structural_phase_effect_calibration() -> DynamicEffectCalibration:
    """Return moderate dynamic penalties for structural paths."""

    return DynamicEffectCalibration(
        minimum_fatigue_effect_multiplier=0.20,
        minimum_effective_capability_multiplier=0.15,
        output=StatePenaltyWeights(
            fatigue=0.35,
            damage=0.15,
            acute_stress=0.20,
        ),
        accuracy=StatePenaltyWeights(
            fatigue=0.15,
            damage=0.25,
            acute_stress=0.30,
        ),
        power=StatePenaltyWeights(
            fatigue=0.20,
            damage=0.30,
            acute_stress=0.15,
        ),
        control=StatePenaltyWeights(
            fatigue=0.30,
            damage=0.15,
            acute_stress=0.15,
        ),
        grappling=StatePenaltyWeights(
            fatigue=0.30,
            damage=0.20,
            acute_stress=0.20,
        ),
        defense=StatePenaltyWeights(
            fatigue=0.20,
            damage=0.35,
            acute_stress=0.25,
        ),
    )


def zero_transition_effect_calibration(
) -> DynamicTransitionEffectCalibration:
    """Return transition effects that preserve baselines."""

    return DynamicTransitionEffectCalibration(
        minimum_fatigue_effect_multiplier=0.20,
        minimum_effective_transition_multiplier=0.10,
        entry=zero_weights(),
        completion=zero_weights(),
        retention=zero_weights(),
        escape=zero_weights(),
        reversal=zero_weights(),
        persistence=zero_weights(),
        imposition=zero_weights(),
        resistance=zero_weights(),
    )


def fatigue_retention_effect_calibration(
) -> DynamicTransitionEffectCalibration:
    """Return calibration where fatigue reduces phase retention."""

    return DynamicTransitionEffectCalibration(
        minimum_fatigue_effect_multiplier=0.20,
        minimum_effective_transition_multiplier=0.10,
        entry=zero_weights(),
        completion=zero_weights(),
        retention=fatigue_weights(1.0),
        escape=zero_weights(),
        reversal=zero_weights(),
        persistence=zero_weights(),
        imposition=zero_weights(),
        resistance=zero_weights(),
    )


def structural_transition_effect_calibration(
) -> DynamicTransitionEffectCalibration:
    """Return moderate transition penalties for structural paths."""

    return DynamicTransitionEffectCalibration(
        minimum_fatigue_effect_multiplier=0.20,
        minimum_effective_transition_multiplier=0.15,
        entry=StatePenaltyWeights(
            fatigue=0.30,
            damage=0.15,
            acute_stress=0.20,
        ),
        completion=StatePenaltyWeights(
            fatigue=0.40,
            damage=0.20,
            acute_stress=0.20,
        ),
        retention=StatePenaltyWeights(
            fatigue=0.35,
            damage=0.20,
            acute_stress=0.15,
        ),
        escape=StatePenaltyWeights(
            fatigue=0.30,
            damage=0.30,
            acute_stress=0.20,
        ),
        reversal=StatePenaltyWeights(
            fatigue=0.40,
            damage=0.25,
            acute_stress=0.20,
        ),
        persistence=StatePenaltyWeights(
            fatigue=0.45,
            damage=0.15,
            acute_stress=0.20,
        ),
        imposition=StatePenaltyWeights(
            fatigue=0.35,
            damage=0.20,
            acute_stress=0.20,
        ),
        resistance=StatePenaltyWeights(
            fatigue=0.25,
            damage=0.35,
            acute_stress=0.25,
        ),
    )


def run_path(
    *,
    seed: int,
    red_transition: FighterTransitionParameters,
    blue_transition: FighterTransitionParameters,
    red_phase: FighterPhaseParameters,
    blue_phase: FighterPhaseParameters,
    red_dynamic: FighterDynamicParameters,
    blue_dynamic: FighterDynamicParameters,
    state_calibration: DynamicStateCalibration,
    phase_effect: DynamicEffectCalibration,
    transition_effect: DynamicTransitionEffectCalibration,
) -> DynamicActivityPath:
    """Run one controlled three-round dynamic path."""

    return run_dynamic_activity_path(
        red_transition,
        blue_transition,
        red_phase,
        blue_phase,
        red_dynamic,
        blue_dynamic,
        dynamic_state_calibration=state_calibration,
        phase_effect_calibration=phase_effect,
        transition_effect_calibration=transition_effect,
        scheduled_rounds=3,
        seed=seed,
    )


def same_phase_and_owner_transition(record: object) -> bool:
    """Return whether a transition preserves phase and ownership."""

    transition = record.transition

    if transition is None:
        return False

    return (
        transition.next_state.phase
        is record.state.phase
        and transition.next_state.phase_owner
        is record.state.phase_owner
    )


def count_structural_violations(
    path: DynamicActivityPath,
) -> int:
    """Count shared-state, activity, exposure, and dynamic-chain violations."""

    violations = 0

    for index, record in enumerate(path.segments):
        if record.activity.state != record.state:
            violations += 1

        if record.exposure.state != record.state:
            violations += 1

        if record.state.phase is FightPhase.DISTANCE:
            if not isinstance(
                record.activity,
                DistanceSegmentActivity,
            ):
                violations += 1

        elif record.state.phase is FightPhase.CLINCH:
            if not isinstance(
                record.activity,
                ClinchSegmentActivity,
            ):
                violations += 1
            else:
                if record.state.phase_owner is FighterSide.RED:
                    if record.activity.blue.control_seconds != 0:
                        violations += 1
                elif record.state.phase_owner is FighterSide.BLUE:
                    if record.activity.red.control_seconds != 0:
                        violations += 1
                else:
                    violations += 1

        elif record.state.phase is FightPhase.GROUND:
            if not isinstance(
                record.activity,
                GroundSegmentActivity,
            ):
                violations += 1
            else:
                if record.state.phase_owner is FighterSide.RED:
                    owner = record.activity.red
                    defender = record.activity.blue
                elif record.state.phase_owner is FighterSide.BLUE:
                    owner = record.activity.blue
                    defender = record.activity.red
                else:
                    violations += 1
                    continue

                if (
                    owner.escape_attempts != 0
                    or owner.reversal_attempts != 0
                    or owner.scramble_attempts != 0
                ):
                    violations += 1

                if (
                    defender.ground_str_attempted != 0
                    or defender.ground_str_landed != 0
                    or defender.control_seconds != 0
                    or defender.submission_attempts != 0
                    or defender.position_advancements != 0
                ):
                    violations += 1

        for fighter_state in (
            record.dynamic_state_before.red,
            record.dynamic_state_before.blue,
            record.dynamic_state_after_activity.red,
            record.dynamic_state_after_activity.blue,
            record.dynamic_state_after_segment.red,
            record.dynamic_state_after_segment.blue,
        ):
            for value in (
                fighter_state.fatigue,
                fighter_state.damage,
                fighter_state.acute_stress,
            ):
                if not 0.0 <= value <= 1.0:
                    violations += 1

        is_round_end = (
            record.state.segment_number
            == SEGMENTS_PER_ROUND
        )

        if is_round_end:
            if record.transition is not None:
                violations += 1
        else:
            if record.transition is None:
                violations += 1

        if index == len(path.segments) - 1:
            continue

        following = path.segments[index + 1]

        if (
            following.dynamic_state_before
            != record.dynamic_state_after_segment
        ):
            violations += 1

        if not is_round_end:
            if (
                record.transition is None
                or record.transition.next_state
                != following.state
            ):
                violations += 1

        if following.state.segment_number == 1:
            if following.state.phase is not FightPhase.DISTANCE:
                violations += 1

            if following.state.phase_owner is not None:
                violations += 1

    return violations


def run_audit(
    *,
    path_count: int,
    seed_start: int,
) -> int:
    """Run all controlled population scenarios."""

    if path_count <= 0:
        raise ValueError(
            "path_count must be positive"
        )

    distance_transition = distance_only_transition_parameters()
    neutral_transition = neutral_transition_parameters()

    standard_phase = phase_parameters()
    zero_phase_effect = zero_phase_effect_calibration()
    zero_transition_effect = zero_transition_effect_calibration()

    low_resistance_final: list[float] = []
    high_resistance_final: list[float] = []

    low_resilience_late_output: list[float] = []
    high_resilience_late_output: list[float] = []

    poor_recovery_round_two: list[float] = []
    strong_recovery_round_two: list[float] = []

    damage_final: list[float] = []
    damage_monotonic_violations = 0
    damage_round_break_violations = 0

    poor_stress_recovery_final: list[float] = []
    strong_stress_recovery_final: list[float] = []

    baseline_stay_transitions = 0
    dynamic_stay_transitions = 0
    baseline_transitions = 0
    dynamic_transitions = 0

    structural_violations = 0

    fatigue_calibration = controlled_state_calibration(
        phase_workload=0.030,
    )
    output_fatigue_calibration = controlled_state_calibration(
        phase_workload=0.025,
    )
    recovery_calibration = controlled_state_calibration(
        phase_workload=0.030,
        round_break_fatigue_recovery=0.200,
    )
    damage_calibration = controlled_state_calibration(
        distance_damage=0.001,
    )
    stress_calibration = controlled_state_calibration(
        distance_stress=0.002,
        segment_stress_recovery=0.050,
    )
    transition_feedback_calibration = controlled_state_calibration(
        phase_workload=0.030,
    )

    passive_phase = phase_parameters(
        distance_attempt_rate=0.0,
        distance_accuracy=1.0,
    )
    active_phase = phase_parameters(
        distance_attempt_rate=6.0,
        distance_accuracy=1.0,
    )

    for path_index in range(path_count):
        seed = seed_start + path_index

        low_resistance_path = run_path(
            seed=seed,
            red_transition=distance_transition,
            blue_transition=distance_transition,
            red_phase=standard_phase,
            blue_phase=standard_phase,
            red_dynamic=dynamic_parameters(
                fatigue_accumulation_resistance=0.0,
            ),
            blue_dynamic=dynamic_parameters(),
            state_calibration=fatigue_calibration,
            phase_effect=zero_phase_effect,
            transition_effect=zero_transition_effect,
        )
        high_resistance_path = run_path(
            seed=seed,
            red_transition=distance_transition,
            blue_transition=distance_transition,
            red_phase=standard_phase,
            blue_phase=standard_phase,
            red_dynamic=dynamic_parameters(
                fatigue_accumulation_resistance=1.0,
            ),
            blue_dynamic=dynamic_parameters(),
            state_calibration=fatigue_calibration,
            phase_effect=zero_phase_effect,
            transition_effect=zero_transition_effect,
        )

        low_resistance_final.append(
            low_resistance_path.segments[-1]
            .dynamic_state_after_segment.red.fatigue
        )
        high_resistance_final.append(
            high_resistance_path.segments[-1]
            .dynamic_state_after_segment.red.fatigue
        )

        low_resilience_path = run_path(
            seed=seed,
            red_transition=distance_transition,
            blue_transition=distance_transition,
            red_phase=standard_phase,
            blue_phase=standard_phase,
            red_dynamic=dynamic_parameters(
                fatigue_performance_resilience=0.0,
            ),
            blue_dynamic=dynamic_parameters(),
            state_calibration=output_fatigue_calibration,
            phase_effect=fatigue_output_effect_calibration(),
            transition_effect=zero_transition_effect,
        )
        high_resilience_path = run_path(
            seed=seed,
            red_transition=distance_transition,
            blue_transition=distance_transition,
            red_phase=standard_phase,
            blue_phase=standard_phase,
            red_dynamic=dynamic_parameters(
                fatigue_performance_resilience=1.0,
            ),
            blue_dynamic=dynamic_parameters(),
            state_calibration=output_fatigue_calibration,
            phase_effect=fatigue_output_effect_calibration(),
            transition_effect=zero_transition_effect,
        )

        low_resilience_late_output.append(
            mean(
                record.red_effective_phase
                .distance.sig_strike_attempt_rate
                for record in low_resilience_path.segments[20:30]
            )
        )
        high_resilience_late_output.append(
            mean(
                record.red_effective_phase
                .distance.sig_strike_attempt_rate
                for record in high_resilience_path.segments[20:30]
            )
        )

        poor_recovery_path = run_path(
            seed=seed,
            red_transition=distance_transition,
            blue_transition=distance_transition,
            red_phase=standard_phase,
            blue_phase=standard_phase,
            red_dynamic=dynamic_parameters(
                recovery_ability=0.0,
            ),
            blue_dynamic=dynamic_parameters(),
            state_calibration=recovery_calibration,
            phase_effect=zero_phase_effect,
            transition_effect=zero_transition_effect,
        )
        strong_recovery_path = run_path(
            seed=seed,
            red_transition=distance_transition,
            blue_transition=distance_transition,
            red_phase=standard_phase,
            blue_phase=standard_phase,
            red_dynamic=dynamic_parameters(
                recovery_ability=1.0,
            ),
            blue_dynamic=dynamic_parameters(),
            state_calibration=recovery_calibration,
            phase_effect=zero_phase_effect,
            transition_effect=zero_transition_effect,
        )

        poor_recovery_round_two.append(
            poor_recovery_path.segments[10]
            .dynamic_state_before.red.fatigue
        )
        strong_recovery_round_two.append(
            strong_recovery_path.segments[10]
            .dynamic_state_before.red.fatigue
        )

        damage_path = run_path(
            seed=seed,
            red_transition=distance_transition,
            blue_transition=distance_transition,
            red_phase=passive_phase,
            blue_phase=active_phase,
            red_dynamic=dynamic_parameters(),
            blue_dynamic=dynamic_parameters(),
            state_calibration=damage_calibration,
            phase_effect=zero_phase_effect,
            transition_effect=zero_transition_effect,
        )

        previous_damage = 0.0

        for record in damage_path.segments:
            before_damage = (
                record.dynamic_state_before.red.damage
            )
            after_damage = (
                record.dynamic_state_after_activity.red.damage
            )

            if before_damage < previous_damage:
                damage_monotonic_violations += 1

            if after_damage < before_damage:
                damage_monotonic_violations += 1

            if (
                record.round_break_recovery_applied
                and record.dynamic_state_after_segment.red.damage
                != after_damage
            ):
                damage_round_break_violations += 1

            previous_damage = (
                record.dynamic_state_after_segment.red.damage
            )

        damage_final.append(
            damage_path.segments[-1]
            .dynamic_state_after_segment.red.damage
        )

        poor_stress_path = run_path(
            seed=seed,
            red_transition=distance_transition,
            blue_transition=distance_transition,
            red_phase=passive_phase,
            blue_phase=active_phase,
            red_dynamic=dynamic_parameters(
                acute_stress_recovery=0.0,
            ),
            blue_dynamic=dynamic_parameters(),
            state_calibration=stress_calibration,
            phase_effect=zero_phase_effect,
            transition_effect=zero_transition_effect,
        )
        strong_stress_path = run_path(
            seed=seed,
            red_transition=distance_transition,
            blue_transition=distance_transition,
            red_phase=passive_phase,
            blue_phase=active_phase,
            red_dynamic=dynamic_parameters(
                acute_stress_recovery=1.0,
            ),
            blue_dynamic=dynamic_parameters(),
            state_calibration=stress_calibration,
            phase_effect=zero_phase_effect,
            transition_effect=zero_transition_effect,
        )

        poor_stress_recovery_final.append(
            poor_stress_path.segments[-1]
            .dynamic_state_after_segment.red.acute_stress
        )
        strong_stress_recovery_final.append(
            strong_stress_path.segments[-1]
            .dynamic_state_after_segment.red.acute_stress
        )

        baseline_transition_path = run_path(
            seed=seed,
            red_transition=neutral_transition,
            blue_transition=neutral_transition,
            red_phase=standard_phase,
            blue_phase=standard_phase,
            red_dynamic=dynamic_parameters(),
            blue_dynamic=dynamic_parameters(),
            state_calibration=transition_feedback_calibration,
            phase_effect=zero_phase_effect,
            transition_effect=zero_transition_effect,
        )
        dynamic_transition_path = run_path(
            seed=seed,
            red_transition=neutral_transition,
            blue_transition=neutral_transition,
            red_phase=standard_phase,
            blue_phase=standard_phase,
            red_dynamic=dynamic_parameters(),
            blue_dynamic=dynamic_parameters(),
            state_calibration=transition_feedback_calibration,
            phase_effect=zero_phase_effect,
            transition_effect=(
                fatigue_retention_effect_calibration()
            ),
        )

        for record in baseline_transition_path.segments:
            if record.transition is None:
                continue

            baseline_transitions += 1

            if same_phase_and_owner_transition(record):
                baseline_stay_transitions += 1

        for record in dynamic_transition_path.segments:
            if record.transition is None:
                continue

            dynamic_transitions += 1

            if same_phase_and_owner_transition(record):
                dynamic_stay_transitions += 1

        structural_path = run_path(
            seed=seed,
            red_transition=neutral_transition,
            blue_transition=neutral_transition,
            red_phase=standard_phase,
            blue_phase=standard_phase,
            red_dynamic=dynamic_parameters(
                fatigue_accumulation_resistance=0.50,
                fatigue_performance_resilience=0.50,
                recovery_ability=0.50,
                damage_resistance=0.50,
                acute_stress_resistance=0.50,
                acute_stress_recovery=0.50,
            ),
            blue_dynamic=dynamic_parameters(
                fatigue_accumulation_resistance=0.50,
                fatigue_performance_resilience=0.50,
                recovery_ability=0.50,
                damage_resistance=0.50,
                acute_stress_resistance=0.50,
                acute_stress_recovery=0.50,
            ),
            state_calibration=structural_state_calibration(),
            phase_effect=structural_phase_effect_calibration(),
            transition_effect=(
                structural_transition_effect_calibration()
            ),
        )

        structural_violations += count_structural_violations(
            structural_path
        )

    low_resistance_mean = mean(low_resistance_final)
    high_resistance_mean = mean(high_resistance_final)

    low_resilience_output_mean = mean(
        low_resilience_late_output
    )
    high_resilience_output_mean = mean(
        high_resilience_late_output
    )

    poor_recovery_mean = mean(
        poor_recovery_round_two
    )
    strong_recovery_mean = mean(
        strong_recovery_round_two
    )

    final_damage_mean = mean(damage_final)

    poor_stress_mean = mean(
        poor_stress_recovery_final
    )
    strong_stress_mean = mean(
        strong_stress_recovery_final
    )

    baseline_stay_rate = (
        baseline_stay_transitions
        / baseline_transitions
    )
    dynamic_stay_rate = (
        dynamic_stay_transitions
        / dynamic_transitions
    )

    checks = [
        AuditCheck(
            name="fatigue accumulates under sustained workload",
            passed=low_resistance_mean > 0.50,
            detail=(
                f"mean final fatigue={low_resistance_mean:.4f}"
            ),
        ),
        AuditCheck(
            name="fatigue resistance reduces accumulation",
            passed=(
                high_resistance_mean
                < low_resistance_mean
            ),
            detail=(
                f"low resistance={low_resistance_mean:.4f}, "
                f"high resistance={high_resistance_mean:.4f}"
            ),
        ),
        AuditCheck(
            name="performance resilience preserves late output",
            passed=(
                high_resilience_output_mean
                > low_resilience_output_mean
            ),
            detail=(
                f"low resilience={low_resilience_output_mean:.4f}, "
                f"high resilience={high_resilience_output_mean:.4f}"
            ),
        ),
        AuditCheck(
            name="round-break recovery lowers next-round fatigue",
            passed=(
                strong_recovery_mean
                < poor_recovery_mean
            ),
            detail=(
                f"poor recovery={poor_recovery_mean:.4f}, "
                f"strong recovery={strong_recovery_mean:.4f}"
            ),
        ),
        AuditCheck(
            name="damage accumulates from opponent offense",
            passed=final_damage_mean > 0.0,
            detail=(
                f"mean final damage={final_damage_mean:.4f}"
            ),
        ),
        AuditCheck(
            name="persistent damage is monotonic",
            passed=damage_monotonic_violations == 0,
            detail=(
                f"violations={damage_monotonic_violations}"
            ),
        ),
        AuditCheck(
            name="round breaks do not recover persistent damage",
            passed=damage_round_break_violations == 0,
            detail=(
                f"violations={damage_round_break_violations}"
            ),
        ),
        AuditCheck(
            name="acute-stress recovery reduces final stress",
            passed=(
                strong_stress_mean
                < poor_stress_mean
            ),
            detail=(
                f"poor recovery={poor_stress_mean:.4f}, "
                f"strong recovery={strong_stress_mean:.4f}"
            ),
        ),
        AuditCheck(
            name="fatigue-sensitive retention reduces phase persistence",
            passed=dynamic_stay_rate < baseline_stay_rate,
            detail=(
                f"baseline stay rate={baseline_stay_rate:.4%}, "
                f"dynamic stay rate={dynamic_stay_rate:.4%}"
            ),
        ),
        AuditCheck(
            name="dynamic structural legality",
            passed=structural_violations == 0,
            detail=(
                f"violations={structural_violations}"
            ),
        ),
    ]

    print("=" * 80)
    print("RFS MONTE CARLO V2 DYNAMIC PATH AUDIT")
    print("=" * 80)
    print(f"Paths per scenario: {path_count:,}")
    print(f"Seed start:         {seed_start:,}")
    print()

    all_passed = True

    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        all_passed = all_passed and check.passed

        print(
            f"[{status}] {check.name}"
        )
        print(
            f"       {check.detail}"
        )

    print()
    print("=" * 80)
    print(
        "AUDIT PASS"
        if all_passed
        else "AUDIT FAIL"
    )
    print("=" * 80)

    return 0 if all_passed else 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Audit dynamically evolving RFS Monte Carlo V2 paths."
        )
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=1_000,
        help="Number of paths per controlled scenario.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First deterministic path seed.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the command-line audit."""

    args = parse_args()

    return run_audit(
        path_count=args.paths,
        seed_start=args.seed_start,
    )


if __name__ == "__main__":
    raise SystemExit(main())
