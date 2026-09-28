"""Finish-enabled dynamic path runner for RFS Monte Carlo V2.

This runner preserves the validated dynamic segment lifecycle while adding
finish evaluation:

1. read current shared and dynamic states
2. build effective phase parameters
3. generate phase-legal activity
4. calculate workload and adversity exposure
5. update dynamic state
6. calculate deterministic finish probabilities
7. sample a finish using an independent RNG stream
8. stop immediately if a finish occurs
9. otherwise sample the normal end-of-segment transition
10. apply round-break recovery after unfinished nonfinal rounds

The existing full-length dynamic runner remains unchanged.
"""

from __future__ import annotations

import numpy as np

from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    SEGMENTS_PER_ROUND,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_calibration import (
    DynamicStateCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_effect_calibration import (
    DynamicEffectCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_exposure import (
    calculate_segment_dynamic_exposure,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_parameters import (
    FighterDynamicParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_state import (
    FightDynamicState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_state_updater import (
    apply_round_break_recovery,
    update_fight_dynamic_state,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_transition_effect_calibration import (
    DynamicTransitionEffectCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.effective_phase_parameters import (
    build_effective_phase_parameters,
    calculate_capability_multipliers,
)
from pipeline.simulation.rfs_mc_v2_shared_state.effective_transition_parameters import (
    build_effective_transition_parameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_calibration import (
    FinishProbabilityCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_path_contracts import (
    FinishEnabledDynamicPath,
    FinishEvaluatedPathSegment,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_probability import (
    calculate_segment_finish_probabilities,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_sampler import (
    sample_segment_finish,
)
from pipeline.simulation.rfs_mc_v2_shared_state.path_runner import (
    SharedPathCalibration,
    select_transition_distribution,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_activity_dispatch import (
    generate_phase_segment_activity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    FighterPhaseParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_sampler import (
    sample_and_apply_transition,
)


def run_finish_enabled_dynamic_path(
    red_transition_baseline: FighterTransitionParameters,
    blue_transition_baseline: FighterTransitionParameters,
    red_phase_baseline: FighterPhaseParameters,
    blue_phase_baseline: FighterPhaseParameters,
    red_dynamic_parameters: FighterDynamicParameters,
    blue_dynamic_parameters: FighterDynamicParameters,
    *,
    dynamic_state_calibration: DynamicStateCalibration,
    phase_effect_calibration: DynamicEffectCalibration,
    transition_effect_calibration: (
        DynamicTransitionEffectCalibration
    ),
    finish_probability_calibration: FinishProbabilityCalibration,
    scheduled_rounds: int,
    seed: int,
    red_intrinsic_power_multiplier: float = 1.0,
    blue_intrinsic_power_multiplier: float = 1.0,
    red_intrinsic_ko_vulnerability_multiplier: float = 1.0,
    blue_intrinsic_ko_vulnerability_multiplier: float = 1.0,
    shared_path_calibration: SharedPathCalibration | None = None,
) -> FinishEnabledDynamicPath:
    """Generate one dynamic fight path that stops at a sampled finish."""

    if scheduled_rounds not in {3, 5}:
        raise ValueError(
            "scheduled_rounds must be 3 or 5"
        )

    if not isinstance(
        seed,
        int,
    ):
        raise TypeError(
            "seed must be an integer"
        )

    if seed < 0:
        raise ValueError(
            "seed cannot be negative"
        )

    if not isinstance(
        finish_probability_calibration,
        FinishProbabilityCalibration,
    ):
        raise TypeError(
            "finish_probability_calibration must be "
            "FinishProbabilityCalibration"
        )

    selected_shared_calibration = (
        shared_path_calibration
        if shared_path_calibration is not None
        else SharedPathCalibration()
    )

    # Preserve the existing transition and activity streams exactly.
    transition_rng = np.random.default_rng(seed)
    activity_rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                seed,
                0xA11CE,
            ]
        )
    )

    # Finish sampling is isolated so it cannot perturb activity or transitions.
    finish_rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                seed,
                0xF1A15,
            ]
        )
    )

    records: list[FinishEvaluatedPathSegment] = []
    current_dynamic_state = FightDynamicState.opening_state()

    for round_number in range(
        1,
        scheduled_rounds + 1,
    ):
        current_shared_state = SharedFightState.opening_state(
            round_number=round_number,
        )

        for segment_number in range(
            1,
            SEGMENTS_PER_ROUND + 1,
        ):
            if (
                current_shared_state.segment_number
                != segment_number
            ):
                raise RuntimeError(
                    "shared-state segment sequence drifted"
                )

            dynamic_state_before = current_dynamic_state

            # Resolve the same live capability multipliers used to build
            # temporary effective phase parameters. Finish conversion needs
            # the attacker's current power separately so generic landed-
            # strike KO hazard decays with finishing potency.
            red_capability_multipliers = (
                calculate_capability_multipliers(
                    dynamic_state_before.red,
                    red_dynamic_parameters,
                    phase_effect_calibration,
                )
            )
            blue_capability_multipliers = (
                calculate_capability_multipliers(
                    dynamic_state_before.blue,
                    blue_dynamic_parameters,
                    phase_effect_calibration,
                )
            )

            red_effective_phase = (
                build_effective_phase_parameters(
                    red_phase_baseline,
                    dynamic_state_before.red,
                    red_dynamic_parameters,
                    phase_effect_calibration,
                )
            )
            blue_effective_phase = (
                build_effective_phase_parameters(
                    blue_phase_baseline,
                    dynamic_state_before.blue,
                    blue_dynamic_parameters,
                    phase_effect_calibration,
                )
            )

            activity = generate_phase_segment_activity(
                current_shared_state,
                red_effective_phase,
                blue_effective_phase,
                activity_rng,
            )

            exposure = calculate_segment_dynamic_exposure(
                activity,
                dynamic_state_calibration,
            )

            dynamic_state_after_activity = (
                update_fight_dynamic_state(
                    dynamic_state_before,
                    exposure,
                    red_dynamic_parameters,
                    blue_dynamic_parameters,
                    dynamic_state_calibration,
                )
            )

            # Finish conversion occurs after the segment's activity and
            # dynamic exposure have been realized.
            finish_probabilities = (
                calculate_segment_finish_probabilities(
                    activity,
                    dynamic_state_after_activity,
                    red_effective_phase,
                    blue_effective_phase,
                    finish_probability_calibration,
                    red_power_multiplier=(
                        red_intrinsic_power_multiplier
                        * red_capability_multipliers.power
                    ),
                    blue_power_multiplier=(
                        blue_intrinsic_power_multiplier
                        * blue_capability_multipliers.power
                    ),
                    red_ko_vulnerability_multiplier=(
                        red_intrinsic_ko_vulnerability_multiplier
                    ),
                    blue_ko_vulnerability_multiplier=(
                        blue_intrinsic_ko_vulnerability_multiplier
                    ),
                )
            )

            finish = sample_segment_finish(
                finish_probabilities,
                finish_rng,
            )

            if finish is not None:
                records.append(
                    FinishEvaluatedPathSegment(
                        state=current_shared_state,
                        dynamic_state_before=dynamic_state_before,
                        red_effective_phase=red_effective_phase,
                        blue_effective_phase=blue_effective_phase,
                        activity=activity,
                        exposure=exposure,
                        dynamic_state_after_activity=(
                            dynamic_state_after_activity
                        ),
                        finish_probabilities=finish_probabilities,
                        finish=finish,
                        red_effective_transition=None,
                        blue_effective_transition=None,
                        transition=None,
                        round_break_recovery_applied=False,
                        dynamic_state_after_segment=(
                            dynamic_state_after_activity
                        ),
                    )
                )

                return FinishEnabledDynamicPath(
                    scheduled_rounds=scheduled_rounds,
                    seed=seed,
                    segments=tuple(records),
                    finish=finish,
                )

            if segment_number < SEGMENTS_PER_ROUND:
                red_effective_transition = (
                    build_effective_transition_parameters(
                        red_transition_baseline,
                        dynamic_state_after_activity.red,
                        red_dynamic_parameters,
                        transition_effect_calibration,
                    )
                )
                blue_effective_transition = (
                    build_effective_transition_parameters(
                        blue_transition_baseline,
                        dynamic_state_after_activity.blue,
                        blue_dynamic_parameters,
                        transition_effect_calibration,
                    )
                )

                distribution = select_transition_distribution(
                    current_shared_state,
                    red_effective_transition,
                    blue_effective_transition,
                    calibration=selected_shared_calibration,
                )

                transition = sample_and_apply_transition(
                    current_shared_state,
                    distribution,
                    transition_rng,
                    calibration=(
                        selected_shared_calibration.transition_state
                    ),
                )

                round_break_applied = False
                dynamic_state_after_segment = (
                    dynamic_state_after_activity
                )

            else:
                red_effective_transition = None
                blue_effective_transition = None
                transition = None

                if round_number < scheduled_rounds:
                    dynamic_state_after_segment = (
                        apply_round_break_recovery(
                            dynamic_state_after_activity,
                            red_dynamic_parameters,
                            blue_dynamic_parameters,
                            dynamic_state_calibration,
                        )
                    )
                    round_break_applied = True
                else:
                    dynamic_state_after_segment = (
                        dynamic_state_after_activity
                    )
                    round_break_applied = False

            records.append(
                FinishEvaluatedPathSegment(
                    state=current_shared_state,
                    dynamic_state_before=dynamic_state_before,
                    red_effective_phase=red_effective_phase,
                    blue_effective_phase=blue_effective_phase,
                    activity=activity,
                    exposure=exposure,
                    dynamic_state_after_activity=(
                        dynamic_state_after_activity
                    ),
                    finish_probabilities=finish_probabilities,
                    finish=None,
                    red_effective_transition=(
                        red_effective_transition
                    ),
                    blue_effective_transition=(
                        blue_effective_transition
                    ),
                    transition=transition,
                    round_break_recovery_applied=(
                        round_break_applied
                    ),
                    dynamic_state_after_segment=(
                        dynamic_state_after_segment
                    ),
                )
            )

            current_dynamic_state = (
                dynamic_state_after_segment
            )

            if transition is not None:
                current_shared_state = (
                    transition.next_state
                )

    return FinishEnabledDynamicPath(
        scheduled_rounds=scheduled_rounds,
        seed=seed,
        segments=tuple(records),
        finish=None,
    )
