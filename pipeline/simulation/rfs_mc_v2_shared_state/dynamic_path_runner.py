"""Dynamic full-path integration for RFS Monte Carlo V2.

Each segment follows this timing contract:

1. read the authoritative shared and dynamic states
2. build temporary effective phase parameters
3. generate phase-legal activity
4. calculate raw workload and adversity exposure
5. update both fighters' dynamic states
6. build temporary effective transition parameters
7. sample the end-of-segment shared-state transition

After segment ten, between-round recovery is applied before the next round
opens at distance.

Baseline phase and transition parameters remain immutable. Activity and
transition randomness use separate deterministic streams.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.clinch_activity_engine import (
    ClinchSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    SEGMENTS_PER_ROUND,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.distance_activity_engine import (
    DistanceSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_calibration import (
    DynamicStateCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_effect_calibration import (
    DynamicEffectCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_exposure import (
    SegmentDynamicExposure,
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
)
from pipeline.simulation.rfs_mc_v2_shared_state.effective_transition_parameters import (
    build_effective_transition_parameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.ground_activity_engine import (
    GroundSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.path_runner import (
    SharedPathCalibration,
    select_transition_distribution,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_activity_dispatch import (
    PhaseSegmentActivity,
    generate_phase_segment_activity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    FighterPhaseParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_contracts import (
    SharedTransition,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_sampler import (
    sample_and_apply_transition,
)


@dataclass(frozen=True)
class DynamicPathSegment:
    """One fully integrated dynamic simulation segment."""

    state: SharedFightState
    dynamic_state_before: FightDynamicState

    red_effective_phase: FighterPhaseParameters
    blue_effective_phase: FighterPhaseParameters
    activity: PhaseSegmentActivity

    exposure: SegmentDynamicExposure
    dynamic_state_after_activity: FightDynamicState

    red_effective_transition: FighterTransitionParameters | None
    blue_effective_transition: FighterTransitionParameters | None
    transition: SharedTransition | None

    round_break_recovery_applied: bool
    dynamic_state_after_segment: FightDynamicState

    def __post_init__(self) -> None:
        """Validate the segment timing and phase contracts."""

        if self.activity.state != self.state:
            raise ValueError(
                "activity state must equal the segment state"
            )

        if self.exposure.state != self.state:
            raise ValueError(
                "exposure state must equal the segment state"
            )

        if self.state.phase is FightPhase.DISTANCE:
            expected_activity_type = DistanceSegmentActivity
        elif self.state.phase is FightPhase.CLINCH:
            expected_activity_type = ClinchSegmentActivity
        elif self.state.phase is FightPhase.GROUND:
            expected_activity_type = GroundSegmentActivity
        else:
            raise ValueError(
                f"unsupported fight phase: {self.state.phase}"
            )

        if not isinstance(
            self.activity,
            expected_activity_type,
        ):
            raise ValueError(
                "activity type does not match shared phase"
            )

        if not isinstance(
            self.round_break_recovery_applied,
            bool,
        ):
            raise TypeError(
                "round_break_recovery_applied must be boolean"
            )

        is_round_end = (
            self.state.segment_number == SEGMENTS_PER_ROUND
        )

        if is_round_end:
            if self.transition is not None:
                raise ValueError(
                    "round-ending segment cannot have a transition"
                )

            if (
                self.red_effective_transition is not None
                or self.blue_effective_transition is not None
            ):
                raise ValueError(
                    "round-ending segment cannot have effective "
                    "transition parameters"
                )

        else:
            if self.transition is None:
                raise ValueError(
                    "non-final segment requires a transition"
                )

            if (
                self.red_effective_transition is None
                or self.blue_effective_transition is None
            ):
                raise ValueError(
                    "non-final segment requires effective "
                    "transition parameters"
                )

            if self.round_break_recovery_applied:
                raise ValueError(
                    "round-break recovery can only follow segment ten"
                )

            if (
                self.dynamic_state_after_segment
                != self.dynamic_state_after_activity
            ):
                raise ValueError(
                    "non-final segment cannot change dynamic state "
                    "after activity"
                )

        if (
            self.transition is not None
            and self.transition.previous_state != self.state
        ):
            raise ValueError(
                "transition previous state must equal segment state"
            )


@dataclass(frozen=True)
class DynamicActivityPath:
    """One complete dynamic three- or five-round fight path."""

    scheduled_rounds: int
    seed: int
    segments: tuple[DynamicPathSegment, ...]

    def __post_init__(self) -> None:
        """Validate complete shared- and dynamic-state timelines."""

        if self.scheduled_rounds not in {3, 5}:
            raise ValueError(
                "scheduled_rounds must be 3 or 5"
            )

        if self.seed < 0:
            raise ValueError(
                "seed cannot be negative"
            )

        expected_count = (
            self.scheduled_rounds
            * SEGMENTS_PER_ROUND
        )

        if len(self.segments) != expected_count:
            raise ValueError(
                "path contains an unexpected number of segments"
            )

        if (
            self.segments[0].dynamic_state_before
            != FightDynamicState.opening_state()
        ):
            raise ValueError(
                "fight must begin with fresh dynamic state"
            )

        for index, record in enumerate(self.segments):
            expected_round = (
                index // SEGMENTS_PER_ROUND
            ) + 1
            expected_segment = (
                index % SEGMENTS_PER_ROUND
            ) + 1

            if record.state.round_number != expected_round:
                raise ValueError(
                    "path round sequence is inconsistent"
                )

            if record.state.segment_number != expected_segment:
                raise ValueError(
                    "path segment sequence is inconsistent"
                )

            if expected_segment == 1:
                expected_opening = (
                    SharedFightState.opening_state(
                        round_number=expected_round,
                    )
                )

                if record.state != expected_opening:
                    raise ValueError(
                        "every round must begin at distance"
                    )

            expected_round_break = (
                expected_segment == SEGMENTS_PER_ROUND
                and expected_round < self.scheduled_rounds
            )

            if (
                record.round_break_recovery_applied
                != expected_round_break
            ):
                raise ValueError(
                    "round-break recovery timing is inconsistent"
                )

            if index == len(self.segments) - 1:
                continue

            next_record = self.segments[index + 1]

            if (
                next_record.dynamic_state_before
                != record.dynamic_state_after_segment
            ):
                raise ValueError(
                    "dynamic-state result must feed the next segment"
                )

            if expected_segment < SEGMENTS_PER_ROUND:
                if (
                    record.transition is None
                    or record.transition.next_state
                    != next_record.state
                ):
                    raise ValueError(
                        "transition result must equal the following "
                        "segment state"
                    )


def run_dynamic_activity_path(
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
    scheduled_rounds: int,
    seed: int,
    shared_path_calibration: SharedPathCalibration | None = None,
) -> DynamicActivityPath:
    """Generate one complete dynamically evolving fight path."""

    if scheduled_rounds not in {3, 5}:
        raise ValueError(
            "scheduled_rounds must be 3 or 5"
        )

    if seed < 0:
        raise ValueError(
            "seed cannot be negative"
        )

    selected_shared_calibration = (
        shared_path_calibration
        if shared_path_calibration is not None
        else SharedPathCalibration()
    )

    # Preserve independent deterministic streams for activity and transitions.
    transition_rng = np.random.default_rng(seed)
    activity_rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                seed,
                0xA11CE,
            ]
        )
    )

    records: list[DynamicPathSegment] = []
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
                DynamicPathSegment(
                    state=current_shared_state,
                    dynamic_state_before=dynamic_state_before,
                    red_effective_phase=red_effective_phase,
                    blue_effective_phase=blue_effective_phase,
                    activity=activity,
                    exposure=exposure,
                    dynamic_state_after_activity=(
                        dynamic_state_after_activity
                    ),
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

    return DynamicActivityPath(
        scheduled_rounds=scheduled_rounds,
        seed=seed,
        segments=tuple(records),
    )
