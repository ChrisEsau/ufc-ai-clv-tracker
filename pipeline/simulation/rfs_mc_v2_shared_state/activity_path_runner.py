"""Static shared-state activity paths for RFS Monte Carlo V2.

Each 30-second segment contains:

1. the authoritative shared fight state
2. phase-legal activity generated from that state
3. an end-of-segment transition, except after segment ten

Transition and activity random generators are intentionally separate.
Changing activity-generation logic must not change the shared phase timeline
for the same seed.

Dynamic-state modifiers, finishes, and scoring remain out of scope.
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
class StaticActivityPathSegment:
    """One shared-state segment with legal activity and transition."""

    state: SharedFightState
    activity: PhaseSegmentActivity
    transition: SharedTransition | None

    def __post_init__(self) -> None:
        """Validate phase, activity, and transition consistency."""

        if self.activity.state != self.state:
            raise ValueError(
                "activity state must equal the segment state"
            )

        expected_activity_type: type[
            DistanceSegmentActivity
            | ClinchSegmentActivity
            | GroundSegmentActivity
        ]

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

        is_round_end = (
            self.state.segment_number == SEGMENTS_PER_ROUND
        )

        if is_round_end and self.transition is not None:
            raise ValueError(
                "round-ending segment cannot have a "
                "same-round transition"
            )

        if not is_round_end and self.transition is None:
            raise ValueError(
                "non-final segment requires a transition"
            )

        if (
            self.transition is not None
            and self.transition.previous_state != self.state
        ):
            raise ValueError(
                "transition previous state must equal "
                "the segment state"
            )


@dataclass(frozen=True)
class StaticActivityPath:
    """One complete static shared-state fight path."""

    scheduled_rounds: int
    seed: int
    segments: tuple[StaticActivityPathSegment, ...]

    def __post_init__(self) -> None:
        """Validate the full round and segment timeline."""

        if self.scheduled_rounds not in {3, 5}:
            raise ValueError(
                "scheduled_rounds must be 3 or 5"
            )

        if self.seed < 0:
            raise ValueError(
                "seed cannot be negative"
            )

        expected_count = (
            self.scheduled_rounds * SEGMENTS_PER_ROUND
        )

        if len(self.segments) != expected_count:
            raise ValueError(
                "path contains an unexpected number "
                "of segments"
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

            if expected_segment < SEGMENTS_PER_ROUND:
                next_record = self.segments[index + 1]

                if (
                    record.transition is None
                    or record.transition.next_state
                    != next_record.state
                ):
                    raise ValueError(
                        "transition result must equal the "
                        "following segment state"
                    )


def run_static_activity_path(
    red_transition: FighterTransitionParameters,
    blue_transition: FighterTransitionParameters,
    red_phase: FighterPhaseParameters,
    blue_phase: FighterPhaseParameters,
    *,
    scheduled_rounds: int,
    seed: int,
    calibration: SharedPathCalibration | None = None,
) -> StaticActivityPath:
    """Generate one coherent static V2 fight path."""

    if scheduled_rounds not in {3, 5}:
        raise ValueError(
            "scheduled_rounds must be 3 or 5"
        )

    if seed < 0:
        raise ValueError(
            "seed cannot be negative"
        )

    selected_calibration = (
        calibration
        if calibration is not None
        else SharedPathCalibration()
    )

    # Preserve the exact transition stream used by the state-only runner.
    transition_rng = np.random.default_rng(seed)

    # Keep activity randomness independent of transition randomness.
    activity_rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                seed,
                0xA11CE,
            ]
        )
    )

    records: list[StaticActivityPathSegment] = []

    for round_number in range(
        1,
        scheduled_rounds + 1,
    ):
        current_state = SharedFightState.opening_state(
            round_number=round_number,
        )

        for segment_number in range(
            1,
            SEGMENTS_PER_ROUND + 1,
        ):
            if current_state.segment_number != segment_number:
                raise RuntimeError(
                    "shared-state segment sequence drifted"
                )

            activity = generate_phase_segment_activity(
                current_state,
                red_phase,
                blue_phase,
                activity_rng,
            )

            if segment_number == SEGMENTS_PER_ROUND:
                transition = None
            else:
                distribution = select_transition_distribution(
                    current_state,
                    red_transition,
                    blue_transition,
                    calibration=selected_calibration,
                )

                transition = sample_and_apply_transition(
                    current_state,
                    distribution,
                    transition_rng,
                    calibration=(
                        selected_calibration.transition_state
                    ),
                )

            records.append(
                StaticActivityPathSegment(
                    state=current_state,
                    activity=activity,
                    transition=transition,
                )
            )

            if transition is not None:
                current_state = transition.next_state

    return StaticActivityPath(
        scheduled_rounds=scheduled_rounds,
        seed=seed,
        segments=tuple(records),
    )
