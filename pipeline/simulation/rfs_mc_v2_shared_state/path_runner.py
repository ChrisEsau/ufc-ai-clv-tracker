"""Shared fight-state path runner for RFS Monte Carlo V2.

Milestone 2E generates only the physical phase timeline:

- every round begins at distance
- every segment has one authoritative shared state
- non-final segments sample exactly one transition
- segment ten ends the round without a same-round transition
- the next round resets to distance
- no activity, damage, finishes, or scoring are generated yet
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    SEGMENTS_PER_ROUND,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_contracts import (
    SharedTransition,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
    ClinchTransitionCalibration,
    DistanceTransitionCalibration,
    GroundTransitionCalibration,
    TransitionDistribution,
    build_clinch_transition_distribution,
    build_distance_transition_distribution,
    build_ground_transition_distribution,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_sampler import (
    TransitionStateCalibration,
    sample_and_apply_transition,
)


@dataclass(frozen=True)
class SharedPathCalibration:
    """Static calibration bundle used by one V2 state path."""

    distance: DistanceTransitionCalibration = field(
        default_factory=DistanceTransitionCalibration
    )
    clinch: ClinchTransitionCalibration = field(
        default_factory=ClinchTransitionCalibration
    )
    ground: GroundTransitionCalibration = field(
        default_factory=GroundTransitionCalibration
    )
    transition_state: TransitionStateCalibration = field(
        default_factory=TransitionStateCalibration
    )


@dataclass(frozen=True)
class SharedPathSegment:
    """Shared physical state and its end-of-segment transition."""

    state: SharedFightState
    transition: SharedTransition | None

    def __post_init__(self) -> None:
        """Validate segment and transition timing."""

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
class SharedStatePath:
    """One complete shared-state Monte Carlo fight path."""

    scheduled_rounds: int
    seed: int
    segments: tuple[SharedPathSegment, ...]

    def __post_init__(self) -> None:
        """Validate the complete round and segment timeline."""

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


def select_transition_distribution(
    current_state: SharedFightState,
    red: FighterTransitionParameters,
    blue: FighterTransitionParameters,
    *,
    calibration: SharedPathCalibration | None = None,
) -> TransitionDistribution:
    """Select and build the legal distribution for the current phase."""

    selected_calibration = (
        calibration
        if calibration is not None
        else SharedPathCalibration()
    )

    if current_state.phase is FightPhase.DISTANCE:
        return build_distance_transition_distribution(
            red,
            blue,
            calibration=selected_calibration.distance,
        )

    if current_state.phase is FightPhase.CLINCH:
        if current_state.phase_owner is None:
            raise ValueError(
                "clinch state requires a phase owner"
            )

        return build_clinch_transition_distribution(
            red,
            blue,
            current_owner=current_state.phase_owner,
            calibration=selected_calibration.clinch,
        )

    if current_state.phase is FightPhase.GROUND:
        if current_state.phase_owner is None:
            raise ValueError(
                "ground state requires a phase owner"
            )

        return build_ground_transition_distribution(
            red,
            blue,
            current_owner=current_state.phase_owner,
            calibration=selected_calibration.ground,
        )

    raise ValueError(
        f"unsupported fight phase: {current_state.phase}"
    )


def run_shared_state_path(
    red: FighterTransitionParameters,
    blue: FighterTransitionParameters,
    *,
    scheduled_rounds: int,
    seed: int,
    calibration: SharedPathCalibration | None = None,
) -> SharedStatePath:
    """Generate one deterministic shared-state fight path."""

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

    rng = np.random.default_rng(seed)
    records: list[SharedPathSegment] = []

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

            if segment_number == SEGMENTS_PER_ROUND:
                records.append(
                    SharedPathSegment(
                        state=current_state,
                        transition=None,
                    )
                )
                continue

            distribution = select_transition_distribution(
                current_state,
                red,
                blue,
                calibration=selected_calibration,
            )

            transition = sample_and_apply_transition(
                current_state,
                distribution,
                rng,
                calibration=(
                    selected_calibration.transition_state
                ),
            )

            records.append(
                SharedPathSegment(
                    state=current_state,
                    transition=transition,
                )
            )

            current_state = transition.next_state

    return SharedStatePath(
        scheduled_rounds=scheduled_rounds,
        seed=seed,
        segments=tuple(records),
    )
