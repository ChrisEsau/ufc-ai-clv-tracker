"""Finish-enabled dynamic path contracts for RFS Monte Carlo V2.

A finishing segment differs from an ordinary dynamic segment:

- activity and dynamic exposure are completed
- finish probabilities are calculated
- a finish may be sampled
- no shared-state transition occurs after a finish
- no round-break recovery occurs after a finish
- no later segments exist

These contracts support both completed finishes and fights that reach the end
of their scheduled rounds without a finish.
"""

from __future__ import annotations

from dataclasses import dataclass

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
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_exposure import (
    SegmentDynamicExposure,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_state import (
    FightDynamicState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_contracts import (
    FinishResult,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_probability import (
    SegmentFinishProbabilities,
)
from pipeline.simulation.rfs_mc_v2_shared_state.ground_activity_engine import (
    GroundSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_activity_dispatch import (
    PhaseSegmentActivity,
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


@dataclass(frozen=True)
class FinishEvaluatedPathSegment:
    """One dynamic segment after finish evaluation."""

    state: SharedFightState
    dynamic_state_before: FightDynamicState

    red_effective_phase: FighterPhaseParameters
    blue_effective_phase: FighterPhaseParameters
    activity: PhaseSegmentActivity

    exposure: SegmentDynamicExposure
    dynamic_state_after_activity: FightDynamicState

    finish_probabilities: SegmentFinishProbabilities
    finish: FinishResult | None

    red_effective_transition: FighterTransitionParameters | None
    blue_effective_transition: FighterTransitionParameters | None
    transition: SharedTransition | None

    round_break_recovery_applied: bool
    dynamic_state_after_segment: FightDynamicState

    def __post_init__(self) -> None:
        """Validate segment state, finish, and transition timing."""

        if not isinstance(
            self.state,
            SharedFightState,
        ):
            raise TypeError(
                "state must be SharedFightState"
            )

        if not isinstance(
            self.dynamic_state_before,
            FightDynamicState,
        ):
            raise TypeError(
                "dynamic_state_before must be FightDynamicState"
            )

        if not isinstance(
            self.red_effective_phase,
            FighterPhaseParameters,
        ):
            raise TypeError(
                "red_effective_phase must be FighterPhaseParameters"
            )

        if not isinstance(
            self.blue_effective_phase,
            FighterPhaseParameters,
        ):
            raise TypeError(
                "blue_effective_phase must be FighterPhaseParameters"
            )

        if not isinstance(
            self.activity,
            (
                DistanceSegmentActivity,
                ClinchSegmentActivity,
                GroundSegmentActivity,
            ),
        ):
            raise TypeError(
                "activity must be a supported phase segment activity"
            )

        if not isinstance(
            self.exposure,
            SegmentDynamicExposure,
        ):
            raise TypeError(
                "exposure must be SegmentDynamicExposure"
            )

        if not isinstance(
            self.dynamic_state_after_activity,
            FightDynamicState,
        ):
            raise TypeError(
                "dynamic_state_after_activity must be FightDynamicState"
            )

        if not isinstance(
            self.finish_probabilities,
            SegmentFinishProbabilities,
        ):
            raise TypeError(
                "finish_probabilities must be "
                "SegmentFinishProbabilities"
            )

        if (
            self.finish is not None
            and not isinstance(
                self.finish,
                FinishResult,
            )
        ):
            raise TypeError(
                "finish must be FinishResult or None"
            )

        if not isinstance(
            self.round_break_recovery_applied,
            bool,
        ):
            raise TypeError(
                "round_break_recovery_applied must be boolean"
            )

        if not isinstance(
            self.dynamic_state_after_segment,
            FightDynamicState,
        ):
            raise TypeError(
                "dynamic_state_after_segment must be FightDynamicState"
            )

        for name, value in (
            (
                "red_effective_transition",
                self.red_effective_transition,
            ),
            (
                "blue_effective_transition",
                self.blue_effective_transition,
            ),
        ):
            if (
                value is not None
                and not isinstance(
                    value,
                    FighterTransitionParameters,
                )
            ):
                raise TypeError(
                    f"{name} must be "
                    "FighterTransitionParameters or None"
                )

        if (
            self.transition is not None
            and not isinstance(
                self.transition,
                SharedTransition,
            )
        ):
            raise TypeError(
                "transition must be SharedTransition or None"
            )

        if self.activity.state != self.state:
            raise ValueError(
                "activity state must equal the segment state"
            )

        if self.exposure.state != self.state:
            raise ValueError(
                "exposure state must equal the segment state"
            )

        if self.finish_probabilities.state != self.state:
            raise ValueError(
                "finish-probability state must equal segment state"
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

        if (
            self.transition is not None
            and self.transition.previous_state != self.state
        ):
            raise ValueError(
                "transition previous state must equal segment state"
            )

        is_round_end = (
            self.state.segment_number
            == SEGMENTS_PER_ROUND
        )

        if self.finish is not None:
            if self.finish.state != self.state:
                raise ValueError(
                    "finish state must equal segment state"
                )

            if (
                self.red_effective_transition is not None
                or self.blue_effective_transition is not None
                or self.transition is not None
            ):
                raise ValueError(
                    "finishing segment cannot have a transition"
                )

            if self.round_break_recovery_applied:
                raise ValueError(
                    "finishing segment cannot apply round-break recovery"
                )

            if (
                self.dynamic_state_after_segment
                != self.dynamic_state_after_activity
            ):
                raise ValueError(
                    "finishing segment cannot alter dynamic state "
                    "after finish evaluation"
                )

            return

        if is_round_end:
            if (
                self.red_effective_transition is not None
                or self.blue_effective_transition is not None
                or self.transition is not None
            ):
                raise ValueError(
                    "round-ending segment cannot have a transition"
                )

            if (
                not self.round_break_recovery_applied
                and self.dynamic_state_after_segment
                != self.dynamic_state_after_activity
            ):
                raise ValueError(
                    "round-ending segment without recovery cannot "
                    "alter dynamic state after activity"
                )

        else:
            if (
                self.red_effective_transition is None
                or self.blue_effective_transition is None
                or self.transition is None
            ):
                raise ValueError(
                    "unfinished non-final segment requires a transition"
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
                    "non-final segment cannot alter dynamic state "
                    "after activity"
                )


@dataclass(frozen=True)
class FinishEnabledDynamicPath:
    """One finish-evaluated three- or five-round dynamic fight path."""

    scheduled_rounds: int
    seed: int
    segments: tuple[FinishEvaluatedPathSegment, ...]
    finish: FinishResult | None

    def __post_init__(self) -> None:
        """Validate timeline continuity and terminal result."""

        if self.scheduled_rounds not in {3, 5}:
            raise ValueError(
                "scheduled_rounds must be 3 or 5"
            )

        if not isinstance(
            self.seed,
            int,
        ):
            raise TypeError(
                "seed must be an integer"
            )

        if self.seed < 0:
            raise ValueError(
                "seed cannot be negative"
            )

        if not isinstance(
            self.segments,
            tuple,
        ):
            raise TypeError(
                "segments must be a tuple"
            )

        if not self.segments:
            raise ValueError(
                "path must contain at least one segment"
            )

        for record in self.segments:
            if not isinstance(
                record,
                FinishEvaluatedPathSegment,
            ):
                raise TypeError(
                    "segments must contain "
                    "FinishEvaluatedPathSegment values"
                )

        maximum_segments = (
            self.scheduled_rounds
            * SEGMENTS_PER_ROUND
        )

        if len(self.segments) > maximum_segments:
            raise ValueError(
                "path contains too many segments"
            )

        if (
            self.segments[0].dynamic_state_before
            != FightDynamicState.opening_state()
        ):
            raise ValueError(
                "fight must begin with fresh dynamic state"
            )

        finish_indices = [
            index
            for index, record in enumerate(self.segments)
            if record.finish is not None
        ]

        if self.finish is None:
            if finish_indices:
                raise ValueError(
                    "path-level finish is missing"
                )

            if len(self.segments) != maximum_segments:
                raise ValueError(
                    "unfinished path must contain all scheduled segments"
                )

        else:
            if not isinstance(
                self.finish,
                FinishResult,
            ):
                raise TypeError(
                    "finish must be FinishResult or None"
                )

            if finish_indices != [
                len(self.segments) - 1
            ]:
                raise ValueError(
                    "finish must occur only on the final stored segment"
                )

            if self.segments[-1].finish != self.finish:
                raise ValueError(
                    "path finish must equal final segment finish"
                )

        # Validate the index-derived timeline before checking
        # transitions between neighboring records. This produces the
        # most precise error when a record has the wrong round or segment.
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

            has_following_segment = (
                index < len(self.segments) - 1
            )

            expected_round_break = (
                expected_segment == SEGMENTS_PER_ROUND
                and has_following_segment
            )

            if (
                record.round_break_recovery_applied
                != expected_round_break
            ):
                raise ValueError(
                    "round-break recovery timing is inconsistent"
                )

            if not has_following_segment:
                continue

            following = self.segments[index + 1]

            if record.finish is not None:
                raise ValueError(
                    "no segment may follow a finish"
                )

            if (
                following.dynamic_state_before
                != record.dynamic_state_after_segment
            ):
                raise ValueError(
                    "dynamic-state result must feed the next segment"
                )

            if expected_segment < SEGMENTS_PER_ROUND:
                if (
                    record.transition is None
                    or record.transition.next_state
                    != following.state
                ):
                    raise ValueError(
                        "transition result must equal the following "
                        "segment state"
                    )

    @property
    def reached_scheduled_distance(self) -> bool:
        """Return whether the fight ended without a sampled finish."""

        return self.finish is None
