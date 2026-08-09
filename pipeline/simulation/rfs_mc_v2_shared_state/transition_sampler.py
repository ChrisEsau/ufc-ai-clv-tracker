"""Seeded transition sampling and shared-state application for V2.

The current phase describes activity during the current 30-second segment.
A sampled transition is applied at the end of that segment and produces the
authoritative shared state for the following segment.

This module does not generate phase activity or modify dynamic fighter state.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    SEGMENTS_PER_ROUND,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_contracts import (
    SharedTransition,
    TransitionEvent,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
    TransitionDistribution,
    TransitionProbability,
)


@dataclass(frozen=True)
class TransitionStateCalibration:
    """Initial position quality assigned after ownership transitions.

    These values initialize a newly established position. They are not
    historical estimates and are intentionally isolated for later
    calibration.
    """

    clinch_entry_position_quality: float = 0.30
    takedown_position_quality: float = 0.55
    ownership_change_position_quality: float = 0.30
    scramble_position_quality: float = 0.25
    reversal_position_quality: float = 0.40

    def __post_init__(self) -> None:
        """Require all initial position qualities to be bounded."""

        for name, value in vars(self).items():
            if (
                not isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )


def sample_transition_option(
    distribution: TransitionDistribution,
    rng: np.random.Generator,
) -> TransitionProbability:
    """Sample one option from a normalized transition distribution."""

    draw = float(rng.random())

    if not 0.0 <= draw < 1.0:
        raise ValueError(
            "random draw must be between 0 inclusive "
            "and 1 exclusive"
        )

    cumulative_probability = 0.0

    for option in distribution.options:
        cumulative_probability += option.probability

        if draw < cumulative_probability:
            return option

    # Protect against floating-point accumulation ending microscopically
    # below one. TransitionDistribution already validates the true total.
    return distribution.options[-1]


def apply_transition_option(
    current_state: SharedFightState,
    option: TransitionProbability,
    *,
    calibration: TransitionStateCalibration | None = None,
) -> SharedTransition:
    """Apply one sampled option to produce the next shared fight state."""

    if current_state.segment_number >= SEGMENTS_PER_ROUND:
        raise ValueError(
            "end-of-round state must use the round reset"
        )

    selected_calibration = (
        calibration
        if calibration is not None
        else TransitionStateCalibration()
    )

    event = option.event
    actor = option.actor

    if event is TransitionEvent.STAY:
        next_phase = current_state.phase
        next_owner = current_state.phase_owner
        next_position_quality = current_state.position_quality
        next_phase_age = current_state.phase_age_segments + 1

    elif event is TransitionEvent.CLINCH_ENTRY:
        next_phase = FightPhase.CLINCH
        next_owner = actor
        next_position_quality = (
            selected_calibration.clinch_entry_position_quality
        )
        next_phase_age = 0

    elif event is TransitionEvent.TAKEDOWN:
        next_phase = FightPhase.GROUND
        next_owner = actor
        next_position_quality = (
            selected_calibration.takedown_position_quality
        )
        next_phase_age = 0

    elif event is TransitionEvent.TAKEDOWN_ATTEMPT_FAILED:
        # A failed wrestling sequence is a real physical exchange but does not
        # establish a new broad phase. Distance remains ownerless; clinch
        # ownership stays with whoever controlled the clinch before the chain.
        next_phase = current_state.phase
        next_owner = current_state.phase_owner
        next_position_quality = current_state.position_quality
        next_phase_age = 0

    elif event is TransitionEvent.CLINCH_BREAK:
        next_phase = FightPhase.DISTANCE
        next_owner = None
        next_position_quality = 0.0
        next_phase_age = 0

    elif event is TransitionEvent.OWNERSHIP_CHANGE:
        next_phase = FightPhase.CLINCH
        next_owner = actor
        next_position_quality = (
            selected_calibration
            .ownership_change_position_quality
        )
        next_phase_age = 0

    elif event is TransitionEvent.GROUND_ESCAPE:
        next_phase = FightPhase.DISTANCE
        next_owner = None
        next_position_quality = 0.0
        next_phase_age = 0

    elif event is TransitionEvent.SCRAMBLE_TO_CLINCH:
        next_phase = FightPhase.CLINCH
        next_owner = actor
        next_position_quality = (
            selected_calibration.scramble_position_quality
        )
        next_phase_age = 0

    elif event is TransitionEvent.REVERSAL:
        next_phase = FightPhase.GROUND
        next_owner = actor
        next_position_quality = (
            selected_calibration.reversal_position_quality
        )
        next_phase_age = 0

    else:
        raise ValueError(
            f"unsupported transition event: {event}"
        )

    next_state = SharedFightState(
        phase=next_phase,
        phase_owner=next_owner,
        phase_age_segments=next_phase_age,
        position_quality=next_position_quality,
        round_number=current_state.round_number,
        segment_number=current_state.segment_number + 1,
    )

    # SharedTransition performs the final legal-phase, actor, ownership,
    # timing, phase-age, and attempt-count validation.
    return SharedTransition(
        previous_state=current_state,
        next_state=next_state,
        event=event,
        actor=actor,
        attempt_count=option.attempt_count,
    )


def sample_and_apply_transition(
    current_state: SharedFightState,
    distribution: TransitionDistribution,
    rng: np.random.Generator,
    *,
    calibration: TransitionStateCalibration | None = None,
) -> SharedTransition:
    """Sample and apply one end-of-segment shared transition."""

    if distribution.source_phase is not current_state.phase:
        raise ValueError(
            "transition distribution source phase does not "
            "match current shared phase"
        )

    option = sample_transition_option(
        distribution,
        rng,
    )

    return apply_transition_option(
        current_state,
        option,
        calibration=calibration,
    )
