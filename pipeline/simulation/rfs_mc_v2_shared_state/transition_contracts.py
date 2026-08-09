"""Transition contracts for the V2 shared fight-state engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    SEGMENTS_PER_ROUND,
    FighterSide,
    SharedFightState,
)


class TransitionEvent(str, Enum):
    """Supported physical transitions between shared fight phases."""

    STAY = "stay"

    CLINCH_ENTRY = "clinch_entry"
    TAKEDOWN = "takedown"
    TAKEDOWN_ATTEMPT_FAILED = "takedown_attempt_failed"

    CLINCH_BREAK = "clinch_break"
    OWNERSHIP_CHANGE = "ownership_change"

    GROUND_ESCAPE = "ground_escape"
    SCRAMBLE_TO_CLINCH = "scramble_to_clinch"
    REVERSAL = "reversal"


TAKEDOWN_EVENTS = {
    TransitionEvent.TAKEDOWN,
    TransitionEvent.TAKEDOWN_ATTEMPT_FAILED,
}


LEGAL_EVENT_PHASES = {
    TransitionEvent.STAY: {
        (FightPhase.DISTANCE, FightPhase.DISTANCE),
        (FightPhase.CLINCH, FightPhase.CLINCH),
        (FightPhase.GROUND, FightPhase.GROUND),
    },
    TransitionEvent.CLINCH_ENTRY: {
        (FightPhase.DISTANCE, FightPhase.CLINCH),
    },
    TransitionEvent.TAKEDOWN: {
        (FightPhase.DISTANCE, FightPhase.GROUND),
        (FightPhase.CLINCH, FightPhase.GROUND),
    },
    TransitionEvent.TAKEDOWN_ATTEMPT_FAILED: {
        (FightPhase.DISTANCE, FightPhase.DISTANCE),
        (FightPhase.CLINCH, FightPhase.CLINCH),
    },
    TransitionEvent.CLINCH_BREAK: {
        (FightPhase.CLINCH, FightPhase.DISTANCE),
    },
    TransitionEvent.OWNERSHIP_CHANGE: {
        (FightPhase.CLINCH, FightPhase.CLINCH),
    },
    TransitionEvent.GROUND_ESCAPE: {
        (FightPhase.GROUND, FightPhase.DISTANCE),
    },
    TransitionEvent.SCRAMBLE_TO_CLINCH: {
        (FightPhase.GROUND, FightPhase.CLINCH),
    },
    TransitionEvent.REVERSAL: {
        (FightPhase.GROUND, FightPhase.GROUND),
    },
}


@dataclass(frozen=True)
class SharedTransition:
    """One coherent phase transition shared by both fighters.

    ``actor`` is the fighter responsible for the transition when one fighter
    must be identified. ``attempt_count`` records how many takedown attempts
    occurred inside a sampled wrestling sequence. Simulator-generated takedown
    events always populate it; zero remains accepted for legacy hand-built test
    fixtures and is interpreted as one attempt by downstream audit code.
    """

    previous_state: SharedFightState
    next_state: SharedFightState

    event: TransitionEvent
    actor: FighterSide | None
    attempt_count: int = 0

    def __post_init__(self) -> None:
        """Validate physical and ownership consistency."""

        previous = self.previous_state
        next_state = self.next_state

        if type(self.attempt_count) is not int:
            raise TypeError("attempt_count must be an integer")
        if self.attempt_count < 0:
            raise ValueError("attempt_count cannot be negative")
        if self.event not in TAKEDOWN_EVENTS and self.attempt_count != 0:
            raise ValueError(
                "attempt_count is only valid for takedown events"
            )

        if previous.round_number != next_state.round_number:
            raise ValueError(
                "a transition cannot cross round boundaries"
            )

        if previous.segment_number >= SEGMENTS_PER_ROUND:
            raise ValueError(
                "end-of-round state must use the round reset"
            )

        expected_next_segment = previous.segment_number + 1

        if next_state.segment_number != expected_next_segment:
            raise ValueError(
                "a transition must advance to the next segment"
            )

        phase_pair = (
            previous.phase,
            next_state.phase,
        )

        if phase_pair not in LEGAL_EVENT_PHASES[self.event]:
            raise ValueError(
                f"illegal phase pair for {self.event.value}: "
                f"{previous.phase.value} -> "
                f"{next_state.phase.value}"
            )

        # A failed shot/chain is a new physical exchange even though the broad
        # phase is unchanged, so it resets phase age rather than behaving as
        # a neutral STAY.
        expected_phase_age = (
            previous.phase_age_segments + 1
            if self.event is TransitionEvent.STAY
            else 0
        )

        if next_state.phase_age_segments != expected_phase_age:
            raise ValueError(
                "next phase_age_segments is inconsistent "
                "with the transition event"
            )

        if self.event is TransitionEvent.STAY:
            if self.actor is not None:
                raise ValueError(
                    "stay transition cannot have an actor"
                )

            if previous.phase_owner != next_state.phase_owner:
                raise ValueError(
                    "stay transition cannot change phase owner"
                )

        elif self.event in {
            TransitionEvent.CLINCH_ENTRY,
            TransitionEvent.TAKEDOWN,
            TransitionEvent.SCRAMBLE_TO_CLINCH,
        }:
            if self.actor is None:
                raise ValueError(
                    f"{self.event.value} requires an actor"
                )

            if next_state.phase_owner is not self.actor:
                raise ValueError(
                    f"{self.event.value} actor must own "
                    "the resulting phase"
                )

        elif self.event is TransitionEvent.TAKEDOWN_ATTEMPT_FAILED:
            if self.actor is None:
                raise ValueError(
                    "takedown_attempt_failed requires an actor"
                )

            if previous.phase_owner != next_state.phase_owner:
                raise ValueError(
                    "failed takedown cannot change phase ownership"
                )

        elif self.event is TransitionEvent.CLINCH_BREAK:
            if self.actor is not None:
                raise ValueError(
                    "clinch break actor is not modeled yet"
                )

        elif self.event in {
            TransitionEvent.OWNERSHIP_CHANGE,
            TransitionEvent.REVERSAL,
        }:
            if self.actor is None:
                raise ValueError(
                    f"{self.event.value} requires an actor"
                )

            if previous.phase_owner == next_state.phase_owner:
                raise ValueError(
                    f"{self.event.value} must change ownership"
                )

            if next_state.phase_owner is not self.actor:
                raise ValueError(
                    f"{self.event.value} actor must become "
                    "the new phase owner"
                )

        elif self.event is TransitionEvent.GROUND_ESCAPE:
            if previous.phase_owner is None:
                raise ValueError(
                    "ground escape requires a prior owner"
                )

            expected_actor = previous.phase_owner.opponent

            if self.actor is not expected_actor:
                raise ValueError(
                    "ground escape actor must be the "
                    "previous ground defender"
                )
