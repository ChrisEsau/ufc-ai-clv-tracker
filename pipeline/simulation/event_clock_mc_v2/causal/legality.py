"""Single authoritative action-legality contract for causal Event Clock V2."""

from __future__ import annotations

from .events import ActionEvent, ActionFamily
from .state import FightState, Phase, Side


STANDING_ACTIONS = (
    ActionFamily.STAND_ATTACK,
    ActionFamily.STAND_COUNTER,
    ActionFamily.PRESSURE,
    ActionFamily.RESET_RANGE,
    ActionFamily.CLINCH_ENTRY,
    ActionFamily.TAKEDOWN_ENTRY,
)

CLINCH_ACTIONS = (
    ActionFamily.CLINCH_STRIKE,
    ActionFamily.CLINCH_CONTROL,
    ActionFamily.CLINCH_TAKEDOWN,
    ActionFamily.BREAK_CLINCH,
)

GROUND_TOP_ACTIONS = (
    ActionFamily.GROUND_STRIKE,
    ActionFamily.ADVANCE_POSITION,
    ActionFamily.SUBMISSION_ATTACK,
    ActionFamily.CONTROL,
    ActionFamily.DISENGAGE,
)

GROUND_BOTTOM_ACTIONS = (
    ActionFamily.ESCAPE_STAND,
    ActionFamily.IMPROVE_POSITION,
    ActionFamily.REVERSAL,
    ActionFamily.SUBMISSION_ATTACK,
    ActionFamily.BOTTOM_STRIKE,
)


def legal_actions(state: FightState, actor: Side) -> tuple[ActionFamily, ...]:
    """Return the complete legal menu for ``actor`` in authoritative state."""
    if not isinstance(state, FightState):
        raise ValueError("state must be a FightState")
    if not isinstance(actor, Side):
        raise ValueError("actor must be a Side value")

    if state.phase is Phase.STANDING:
        return STANDING_ACTIONS
    if state.phase is Phase.CLINCH:
        return CLINCH_ACTIONS
    if state.phase is Phase.GROUND:
        return GROUND_TOP_ACTIONS if actor is state.ground_controller else GROUND_BOTTOM_ACTIONS
    raise ValueError(f"unsupported phase: {state.phase!r}")


def validate_action_event(event: ActionEvent, state: FightState) -> None:
    """Fail loudly unless an attempted action is legal in authoritative state."""
    if not isinstance(event, ActionEvent):
        raise ValueError("event must be an ActionEvent")
    if not isinstance(state, FightState):
        raise ValueError("state must be a FightState")
    if event.source_phase is not state.phase:
        raise ValueError("event source_phase does not match authoritative state")
    if event.timestamp_seconds < state.fight_time_seconds:
        raise ValueError("event timestamp cannot precede current fight time")
    if event.action_family not in legal_actions(state, event.actor):
        raise ValueError(
            f"{event.action_family.value} is not legal for {event.actor.value} "
            f"in {state.phase.value}"
        )
