"""Explicit Stage 1 phase transitions for the causal Event Clock V2."""

from __future__ import annotations

from dataclasses import replace

from .state import FightState, Phase, Side
from .timeline import PhaseTimeline


def enter_clinch(
    state: FightState, timeline: PhaseTimeline, timestamp: float, controller: Side
) -> FightState:
    _require_phase(state, Phase.STANDING, "enter clinch")
    return _apply_transition(
        state, timeline, timestamp, Phase.CLINCH, controller, "clinch_entry", "clinch_entry"
    )


def direct_takedown(
    state: FightState, timeline: PhaseTimeline, timestamp: float, controller: Side
) -> FightState:
    _require_phase(state, Phase.STANDING, "complete a direct takedown")
    return _apply_transition(
        state, timeline, timestamp, Phase.GROUND, controller, "direct_takedown", "direct_takedown"
    )


def separate_clinch(state: FightState, timeline: PhaseTimeline, timestamp: float) -> FightState:
    _require_phase(state, Phase.CLINCH, "separate from clinch")
    return _apply_transition(
        state, timeline, timestamp, Phase.STANDING, None, "clinch_separation", "clinch_separation"
    )


def clinch_takedown(
    state: FightState, timeline: PhaseTimeline, timestamp: float, controller: Side
) -> FightState:
    _require_phase(state, Phase.CLINCH, "complete a clinch takedown")
    return _apply_transition(
        state, timeline, timestamp, Phase.GROUND, controller, "clinch_takedown", "clinch_takedown"
    )


def escape_ground(state: FightState, timeline: PhaseTimeline, timestamp: float) -> FightState:
    _require_phase(state, Phase.GROUND, "escape ground")
    return _apply_transition(
        state, timeline, timestamp, Phase.STANDING, None, "ground_escape", "ground_escape"
    )


def reverse_ground(
    state: FightState, timeline: PhaseTimeline, timestamp: float, new_controller: Side
) -> FightState:
    _require_phase(state, Phase.GROUND, "reverse ground control")
    if new_controller is state.ground_controller:
        raise ValueError("ground reversal must change the controller")
    return _apply_transition(
        state, timeline, timestamp, Phase.GROUND, new_controller, "ground_reversal", "ground_reversal"
    )


def start_next_round(
    state: FightState, timeline: PhaseTimeline, timestamp: float
) -> FightState:
    """Close the prior round and reset physical state at the exact boundary."""
    return _apply_transition(
        state,
        timeline,
        timestamp,
        Phase.STANDING,
        None,
        "round_end",
        "round_start",
        round_number=state.round_number + 1,
    )


def close_at_horizon(
    state: FightState,
    timeline: PhaseTimeline,
    timestamp: float,
    *,
    exit_reason: str = "validation_horizon",
) -> FightState:
    """Advance state and close its active segment without opening a successor."""
    if timestamp < state.fight_time_seconds:
        raise ValueError("time cannot move backward")
    if timeline.active is None:
        raise ValueError("timeline has no active phase")
    if (
        timeline.active.phase is not state.phase
        or timeline.active.start_time != state.phase_started_at
        or timeline.active.controller != _state_controller(state)
    ):
        raise ValueError("fight state and active timeline phase are inconsistent")
    timeline.close(timestamp=timestamp, exit_reason=exit_reason)
    return replace(state, fight_time_seconds=timestamp)


def _apply_transition(
    state: FightState,
    timeline: PhaseTimeline,
    timestamp: float,
    phase: Phase,
    controller: Side | None,
    exit_reason: str,
    entry_reason: str,
    *,
    round_number: int | None = None,
) -> FightState:
    if timestamp < state.fight_time_seconds:
        raise ValueError("time cannot move backward")
    if timeline.active is None:
        raise ValueError("timeline has no active phase")
    if (
        timeline.active.phase is not state.phase
        or timeline.active.start_time != state.phase_started_at
        or timeline.active.controller != _state_controller(state)
    ):
        raise ValueError("fight state and active timeline phase are inconsistent")

    clinch_controller = controller if phase is Phase.CLINCH else None
    ground_controller = controller if phase is Phase.GROUND else None
    next_state = replace(
        state,
        fight_time_seconds=timestamp,
        round_number=state.round_number if round_number is None else round_number,
        phase=phase,
        phase_started_at=timestamp,
        clinch_controller=clinch_controller,
        ground_controller=ground_controller,
    )
    timeline.transition(
        timestamp=timestamp,
        phase=phase,
        controller=controller,
        exit_reason=exit_reason,
        entry_reason=entry_reason,
    )
    return next_state


def _require_phase(state: FightState, expected: Phase, operation: str) -> None:
    if state.phase is not expected:
        raise ValueError(f"cannot {operation} from {state.phase.value}")


def _state_controller(state: FightState) -> Side | None:
    if state.phase is Phase.CLINCH:
        return state.clinch_controller
    if state.phase is Phase.GROUND:
        return state.ground_controller
    return None
