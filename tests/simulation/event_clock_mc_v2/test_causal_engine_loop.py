"""Stage 6 structural tests for the authoritative brain-driven engine loop."""

from dataclasses import fields
import inspect

import numpy as np
import pytest

from pipeline.simulation.event_clock_mc_v2.brain.capabilities import BrainCapabilities
from pipeline.simulation.event_clock_mc_v2.brain.policy import BrainDecisionContext
from pipeline.simulation.event_clock_mc_v2.brain.timing import BrainTimingContext
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionEvent, ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.legality import legal_actions
from pipeline.simulation.event_clock_mc_v2.causal.state import FightState, Phase, Side
from pipeline.simulation.event_clock_mc_v2.engine import causal_engine
from pipeline.simulation.event_clock_mc_v2.engine.causal_engine import (
    EngineConfig,
    EngineFunctions,
    EngineInputs,
    EngineRNGs,
    FighterEngineInputs,
    PendingAction,
    initialize_pending_actions,
    run_causal_path,
)
from pipeline.simulation.event_clock_mc_v2.mechanics.config import (
    FighterMechanics,
    StructuralMVPPlaceholders,
)
from pipeline.simulation.event_clock_mc_v2.mechanics.resolution import (
    ActionOutcome,
    FinishMethod,
    TransitionKind,
)
from pipeline.simulation.event_clock_mc_v2.mechanics.resolver import resolve_action


RED_TIMING = BrainTimingContext(own_fatigue=.1)
BLUE_TIMING = BrainTimingContext(own_fatigue=.2)
CAPABILITIES = BrainCapabilities(.5, .5, .5, .5, .5, .5, .5, .5, .5)
DECISION = BrainDecisionContext()
SUCCESS_MECHANICS = FighterMechanics(1, 1, 1, 1, 1, 1)
FAILURE_MECHANICS = FighterMechanics(0, 0, 0, 0, 0, 0)
SUCCESS_PLACEHOLDERS = StructuralMVPPlaceholders(1, 1, 1)
FAILURE_PLACEHOLDERS = StructuralMVPPlaceholders(0, 0, 0)


def _inputs(
    *, mechanics: FighterMechanics = SUCCESS_MECHANICS,
    placeholders: StructuralMVPPlaceholders = SUCCESS_PLACEHOLDERS,
) -> EngineInputs:
    return EngineInputs(
        FighterEngineInputs(CAPABILITIES, RED_TIMING, DECISION, mechanics),
        FighterEngineInputs(CAPABILITIES, BLUE_TIMING, DECISION, mechanics),
        mechanics_placeholders=placeholders,
    )


class ScriptTiming:
    def __init__(self, red: list[float], blue: list[float]) -> None:
        self.remaining = {Side.RED: list(red), Side.BLUE: list(blue)}
        self.calls: list[tuple[Side, Phase, float]] = []

    def __call__(self, state, context, rng, config) -> float:
        side = Side.RED if context is RED_TIMING else Side.BLUE
        self.calls.append((side, state.phase, state.fight_time_seconds))
        return self.remaining[side].pop(0)


class ScriptChooser:
    def __init__(self, red: list[ActionFamily], blue: list[ActionFamily]) -> None:
        self.remaining = {Side.RED: list(red), Side.BLUE: list(blue)}
        self.calls: list[tuple[Side, Phase, float, ActionFamily]] = []

    def __call__(self, state, actor, capabilities, context, rng, config) -> ActionFamily:
        action = self.remaining[actor].pop(0)
        assert action in legal_actions(state, actor)
        self.calls.append((actor, state.phase, state.fight_time_seconds, action))
        return action


def _functions(timing: ScriptTiming, chooser: ScriptChooser, resolver=resolve_action):
    return EngineFunctions(timing, chooser, resolver)


def test_pending_action_contains_actor_and_time_only_and_initializes_one_per_side() -> None:
    timing = ScriptTiming([1.0], [2.0])
    pending = initialize_pending_actions(
        FightState(), _inputs(), EngineRNGs.from_seed(1), _functions(timing, ScriptChooser([], []))
    )

    assert tuple(field.name for field in fields(PendingAction)) == (
        "actor",
        "scheduled_time_seconds",
    )
    assert pending == (PendingAction(Side.RED, 1.0), PendingAction(Side.BLUE, 2.0))
    assert timing.calls == [
        (Side.RED, Phase.STANDING, 0.0),
        (Side.BLUE, Phase.STANDING, 0.0),
    ]


def test_earliest_actor_executes_at_authoritative_time_and_exact_event_reaches_mechanics() -> None:
    timing = ScriptTiming([2.0, 10.0], [1.0, 10.0])
    chooser = ScriptChooser([], [ActionFamily.PRESSURE])
    received: list[ActionEvent] = []

    def spy(event, state, inputs, rng, placeholders):
        received.append(event)
        assert event.timestamp_seconds == state.fight_time_seconds
        assert event.source_phase is state.phase
        return resolve_action(event, state, inputs, rng, placeholders)

    result = run_causal_path(
        _inputs(),
        seed=2,
        horizon_seconds=1.5,
        functions=_functions(timing, chooser, spy),
    )

    assert result.events[0].actor is Side.BLUE
    assert received == [ActionEvent(1.0, Side.BLUE, ActionFamily.PRESSURE, Phase.STANDING)]
    assert chooser.calls == [(Side.BLUE, Phase.STANDING, 1.0, ActionFamily.PRESSURE)]


def test_no_material_change_reschedules_actor_and_preserves_opponent_pending() -> None:
    timing = ScriptTiming([1.0, 5.0], [3.0])
    result = run_causal_path(
        _inputs(),
        seed=3,
        horizon_seconds=1.5,
        functions=_functions(timing, ScriptChooser([ActionFamily.PRESSURE], [])),
    )

    assert result.final_pending_actions == (
        PendingAction(Side.BLUE, 3.0),
        PendingAction(Side.RED, 6.0),
    )
    assert timing.calls == [
        (Side.RED, Phase.STANDING, 0.0),
        (Side.BLUE, Phase.STANDING, 0.0),
        (Side.RED, Phase.STANDING, 1.0),
    ]
    assert result.final_state.phase is Phase.STANDING
    assert len(result.timeline_segments) == 1


@pytest.mark.parametrize(
    ("action", "initial_state", "actor", "expected_phase", "expected_controller"),
    [
        (ActionFamily.CLINCH_ENTRY, FightState(), Side.RED, Phase.CLINCH, Side.RED),
        (ActionFamily.TAKEDOWN_ENTRY, FightState(), Side.BLUE, Phase.GROUND, Side.BLUE),
        (
            ActionFamily.ESCAPE_STAND,
            FightState(phase=Phase.GROUND, ground_controller=Side.RED),
            Side.BLUE,
            Phase.STANDING,
            None,
        ),
        (
            ActionFamily.REVERSAL,
            FightState(phase=Phase.GROUND, ground_controller=Side.RED),
            Side.BLUE,
            Phase.GROUND,
            Side.BLUE,
        ),
    ],
)
def test_material_transition_updates_timeline_and_resamples_both(
    action, initial_state, actor, expected_phase, expected_controller
) -> None:
    red_first = actor is Side.RED
    timing = ScriptTiming(
        [1.0, 10.0] if red_first else [2.0, 10.0],
        [2.0, 10.0] if red_first else [1.0, 10.0],
    )
    chooser = ScriptChooser([action] if red_first else [], [] if red_first else [action])
    result = run_causal_path(
        _inputs(),
        seed=4,
        horizon_seconds=1.1,
        initial_state=initial_state,
        functions=_functions(timing, chooser),
    )

    assert result.final_state.phase is expected_phase
    assert result.events[0].resulting_controller is expected_controller
    assert result.timeline_segments[0].end_time == 1.0
    assert result.timeline_segments[1].start_time == 1.0
    assert timing.calls[-2:] == [
        (Side.RED, expected_phase, 1.0),
        (Side.BLUE, expected_phase, 1.0),
    ]


@pytest.mark.parametrize(
    ("action", "initial_state", "actor"),
    [
        (ActionFamily.CLINCH_ENTRY, FightState(), Side.RED),
        (ActionFamily.TAKEDOWN_ENTRY, FightState(), Side.RED),
        (
            ActionFamily.ESCAPE_STAND,
            FightState(phase=Phase.GROUND, ground_controller=Side.RED),
            Side.BLUE,
        ),
    ],
)
def test_failed_transition_preserves_phase_and_reschedules_only_actor(
    action, initial_state, actor
) -> None:
    timing = ScriptTiming([1.0, 5.0], [3.0]) if actor is Side.RED else ScriptTiming([3.0], [1.0, 5.0])
    chooser = ScriptChooser([action], []) if actor is Side.RED else ScriptChooser([], [action])
    result = run_causal_path(
        _inputs(mechanics=FAILURE_MECHANICS, placeholders=FAILURE_PLACEHOLDERS),
        seed=5,
        horizon_seconds=1.1,
        initial_state=initial_state,
        functions=_functions(timing, chooser),
    )

    assert result.events[0].transition_kind is None
    assert result.final_state.phase is initial_state.phase
    assert len(timing.calls) == 3
    assert timing.calls[-1][0] is actor


def test_successful_submission_terminates_immediately_without_resampling() -> None:
    initial = FightState(phase=Phase.GROUND, ground_controller=Side.RED)
    timing = ScriptTiming([1.0], [2.0])
    result = run_causal_path(
        _inputs(),
        seed=6,
        horizon_seconds=10.0,
        initial_state=initial,
        functions=_functions(timing, ScriptChooser([ActionFamily.SUBMISSION_ATTACK], [])),
    )

    assert len(result.events) == 1
    assert result.final_state.finished is True
    assert result.final_state.winner is Side.RED
    assert result.final_state.finish_method == FinishMethod.SUBMISSION.value
    assert result.termination.winner is Side.RED
    assert result.termination.finish_method is FinishMethod.SUBMISSION
    assert result.final_pending_actions == ()
    assert len(timing.calls) == 2
    assert result.reported_through_seconds == 1.0


def test_round_boundary_wins_exact_tie_resets_state_and_resamples_both() -> None:
    initial = FightState(phase=Phase.GROUND, ground_controller=Side.RED)
    timing = ScriptTiming([300.0, 10.0], [300.0, 10.0])
    chooser = ScriptChooser([], [])
    result = run_causal_path(
        _inputs(),
        seed=7,
        horizon_seconds=301.0,
        initial_state=initial,
        config=EngineConfig(round_length_seconds=300.0, number_of_rounds=3),
        functions=_functions(timing, chooser),
    )

    assert result.events == ()
    assert result.round_boundaries[0].timestamp_seconds == 300.0
    assert result.final_state.round_number == 2
    assert result.final_state.phase is Phase.STANDING
    assert result.final_state.fight_time_seconds == 300.0
    assert len(timing.calls) == 4
    assert result.timeline_segments[0].end_time == 300.0
    assert result.timeline_segments[1].start_time == 300.0
    assert sum(segment.duration for segment in result.timeline_segments) == 301.0


def test_horizon_does_not_execute_or_advance_state_to_future_pending_action() -> None:
    timing = ScriptTiming([10.0], [12.0])
    result = run_causal_path(
        _inputs(),
        seed=8,
        horizon_seconds=5.0,
        functions=_functions(timing, ScriptChooser([], [])),
    )

    assert result.events == ()
    assert result.final_state.fight_time_seconds == 0.0
    assert result.reported_through_seconds == 5.0
    assert result.timeline_segments[0].start_time == 0.0
    assert result.timeline_segments[0].end_time == 5.0


def test_exact_pending_tie_is_red_before_blue() -> None:
    result = run_causal_path(
        _inputs(),
        seed=9,
        horizon_seconds=1.0,
        functions=_functions(
            ScriptTiming([1.0, 10.0], [1.0]),
            ScriptChooser([ActionFamily.PRESSURE], []),
        ),
    )

    assert [event.actor for event in result.events] == [Side.RED]


def test_same_seed_reproduces_path_and_different_seed_changes_it() -> None:
    first = run_causal_path(_inputs(), seed=101, horizon_seconds=12.0)
    repeated = run_causal_path(_inputs(), seed=101, horizon_seconds=12.0)
    different = run_causal_path(_inputs(), seed=102, horizon_seconds=12.0)

    assert first == repeated
    assert first.events != different.events


def test_scripted_causal_path_transitions_and_role_legality_are_exact() -> None:
    timing = ScriptTiming(
        [1.0, 1.0, 10.0, 1.0, 10.0],
        [10.0, 10.0, 1.0, 10.0, 10.0],
    )
    chooser = ScriptChooser(
        [ActionFamily.CLINCH_ENTRY, ActionFamily.CLINCH_TAKEDOWN, ActionFamily.ESCAPE_STAND],
        [ActionFamily.REVERSAL],
    )
    result = run_causal_path(
        _inputs(),
        seed=10,
        horizon_seconds=5.0,
        functions=_functions(timing, chooser),
    )

    assert [
        (
            event.timestamp_seconds,
            event.actor,
            event.source_phase,
            event.selected_action,
            event.transition_kind,
            event.resulting_phase,
            event.resulting_controller,
        )
        for event in result.events
    ] == [
        (1.0, Side.RED, Phase.STANDING, ActionFamily.CLINCH_ENTRY, TransitionKind.ENTER_CLINCH, Phase.CLINCH, Side.RED),
        (2.0, Side.RED, Phase.CLINCH, ActionFamily.CLINCH_TAKEDOWN, TransitionKind.CLINCH_TAKEDOWN, Phase.GROUND, Side.RED),
        (3.0, Side.BLUE, Phase.GROUND, ActionFamily.REVERSAL, TransitionKind.REVERSE_GROUND, Phase.GROUND, Side.BLUE),
        (4.0, Side.RED, Phase.GROUND, ActionFamily.ESCAPE_STAND, TransitionKind.ESCAPE_GROUND, Phase.STANDING, None),
    ]
    assert [segment.start_time for segment in result.timeline_segments] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [segment.end_time for segment in result.timeline_segments] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert all(result.events[index].timestamp_seconds < result.events[index + 1].timestamp_seconds for index in range(3))
    assert all(event.selected_action in legal_actions(_state_before(event), event.actor) for event in result.events)
    assert not any(
        event.source_phase is Phase.GROUND
        and event.selected_action in {ActionFamily.STAND_ATTACK, ActionFamily.STAND_COUNTER}
        for event in result.events
    )


def _state_before(event) -> FightState:
    if event.source_phase is Phase.CLINCH:
        return FightState(
            fight_time_seconds=event.timestamp_seconds,
            phase=Phase.CLINCH,
            phase_started_at=event.timestamp_seconds,
            clinch_controller=Side.RED,
        )
    if event.source_phase is Phase.GROUND:
        controller = Side.RED if event.timestamp_seconds == 3.0 else Side.BLUE
        return FightState(
            fight_time_seconds=event.timestamp_seconds,
            phase=Phase.GROUND,
            phase_started_at=event.timestamp_seconds,
            ground_controller=controller,
        )
    return FightState(fight_time_seconds=event.timestamp_seconds)


def test_engine_owns_orchestration_without_legacy_runtime_dependencies() -> None:
    source = inspect.getsource(causal_engine)

    assert "event_mc_v1" not in source
    assert "event_clock_mc_v1" not in source
    assert "scheduler" not in source.lower()
    assert "ActionRate" not in source
    assert "pending_action" in source.lower()
