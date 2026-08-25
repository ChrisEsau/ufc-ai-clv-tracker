"""Stage 3 tests for the clean, non-mutating mechanics boundary."""

from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from pipeline.simulation.event_clock_mc_v2.causal.events import ActionEvent, ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import FightState, Phase, Side
from pipeline.simulation.event_clock_mc_v2.causal.timeline import PhaseTimeline
from pipeline.simulation.event_clock_mc_v2.mechanics.config import (
    FighterMechanics,
    MechanicsInputs,
    StructuralMVPPlaceholders,
)
from pipeline.simulation.event_clock_mc_v2.mechanics.resolution import (
    ActionOutcome,
    ActionResolution,
    FightTerminationRequest,
    StrikeConsequence,
    TransitionKind,
    TransitionRequest,
)
from pipeline.simulation.event_clock_mc_v2.mechanics.resolver import resolve_action


SUCCESS_INPUTS = MechanicsInputs(
    red=FighterMechanics(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    blue=FighterMechanics(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
)
FAILURE_INPUTS = MechanicsInputs(
    red=FighterMechanics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    blue=FighterMechanics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)
SUCCESS_PLACEHOLDERS = StructuralMVPPlaceholders(1.0, 1.0, 1.0)
FAILURE_PLACEHOLDERS = StructuralMVPPlaceholders(0.0, 0.0, 0.0)


def _event(family: ActionFamily, phase: Phase, actor: Side = Side.RED) -> ActionEvent:
    return ActionEvent(10.0, actor, family, phase)


def _state(
    phase: Phase,
    *,
    clinch_controller: Side | None = None,
    ground_controller: Side | None = None,
) -> FightState:
    return FightState(
        fight_time_seconds=10.0,
        phase=phase,
        phase_started_at=10.0,
        clinch_controller=clinch_controller,
        ground_controller=ground_controller,
    )


def test_resolution_contract_is_frozen_and_typed() -> None:
    event = _event(ActionFamily.PRESSURE, Phase.STANDING)
    result = ActionResolution(event, ActionOutcome.TACTICAL)

    assert tuple(field.name for field in fields(ActionResolution)) == (
        "event",
        "outcome",
        "transition",
        "consequence",
    )
    with pytest.raises(FrozenInstanceError):
        result.outcome = ActionOutcome.SUCCESS
    with pytest.raises(ValueError, match="typed mechanics consequence"):
        ActionResolution(event, ActionOutcome.TACTICAL, consequence={"damage": 1})


def test_resolver_rejects_illegal_and_source_phase_mismatched_events() -> None:
    with pytest.raises(ValueError, match="is not legal"):
        resolve_action(
            _event(ActionFamily.CLINCH_STRIKE, Phase.STANDING),
            _state(Phase.STANDING),
            SUCCESS_INPUTS,
            np.random.default_rng(1),
        )
    with pytest.raises(ValueError, match="does not match"):
        resolve_action(
            _event(ActionFamily.STAND_ATTACK, Phase.CLINCH),
            _state(Phase.STANDING),
            SUCCESS_INPUTS,
            np.random.default_rng(1),
        )


def test_resolution_does_not_mutate_state_or_timeline() -> None:
    state = _state(Phase.STANDING)
    timeline = PhaseTimeline.from_state(state)
    state_before = state
    segments_before, active_before = timeline.segments, timeline.active

    result = resolve_action(
        _event(ActionFamily.CLINCH_ENTRY, Phase.STANDING),
        state,
        SUCCESS_INPUTS,
        np.random.default_rng(5),
        SUCCESS_PLACEHOLDERS,
    )

    assert result.transition is not None
    assert state == state_before
    assert timeline.segments == segments_before
    assert timeline.active is active_before


def test_same_seed_and_inputs_produce_identical_result() -> None:
    event = _event(ActionFamily.STAND_ATTACK, Phase.STANDING)
    inputs = MechanicsInputs(
        FighterMechanics(0.4, 0.4, 0.4, 0.4, 0.4, 0.4),
        FighterMechanics(0.6, 0.6, 0.6, 0.6, 0.6, 0.6),
    )

    first = resolve_action(event, _state(Phase.STANDING), inputs, np.random.default_rng(42))
    second = resolve_action(event, _state(Phase.STANDING), inputs, np.random.default_rng(42))

    assert first == second


@pytest.mark.parametrize(
    ("family", "state", "actor", "expected"),
    [
        (
            ActionFamily.CLINCH_ENTRY,
            _state(Phase.STANDING),
            Side.RED,
            TransitionRequest(TransitionKind.ENTER_CLINCH, Phase.STANDING, Phase.CLINCH, Side.RED),
        ),
        (
            ActionFamily.TAKEDOWN_ENTRY,
            _state(Phase.STANDING),
            Side.BLUE,
            TransitionRequest(TransitionKind.DIRECT_TAKEDOWN, Phase.STANDING, Phase.GROUND, Side.BLUE),
        ),
        (
            ActionFamily.CLINCH_TAKEDOWN,
            _state(Phase.CLINCH, clinch_controller=Side.RED),
            Side.BLUE,
            TransitionRequest(TransitionKind.CLINCH_TAKEDOWN, Phase.CLINCH, Phase.GROUND, Side.BLUE),
        ),
        (
            ActionFamily.BREAK_CLINCH,
            _state(Phase.CLINCH, clinch_controller=Side.RED),
            Side.BLUE,
            TransitionRequest(TransitionKind.BREAK_CLINCH, Phase.CLINCH, Phase.STANDING),
        ),
        (
            ActionFamily.ESCAPE_STAND,
            _state(Phase.GROUND, ground_controller=Side.RED),
            Side.BLUE,
            TransitionRequest(TransitionKind.ESCAPE_GROUND, Phase.GROUND, Phase.STANDING),
        ),
        (
            ActionFamily.REVERSAL,
            _state(Phase.GROUND, ground_controller=Side.RED),
            Side.BLUE,
            TransitionRequest(TransitionKind.REVERSE_GROUND, Phase.GROUND, Phase.GROUND, Side.BLUE),
        ),
        (
            ActionFamily.DISENGAGE,
            _state(Phase.GROUND, ground_controller=Side.RED),
            Side.RED,
            TransitionRequest(TransitionKind.DISENGAGE_GROUND, Phase.GROUND, Phase.STANDING),
        ),
    ],
)
def test_successful_physical_actions_request_explicit_transitions(
    family: ActionFamily,
    state: FightState,
    actor: Side,
    expected: TransitionRequest,
) -> None:
    result = resolve_action(
        _event(family, state.phase, actor),
        state,
        SUCCESS_INPUTS,
        np.random.default_rng(1),
        SUCCESS_PLACEHOLDERS,
    )

    assert result.transition == expected


@pytest.mark.parametrize(
    ("family", "state", "actor", "expected_outcome"),
    [
        (ActionFamily.CLINCH_ENTRY, _state(Phase.STANDING), Side.RED, ActionOutcome.FAILURE),
        (ActionFamily.TAKEDOWN_ENTRY, _state(Phase.STANDING), Side.RED, ActionOutcome.STUFFED),
        (
            ActionFamily.CLINCH_TAKEDOWN,
            _state(Phase.CLINCH, clinch_controller=Side.RED),
            Side.BLUE,
            ActionOutcome.STUFFED,
        ),
        (
            ActionFamily.BREAK_CLINCH,
            _state(Phase.CLINCH, clinch_controller=Side.RED),
            Side.BLUE,
            ActionOutcome.FAILURE,
        ),
        (
            ActionFamily.ESCAPE_STAND,
            _state(Phase.GROUND, ground_controller=Side.RED),
            Side.BLUE,
            ActionOutcome.FAILURE,
        ),
        (
            ActionFamily.REVERSAL,
            _state(Phase.GROUND, ground_controller=Side.RED),
            Side.BLUE,
            ActionOutcome.FAILURE,
        ),
    ],
)
def test_failed_physical_actions_request_no_transition(
    family: ActionFamily,
    state: FightState,
    actor: Side,
    expected_outcome: ActionOutcome,
) -> None:
    result = resolve_action(
        _event(family, state.phase, actor),
        state,
        FAILURE_INPUTS,
        np.random.default_rng(1),
        FAILURE_PLACEHOLDERS,
    )

    assert result.outcome is expected_outcome
    assert result.transition is None


@pytest.mark.parametrize(
    ("family", "state", "actor"),
    [
        (ActionFamily.STAND_ATTACK, _state(Phase.STANDING), Side.RED),
        (
            ActionFamily.STAND_COUNTER,
            _state(Phase.STANDING),
            Side.BLUE,
        ),
        (
            ActionFamily.CLINCH_STRIKE,
            _state(Phase.CLINCH, clinch_controller=Side.RED),
            Side.BLUE,
        ),
        (
            ActionFamily.GROUND_STRIKE,
            _state(Phase.GROUND, ground_controller=Side.RED),
            Side.RED,
        ),
        (
            ActionFamily.BOTTOM_STRIKE,
            _state(Phase.GROUND, ground_controller=Side.RED),
            Side.BLUE,
        ),
    ],
)
def test_strikes_return_landed_or_missed_without_phase_transition(
    family: ActionFamily, state: FightState, actor: Side
) -> None:
    landed = resolve_action(
        _event(family, state.phase, actor),
        state,
        SUCCESS_INPUTS,
        np.random.default_rng(2),
        SUCCESS_PLACEHOLDERS,
    )
    missed = resolve_action(
        _event(family, state.phase, actor),
        state,
        FAILURE_INPUTS,
        np.random.default_rng(2),
        FAILURE_PLACEHOLDERS,
    )

    assert landed.outcome is ActionOutcome.LANDED
    assert landed.consequence == StrikeConsequence(True)
    assert landed.transition is None
    assert missed.outcome is ActionOutcome.MISSED
    assert missed.consequence == StrikeConsequence(False)
    assert missed.transition is None


def test_submission_success_requests_termination_without_mutating_state() -> None:
    state = _state(Phase.GROUND, ground_controller=Side.RED)
    event = _event(ActionFamily.SUBMISSION_ATTACK, Phase.GROUND, Side.BLUE)
    state_before = state

    succeeded = resolve_action(event, state, SUCCESS_INPUTS, np.random.default_rng(3))
    failed = resolve_action(event, state, FAILURE_INPUTS, np.random.default_rng(3))

    assert succeeded.outcome is ActionOutcome.SUCCESS
    assert succeeded.consequence == FightTerminationRequest(Side.BLUE)
    assert failed.outcome is ActionOutcome.FAILURE
    assert failed.consequence is None
    assert state == state_before
    assert state.finished is False


@pytest.mark.parametrize(
    ("family", "state", "actor", "outcome"),
    [
        (ActionFamily.PRESSURE, _state(Phase.STANDING), Side.RED, ActionOutcome.TACTICAL),
        (ActionFamily.RESET_RANGE, _state(Phase.STANDING), Side.BLUE, ActionOutcome.TACTICAL),
        (
            ActionFamily.CLINCH_CONTROL,
            _state(Phase.CLINCH, clinch_controller=Side.RED),
            Side.BLUE,
            ActionOutcome.CONTROLLED,
        ),
        (
            ActionFamily.CONTROL,
            _state(Phase.GROUND, ground_controller=Side.RED),
            Side.RED,
            ActionOutcome.CONTROLLED,
        ),
        (
            ActionFamily.ADVANCE_POSITION,
            _state(Phase.GROUND, ground_controller=Side.RED),
            Side.RED,
            ActionOutcome.MAINTAINED,
        ),
        (
            ActionFamily.IMPROVE_POSITION,
            _state(Phase.GROUND, ground_controller=Side.RED),
            Side.BLUE,
            ActionOutcome.MAINTAINED,
        ),
    ],
)
def test_mode_actions_have_neutral_non_fabricated_semantics(
    family: ActionFamily, state: FightState, actor: Side, outcome: ActionOutcome
) -> None:
    result = resolve_action(
        _event(family, state.phase, actor), state, SUCCESS_INPUTS, np.random.default_rng(8)
    )

    assert result.outcome is outcome
    assert result.transition is None
    assert result.consequence is None


def test_ground_role_legality_is_defensively_enforced() -> None:
    state = _state(Phase.GROUND, ground_controller=Side.RED)
    with pytest.raises(ValueError, match="is not legal"):
        resolve_action(
            _event(ActionFamily.GROUND_STRIKE, Phase.GROUND, Side.BLUE),
            state,
            SUCCESS_INPUTS,
            np.random.default_rng(1),
        )
    with pytest.raises(ValueError, match="is not legal"):
        resolve_action(
            _event(ActionFamily.REVERSAL, Phase.GROUND, Side.RED),
            state,
            SUCCESS_INPUTS,
            np.random.default_rng(1),
        )


def test_every_action_family_has_a_resolution_path() -> None:
    cases = {
        **{family: (_state(Phase.STANDING), Side.RED) for family in (
            ActionFamily.STAND_ATTACK,
            ActionFamily.STAND_COUNTER,
            ActionFamily.PRESSURE,
            ActionFamily.RESET_RANGE,
            ActionFamily.CLINCH_ENTRY,
            ActionFamily.TAKEDOWN_ENTRY,
        )},
        **{family: (_state(Phase.CLINCH, clinch_controller=Side.RED), Side.BLUE) for family in (
            ActionFamily.CLINCH_STRIKE,
            ActionFamily.CLINCH_CONTROL,
            ActionFamily.CLINCH_TAKEDOWN,
            ActionFamily.BREAK_CLINCH,
        )},
        **{family: (_state(Phase.GROUND, ground_controller=Side.RED), Side.RED) for family in (
            ActionFamily.GROUND_STRIKE,
            ActionFamily.ADVANCE_POSITION,
            ActionFamily.SUBMISSION_ATTACK,
            ActionFamily.CONTROL,
            ActionFamily.DISENGAGE,
        )},
        **{family: (_state(Phase.GROUND, ground_controller=Side.RED), Side.BLUE) for family in (
            ActionFamily.ESCAPE_STAND,
            ActionFamily.IMPROVE_POSITION,
            ActionFamily.REVERSAL,
            ActionFamily.BOTTOM_STRIKE,
        )},
    }

    assert set(cases) == set(ActionFamily)
    for family, (state, actor) in cases.items():
        assert isinstance(
            resolve_action(
                _event(family, state.phase, actor),
                state,
                SUCCESS_INPUTS,
                np.random.default_rng(11),
                SUCCESS_PLACEHOLDERS,
            ),
            ActionResolution,
        )
