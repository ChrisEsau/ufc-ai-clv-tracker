"""Stage 5 tests for the generic legal-action selection policy."""

from dataclasses import FrozenInstanceError, fields, replace
import inspect
import math

import numpy as np
import pytest

from pipeline.simulation.event_clock_mc_v2.brain import policy
from pipeline.simulation.event_clock_mc_v2.brain.capabilities import (
    BrainCapabilities,
    capabilities_from_percentiles,
)
from pipeline.simulation.event_clock_mc_v2.brain.policy import (
    ActionProbability,
    BrainDecisionContext,
    BrainPolicyConfig,
    action_probabilities,
    action_utilities,
    choose_action,
)
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.legality import legal_actions
from pipeline.simulation.event_clock_mc_v2.causal.state import FightState, Phase, Side
from pipeline.simulation.event_clock_mc_v2.causal.timeline import PhaseTimeline
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1 import policy as research


BALANCED = BrainCapabilities(.45, .35, .35, .35, .35, .35, .30, .40, .30)
STRIKER = BrainCapabilities(.85, .70, .65, .20, .10, .35, .30, .40, .30)
WRESTLER = BrainCapabilities(.30, .20, .55, .75, .90, .85, .30, .40, .30)
WEAK_WRESTLER = replace(STRIKER, takedown=0.0, clinch=.05)
NEUTRAL = BrainDecisionContext()
CONFIG = BrainPolicyConfig()


def _state(phase: Phase, controller: Side = Side.RED) -> FightState:
    if phase is Phase.CLINCH:
        return FightState(phase=phase, clinch_controller=controller)
    if phase is Phase.GROUND:
        return FightState(phase=phase, ground_controller=controller)
    return FightState()


def _prob(
    action: ActionFamily,
    capabilities: BrainCapabilities = BALANCED,
    context: BrainDecisionContext = NEUTRAL,
    state: FightState | None = None,
    actor: Side = Side.RED,
) -> float:
    rows = action_probabilities(state or FightState(), actor, capabilities, context, CONFIG)
    return next(row.probability for row in rows if row.action_family is action)


def _utility(
    action: ActionFamily,
    capabilities: BrainCapabilities = BALANCED,
    context: BrainDecisionContext = NEUTRAL,
    state: FightState | None = None,
    actor: Side = Side.RED,
) -> float:
    return dict(action_utilities(state or FightState(), actor, capabilities, context))[action]


def test_capabilities_context_config_and_probability_are_frozen() -> None:
    for record, field_name in (
        (BALANCED, "standing"),
        (NEUTRAL, "own_hurt"),
        (CONFIG, "softmax_temperature"),
        (ActionProbability(ActionFamily.PRESSURE, 0.0, 0.5), "probability"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(record, field_name, 1.0)


@pytest.mark.parametrize("field", [field.name for field in fields(BrainCapabilities)])
@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_capabilities_are_bounded(field: str, value) -> None:
    with pytest.raises(ValueError):
        replace(BALANCED, **{field: value})


@pytest.mark.parametrize("field", [field.name for field in fields(BrainDecisionContext)])
def test_context_fields_reject_non_finite_values(field: str) -> None:
    with pytest.raises(ValueError):
        replace(NEUTRAL, **{field: float("nan")})


@pytest.mark.parametrize("field", ["striking_edge", "score_state"])
@pytest.mark.parametrize("value", [-1.01, 1.01])
def test_signed_context_fields_are_bounded(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        replace(NEUTRAL, **{field: value})


def test_policy_config_is_validated() -> None:
    with pytest.raises(ValueError):
        BrainPolicyConfig(0.0)
    with pytest.raises(ValueError):
        BrainPolicyConfig(float("nan"))


@pytest.mark.parametrize(
    ("state", "actor"),
    [
        (FightState(), Side.RED),
        (_state(Phase.CLINCH), Side.BLUE),
        (_state(Phase.GROUND), Side.RED),
        (_state(Phase.GROUND), Side.BLUE),
    ],
)
def test_distribution_is_typed_finite_normalized_and_exactly_legal(
    state: FightState, actor: Side
) -> None:
    rows = action_probabilities(state, actor, BALANCED, NEUTRAL)

    assert isinstance(rows, tuple)
    assert all(isinstance(row, ActionProbability) for row in rows)
    assert tuple(row.action_family for row in rows) == legal_actions(state, actor)
    assert sum(row.probability for row in rows) == pytest.approx(1.0, abs=1e-12)
    assert all(row.probability >= 0.0 for row in rows)
    assert all(math.isfinite(row.utility) and math.isfinite(row.probability) for row in rows)


def test_choose_action_is_legal_typed_and_seeded_deterministic() -> None:
    state = _state(Phase.GROUND)
    first = choose_action(state, Side.BLUE, BALANCED, NEUTRAL, np.random.default_rng(42))
    second = choose_action(state, Side.BLUE, BALANCED, NEUTRAL, np.random.default_rng(42))

    assert isinstance(first, ActionFamily)
    assert first in legal_actions(state, Side.BLUE)
    assert first is second


def test_policy_is_deterministic_and_does_not_mutate_state_or_timeline() -> None:
    state = FightState()
    timeline = PhaseTimeline.from_state(state)
    state_before = state
    segments_before, active_before = timeline.segments, timeline.active

    first = action_probabilities(state, Side.RED, BALANCED, NEUTRAL)
    second = action_probabilities(state, Side.RED, BALANCED, NEUTRAL)

    assert first == second
    assert state == state_before
    assert timeline.segments == segments_before
    assert timeline.active is active_before


def test_policy_has_no_timing_event_or_mechanics_dependency() -> None:
    source = inspect.getsource(policy)

    assert "ActionEvent" not in source
    assert "expected_action_delay" not in source
    assert "sample_next_action_delay" not in source
    assert "resolve_action" not in source
    assert "standard_fighter_v1" not in source


@pytest.mark.parametrize(
    ("field", "action"),
    [
        ("standing", ActionFamily.STAND_ATTACK),
        ("counter", ActionFamily.STAND_COUNTER),
        ("pressure", ActionFamily.PRESSURE),
        ("clinch", ActionFamily.CLINCH_ENTRY),
        ("takedown", ActionFamily.TAKEDOWN_ENTRY),
    ],
)
def test_standing_capabilities_raise_matching_action_utility(
    field: str, action: ActionFamily
) -> None:
    low, high = replace(BALANCED, **{field: 0.0}), replace(BALANCED, **{field: 1.0})
    assert _utility(action, high) > _utility(action, low)


def test_standing_tactical_directionality() -> None:
    losing = BrainDecisionContext(striking_edge=-.85)
    winning = BrainDecisionContext(striking_edge=.85)
    own_hurt = BrainDecisionContext(own_hurt=.9)
    opponent_hurt = BrainDecisionContext(opponent_hurt=.9)
    td_success = BrainDecisionContext(td_success_recent=.9)
    td_failure = BrainDecisionContext(td_failure_recent=.9)
    tired = BrainDecisionContext(fatigue=1.0)
    behind = BrainDecisionContext(score_state=-1.0, late_urgency=1.0)
    ahead = BrainDecisionContext(score_state=1.0, late_urgency=1.0)

    assert _prob(ActionFamily.STAND_ATTACK, STRIKER) > _prob(ActionFamily.STAND_ATTACK, BALANCED)
    assert _prob(ActionFamily.RESET_RANGE, context=losing) > _prob(ActionFamily.RESET_RANGE)
    assert _prob(ActionFamily.TAKEDOWN_ENTRY, WRESTLER, losing) > _prob(ActionFamily.TAKEDOWN_ENTRY, WRESTLER)
    assert _prob(ActionFamily.TAKEDOWN_ENTRY, WRESTLER, losing) > _prob(ActionFamily.TAKEDOWN_ENTRY, WEAK_WRESTLER, losing)
    assert _prob(ActionFamily.RESET_RANGE, context=own_hurt) > _prob(ActionFamily.RESET_RANGE)
    assert _prob(ActionFamily.STAND_ATTACK, context=opponent_hurt) > _prob(ActionFamily.STAND_ATTACK)
    assert _prob(ActionFamily.PRESSURE, context=opponent_hurt) > _prob(ActionFamily.PRESSURE)
    assert _prob(ActionFamily.TAKEDOWN_ENTRY, context=td_success) > _prob(ActionFamily.TAKEDOWN_ENTRY)
    assert _prob(ActionFamily.TAKEDOWN_ENTRY, context=td_failure) < _prob(ActionFamily.TAKEDOWN_ENTRY)
    assert _utility(ActionFamily.PRESSURE, context=tired) < _utility(ActionFamily.PRESSURE)
    assert _utility(ActionFamily.TAKEDOWN_ENTRY, context=tired) < _utility(ActionFamily.TAKEDOWN_ENTRY)
    assert _prob(ActionFamily.STAND_ATTACK, context=behind) > _prob(ActionFamily.STAND_ATTACK)
    assert _prob(ActionFamily.RESET_RANGE, context=ahead) > _prob(ActionFamily.RESET_RANGE)
    assert _prob(ActionFamily.STAND_ATTACK, context=winning) > _prob(ActionFamily.STAND_ATTACK)


def test_cage_context_directionality() -> None:
    own_cage = BrainDecisionContext(own_back_to_cage=1.0)
    opponent_cage = BrainDecisionContext(opponent_back_to_cage=1.0)

    assert _prob(ActionFamily.RESET_RANGE, context=own_cage) > _prob(ActionFamily.RESET_RANGE)
    assert _prob(ActionFamily.CLINCH_ENTRY, WRESTLER, opponent_cage) > _prob(
        ActionFamily.CLINCH_ENTRY, WRESTLER
    )


def test_same_state_context_different_capability_produces_heterogeneous_policy() -> None:
    context = BrainDecisionContext(striking_edge=-.9, td_failure_recent=.8)
    strong = action_probabilities(FightState(), Side.RED, WRESTLER, context)
    weak = action_probabilities(FightState(), Side.RED, WEAK_WRESTLER, context)
    strong_map = {row.action_family: row.probability for row in strong}
    weak_map = {row.action_family: row.probability for row in weak}

    assert strong != weak
    assert strong_map[ActionFamily.TAKEDOWN_ENTRY] > weak_map[ActionFamily.TAKEDOWN_ENTRY]
    assert strong_map[ActionFamily.CLINCH_ENTRY] > weak_map[ActionFamily.CLINCH_ENTRY]
    assert weak_map[ActionFamily.RESET_RANGE] > strong_map[ActionFamily.RESET_RANGE]
    assert not hasattr(BRAIN_CAPABILITY_TYPE := BrainCapabilities, "archetype")
    assert BRAIN_CAPABILITY_TYPE is BrainCapabilities


def test_clinch_behavior_and_menu() -> None:
    state = _state(Phase.CLINCH)
    strong = _prob(ActionFamily.CLINCH_TAKEDOWN, WRESTLER, state=state, actor=Side.BLUE)
    weak = _prob(ActionFamily.CLINCH_TAKEDOWN, WEAK_WRESTLER, state=state, actor=Side.BLUE)
    urgent = BrainDecisionContext(score_state=-1.0, late_urgency=1.0)
    hurt = BrainDecisionContext(own_hurt=1.0)

    assert tuple(row.action_family for row in action_probabilities(state, Side.BLUE, BALANCED, NEUTRAL)) == legal_actions(state, Side.BLUE)
    assert strong > weak
    assert _prob(ActionFamily.CLINCH_TAKEDOWN, WRESTLER, urgent, state, Side.BLUE) > strong
    assert _prob(ActionFamily.CLINCH_CONTROL, BALANCED, hurt, state, Side.BLUE) > _prob(
        ActionFamily.CLINCH_CONTROL, BALANCED, NEUTRAL, state, Side.BLUE
    )


def test_ground_top_behavior() -> None:
    state = _state(Phase.GROUND)
    weak_top = replace(BALANCED, ground_top=0.0)
    strong_top = replace(BALANCED, ground_top=1.0)
    high_sub = replace(BALANCED, submission=1.0)
    dominant = BrainDecisionContext(dominant_top_position=1.0)
    behind = BrainDecisionContext(score_state=-1.0, late_urgency=1.0)

    assert tuple(row.action_family for row in action_probabilities(state, Side.RED, BALANCED, NEUTRAL)) == legal_actions(state, Side.RED)
    for action in (ActionFamily.GROUND_STRIKE, ActionFamily.ADVANCE_POSITION, ActionFamily.CONTROL):
        assert _utility(action, strong_top, state=state) > _utility(action, weak_top, state=state)
    assert _prob(ActionFamily.SUBMISSION_ATTACK, high_sub, state=state) > _prob(
        ActionFamily.SUBMISSION_ATTACK, BALANCED, state=state
    )
    assert _prob(ActionFamily.DISENGAGE, context=dominant, state=state) < _prob(
        ActionFamily.DISENGAGE, state=state
    )
    assert _prob(ActionFamily.SUBMISSION_ATTACK, context=behind, state=state) > _prob(
        ActionFamily.SUBMISSION_ATTACK, state=state
    )


def test_ground_bottom_behavior_and_no_top_actions() -> None:
    state = _state(Phase.GROUND)
    high_escape = replace(BALANCED, escape=1.0)
    high_reversal = replace(BALANCED, reversal=1.0)
    high_sub = replace(BALANCED, submission=1.0)
    danger = BrainDecisionContext(bad_bottom_position=1.0)

    rows = action_probabilities(state, Side.BLUE, BALANCED, NEUTRAL)
    assert tuple(row.action_family for row in rows) == legal_actions(state, Side.BLUE)
    assert ActionFamily.CONTROL not in {row.action_family for row in rows}
    assert _prob(ActionFamily.ESCAPE_STAND, high_escape, state=state, actor=Side.BLUE) > _prob(ActionFamily.ESCAPE_STAND, state=state, actor=Side.BLUE)
    assert _prob(ActionFamily.REVERSAL, high_reversal, state=state, actor=Side.BLUE) > _prob(ActionFamily.REVERSAL, state=state, actor=Side.BLUE)
    assert _prob(ActionFamily.SUBMISSION_ATTACK, high_sub, state=state, actor=Side.BLUE) > _prob(ActionFamily.SUBMISSION_ATTACK, state=state, actor=Side.BLUE)
    assert _utility(ActionFamily.IMPROVE_POSITION, context=danger, state=state, actor=Side.BLUE) > _utility(ActionFamily.IMPROVE_POSITION, state=state, actor=Side.BLUE)


def test_capability_percentile_translation_preserves_formal_semantics() -> None:
    cap = capabilities_from_percentiles(
        standing_rate_percentile=.8,
        standing_accuracy_percentile=.6,
        takedown_rate_percentile=.2,
        takedown_completion_percentile=.4,
        ground_rate_percentile=.9,
        ground_accuracy_percentile=.7,
    )

    assert {
        field.name: getattr(cap, field.name) for field in fields(cap)
    } == pytest.approx({
        "standing": .7,
        "counter": .6,
        "pressure": .8,
        "clinch": .35,
        "takedown": .3,
        "ground_top": .8,
        "submission": .30,
        "escape": .40,
        "reversal": .30,
    })


@pytest.mark.parametrize(
    ("phase", "actor", "research_phase"),
    [
        (Phase.STANDING, Side.RED, research.Phase.STANDING),
        (Phase.CLINCH, Side.RED, research.Phase.CLINCH),
        (Phase.GROUND, Side.RED, research.Phase.GROUND_TOP),
        (Phase.GROUND, Side.BLUE, research.Phase.GROUND_BOTTOM),
    ],
)
def test_neutral_policy_migration_matches_research_exactly(
    phase: Phase, actor: Side, research_phase: research.Phase
) -> None:
    state = _state(phase)
    new = action_probabilities(state, actor, BALANCED, NEUTRAL)
    old_cap = research.Capability(**{field.name: getattr(BALANCED, field.name) for field in fields(BALANCED)})
    old = research.action_probabilities(research.FightState(phase=research_phase), old_cap)

    assert {row.action_family.value: row.probability for row in new} == pytest.approx(
        {action.value: probability for action, probability in old.items()}, abs=1e-15
    )


def test_real_fsr_regression_fixture_has_distinct_generic_behavior() -> None:
    rakic = BrainCapabilities(.789, .789, .789, .35, .065, .662, .30, .40, .30)
    tybura = BrainCapabilities(.203, .203, .203, .35, .195, .958, .30, .40, .30)
    context = BrainDecisionContext(striking_edge=-.85)

    assert _prob(ActionFamily.STAND_ATTACK, rakic, context) > _prob(
        ActionFamily.STAND_ATTACK, tybura, context
    )
    assert _prob(ActionFamily.TAKEDOWN_ENTRY, tybura, context) > _prob(
        ActionFamily.TAKEDOWN_ENTRY, rakic, context
    )
