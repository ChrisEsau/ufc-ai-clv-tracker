"""Stage 4 tests for the isolated one-clock brain timing model."""

from dataclasses import FrozenInstanceError, replace
import inspect
import math

import numpy as np
import pytest

from pipeline.simulation.event_clock_mc_v2.brain import timing
from pipeline.simulation.event_clock_mc_v2.brain.timing import (
    BrainTimingConfig,
    BrainTimingContext,
    expected_action_delay,
    sample_next_action_delay,
)
from pipeline.simulation.event_clock_mc_v2.causal.state import FightState, Phase, Side
from pipeline.simulation.event_clock_mc_v2.causal.timeline import PhaseTimeline


CONFIG = BrainTimingConfig()
NEUTRAL = BrainTimingContext()


def _state(phase: Phase) -> FightState:
    controllers = {
        Phase.STANDING: {},
        Phase.CLINCH: {"clinch_controller": Side.RED},
        Phase.GROUND: {"ground_controller": Side.RED},
    }
    return FightState(phase=phase, **controllers[phase])


def test_context_and_config_are_frozen_typed_records() -> None:
    with pytest.raises(FrozenInstanceError):
        NEUTRAL.own_hurt = 1.0
    with pytest.raises(FrozenInstanceError):
        CONFIG.gamma_shape = 3.0
    with pytest.raises(ValueError, match="context must"):
        expected_action_delay(FightState(), object())
    with pytest.raises(ValueError, match="config must"):
        expected_action_delay(FightState(), NEUTRAL, object())


@pytest.mark.parametrize("field", ["own_fatigue", "own_hurt", "opponent_hurt", "late_urgency"])
@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_normalized_context_values_are_validated(field: str, value) -> None:
    with pytest.raises(ValueError):
        BrainTimingContext(**{field: value})


@pytest.mark.parametrize("value", [-1.01, 1.01, float("nan"), float("inf"), True])
def test_score_state_is_bounded(value) -> None:
    with pytest.raises(ValueError):
        BrainTimingContext(score_state=value)


@pytest.mark.parametrize(
    "config",
    [
        BrainTimingConfig(base_mean_delay_seconds=4.0),
        replace(CONFIG, gamma_shape=0.01),
        replace(CONFIG, minimum_delay_seconds=1.0, maximum_delay_seconds=1.0),
    ],
)
def test_valid_configurations_are_accepted(config: BrainTimingConfig) -> None:
    assert isinstance(config, BrainTimingConfig)


@pytest.mark.parametrize(
    "changes",
    [
        {"base_mean_delay_seconds": 0.0},
        {"gamma_shape": 0.0},
        {"standing_phase_factor": -1.0},
        {"maximum_fatigue_slowdown": 1.0},
        {"maximum_own_hurt_speedup": -0.1},
        {"minimum_delay_seconds": 2.0, "maximum_delay_seconds": 1.0},
    ],
)
def test_invalid_configurations_are_rejected(changes) -> None:
    with pytest.raises(ValueError):
        replace(CONFIG, **changes)


def test_sample_is_finite_positive_and_structurally_bounded() -> None:
    delay = sample_next_action_delay(
        FightState(), NEUTRAL, np.random.default_rng(20260825), CONFIG
    )

    assert isinstance(delay, float)
    assert math.isfinite(delay)
    assert CONFIG.minimum_delay_seconds <= delay <= CONFIG.maximum_delay_seconds


def test_timing_does_not_mutate_authoritative_state_or_timeline() -> None:
    state = FightState()
    timeline = PhaseTimeline.from_state(state)
    state_before = state
    segments_before, active_before = timeline.segments, timeline.active

    sample_next_action_delay(state, NEUTRAL, np.random.default_rng(3), CONFIG)

    assert state == state_before
    assert timeline.segments == segments_before
    assert timeline.active is active_before


def test_timing_module_knows_when_not_what() -> None:
    source = inspect.getsource(timing)

    assert "ActionFamily" not in source
    assert "ActionEvent" not in source
    assert "legal_actions" not in source
    assert "standard_fighter_v1" not in source
    assert "choose_action" not in source


def test_seeded_sampling_and_expected_delay_are_deterministic() -> None:
    state = FightState()
    first = sample_next_action_delay(state, NEUTRAL, np.random.default_rng(42), CONFIG)
    second = sample_next_action_delay(state, NEUTRAL, np.random.default_rng(42), CONFIG)

    assert first == second
    assert expected_action_delay(state, NEUTRAL, CONFIG) == expected_action_delay(
        state, NEUTRAL, CONFIG
    )


def test_expected_delay_directional_contract() -> None:
    state = FightState()
    neutral = expected_action_delay(state, NEUTRAL, CONFIG)

    assert expected_action_delay(state, BrainTimingContext(own_fatigue=1.0), CONFIG) > neutral
    assert expected_action_delay(state, BrainTimingContext(opponent_hurt=1.0), CONFIG) < neutral
    assert expected_action_delay(state, BrainTimingContext(own_hurt=1.0), CONFIG) < neutral
    assert expected_action_delay(
        state, BrainTimingContext(score_state=-1.0, late_urgency=1.0), CONFIG
    ) < neutral
    assert expected_action_delay(
        state, BrainTimingContext(score_state=1.0, late_urgency=1.0), CONFIG
    ) > neutral


def test_stacked_modifiers_remain_finite_and_bounded() -> None:
    contexts = (
        BrainTimingContext(opponent_hurt=1.0, score_state=-1.0, late_urgency=1.0),
        BrainTimingContext(own_fatigue=1.0, score_state=1.0, late_urgency=1.0),
        BrainTimingContext(own_fatigue=1.0),
    )
    for context in contexts:
        delay = expected_action_delay(FightState(), context, CONFIG)
        assert math.isfinite(delay)
        assert CONFIG.minimum_delay_seconds <= delay <= CONFIG.maximum_delay_seconds


def test_phase_baselines_are_exact() -> None:
    assert expected_action_delay(_state(Phase.STANDING), NEUTRAL, CONFIG) == 4.0
    assert expected_action_delay(_state(Phase.CLINCH), NEUTRAL, CONFIG) == 3.6
    assert expected_action_delay(_state(Phase.GROUND), NEUTRAL, CONFIG) == 4.4


def test_unsupported_phase_is_not_silently_accepted() -> None:
    with pytest.raises(ValueError, match="phase must be"):
        FightState(phase="distance")


def test_seeded_batch_has_variance_mean_near_expectation_and_respects_bounds() -> None:
    state = FightState()
    rng = np.random.default_rng(12345)
    samples = np.array(
        [sample_next_action_delay(state, NEUTRAL, rng, CONFIG) for _ in range(20_000)]
    )

    assert np.var(samples) > 0.0
    assert np.mean(samples) == pytest.approx(
        expected_action_delay(state, NEUTRAL, CONFIG), rel=0.03
    )
    assert np.all(samples >= CONFIG.minimum_delay_seconds)
    assert np.all(samples <= CONFIG.maximum_delay_seconds)


def test_standing_scenario_audit_exact_expected_delays() -> None:
    state = FightState()
    scenarios = {
        "neutral": NEUTRAL,
        "fatigued": BrainTimingContext(own_fatigue=1.0),
        "own_hurt": BrainTimingContext(own_hurt=1.0),
        "opponent_hurt": BrainTimingContext(opponent_hurt=1.0),
        "behind_late": BrainTimingContext(score_state=-1.0, late_urgency=1.0),
        "ahead_late": BrainTimingContext(score_state=1.0, late_urgency=1.0),
        "stacked_urgency": BrainTimingContext(
            opponent_hurt=1.0, score_state=-1.0, late_urgency=1.0
        ),
        "stacked_slowdown": BrainTimingContext(
            own_fatigue=1.0, score_state=1.0, late_urgency=1.0
        ),
    }

    assert {
        name: expected_action_delay(state, context, CONFIG)
        for name, context in scenarios.items()
    } == {
        "neutral": 4.0,
        "fatigued": 5.8,
        "own_hurt": 3.4,
        "opponent_hurt": 2.8,
        "behind_late": 3.0,
        "ahead_late": 4.6,
        "stacked_urgency": 2.0999999999999996,
        "stacked_slowdown": 6.669999999999999,
    }
