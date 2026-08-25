"""One-action-clock timing model for the causal Event Clock V2 brain.

Timing answers only how soon a fighter reacts. In particular, a shorter delay
while hurt means an earlier response, not a more aggressive action; Stage 5
will separately decide what that response is.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np

from pipeline.simulation.event_clock_mc_v2.causal.state import FightState, Phase


def _validate_bounded(value: float, name: str, lower: float, upper: float) -> None:
    _validate_finite_number(value, name)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")


def _validate_finite_number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


@dataclass(frozen=True)
class BrainTimingContext:
    """Normalized read-only signals for one fighter's next timing decision.

    Severity fields use 0 for absent/fresh and 1 for maximum modeled severity.
    Score state uses -1 for fully behind, 0 for neutral/unknown, and +1 for
    fully ahead. ``late_urgency`` controls how strongly score affects cadence.
    The authoritative phase remains on ``FightState`` rather than being copied.
    """

    own_fatigue: float = 0.0
    own_hurt: float = 0.0
    opponent_hurt: float = 0.0
    score_state: float = 0.0
    late_urgency: float = 0.0
    activity_rate_ratio: float = 1.0

    def __post_init__(self) -> None:
        for name in ("own_fatigue", "own_hurt", "opponent_hurt", "late_urgency"):
            _validate_bounded(getattr(self, name), name, 0.0, 1.0)
        _validate_bounded(self.score_state, "score_state", -1.0, 1.0)
        _validate_finite_number(self.activity_rate_ratio, "activity_rate_ratio")
        if self.activity_rate_ratio <= 0.0:
            raise ValueError("activity_rate_ratio must be positive")


@dataclass(frozen=True)
class BrainTimingConfig:
    """Uncalibrated structural MVP cadence and modest directional effects."""

    base_mean_delay_seconds: float = 4.0
    standing_phase_factor: float = 1.00
    clinch_phase_factor: float = 0.90
    ground_phase_factor: float = 1.10
    maximum_fatigue_slowdown: float = 0.45
    maximum_own_hurt_speedup: float = 0.15
    maximum_opponent_hurt_speedup: float = 0.30
    maximum_late_behind_speedup: float = 0.25
    maximum_late_ahead_slowdown: float = 0.15
    gamma_shape: float = 2.0
    minimum_delay_seconds: float = 0.50
    maximum_delay_seconds: float = 20.0

    def __post_init__(self) -> None:
        for field in fields(self):
            _validate_finite_number(getattr(self, field.name), field.name)
        for name in (
            "base_mean_delay_seconds",
            "standing_phase_factor",
            "clinch_phase_factor",
            "ground_phase_factor",
            "gamma_shape",
            "minimum_delay_seconds",
            "maximum_delay_seconds",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "maximum_fatigue_slowdown",
            "maximum_own_hurt_speedup",
            "maximum_opponent_hurt_speedup",
            "maximum_late_behind_speedup",
            "maximum_late_ahead_slowdown",
        ):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.maximum_delay_seconds < self.minimum_delay_seconds:
            raise ValueError("maximum_delay_seconds cannot be below minimum_delay_seconds")


DEFAULT_BRAIN_TIMING_CONFIG = BrainTimingConfig()


def expected_action_delay(
    state: FightState,
    context: BrainTimingContext,
    config: BrainTimingConfig = DEFAULT_BRAIN_TIMING_CONFIG,
) -> float:
    """Return the bounded mean delay using transparent multiplicative scaling."""
    _validate_inputs(state, context, config)
    phase_factor = {
        Phase.STANDING: config.standing_phase_factor,
        Phase.CLINCH: config.clinch_phase_factor,
        Phase.GROUND: config.ground_phase_factor,
    }[state.phase]
    fatigue_factor = 1.0 + config.maximum_fatigue_slowdown * context.own_fatigue
    own_hurt_factor = 1.0 - config.maximum_own_hurt_speedup * context.own_hurt
    opponent_hurt_factor = (
        1.0 - config.maximum_opponent_hurt_speedup * context.opponent_hurt
    )
    behind = max(-context.score_state, 0.0)
    ahead = max(context.score_state, 0.0)
    score_factor = (
        1.0
        - config.maximum_late_behind_speedup * context.late_urgency * behind
        + config.maximum_late_ahead_slowdown * context.late_urgency * ahead
    )
    mean = (
        config.base_mean_delay_seconds
        * phase_factor
        * fatigue_factor
        * own_hurt_factor
        * opponent_hurt_factor
        * score_factor
        / context.activity_rate_ratio
    )
    return float(np.clip(mean, config.minimum_delay_seconds, config.maximum_delay_seconds))


def sample_next_action_delay(
    state: FightState,
    context: BrainTimingContext,
    rng: np.random.Generator,
    config: BrainTimingConfig = DEFAULT_BRAIN_TIMING_CONFIG,
) -> float:
    """Draw one bounded Gamma waiting time for a fighter's next action opportunity."""
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")
    mean = expected_action_delay(state, context, config)
    sampled = rng.gamma(shape=config.gamma_shape, scale=mean / config.gamma_shape)
    return float(
        np.clip(sampled, config.minimum_delay_seconds, config.maximum_delay_seconds)
    )


def _validate_inputs(
    state: FightState, context: BrainTimingContext, config: BrainTimingConfig
) -> None:
    if not isinstance(state, FightState):
        raise ValueError("state must be a FightState")
    if not isinstance(context, BrainTimingContext):
        raise ValueError("context must be a BrainTimingContext")
    if not isinstance(config, BrainTimingConfig):
        raise ValueError("config must be a BrainTimingConfig")
