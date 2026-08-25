"""Generic causal fighter policy: choose what, never when or whether it succeeds."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np

from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.legality import legal_actions
from pipeline.simulation.event_clock_mc_v2.causal.state import FightState, Phase, Side

from .capabilities import BrainCapabilities


@dataclass(frozen=True)
class BrainDecisionContext:
    """Bounded tactical signals read by one generic action-selection policy."""

    striking_edge: float = 0.0
    own_hurt: float = 0.0
    opponent_hurt: float = 0.0
    td_success_recent: float = 0.0
    td_failure_recent: float = 0.0
    td_defense_success_recent: float = 0.0
    control_success_recent: float = 0.0
    own_back_to_cage: float = 0.0
    opponent_back_to_cage: float = 0.0
    fatigue: float = 0.0
    score_state: float = 0.0
    late_urgency: float = 0.0
    bad_bottom_position: float = 0.0
    dominant_top_position: float = 0.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            lower = -1.0 if field.name in {"striking_edge", "score_state"} else 0.0
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not lower <= value <= 1.0
            ):
                raise ValueError(f"{field.name} must be a finite value in [{lower}, 1]")


@dataclass(frozen=True)
class BrainPolicyConfig:
    """Structural research configuration; not predictive calibration."""

    softmax_temperature: float = 0.55

    def __post_init__(self) -> None:
        value = self.softmax_temperature
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError("softmax_temperature must be finite and positive")


@dataclass(frozen=True)
class ActionProbability:
    action_family: ActionFamily
    utility: float
    probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.action_family, ActionFamily):
            raise ValueError("action_family must be an ActionFamily")
        if not math.isfinite(self.utility):
            raise ValueError("utility must be finite")
        if not math.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be finite and in [0, 1]")


DEFAULT_BRAIN_POLICY_CONFIG = BrainPolicyConfig()


def action_utilities(
    state: FightState,
    actor: Side,
    capabilities: BrainCapabilities,
    context: BrainDecisionContext,
) -> tuple[tuple[ActionFamily, float], ...]:
    """Return utilities in authoritative ``legal_actions`` order.

    Coefficients are a direct migration of Standard Fighter V1 research. The
    only additions complete late-score responses outside standing; they are
    structural coefficients and have not been tuned to predictive outcomes.
    """
    _validate_inputs(state, actor, capabilities, context)
    menu = legal_actions(state, actor)
    u = {action: 0.0 for action in menu}
    c = capabilities
    x = context

    if state.phase is Phase.STANDING:
        u[ActionFamily.STAND_ATTACK] = 0.35 + 0.85 * c.standing
        u[ActionFamily.STAND_COUNTER] = 0.20 + 0.70 * c.counter
        u[ActionFamily.PRESSURE] = 0.10 + 0.65 * c.pressure
        u[ActionFamily.RESET_RANGE] = 0.00
        u[ActionFamily.CLINCH_ENTRY] = -0.10 + 0.70 * c.clinch
        u[ActionFamily.TAKEDOWN_ENTRY] = -0.15 + 0.90 * c.takedown
        if x.striking_edge > 0.0:
            edge = x.striking_edge
            u[ActionFamily.STAND_ATTACK] += 0.85 * edge
            u[ActionFamily.STAND_COUNTER] += 0.35 * edge
            u[ActionFamily.TAKEDOWN_ENTRY] -= 0.45 * edge
            u[ActionFamily.CLINCH_ENTRY] -= 0.30 * edge
        elif x.striking_edge < 0.0:
            trouble = -x.striking_edge
            u[ActionFamily.RESET_RANGE] += 0.70 * trouble
            u[ActionFamily.CLINCH_ENTRY] += 0.65 * trouble * (0.35 + c.clinch)
            u[ActionFamily.TAKEDOWN_ENTRY] += 0.85 * trouble * (0.30 + c.takedown)
            u[ActionFamily.STAND_ATTACK] -= 0.45 * trouble
        u[ActionFamily.STAND_ATTACK] -= 1.05 * x.own_hurt
        u[ActionFamily.PRESSURE] -= 0.65 * x.own_hurt
        u[ActionFamily.RESET_RANGE] += 1.15 * x.own_hurt
        u[ActionFamily.CLINCH_ENTRY] += 0.65 * x.own_hurt * (0.35 + c.clinch)
        u[ActionFamily.TAKEDOWN_ENTRY] += 0.70 * x.own_hurt * (0.30 + c.takedown)
        u[ActionFamily.STAND_ATTACK] += 1.10 * x.opponent_hurt
        u[ActionFamily.PRESSURE] += 0.95 * x.opponent_hurt
        u[ActionFamily.RESET_RANGE] -= 0.60 * x.opponent_hurt
        u[ActionFamily.TAKEDOWN_ENTRY] += 0.75 * x.td_success_recent
        u[ActionFamily.TAKEDOWN_ENTRY] -= 0.70 * x.td_failure_recent
        u[ActionFamily.STAND_ATTACK] += 0.30 * x.td_defense_success_recent
        u[ActionFamily.TAKEDOWN_ENTRY] += 0.25 * x.td_defense_success_recent * c.takedown
        u[ActionFamily.PRESSURE] += 0.55 * x.opponent_back_to_cage
        u[ActionFamily.CLINCH_ENTRY] += 0.55 * x.opponent_back_to_cage * (0.40 + c.clinch)
        u[ActionFamily.TAKEDOWN_ENTRY] += 0.55 * x.opponent_back_to_cage * (0.35 + c.takedown)
        u[ActionFamily.RESET_RANGE] += 0.85 * x.own_back_to_cage
        u[ActionFamily.PRESSURE] -= 0.40 * x.own_back_to_cage
        u[ActionFamily.PRESSURE] -= 0.55 * x.fatigue
        u[ActionFamily.TAKEDOWN_ENTRY] -= 0.45 * x.fatigue
        u[ActionFamily.RESET_RANGE] += 0.35 * x.fatigue
        urgency, safety = _score_signals(x)
        u[ActionFamily.STAND_ATTACK] += 0.65 * urgency - 0.35 * safety
        u[ActionFamily.PRESSURE] += 0.55 * urgency - 0.25 * safety
        u[ActionFamily.TAKEDOWN_ENTRY] += 0.35 * urgency * (0.25 + c.takedown)
        u[ActionFamily.RESET_RANGE] += -0.45 * urgency + 0.55 * safety

    elif state.phase is Phase.CLINCH:
        u[ActionFamily.CLINCH_STRIKE] = 0.20 + 0.70 * c.clinch + 0.40 * x.opponent_hurt
        u[ActionFamily.CLINCH_CONTROL] = 0.10 + 0.55 * c.clinch + 0.50 * x.control_success_recent
        u[ActionFamily.CLINCH_TAKEDOWN] = 0.80 * c.takedown + 0.55 * x.td_success_recent - 0.55 * x.td_failure_recent
        u[ActionFamily.BREAK_CLINCH] = 0.05 + 0.45 * c.standing + 0.55 * max(x.striking_edge, 0.0)
        u[ActionFamily.CLINCH_CONTROL] += 0.70 * x.own_hurt
        urgency, safety = _score_signals(x)
        u[ActionFamily.CLINCH_STRIKE] += 0.40 * urgency
        u[ActionFamily.CLINCH_TAKEDOWN] += 0.35 * urgency * (0.25 + c.takedown)
        u[ActionFamily.CLINCH_CONTROL] += -0.20 * urgency + 0.45 * safety

    elif state.phase is Phase.GROUND and actor is state.ground_controller:
        u[ActionFamily.GROUND_STRIKE] = 0.20 + 0.75 * c.ground_top + 0.70 * x.opponent_hurt
        u[ActionFamily.ADVANCE_POSITION] = 0.15 + 0.55 * c.ground_top
        u[ActionFamily.SUBMISSION_ATTACK] = 0.80 * c.submission + 0.35 * x.opponent_hurt
        u[ActionFamily.CONTROL] = 0.10 + 0.65 * c.ground_top + 0.50 * x.control_success_recent
        u[ActionFamily.DISENGAGE] = -0.20 + 0.55 * c.standing
        u[ActionFamily.GROUND_STRIKE] += 0.65 * x.dominant_top_position
        u[ActionFamily.SUBMISSION_ATTACK] += 0.55 * x.dominant_top_position
        u[ActionFamily.DISENGAGE] -= 0.90 * x.dominant_top_position
        urgency, safety = _score_signals(x)
        u[ActionFamily.GROUND_STRIKE] += 0.45 * urgency
        u[ActionFamily.SUBMISSION_ATTACK] += 0.55 * urgency
        u[ActionFamily.CONTROL] += -0.30 * urgency + 0.60 * safety
        u[ActionFamily.DISENGAGE] -= 0.20 * safety

    else:
        u[ActionFamily.ESCAPE_STAND] = 0.15 + 0.80 * c.escape
        u[ActionFamily.IMPROVE_POSITION] = 0.20 + 0.55 * c.escape
        u[ActionFamily.REVERSAL] = -0.05 + 0.70 * c.reversal
        u[ActionFamily.SUBMISSION_ATTACK] = -0.10 + 0.75 * c.submission
        u[ActionFamily.BOTTOM_STRIKE] = -0.20 + 0.35 * c.standing
        danger = x.bad_bottom_position
        u[ActionFamily.ESCAPE_STAND] += 0.65 * danger + 0.45 * x.own_hurt
        u[ActionFamily.IMPROVE_POSITION] += 1.00 * danger + 0.55 * x.own_hurt
        u[ActionFamily.REVERSAL] += 0.45 * danger * (0.25 + c.reversal)
        u[ActionFamily.SUBMISSION_ATTACK] -= 0.40 * danger
        u[ActionFamily.BOTTOM_STRIKE] -= 0.70 * danger
        urgency, _ = _score_signals(x)
        u[ActionFamily.ESCAPE_STAND] += 0.20 * urgency
        u[ActionFamily.REVERSAL] += 0.30 * urgency
        u[ActionFamily.SUBMISSION_ATTACK] += 0.35 * urgency

    return tuple((action, float(u[action])) for action in menu)


def action_probabilities(
    state: FightState,
    actor: Side,
    capabilities: BrainCapabilities,
    context: BrainDecisionContext,
    config: BrainPolicyConfig = DEFAULT_BRAIN_POLICY_CONFIG,
) -> tuple[ActionProbability, ...]:
    """Apply a stable softmax and return only the authoritative legal menu."""
    if not isinstance(config, BrainPolicyConfig):
        raise ValueError("config must be a BrainPolicyConfig")
    utilities = action_utilities(state, actor, capabilities, context)
    raw = np.asarray([utility for _, utility in utilities], dtype=float)
    shifted = (raw - np.max(raw)) / config.softmax_temperature
    weights = np.exp(shifted)
    probabilities = weights / np.sum(weights)
    return tuple(
        ActionProbability(action, utility, float(probability))
        for (action, utility), probability in zip(utilities, probabilities, strict=True)
    )


def choose_action(
    state: FightState,
    actor: Side,
    capabilities: BrainCapabilities,
    context: BrainDecisionContext,
    rng: np.random.Generator,
    config: BrainPolicyConfig = DEFAULT_BRAIN_POLICY_CONFIG,
) -> ActionFamily:
    """Select exactly one legal attempted action with one explicit categorical draw."""
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")
    distribution = action_probabilities(state, actor, capabilities, context, config)
    index = int(rng.choice(len(distribution), p=[row.probability for row in distribution]))
    return distribution[index].action_family


def _validate_inputs(
    state: FightState,
    actor: Side,
    capabilities: BrainCapabilities,
    context: BrainDecisionContext,
) -> None:
    if not isinstance(state, FightState):
        raise ValueError("state must be a FightState")
    if not isinstance(actor, Side):
        raise ValueError("actor must be a Side")
    if not isinstance(capabilities, BrainCapabilities):
        raise ValueError("capabilities must be BrainCapabilities")
    if not isinstance(context, BrainDecisionContext):
        raise ValueError("context must be BrainDecisionContext")


def _score_signals(context: BrainDecisionContext) -> tuple[float, float]:
    return (
        context.late_urgency * max(-context.score_state, 0.0),
        context.late_urgency * max(context.score_state, 0.0),
    )
