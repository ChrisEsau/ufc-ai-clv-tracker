"""Candidate separation of fighter intent priors from execution capability.

Research-only during Stage 8 validation. FSR matchup attempt rates describe
what a fighter tends to try; completion/accuracy remain mechanics. Clinch entry
has no validated fighter-specific FSR trait yet, so its neutral choice rate is
represented by an explicit population structural prior instead of inheriting
wrestling capability.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from pipeline.simulation.event_clock_mc_v2.brain.capabilities import BrainCapabilities
from pipeline.simulation.event_clock_mc_v2.brain.policy import (
    ActionProbability,
    BrainDecisionContext,
    BrainPolicyConfig,
    DEFAULT_BRAIN_POLICY_CONFIG,
    action_utilities,
)
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import FightState, Phase, Side

EPS = 1e-12


@dataclass(frozen=True)
class BrainIntentPriors:
    """Attempt-rate / structural priors used only for action choice.

    ``standing_attempt_rate_15m`` and ``takedown_attempt_rate_15m`` are
    matchup-effective FSR quantities. ``clinch_entry_to_standing_ratio`` is a
    population structural prior because no validated fighter-specific clinch
    tendency trait exists yet. ``None`` preserves the Stage-5 clinch-entry
    utility for diagnostic comparisons.
    """

    standing_attempt_rate_15m: float
    takedown_attempt_rate_15m: float
    clinch_entry_to_standing_ratio: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("standing_attempt_rate_15m", self.standing_attempt_rate_15m),
            ("takedown_attempt_rate_15m", self.takedown_attempt_rate_15m),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.standing_attempt_rate_15m <= 0.0:
            raise ValueError("standing_attempt_rate_15m must be positive")
        ratio = self.clinch_entry_to_standing_ratio
        if ratio is not None and (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or ratio < 0.0
        ):
            raise ValueError("clinch_entry_to_standing_ratio must be finite and non-negative")

    @property
    def takedown_to_standing_ratio(self) -> float:
        return float(self.takedown_attempt_rate_15m / self.standing_attempt_rate_15m)


def action_probabilities_with_intent_priors(
    state: FightState,
    actor: Side,
    capabilities: BrainCapabilities,
    context: BrainDecisionContext,
    priors: BrainIntentPriors,
    config: BrainPolicyConfig = DEFAULT_BRAIN_POLICY_CONFIG,
) -> tuple[ActionProbability, ...]:
    """Apply explicit intent priors while preserving tactical modulation.

    In STANDING, neutral odds satisfy::

        P(TD) / (P(STAND_ATTACK) + P(STAND_COUNTER)) = TD_rate / standing_rate

    and, when a clinch prior is supplied::

        P(CLINCH_ENTRY) / (P(STAND_ATTACK) + P(STAND_COUNTER)) = clinch prior

    In CLINCH, the matchup TD/standing rate ratio anchors clinch takedown vs
    clinch strike. Existing Stage-5 tactical context acts multiplicatively on
    these neutral odds. No completion probability enters this calculation.
    """
    if not isinstance(priors, BrainIntentPriors):
        raise ValueError("priors must be BrainIntentPriors")
    if not isinstance(config, BrainPolicyConfig):
        raise ValueError("config must be BrainPolicyConfig")

    rows = action_utilities(state, actor, capabilities, context)
    neutral_rows = action_utilities(state, actor, capabilities, BrainDecisionContext())
    utilities = dict(rows)
    neutral = dict(neutral_rows)
    actions = [action for action, _ in rows]
    temperature = config.softmax_temperature
    log_weights = {action: utilities[action] / temperature for action in actions}
    td_ratio = priors.takedown_to_standing_ratio

    if state.phase is Phase.STANDING:
        strike_log_weight = float(
            np.logaddexp(
                log_weights[ActionFamily.STAND_ATTACK],
                log_weights[ActionFamily.STAND_COUNTER],
            )
        )
        if ActionFamily.TAKEDOWN_ENTRY in log_weights:
            context_delta = (
                utilities[ActionFamily.TAKEDOWN_ENTRY]
                - neutral[ActionFamily.TAKEDOWN_ENTRY]
            ) / temperature
            log_weights[ActionFamily.TAKEDOWN_ENTRY] = (
                math.log(max(td_ratio, EPS)) + strike_log_weight + context_delta
            )
        clinch_ratio = priors.clinch_entry_to_standing_ratio
        if clinch_ratio is not None and ActionFamily.CLINCH_ENTRY in log_weights:
            context_delta = (
                utilities[ActionFamily.CLINCH_ENTRY]
                - neutral[ActionFamily.CLINCH_ENTRY]
            ) / temperature
            log_weights[ActionFamily.CLINCH_ENTRY] = (
                math.log(max(clinch_ratio, EPS)) + strike_log_weight + context_delta
            )
    elif state.phase is Phase.CLINCH and ActionFamily.CLINCH_TAKEDOWN in log_weights:
        context_delta = (
            utilities[ActionFamily.CLINCH_TAKEDOWN]
            - neutral[ActionFamily.CLINCH_TAKEDOWN]
        ) / temperature
        log_weights[ActionFamily.CLINCH_TAKEDOWN] = (
            math.log(max(td_ratio, EPS))
            + log_weights[ActionFamily.CLINCH_STRIKE]
            + context_delta
        )

    raw = np.asarray([log_weights[action] for action in actions], dtype=float)
    shifted = raw - np.max(raw)
    weights = np.exp(shifted)
    probabilities = weights / weights.sum()
    return tuple(
        ActionProbability(action, float(utilities[action]), float(probability))
        for action, probability in zip(actions, probabilities, strict=True)
    )


def choose_action_with_intent_priors(
    state: FightState,
    actor: Side,
    capabilities: BrainCapabilities,
    context: BrainDecisionContext,
    priors: BrainIntentPriors,
    rng: np.random.Generator,
    config: BrainPolicyConfig = DEFAULT_BRAIN_POLICY_CONFIG,
) -> ActionFamily:
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")
    rows = action_probabilities_with_intent_priors(
        state, actor, capabilities, context, priors, config
    )
    return rows[int(rng.choice(len(rows), p=[row.probability for row in rows]))].action_family
