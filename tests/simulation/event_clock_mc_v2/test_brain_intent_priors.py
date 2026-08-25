import pytest

from pipeline.simulation.event_clock_mc_v2.brain.capabilities import BrainCapabilities
from pipeline.simulation.event_clock_mc_v2.brain.intent_priors import (
    BrainIntentPriors,
    action_probabilities_with_intent_priors,
)
from pipeline.simulation.event_clock_mc_v2.brain.policy import BrainDecisionContext
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import FightState, Side


def test_neutral_standing_td_to_strike_odds_equal_fsr_rate_ratio() -> None:
    capabilities = BrainCapabilities(.5, .5, .5, .5, .95, .5, .3, .4, .3)
    priors = BrainIntentPriors(standing_attempt_rate_15m=90.0, takedown_attempt_rate_15m=4.5)
    rows = action_probabilities_with_intent_priors(
        FightState(), Side.RED, capabilities, BrainDecisionContext(), priors
    )
    probs = {row.action_family: row.probability for row in rows}
    strike = probs[ActionFamily.STAND_ATTACK] + probs[ActionFamily.STAND_COUNTER]
    assert probs[ActionFamily.TAKEDOWN_ENTRY] / strike == pytest.approx(4.5 / 90.0, rel=1e-12)


def test_completion_like_capability_does_not_change_neutral_td_odds_when_priors_fixed() -> None:
    low = BrainCapabilities(.5, .5, .5, .5, 0.0, .5, .3, .4, .3)
    high = BrainCapabilities(.5, .5, .5, .5, 1.0, .5, .3, .4, .3)
    priors = BrainIntentPriors(80.0, 4.0)
    low_rows = action_probabilities_with_intent_priors(
        FightState(), Side.RED, low, BrainDecisionContext(), priors
    )
    high_rows = action_probabilities_with_intent_priors(
        FightState(), Side.RED, high, BrainDecisionContext(), priors
    )
    low_p = next(row.probability for row in low_rows if row.action_family is ActionFamily.TAKEDOWN_ENTRY)
    high_p = next(row.probability for row in high_rows if row.action_family is ActionFamily.TAKEDOWN_ENTRY)
    assert high_p == pytest.approx(low_p, abs=1e-15)


def test_recent_success_and_failure_still_modulate_td_choice() -> None:
    capabilities = BrainCapabilities(.5, .5, .5, .5, .5, .5, .3, .4, .3)
    priors = BrainIntentPriors(80.0, 4.0)

    def p(context):
        rows = action_probabilities_with_intent_priors(
            FightState(), Side.RED, capabilities, context, priors
        )
        return next(row.probability for row in rows if row.action_family is ActionFamily.TAKEDOWN_ENTRY)

    neutral = p(BrainDecisionContext())
    assert p(BrainDecisionContext(td_success_recent=.8)) > neutral
    assert p(BrainDecisionContext(td_failure_recent=.8)) < neutral


def test_intent_priors_validate_rate_semantics() -> None:
    with pytest.raises(ValueError):
        BrainIntentPriors(0.0, 1.0)
    with pytest.raises(ValueError):
        BrainIntentPriors(10.0, -1.0)
