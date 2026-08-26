"""Matched-seed Leavitt-Brito Brain intent-rate shadow.

Research only. Production Brain/engine/mechanics are unchanged.

This diagnostic keeps the validated FSR matchup transforms and all mechanics
frozen, but replaces the standing generic readiness + multinomial policy with a
Brain-owned rate clock:

- standing strike intent starts from matchup-effective FSR standing rate;
- takedown intent starts from matchup-effective FSR takedown rate;
- clinch intent preserves the current 0.06 structural ratio;
- dynamic pressure modifies standing-strike intent rather than consuming an action;
- current policy context contributes only live tactical deltas, not neutral
  capability weights, to strike/TD/clinch intent rates;
- RESET_RANGE remains a tactical Brain event, anchored to its current neutral
  odds relative to standing offense and modulated by live context;
- non-standing phases retain the current intent-prior chooser and timing.

The same 500 matched seeds are run for the pressure-only control and the
intent-rate shadow.
"""
from __future__ import annotations

from collections import Counter
import json
import math

import numpy as np

from pipeline.simulation.event_clock_mc_v2.brain.intent_priors import (
    BrainIntentPriors,
    action_probabilities_with_intent_priors,
)
from pipeline.simulation.event_clock_mc_v2.brain.memory import decision_context
from pipeline.simulation.event_clock_mc_v2.brain.policy import (
    BrainDecisionContext,
    action_utilities,
)
from pipeline.simulation.event_clock_mc_v2.brain.timing import sample_next_action_delay
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineFunctions, run_causal_path
from pipeline.simulation.event_clock_mc_v2.mechanics.resolution import FinishMethod
from pipeline.simulation.event_clock_mc_v2.calibration.seeds import derive_path_seed
from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION
from pipeline.simulation.event_clock_mc_v2.diagnostics.leavitt_brito_dynamic_pressure_shadow import (
    FIGHT_ID,
    PATHS,
    STAND,
    GROUND,
    TD,
    STRIKES,
    build_setup,
    dynamic_pressure,
    run_condition,
)

EPS = 1e-12
CLINCH_TO_STANDING = 0.06


def _context_factor(current_utility: float, neutral_utility: float, temperature: float) -> float:
    """Convert only the live policy delta to a multiplicative intent factor."""
    return float(math.exp((current_utility - neutral_utility) / temperature))


def _standing_rates(state, actor, capabilities, context, priors, config):
    """Return Brain-owned live standing intent rates per 15 minutes.

    Neutral capability utilities do not set the absolute strike/TD rates. They
    are used only to preserve the current neutral RESET_RANGE/strike structural
    ratio because reset has no validated FSR tendency trait.
    """
    current = dict(action_utilities(state, actor, capabilities, context))
    neutral_context = BrainDecisionContext()
    neutral = dict(action_utilities(state, actor, capabilities, neutral_context))
    temp = config.softmax_temperature

    current_strike_log = float(np.logaddexp(
        current[ActionFamily.STAND_ATTACK] / temp,
        current[ActionFamily.STAND_COUNTER] / temp,
    ))
    neutral_strike_log = float(np.logaddexp(
        neutral[ActionFamily.STAND_ATTACK] / temp,
        neutral[ActionFamily.STAND_COUNTER] / temp,
    ))
    strike_context_factor = float(math.exp(current_strike_log - neutral_strike_log))

    p = dynamic_pressure(capabilities, context)
    pressure_factor = 0.75 + 1.00 * p

    strike_rate = max(priors.standing_attempt_rate_15m * pressure_factor * strike_context_factor, EPS)

    td_factor = _context_factor(
        current[ActionFamily.TAKEDOWN_ENTRY],
        neutral[ActionFamily.TAKEDOWN_ENTRY],
        temp,
    )
    td_rate = max(priors.takedown_attempt_rate_15m * td_factor, EPS)

    clinch_factor = _context_factor(
        current[ActionFamily.CLINCH_ENTRY],
        neutral[ActionFamily.CLINCH_ENTRY],
        temp,
    )
    clinch_rate = max(
        priors.standing_attempt_rate_15m * CLINCH_TO_STANDING * clinch_factor,
        EPS,
    )

    # Preserve the existing neutral tactical reset odds relative to the combined
    # standing-strike bucket. Live context can raise/lower reset intent, but the
    # absolute offensive cadence is now anchored by FSR rather than the softmax.
    neutral_reset_to_strike = math.exp(
        neutral[ActionFamily.RESET_RANGE] / temp - neutral_strike_log
    )
    reset_factor = _context_factor(
        current[ActionFamily.RESET_RANGE],
        neutral[ActionFamily.RESET_RANGE],
        temp,
    )
    reset_rate = max(
        priors.standing_attempt_rate_15m * neutral_reset_to_strike * reset_factor,
        EPS,
    )

    return {
        ActionFamily.STAND_ATTACK: strike_rate,
        ActionFamily.TAKEDOWN_ENTRY: td_rate,
        ActionFamily.CLINCH_ENTRY: clinch_rate,
        ActionFamily.RESET_RANGE: reset_rate,
    }, p


class IntentRateBrain:
    """Coupled timing + chooser implementing rate-driven standing Brain intent."""

    def __init__(self, inputs, priors, horizon):
        self.inputs = inputs
        self.priors = priors
        self.horizon = float(horizon)
        self.side_by_timing_context_id = {
            id(inputs.red.timing_context): Side.RED,
            id(inputs.blue.timing_context): Side.BLUE,
        }
        if len(self.side_by_timing_context_id) != 2:
            raise RuntimeError("red/blue timing contexts must be distinct objects")
        self.rate_sums = {side: Counter() for side in Side}
        self.rate_n = Counter()
        self.pressure_sum = Counter()

    def timing_sampler(self, state, timing_context, rng, timing_config):
        if state.phase is not Phase.STANDING:
            return sample_next_action_delay(state, timing_context, rng, timing_config)

        side = self.side_by_timing_context_id[id(timing_context)]
        fighter = self.inputs.fighter(side)
        context = decision_context(
            state, side, fighter.decision_context, self.horizon
        )
        rates, p = _standing_rates(
            state,
            side,
            fighter.capabilities,
            context,
            self.priors[side],
            self.inputs.policy_config,
        )
        total_rate_15m = sum(rates.values())
        mean_delay = 900.0 / max(total_rate_15m, EPS)
        mean_delay = float(np.clip(
            mean_delay,
            timing_config.minimum_delay_seconds,
            timing_config.maximum_delay_seconds,
        ))
        sampled = rng.gamma(
            shape=timing_config.gamma_shape,
            scale=mean_delay / timing_config.gamma_shape,
        )
        for action, rate in rates.items():
            self.rate_sums[side][action.value] += rate
        self.rate_sums[side]["total"] += total_rate_15m
        self.pressure_sum[side] += p
        self.rate_n[side] += 1
        return float(np.clip(
            sampled,
            timing_config.minimum_delay_seconds,
            timing_config.maximum_delay_seconds,
        ))

    def action_chooser(self, state, actor, capabilities, context, rng, config):
        if state.phase is not Phase.STANDING:
            rows = action_probabilities_with_intent_priors(
                state, actor, capabilities, context, self.priors[actor], config
            )
            probs = [row.probability for row in rows]
            return rows[int(rng.choice(len(rows), p=probs))].action_family

        rates, _ = _standing_rates(
            state, actor, capabilities, context, self.priors[actor], config
        )
        actions = tuple(rates)
        weights = np.asarray([rates[action] for action in actions], dtype=float)
        probs = weights / weights.sum()
        return actions[int(rng.choice(len(actions), p=probs))]

    def summary(self, side):
        n = self.rate_n[side]
        if not n:
            return {}
        return {
            "mean_live_rate_15m": {
                key: value / n for key, value in sorted(self.rate_sums[side].items())
            },
            "mean_dynamic_pressure": self.pressure_sum[side] / n,
            "timing_samples": int(n),
        }


def run_intent_rate_condition(fight, inputs, priors, horizon, cfg):
    brain = IntentRateBrain(inputs, priors, horizon)
    funcs = EngineFunctions(
        timing_sampler=brain.timing_sampler,
        action_chooser=brain.action_chooser,
    )
    totals = {s: Counter() for s in Side}
    control = {s: 0.0 for s in Side}
    standing_exposure = 0.0
    wins = Counter()
    methods = Counter()
    decision_wins = Counter()

    for path_id in range(PATHS):
        seed = derive_path_seed(SEED_SET_VERSION, FIGHT_ID, path_id)
        out = run_causal_path(
            inputs,
            seed=seed,
            horizon_seconds=horizon,
            config=cfg,
            functions=funcs,
        )
        for seg in out.timeline_segments:
            if seg.phase is Phase.STANDING:
                standing_exposure += seg.duration
            if seg.phase is Phase.GROUND and seg.controller in (Side.RED, Side.BLUE):
                control[seg.controller] += seg.duration
        for ev in out.events:
            side = ev.actor
            action = ev.selected_action
            totals[side][f"action::{action.value}"] += 1
            if action in TD:
                totals[side]["td_attempts"] += 1
                if ev.resulting_phase is Phase.GROUND and ev.resulting_controller is side:
                    totals[side]["td_success"] += 1
            if action in STRIKES:
                totals[side]["strike_attempts"] += 1
                if ev.outcome.value == "landed":
                    totals[side]["strikes_landed"] += 1
            if action in STAND:
                totals[side]["standing_strike_attempts"] += 1
            if action in GROUND:
                totals[side]["ground_strike_attempts"] += 1
                if ev.outcome.value == "landed":
                    totals[side]["ground_strikes_landed"] += 1
            if ev.submission_attempt:
                totals[side]["sub_attempts"] += 1
                if ev.submission_success:
                    totals[side]["sub_success"] += 1
            if ev.knockdown:
                totals[side]["kd"] += 1

        if out.termination is None:
            continue
        winner = out.termination.winner
        method = out.termination.finish_method.value
        wins[winner] += 1
        methods[method] += 1
        if out.termination.finish_method is FinishMethod.DECISION:
            decision_wins[winner] += 1

    standing_seconds_per_path = standing_exposure / PATHS

    def side_summary(side, fighter_name):
        t = totals[side]
        action_counts = {
            key.split("::", 1)[1]: value / PATHS
            for key, value in sorted(t.items())
            if key.startswith("action::")
        }
        standing_attempts_per_path = t["standing_strike_attempts"] / PATHS
        return {
            "fighter": fighter_name,
            "wins": wins[side],
            "win_probability": wins[side] / PATHS,
            "decision_wins": decision_wins[side],
            "standing_seconds_per_path": standing_seconds_per_path,
            "standing_strike_attempts_per_path": standing_attempts_per_path,
            "standing_strike_attempts_per_15m_standing_exposure": (
                standing_attempts_per_path * 900.0 / standing_seconds_per_path
                if standing_seconds_per_path > 0 else 0.0
            ),
            "td_attempts_per_path": t["td_attempts"] / PATHS,
            "td_success_per_path": t["td_success"] / PATHS,
            "td_success_rate": t["td_success"] / t["td_attempts"] if t["td_attempts"] else 0.0,
            "ground_control_seconds_per_path": control[side] / PATHS,
            "sub_attempts_per_path": t["sub_attempts"] / PATHS,
            "sub_conversion": t["sub_success"] / t["sub_attempts"] if t["sub_attempts"] else 0.0,
            "strike_attempts_per_path": t["strike_attempts"] / PATHS,
            "strikes_landed_per_path": t["strikes_landed"] / PATHS,
            "ground_strike_attempts_per_path": t["ground_strike_attempts"] / PATHS,
            "ground_strikes_landed_per_path": t["ground_strikes_landed"] / PATHS,
            "knockdowns_per_path": t["kd"] / PATHS,
            "actions_per_path": action_counts,
            "brain_rate_diagnostics": brain.summary(side),
        }

    return {
        "condition": "dynamic_pressure_plus_fsr_intent_rate_clock",
        "red": side_summary(Side.RED, str(fight.r_name)),
        "blue": side_summary(Side.BLUE, str(fight.b_name)),
        "methods": dict(methods),
    }


def main():
    fight, inputs, priors, horizon, cfg = build_setup()
    pressure_only = run_condition(
        "dynamic_pressure_only",
        fight,
        inputs,
        priors,
        horizon,
        cfg,
        True,
    )
    intent_rate = run_intent_rate_condition(
        fight, inputs, priors, horizon, cfg
    )
    payload = {
        "diagnostic": "Leavitt-Brito Brain intent-rate matched-seed shadow",
        "fight_id": FIGHT_ID,
        "paths_each": PATHS,
        "seed_set": SEED_SET_VERSION,
        "production_changed": False,
        "intervention": {
            "fsr_absolute_standing_rate_drives_standing_brain_cadence": True,
            "fsr_absolute_takedown_rate_drives_td_intent": True,
            "dynamic_pressure_retained": True,
            "pressure_action_consumes_clock": False,
            "stand_counter_independent_baseline_intent": False,
            "clinch_structural_ratio": CLINCH_TO_STANDING,
            "nonstanding_brain_changed": False,
            "mechanics_changed": False,
            "judging_changed": False,
        },
        "pressure_only": pressure_only,
        "intent_rate_shadow": intent_rate,
        "delta_brito_win_probability_vs_pressure_only": (
            intent_rate["blue"]["win_probability"]
            - pressure_only["blue"]["win_probability"]
        ),
        "fsr_effective_rates": {
            "red": {
                "fighter": str(fight.r_name),
                "standing_rate_15m": priors[Side.RED].standing_attempt_rate_15m,
                "takedown_rate_15m": priors[Side.RED].takedown_attempt_rate_15m,
            },
            "blue": {
                "fighter": str(fight.b_name),
                "standing_rate_15m": priors[Side.BLUE].standing_attempt_rate_15m,
                "takedown_rate_15m": priors[Side.BLUE].takedown_attempt_rate_15m,
            },
        },
    }
    print("LEAVITT_BRITO_INTENT_RATE_SHADOW")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
