"""Matched-seed Leavitt-Brito shadow: dynamic pressure + collapsed standing counter.

Research only. Production Brain policy/mechanics are unchanged.
Interventions relative to production:
- standing PRESSURE is a dynamic modifier and does not consume clock;
- STAND_COUNTER no longer independently generates a strike attempt;
- its probability mass is folded into STAND_ATTACK before pressure modulation.
Everything else, including timing and strike mechanics, remains unchanged.
"""
from __future__ import annotations

from collections import Counter
import json

import numpy as np

from pipeline.simulation.event_clock_mc_v2.brain.intent_priors import (
    BrainIntentPriors,
    action_probabilities_with_intent_priors,
)
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineFunctions, run_causal_path
from pipeline.simulation.event_clock_mc_v2.mechanics.resolution import FinishMethod
from pipeline.simulation.event_clock_mc_v2.calibration.seeds import derive_path_seed
from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION
from pipeline.simulation.event_clock_mc_v2.diagnostics.leavitt_brito_dynamic_pressure_shadow import (
    FIGHT_ID,
    PATHS,
    REFERENCE_CUTOFF,
    GROUND,
    TD,
    STRIKES,
    EPS,
    build_setup,
    dynamic_pressure,
    DynamicPressureChooser,
)


class DynamicPressureCounterCollapseChooser:
    """Dynamic pressure plus one standing-strike bucket.

    The production STAND_COUNTER probability is not discarded. Its probability
    mass is merged into STAND_ATTACK so counter capability cannot create a
    second independent strike generator on the actor's own action clock.
    """

    def __init__(self, priors: dict[Side, BrainIntentPriors]) -> None:
        self.priors = priors
        self.pressure_sum = Counter()
        self.pressure_n = Counter()
        self.pressure_min = {Side.RED: 1.0, Side.BLUE: 1.0}
        self.pressure_max = {Side.RED: 0.0, Side.BLUE: 0.0}

    def __call__(self, state, actor, capabilities, context, rng, config):
        rows = action_probabilities_with_intent_priors(
            state, actor, capabilities, context, self.priors[actor], config
        )
        if state.phase is not Phase.STANDING:
            probs = [row.probability for row in rows]
            return rows[int(rng.choice(len(rows), p=probs))].action_family

        p = dynamic_pressure(capabilities, context)
        self.pressure_sum[actor] += p
        self.pressure_n[actor] += 1
        self.pressure_min[actor] = min(self.pressure_min[actor], p)
        self.pressure_max[actor] = max(self.pressure_max[actor], p)

        raw = {row.action_family: float(row.probability) for row in rows}
        # One standing-strike intent bucket: preserve total strike-choice mass,
        # but do not let counter exist as a second actor-clock strike action.
        raw[ActionFamily.STAND_ATTACK] = (
            raw.get(ActionFamily.STAND_ATTACK, 0.0)
            + raw.get(ActionFamily.STAND_COUNTER, 0.0)
        )
        raw[ActionFamily.STAND_COUNTER] = 0.0

        strike_mult = 0.75 + 1.00 * p
        entry_mult = 0.90 + 0.40 * p
        reset_mult = 1.25 - 0.75 * p
        multipliers = {
            ActionFamily.STAND_ATTACK: strike_mult,
            ActionFamily.STAND_COUNTER: 0.0,
            ActionFamily.CLINCH_ENTRY: entry_mult,
            ActionFamily.TAKEDOWN_ENTRY: entry_mult,
            ActionFamily.RESET_RANGE: reset_mult,
            ActionFamily.PRESSURE: EPS,
        }
        actions = [row.action_family for row in rows]
        weights = np.asarray(
            [max(raw.get(action, 0.0), 0.0) * multipliers.get(action, 1.0) for action in actions],
            dtype=float,
        )
        if weights.sum() <= 0.0:
            raise RuntimeError("collapsed standing chooser produced zero probability mass")
        weights /= weights.sum()
        return actions[int(rng.choice(len(actions), p=weights))]

    def summary(self, side: Side):
        n = self.pressure_n[side]
        return {
            "mean": self.pressure_sum[side] / n if n else 0.0,
            "min": self.pressure_min[side] if n else 0.0,
            "max": self.pressure_max[side] if n else 0.0,
            "standing_decisions": int(n),
        }


def run_condition(name, fight, inputs, priors, horizon, cfg, chooser):
    funcs = EngineFunctions(action_chooser=chooser)
    totals = {s: Counter() for s in Side}
    control = {s: 0.0 for s in Side}
    wins = Counter()
    methods = Counter()
    decision_wins = Counter()

    for path_id in range(PATHS):
        seed = derive_path_seed(SEED_SET_VERSION, FIGHT_ID, path_id)
        out = run_causal_path(
            inputs, seed=seed, horizon_seconds=horizon, config=cfg, functions=funcs
        )
        for seg in out.timeline_segments:
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

    def side_summary(side, fighter_name):
        t = totals[side]
        action_counts = {
            key.split("::", 1)[1]: value / PATHS
            for key, value in sorted(t.items())
            if key.startswith("action::")
        }
        return {
            "fighter": fighter_name,
            "wins": wins[side],
            "win_probability": wins[side] / PATHS,
            "decision_wins": decision_wins[side],
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
            "dynamic_pressure": chooser.summary(side),
        }

    return {
        "condition": name,
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
        DynamicPressureChooser(priors),
    )
    collapsed = run_condition(
        "dynamic_pressure_plus_counter_collapse",
        fight,
        inputs,
        priors,
        horizon,
        cfg,
        DynamicPressureCounterCollapseChooser(priors),
    )

    payload = {
        "diagnostic": "Leavitt-Brito dynamic pressure + counter collapse matched-seed shadow",
        "fight_id": FIGHT_ID,
        "paths_each": PATHS,
        "seed_set": SEED_SET_VERSION,
        "reference_cutoff": str(REFERENCE_CUTOFF.date()),
        "production_changed": False,
        "intervention": {
            "dynamic_pressure_retained": True,
            "pressure_action_consumes_clock": False,
            "stand_counter_independent_action": False,
            "counter_probability_mass_folded_into_stand_attack": True,
            "mechanics_changed": False,
            "timing_changed": False,
        },
        "pressure_only": pressure_only,
        "counter_collapsed": collapsed,
        "delta_brito_win_probability_vs_pressure_only": (
            collapsed["blue"]["win_probability"] - pressure_only["blue"]["win_probability"]
        ),
    }
    print("LEAVITT_BRITO_DYNAMIC_PRESSURE_COUNTER_COLLAPSE_SHADOW")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
