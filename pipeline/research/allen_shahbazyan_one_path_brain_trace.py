"""Research-only one-path Brain trace for Brendan Allen vs Edmen Shahbazyan.
Prints every Brain decision distribution and every resulting causal event.
Production mechanics are unchanged.
"""
from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np

from pipeline.simulation.event_clock_mc_v2.brain.intent_priors import action_probabilities_with_intent_priors
from pipeline.simulation.event_clock_mc_v2.brain.policy import action_utilities
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineFunctions, run_causal_path
from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION
from pipeline.simulation.event_clock_mc_v2.calibration.seeds import derive_path_seed
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_dynamic_pressure_shadow as pressure_mod
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_intent_rate_shadow as intent_mod

FIGHT_ID = "419fff06f338f5c6"
PATH_ID = 0


class TraceIntentRateBrain(intent_mod.IntentRateBrain):
    def __init__(self, inputs, priors, horizon):
        super().__init__(inputs, priors, horizon)
        self.decisions = []

    def action_chooser(self, state, actor, capabilities, context, rng, config):
        if state.phase is Phase.STANDING:
            rates, pressure = intent_mod._standing_rates(
                state, actor, capabilities, context, self.priors[actor], config
            )
            actions = tuple(rates)
            weights = np.asarray([rates[a] for a in actions], dtype=float)
            probs = weights / weights.sum()
            utilities = dict(action_utilities(state, actor, capabilities, context))
            idx = int(rng.choice(len(actions), p=probs))
            selected = actions[idx]
            rows = [
                {
                    "action": a.value,
                    "utility": float(utilities.get(a, 0.0)),
                    "live_rate_15m": float(rates[a]),
                    "probability": float(p),
                }
                for a, p in zip(actions, probs, strict=True)
            ]
            mode = "standing_fsr_intent_rate"
        else:
            dist = action_probabilities_with_intent_priors(
                state, actor, capabilities, context, self.priors[actor], config
            )
            probs = [row.probability for row in dist]
            idx = int(rng.choice(len(dist), p=probs))
            selected = dist[idx].action_family
            rows = [
                {
                    "action": row.action_family.value,
                    "utility": float(row.utility),
                    "probability": float(row.probability),
                }
                for row in dist
            ]
            pressure = None
            mode = "intent_prior_softmax"

        self.decisions.append(
            {
                "timestamp_seconds": float(state.fight_time_seconds),
                "round": int(state.round_number),
                "actor": actor.value,
                "phase": state.phase.value,
                "ground_controller": state.ground_controller.value if state.ground_controller else None,
                "clinch_controller": state.clinch_controller.value if state.clinch_controller else None,
                "mode": mode,
                "dynamic_pressure": None if pressure is None else float(pressure),
                "context": asdict(context),
                "probabilities": rows,
                "selected_action": selected.value,
            }
        )
        return selected


def side_inputs(fight, inputs, priors, side):
    fighter = inputs.fighter(side)
    name = str(fight.r_name if side is Side.RED else fight.b_name)
    return {
        "fighter": name,
        "side": side.value,
        "brain_capabilities": asdict(fighter.capabilities),
        "intent_priors": asdict(priors[side]),
        "mechanics": {
            "standing_strike_landing_probability": fighter.mechanics.standing_strike_landing_probability,
            "takedown_completion_probability": fighter.mechanics.takedown_completion_probability,
            "ground_strike_landing_probability": fighter.mechanics.ground_strike_landing_probability,
            "ground_escape_probability": fighter.mechanics.ground_escape_probability,
            "ground_reversal_probability": fighter.mechanics.ground_reversal_probability,
            "submission_conversion_baseline": fighter.mechanics.submission_conversion_baseline,
            "submission_offense": fighter.mechanics.submission_offense,
            "submission_defense": fighter.mechanics.submission_defense,
            "submission_conversion_offset": fighter.mechanics.submission_conversion_offset,
        },
    }


def main():
    pressure_mod.FIGHT_ID = FIGHT_ID
    intent_mod.FIGHT_ID = FIGHT_ID
    fight, inputs, priors, horizon, cfg = pressure_mod.build_setup()
    brain = TraceIntentRateBrain(inputs, priors, horizon)
    funcs = EngineFunctions(
        timing_sampler=brain.timing_sampler,
        action_chooser=brain.action_chooser,
    )
    seed = derive_path_seed(SEED_SET_VERSION, FIGHT_ID, PATH_ID)
    out = run_causal_path(
        inputs,
        seed=seed,
        horizon_seconds=horizon,
        config=cfg,
        functions=funcs,
    )
    if len(brain.decisions) != len(out.events):
        raise RuntimeError(
            f"decision/event mismatch: {len(brain.decisions)} vs {len(out.events)}"
        )

    names = {Side.RED: str(fight.r_name), Side.BLUE: str(fight.b_name)}
    events = []
    for i, (decision, ev) in enumerate(zip(brain.decisions, out.events, strict=True), start=1):
        row = dict(decision)
        row["event_number"] = i
        row["actor_name"] = names[ev.actor]
        row["mechanics_outcome"] = ev.outcome.value
        row["transition_kind"] = ev.transition_kind.value if ev.transition_kind else None
        row["resulting_phase"] = ev.resulting_phase.value
        row["resulting_controller"] = (
            ev.resulting_controller.value if ev.resulting_controller else None
        )
        row["strike"] = {
            "impact": ev.impact,
            "knockdown": ev.knockdown,
            "ko_probability": ev.ko_probability,
            "kd_probability": ev.kd_probability,
            "ko_tko": ev.ko_tko,
        }
        row["submission"] = {
            "attempt": ev.submission_attempt,
            "conversion_probability": ev.submission_probability,
            "success": ev.submission_success,
        }
        events.append(row)

    segments = [
        {
            "start": seg.start_time,
            "end": seg.end_time,
            "duration": seg.duration,
            "phase": seg.phase.value,
            "controller": seg.controller.value if seg.controller else None,
            "entry_reason": seg.entry_reason,
            "exit_reason": seg.exit_reason,
        }
        for seg in out.timeline_segments
    ]

    payload = {
        "diagnostic": "Allen-Shahbazyan one-path full Brain probability/event trace",
        "production_changed": False,
        "fight_id": FIGHT_ID,
        "path_id": PATH_ID,
        "seed_set": SEED_SET_VERSION,
        "seed": seed,
        "red": side_inputs(fight, inputs, priors, Side.RED),
        "blue": side_inputs(fight, inputs, priors, Side.BLUE),
        "events": events,
        "timeline_segments": segments,
        "termination": None if out.termination is None else {
            "winner": names[out.termination.winner],
            "winner_side": out.termination.winner.value,
            "method": out.termination.finish_method.value,
            "reported_through_seconds": out.reported_through_seconds,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
