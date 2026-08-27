"""Research-only Allen-Shahbazyan same-seed trace using OOS-validated expected control duration.

Uses point-in-time EWM top retention x opponent control-allowed history to estimate
expected round control duration conditional on >=1 TD landed.  Individual ground-spell
duration is sampled from the historical positive-TD control-duration ratio distribution,
scaled to the matchup expectation. Escape attempts before the sampled duration fail;
the first escape attempt at/after it succeeds.

RESET_RANGE, IMPROVE_POSITION and ADVANCE_POSITION remain removed in this diagnostic.
Production code is unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.research.allen_shahbazyan_one_path_brain_trace_v1 import (
    FIGHT_ID,
    PATH_ID,
    TraceBrain,
    _enum,
    _standing_rates_no_reset,
)
from pipeline.research.control_duration_oos_validation import _paired, _ewm_prior
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineFunctions, run_causal_path
from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION
from pipeline.simulation.event_clock_mc_v2.calibration.seeds import derive_path_seed
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_dynamic_pressure_shadow as pressure_mod
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_intent_rate_shadow as intent_mod
from pipeline.simulation.event_clock_mc_v2.mechanics.resolver import resolve_action
from pipeline.simulation.event_clock_mc_v2.mechanics.resolution import (
    ActionOutcome,
    ActionResolution,
    TransitionKind,
    TransitionRequest,
)

DATA = Path("data/fight_details/ufc_round_stats.parquet")
MIN_DURATION = 5.0
MAX_DURATION = 300.0


def _expected_control_model(target_date, names):
    df = _paired(pd.read_parquet(DATA))
    hist = df[(df["event_date"] < pd.Timestamp(target_date)) & (df["td_landed"] > 0)].copy()
    if hist.empty:
        raise RuntimeError("no positive-TD control history before target fight")
    global_mean = float(hist["ctrl_sec"].mean())
    ratios = (hist["ctrl_sec"].to_numpy(float) / max(global_mean, 1e-9))
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]

    def matchup(top, bottom):
        top_values = hist.loc[hist["fighter_name"].astype(str) == str(top), "ctrl_sec"].tolist()
        bottom_values = hist.loc[hist["opponent_name"].astype(str) == str(bottom), "ctrl_sec"].tolist()
        top_ewm = _ewm_prior(top_values)
        bottom_ewm = _ewm_prior(bottom_values)
        top_ewm = global_mean if top_ewm is None else float(top_ewm)
        bottom_ewm = global_mean if bottom_ewm is None else float(bottom_ewm)
        expected = float(np.sqrt(max(top_ewm, 0.1) * max(bottom_ewm, 0.1)))
        return {
            "top_ewm_seconds": top_ewm,
            "bottom_allowed_ewm_seconds": bottom_ewm,
            "expected_control_seconds": expected,
            "top_history_n": len(top_values),
            "bottom_allowed_history_n": len(bottom_values),
        }

    return {
        "source": str(DATA),
        "target_date": str(target_date),
        "global_positive_td_mean_seconds": global_mean,
        "empirical_ratio_n": int(len(ratios)),
        "ratios": ratios,
        "matchups": {
            "red_controls_blue": matchup(names[Side.RED], names[Side.BLUE]),
            "blue_controls_red": matchup(names[Side.BLUE], names[Side.RED]),
        },
    }


class ExpectedControlEscapeResolver:
    def __init__(self, model, seed):
        self.model = model
        self.rng = np.random.default_rng(int(seed) ^ 0x5A17C0DE)
        self.spells = {}
        self.escape_checks = []

    def _expected(self, controller):
        key = "red_controls_blue" if controller is Side.RED else "blue_controls_red"
        return float(self.model["matchups"][key]["expected_control_seconds"])

    def _spell(self, state):
        key = (int(state.round_number), state.ground_controller.value, round(float(state.phase_started_at), 9))
        if key not in self.spells:
            ratio = float(self.rng.choice(self.model["ratios"]))
            expected = self._expected(state.ground_controller)
            duration = float(np.clip(expected * ratio, MIN_DURATION, MAX_DURATION))
            self.spells[key] = {
                "round": int(state.round_number),
                "controller": state.ground_controller.value,
                "phase_started_at": float(state.phase_started_at),
                "expected_control_seconds": expected,
                "empirical_duration_ratio": ratio,
                "sampled_escape_threshold_seconds": duration,
            }
        return self.spells[key]

    def __call__(self, event, state, inputs, rng, placeholders, ko_kd_rng=None, submission_rng=None):
        if event.action_family is not ActionFamily.ESCAPE_STAND:
            return resolve_action(event, state, inputs, rng, placeholders, ko_kd_rng, submission_rng)
        spell = self._spell(state)
        elapsed = float(state.fight_time_seconds - state.phase_started_at)
        succeeded = elapsed >= float(spell["sampled_escape_threshold_seconds"])
        self.escape_checks.append({
            "timestamp": float(event.timestamp_seconds),
            "actor": event.actor.value,
            "controller": state.ground_controller.value,
            "elapsed_control_seconds": elapsed,
            **spell,
            "success": bool(succeeded),
        })
        return ActionResolution(
            event,
            ActionOutcome.ESCAPED if succeeded else ActionOutcome.FAILURE,
            TransitionRequest(TransitionKind.ESCAPE_GROUND, Phase.GROUND, Phase.STANDING) if succeeded else None,
        )


def main():
    pressure_mod.FIGHT_ID = FIGHT_ID
    pressure_mod.PATHS = 1
    intent_mod._standing_rates = _standing_rates_no_reset
    fight, inputs, priors, horizon, cfg = pressure_mod.build_setup()
    names = {Side.RED: str(fight.r_name), Side.BLUE: str(fight.b_name)}
    target_date = getattr(fight, "date", None) or getattr(fight, "event_date", None)
    if target_date is None:
        raise RuntimeError("fight date unavailable")

    seed = derive_path_seed(SEED_SET_VERSION, FIGHT_ID, PATH_ID)
    model = _expected_control_model(target_date, names)
    ratios = model.pop("ratios")
    resolver_model = {**model, "ratios": ratios}
    brain = TraceBrain(inputs, priors, horizon)
    escape_resolver = ExpectedControlEscapeResolver(resolver_model, seed)
    funcs = EngineFunctions(
        timing_sampler=brain.timing_sampler,
        action_chooser=brain.action_chooser,
        mechanics_resolver=escape_resolver,
    )
    out = run_causal_path(inputs, seed=seed, horizon_seconds=horizon, config=cfg, functions=funcs)
    if len(brain.decisions) != len(out.events):
        raise RuntimeError(f"decision/event mismatch: {len(brain.decisions)} != {len(out.events)}")

    escape_by_key = {(round(x["timestamp"], 9), x["actor"]): x for x in escape_resolver.escape_checks}
    events = []
    for d, e in zip(brain.decisions, out.events, strict=True):
        events.append({
            **d,
            "actor_name": names[e.actor],
            "event_timestamp": float(e.timestamp_seconds),
            "outcome": e.outcome.value,
            "resulting_phase": e.resulting_phase.value,
            "resulting_controller": _enum(e.resulting_controller),
            "escape_model": escape_by_key.get((round(float(e.timestamp_seconds), 9), e.actor.value)),
            "submission_attempt": bool(e.submission_attempt),
            "submission_probability": float(e.submission_probability),
            "submission_success": bool(e.submission_success),
            "knockdown": bool(e.knockdown),
            "ko_tko": bool(e.ko_tko),
        })

    printable_model = {k: v for k, v in model.items()}
    payload = {
        "study": "Allen-Shahbazyan same-seed OOS-validated expected-control escape trace",
        "production_changed": False,
        "fight_id": FIGHT_ID,
        "path_id": PATH_ID,
        "seed": seed,
        "red": names[Side.RED],
        "blue": names[Side.BLUE],
        "control_model": printable_model,
        "sampled_ground_spells": list(escape_resolver.spells.values()),
        "escape_checks": escape_resolver.escape_checks,
        "termination": None if out.termination is None else {
            "winner": names[out.termination.winner],
            "winner_side": out.termination.winner.value,
            "method": out.termination.finish_method.value,
            "reported_through_seconds": out.reported_through_seconds,
        },
        "timeline_segments": [
            {"start": s.start_time, "end": s.end_time, "duration": s.duration, "phase": s.phase.value,
             "controller": _enum(s.controller), "controller_name": None if s.controller is None else names[s.controller],
             "entry_reason": s.entry_reason, "exit_reason": s.exit_reason}
            for s in out.timeline_segments
        ],
        "events": events,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
