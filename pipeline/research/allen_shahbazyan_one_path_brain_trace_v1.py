"""Research-only Allen-Shahbazyan trace with empirical control-time escape.

Research changes only:
- RESET_RANGE removed from standing intent clock/chooser.
- IMPROVE_POSITION and ADVANCE_POSITION removed from ground choices.
- ESCAPE_STAND success is driven by elapsed continuous ground-control time and
  a historical UFCStats control-duration model instead of the flat 0.40 roll.

Production Brain and mechanics remain unchanged.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import weibull_min

from pipeline.simulation.event_clock_mc_v2.brain.intent_priors import action_probabilities_with_intent_priors
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineFunctions, run_causal_path
from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION
from pipeline.simulation.event_clock_mc_v2.calibration.seeds import derive_path_seed
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_dynamic_pressure_shadow as pressure_mod
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_intent_rate_shadow as intent_mod
from pipeline.simulation.event_clock_mc_v2.diagnostics.leavitt_brito_intent_rate_shadow import IntentRateBrain, _standing_rates as _standing_rates_original
from pipeline.simulation.event_clock_mc_v2.mechanics.resolver import resolve_action
from pipeline.simulation.event_clock_mc_v2.mechanics.resolution import ActionOutcome, ActionResolution, TransitionKind, TransitionRequest

FIGHT_ID = "419fff06f338f5c6"
PATH_ID = 0
ROUND_STATS = Path("data/fight_details/ufc_round_stats.parquet")
REMOVED_GROUND_ACTIONS = {ActionFamily.IMPROVE_POSITION, ActionFamily.ADVANCE_POSITION}
ESCAPE_WINDOW_SECONDS = 5.0
MIN_SCALE_SECONDS = 8.0
MAX_SCALE_SECONDS = 180.0


def _enum(v):
    return None if v is None else getattr(v, "value", str(v))


def _col(df, *names):
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    raise KeyError(f"none of columns {names!r} found; columns={list(df.columns)!r}")


def _to_seconds(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    def one(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        if ":" in s:
            parts = s.split(":")
            try:
                if len(parts) == 2:
                    return 60.0 * float(parts[0]) + float(parts[1])
                if len(parts) == 3:
                    return 3600.0 * float(parts[0]) + 60.0 * float(parts[1]) + float(parts[2])
            except ValueError:
                return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan
    return series.map(one)


def _historical_control_model(target_date, fighter_names):
    df = pd.read_parquet(ROUND_STATS)
    name_col = _col(df, "fighter_name", "fighter", "name")
    opp_col = _col(df, "opponent_name", "opponent")
    ctrl_col = _col(df, "control_seconds", "ctrl_seconds", "ctrl", "control")
    td_col = _col(df, "takedowns_landed", "td_landed", "td_l", "takedown_landed")
    date_col = _col(df, "date", "event_date", "fight_date")

    work = df[[name_col, opp_col, ctrl_col, td_col, date_col]].copy()
    work["date"] = pd.to_datetime(work[date_col], errors="coerce")
    work = work[work["date"] < pd.Timestamp(target_date)].copy()
    work["ctrl"] = _to_seconds(work[ctrl_col]).fillna(0.0).clip(lower=0.0)
    work["td"] = pd.to_numeric(work[td_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    # UFCStats only exposes round aggregate control, not individual spells. For
    # this diagnostic, each round with control is represented by control seconds
    # divided by max(1, takedowns landed) as a spell-duration proxy.
    pos = work[work["ctrl"] > 0].copy()
    pos["spell_proxy"] = pos["ctrl"] / np.maximum(pos["td"], 1.0)
    durations = pos["spell_proxy"].to_numpy(dtype=float)
    durations = durations[np.isfinite(durations) & (durations > 0.25) & (durations <= 300.0)]
    if len(durations) < 100:
        raise RuntimeError(f"insufficient positive control observations: {len(durations)}")

    shape, _, global_scale = weibull_min.fit(durations, floc=0.0)
    shape = float(np.clip(shape, 0.50, 3.00))
    global_scale = float(np.clip(global_scale, MIN_SCALE_SECONDS, MAX_SCALE_SECONDS))

    def retained_scale(top_name, bottom_name):
        top = work[work[name_col].astype(str) == str(top_name)]
        allowed = work[work[opp_col].astype(str) == str(bottom_name)]

        def ratio(frame):
            ctrl = float(frame["ctrl"].sum())
            denom = float(np.maximum(frame["td"], 1.0).sum())
            return ctrl / denom if denom > 0 else np.nan

        top_raw = ratio(top)
        bottom_raw = ratio(allowed)
        # Exposure shrinkage toward global scale. Each TD-equivalent contributes
        # one unit; 10 pseudo-observations stabilize sparse fighters.
        def shrink(raw, frame):
            n = float(np.maximum(frame["td"], 1.0).sum())
            if not np.isfinite(raw):
                return global_scale
            return (n * raw + 10.0 * global_scale) / (n + 10.0)

        top_scale = shrink(top_raw, top)
        bottom_scale = shrink(bottom_raw, allowed)
        matchup = math.sqrt(max(top_scale, 0.1) * max(bottom_scale, 0.1))
        return {
            "top_raw_seconds": None if not np.isfinite(top_raw) else float(top_raw),
            "bottom_allowed_raw_seconds": None if not np.isfinite(bottom_raw) else float(bottom_raw),
            "top_shrunk_seconds": float(top_scale),
            "bottom_allowed_shrunk_seconds": float(bottom_scale),
            "matchup_scale_seconds": float(np.clip(matchup, MIN_SCALE_SECONDS, MAX_SCALE_SECONDS)),
            "top_exposure_units": float(np.maximum(top["td"], 1.0).sum()),
            "bottom_allowed_exposure_units": float(np.maximum(allowed["td"], 1.0).sum()),
        }

    red, blue = fighter_names[Side.RED], fighter_names[Side.BLUE]
    matchups = {
        "red_controls_blue": retained_scale(red, blue),
        "blue_controls_red": retained_scale(blue, red),
    }
    return {
        "source": str(ROUND_STATS),
        "observations": int(len(durations)),
        "weibull_shape": shape,
        "global_scale_seconds": global_scale,
        "escape_window_seconds": ESCAPE_WINDOW_SECONDS,
        "matchups": matchups,
    }


def _standing_rates_no_reset(state, actor, capabilities, context, priors, config):
    rates, pressure = _standing_rates_original(state, actor, capabilities, context, priors, config)
    rates = dict(rates)
    rates.pop(ActionFamily.RESET_RANGE, None)
    return rates, pressure


class TraceBrain(IntentRateBrain):
    def __init__(self, inputs, priors, horizon):
        super().__init__(inputs, priors, horizon)
        self.decisions = []

    def action_chooser(self, state, actor, capabilities, context, rng, config):
        if state.phase is Phase.STANDING:
            rates, pressure = _standing_rates_no_reset(state, actor, capabilities, context, self.priors[actor], config)
            actions = tuple(rates)
            weights = np.asarray([rates[a] for a in actions], dtype=float)
            probs = weights / weights.sum()
            rows = [
                {"action": a.value, "rate_15m": float(rates[a]), "probability": float(p)}
                for a, p in zip(actions, probs, strict=True)
            ]
            selected = actions[int(rng.choice(len(actions), p=probs))]
            extra = {"dynamic_pressure": float(pressure)}
        else:
            raw = action_probabilities_with_intent_priors(state, actor, capabilities, context, self.priors[actor], config)
            dist = [r for r in raw if r.action_family not in REMOVED_GROUND_ACTIONS]
            if not dist:
                raise RuntimeError("all non-standing Brain actions were filtered")
            weights = np.asarray([r.probability for r in dist], dtype=float)
            probs = weights / weights.sum()
            rows = [
                {"action": r.action_family.value, "utility": float(r.utility), "probability": float(p)}
                for r, p in zip(dist, probs, strict=True)
            ]
            selected = dist[int(rng.choice(len(dist), p=probs))].action_family
            extra = {}
        self.decisions.append({
            "decision_index": len(self.decisions),
            "timestamp_before_action": float(state.fight_time_seconds),
            "round": int(state.round_number),
            "phase": state.phase.value,
            "phase_started_at": float(state.phase_started_at),
            "ground_control_elapsed_seconds": float(state.fight_time_seconds - state.phase_started_at) if state.phase is Phase.GROUND else None,
            "ground_controller": _enum(state.ground_controller),
            "clinch_controller": _enum(state.clinch_controller),
            "actor": actor.value,
            "context": asdict(context),
            "brain_options": rows,
            "selected_action": selected.value,
            **extra,
        })
        return selected


class ControlTimeEscapeResolver:
    def __init__(self, model):
        self.model = model
        self.escape_checks = []

    def _scale_for_state(self, state):
        if state.ground_controller is Side.RED:
            return self.model["matchups"]["red_controls_blue"]["matchup_scale_seconds"]
        return self.model["matchups"]["blue_controls_red"]["matchup_scale_seconds"]

    def probability(self, state):
        elapsed = max(0.0, float(state.fight_time_seconds - state.phase_started_at))
        shape = float(self.model["weibull_shape"])
        scale = float(self._scale_for_state(state))
        # Conditional probability that a historically fitted control spell ends
        # in the next ESCAPE_WINDOW_SECONDS, given survival through elapsed time.
        t0 = elapsed
        t1 = elapsed + ESCAPE_WINDOW_SECONDS
        s0 = math.exp(-((t0 / scale) ** shape))
        s1 = math.exp(-((t1 / scale) ** shape))
        p = 1.0 - (s1 / max(s0, 1e-12))
        return float(np.clip(p, 0.01, 0.95)), elapsed, scale

    def __call__(self, event, state, inputs, rng, placeholders, ko_kd_rng=None, submission_rng=None):
        if event.action_family is not ActionFamily.ESCAPE_STAND:
            return resolve_action(event, state, inputs, rng, placeholders, ko_kd_rng, submission_rng)
        p, elapsed, scale = self.probability(state)
        succeeded = bool(rng.random() < p)
        self.escape_checks.append({
            "timestamp": float(event.timestamp_seconds),
            "actor": event.actor.value,
            "controller": _enum(state.ground_controller),
            "elapsed_control_seconds": elapsed,
            "matchup_scale_seconds": scale,
            "escape_probability": p,
            "success": succeeded,
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
        raise RuntimeError("fight object does not expose date/event_date for point-in-time control model")
    control_model = _historical_control_model(target_date, names)

    brain = TraceBrain(inputs, priors, horizon)
    escape_resolver = ControlTimeEscapeResolver(control_model)
    funcs = EngineFunctions(
        timing_sampler=brain.timing_sampler,
        action_chooser=brain.action_chooser,
        mechanics_resolver=escape_resolver,
    )
    seed = derive_path_seed(SEED_SET_VERSION, FIGHT_ID, PATH_ID)
    out = run_causal_path(inputs, seed=seed, horizon_seconds=horizon, config=cfg, functions=funcs)
    if len(brain.decisions) != len(out.events):
        raise RuntimeError(f"decision/event mismatch: {len(brain.decisions)} != {len(out.events)}")

    escape_by_key = {(round(x["timestamp"], 9), x["actor"]): x for x in escape_resolver.escape_checks}
    trace = []
    for d, e in zip(brain.decisions, out.events, strict=True):
        key = (round(float(e.timestamp_seconds), 9), e.actor.value)
        escape = escape_by_key.get(key)
        trace.append({
            **d,
            "actor_name": names[e.actor],
            "event_timestamp": float(e.timestamp_seconds),
            "source_phase": e.source_phase.value,
            "outcome": e.outcome.value,
            "transition_kind": _enum(e.transition_kind),
            "resulting_phase": e.resulting_phase.value,
            "resulting_controller": _enum(e.resulting_controller),
            "escape_model": escape,
            "impact": float(e.impact),
            "ko_probability": float(e.ko_probability),
            "kd_probability": float(e.kd_probability),
            "knockdown": bool(e.knockdown),
            "ko_tko": bool(e.ko_tko),
            "submission_attempt": bool(e.submission_attempt),
            "submission_probability": float(e.submission_probability),
            "submission_success": bool(e.submission_success),
        })

    payload = {
        "study": "Allen-Shahbazyan same-seed control-time escape prototype",
        "production_changed": False,
        "reset_range_removed": True,
        "improve_position_removed": True,
        "advance_position_removed": True,
        "flat_escape_probability_removed": True,
        "fight_id": FIGHT_ID,
        "path_id": PATH_ID,
        "seed_set": SEED_SET_VERSION,
        "seed": seed,
        "red": names[Side.RED],
        "blue": names[Side.BLUE],
        "target_date": str(target_date),
        "control_time_escape_model": control_model,
        "termination": None if out.termination is None else {
            "winner": names[out.termination.winner],
            "winner_side": out.termination.winner.value,
            "method": out.termination.finish_method.value,
            "reported_through_seconds": out.reported_through_seconds,
        },
        "timeline_segments": [
            {
                "start": s.start_time,
                "end": s.end_time,
                "duration": s.duration,
                "phase": s.phase.value,
                "controller": _enum(s.controller),
                "controller_name": None if s.controller is None else names[s.controller],
                "entry_reason": s.entry_reason,
                "exit_reason": s.exit_reason,
            }
            for s in out.timeline_segments
        ],
        "escape_checks": escape_resolver.escape_checks,
        "events": trace,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
