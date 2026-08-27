"""Schnell-Costa KO V3 opportunity funnel audit.

Research-only diagnostic. Re-runs the 500-path two-thirds post-KD hurt shadow
and measures where Costa's KO opportunity is being lost:

    actions -> strike attempts -> landed strikes -> KD-hazard rolls -> KDs -> KO/TKOs

It also compares simulation strike/KD rates with strictly-prefight UFCStats
history for Costa offense and Schnell opponent-allowed defense.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import shutil

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineFunctions, run_causal_path
from pipeline.simulation.event_clock_mc_v2.mechanics.resolution import ActionOutcome
from pipeline.simulation.event_clock_mc_v2.mechanics.resolver import resolve_action
from pipeline.simulation.event_clock_mc_v2.diagnostics import schnell_costa_ko_v3_from_scratch_shadow as base
from pipeline.simulation.event_clock_mc_v2.diagnostics import schnell_costa_ko_v3_hurt_followup_shadow as hurt

STRIKES = hurt.STRIKES
STANDING_STRIKES = frozenset({ActionFamily.STAND_ATTACK, ActionFamily.STAND_COUNTER})


def _duration_map(master: pd.DataFrame) -> dict[str, float]:
    cols = set(master.columns)
    if not {"fight_id", "finish_round", "match_time_sec"}.issubset(cols):
        return {}
    r = pd.to_numeric(master["finish_round"], errors="coerce")
    t = pd.to_numeric(master["match_time_sec"], errors="coerce")
    elapsed = (r - 1.0) * 300.0 + t
    return dict(zip(master["fight_id"].astype(str), elapsed.astype(float)))


def _ewm_summary(fights: pd.DataFrame, decay: float = 0.50) -> dict:
    fields = ["sig_attempted", "sig_landed", "kd"]
    state = {k: 0.0 for k in fields}
    for row in fights.sort_values(["event_date", "fight_id"]).itertuples(index=False):
        for k in fields:
            state[k] = decay * state[k] + float(getattr(row, k))
    return state


def _historical_reference(fight_id: str, fighter_ids: dict[Side, str], names: dict[Side, str]) -> dict:
    cols = ["event_date", "fight_id", "fighter_id", "opponent_id", "round", "kd", "sig_str_landed", "sig_str_attempted"]
    rr = pd.read_parquet(ROUND_STATS_PATH, columns=cols).copy()
    rr["event_date"] = pd.to_datetime(rr["event_date"]).dt.normalize()
    for c in ("fight_id", "fighter_id", "opponent_id"):
        rr[c] = rr[c].astype(str)
    target_date = rr.loc[rr.fight_id.eq(str(fight_id)), "event_date"].iloc[0]
    prior = rr[rr.event_date < target_date].copy()
    agg = prior.groupby(["event_date", "fight_id", "fighter_id", "opponent_id"], as_index=False).agg(
        kd=("kd", "sum"), sig_landed=("sig_str_landed", "sum"), sig_attempted=("sig_str_attempted", "sum")
    )
    master = pd.read_parquet(MASTER_PATH).copy()
    master["fight_id"] = master["fight_id"].astype(str)
    durations = _duration_map(master)

    out = {}
    for side in Side:
        fid = fighter_ids[side]
        own = agg[agg.fighter_id.eq(fid)].copy()
        faced = agg[agg.opponent_id.eq(fid)].copy()  # opponent offense against this fighter
        own["elapsed_sec"] = own.fight_id.map(durations)
        faced["elapsed_sec"] = faced.fight_id.map(durations)

        def block(df: pd.DataFrame) -> dict:
            fights = int(df.fight_id.nunique())
            att = float(df.sig_attempted.sum()); land = float(df.sig_landed.sum()); kd = float(df.kd.sum())
            sec = float(df.elapsed_sec.dropna().sum()) if "elapsed_sec" in df else 0.0
            return {
                "fights": fights,
                "sig_attempts_per_fight": att / fights if fights else None,
                "sig_landed_per_fight": land / fights if fights else None,
                "sig_accuracy": land / att if att > 0 else None,
                "kd_per_fight": kd / fights if fights else None,
                "kd_per_landed": kd / land if land > 0 else None,
                "sig_attempts_per_15m": att / sec * 900.0 if sec > 0 else None,
                "sig_landed_per_15m": land / sec * 900.0 if sec > 0 else None,
                "kd_per_15m": kd / sec * 900.0 if sec > 0 else None,
                "elapsed_seconds_with_duration": sec,
            }

        own_ewm = _ewm_summary(own)
        faced_ewm = _ewm_summary(faced)
        out[names[side]] = {
            "career_offense": block(own),
            "career_opponents_against": block(faced),
            "ewm50_offense": {
                "sig_attempted": own_ewm["sig_attempted"],
                "sig_landed": own_ewm["sig_landed"],
                "kd": own_ewm["kd"],
                "sig_accuracy": own_ewm["sig_landed"] / own_ewm["sig_attempted"] if own_ewm["sig_attempted"] > 0 else None,
                "kd_per_landed": own_ewm["kd"] / own_ewm["sig_landed"] if own_ewm["sig_landed"] > 0 else None,
            },
            "ewm50_opponents_against": {
                "sig_attempted": faced_ewm["sig_attempted"],
                "sig_landed": faced_ewm["sig_landed"],
                "kd": faced_ewm["kd"],
                "sig_accuracy": faced_ewm["sig_landed"] / faced_ewm["sig_attempted"] if faced_ewm["sig_attempted"] > 0 else None,
                "kd_per_landed": faced_ewm["kd"] / faced_ewm["sig_landed"] if faced_ewm["sig_landed"] > 0 else None,
            },
        }
    return {"target_date": str(target_date.date()), "fighters": out}


def main():
    fight_id = base.resolve_fight_id()
    hazards_by_id = base.fit_prefight_hazards(fight_id=fight_id)

    canonical = pd.read_parquet(base.FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    canonical["event_date"] = pd.to_datetime(canonical["event_date"]).dt.normalize()
    canonical["fight_id"] = canonical["fight_id"].astype(str)
    canonical["fighter_id"] = canonical["fighter_id"].astype(str)
    ewm50 = base.build_pure_ewm50_snapshot(canonical)

    shutil.copy2(base.FSR_V3_PREFIGHT_SNAPSHOTS_PATH, base.BACKUP_PATH)
    original_standing_rates = base.intent_mod._standing_rates
    original_empirical_resolver = base.physiology_mod.resolve_empirical_ko_kd
    original_hurt_increment = base.physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT
    try:
        ewm50.to_parquet(base.FSR_V3_PREFIGHT_SNAPSHOTS_PATH, index=False)
        base.pressure_mod.FIGHT_ID = fight_id; base.pressure_mod.PATHS = base.PATHS
        base.intent_mod.FIGHT_ID = fight_id; base.intent_mod.PATHS = base.PATHS

        def calibrated_standing_rates(state, actor, capabilities, context, priors, config):
            rates, pressure = original_standing_rates(state, actor, capabilities, context, priors, config)
            rates = dict(rates); rates[ActionFamily.STAND_ATTACK] *= base.STANDING_ATTEMPT_SCALE
            return rates, pressure

        base.intent_mod._standing_rates = calibrated_standing_rates
        fight, inputs, priors, horizon, cfg = base.pressure_mod.build_setup()
        side_to_id = {Side.RED: str(fight.r_id), Side.BLUE: str(fight.b_id)}
        names = {Side.RED: str(fight.r_name), Side.BLUE: str(fight.b_name)}
        hazards_by_side = {side: hazards_by_id[fid] for side, fid in side_to_id.items()}
        resolver = hurt.HurtFollowupResolver(hazards_by_side)
        base.physiology_mod.resolve_empirical_ko_kd = resolver
        base.physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT = 0.0
        brain = base.intent_mod.IntentRateBrain(inputs, priors, horizon)

        def action_chooser(state, actor, capabilities, context, rng, config):
            return brain.action_chooser(state, actor, capabilities, resolver.brain_context(state, actor, context), rng, config)

        def mechanics_resolver(event, state, mechanics_inputs, rng, placeholders, ko_kd_rng, submission_rng):
            resolver.clear_if_new_round(state)
            out = resolve_action(event, state, mechanics_inputs, rng, placeholders, ko_kd_rng, submission_rng)
            resolver.observe_resolution(event, out)
            return out

        funcs = EngineFunctions(timing_sampler=brain.timing_sampler, action_chooser=action_chooser, mechanics_resolver=mechanics_resolver)
        counts = {side: Counter() for side in Side}
        action_family_counts = {side: Counter() for side in Side}
        phase_action_counts = {side: Counter() for side in Side}
        phase_seconds = Counter()
        total_exposure_seconds = 0.0
        sixway = Counter()

        for path_id in range(base.PATHS):
            seed = base.derive_path_seed(base.SEED_SET_VERSION, fight_id, path_id)
            out = run_causal_path(inputs, seed=seed, horizon_seconds=horizon, config=cfg, functions=funcs)
            total_exposure_seconds += float(out.reported_through_seconds)
            for seg in out.timeline_segments:
                phase_seconds[seg.phase.value] += float(seg.duration)
            for e in out.events:
                c = counts[e.actor]
                c["actions"] += 1
                action_family_counts[e.actor][e.selected_action.value] += 1
                phase_action_counts[e.actor][e.source_phase.value] += 1
                if e.selected_action in STRIKES:
                    c["strike_attempts"] += 1
                    if e.source_phase is Phase.STANDING:
                        c["standing_phase_strike_attempts"] += 1
                    if e.selected_action in STANDING_STRIKES:
                        c["standing_family_attempts"] += 1
                    if e.outcome is ActionOutcome.LANDED:
                        c["landed_strikes"] += 1
                        if e.source_phase is Phase.STANDING:
                            c["standing_phase_landed"] += 1
                if e.knockdown:
                    c["event_recorded_kds"] += 1
                if e.ko_tko:
                    c["event_recorded_kos"] += 1
            if out.termination is not None:
                sixway[(out.termination.winner.value, out.termination.finish_method.value)] += 1

        sim = {}
        for side in Side:
            c = counts[side]
            rs = resolver.summary(side)
            kd_rolls = int(rs["landed_strike_resolutions"] - rs["direct_finishes"] - rs["followup_finishes"])
            mins15 = total_exposure_seconds / 900.0
            sim[names[side]] = {
                "actions": int(c["actions"]),
                "actions_per_path": c["actions"] / base.PATHS,
                "actions_per_15m": c["actions"] / mins15 if mins15 > 0 else None,
                "strike_attempts": int(c["strike_attempts"]),
                "strike_attempts_per_path": c["strike_attempts"] / base.PATHS,
                "strike_attempts_per_15m": c["strike_attempts"] / mins15 if mins15 > 0 else None,
                "standing_phase_strike_attempts": int(c["standing_phase_strike_attempts"]),
                "standing_family_attempts": int(c["standing_family_attempts"]),
                "landed_strikes": int(c["landed_strikes"]),
                "landed_per_path": c["landed_strikes"] / base.PATHS,
                "landed_per_15m": c["landed_strikes"] / mins15 if mins15 > 0 else None,
                "sim_strike_accuracy": c["landed_strikes"] / c["strike_attempts"] if c["strike_attempts"] else None,
                "kd_hazard_rolls": kd_rolls,
                "knockdowns": int(rs["knockdowns"]),
                "kd_per_path": rs["knockdowns"] / base.PATHS,
                "kd_per_15m": rs["knockdowns"] / mins15 if mins15 > 0 else None,
                "kd_per_landed": rs["knockdowns"] / rs["landed_strike_resolutions"] if rs["landed_strike_resolutions"] else None,
                "direct_finishes": int(rs["direct_finishes"]),
                "followup_finishes": int(rs["followup_finishes"]),
                "ko_tko_wins": int(sixway[(side.value, "ko_tko")]),
                "ko_tko_probability": sixway[(side.value, "ko_tko")] / base.PATHS,
                "action_family_counts": dict(action_family_counts[side]),
                "source_phase_action_counts": dict(phase_action_counts[side]),
            }

        payload = {
            "diagnostic": "Schnell-Costa KO V3 opportunity funnel audit",
            "fight_id": fight_id,
            "paths": base.PATHS,
            "standing_attempt_scale": base.STANDING_ATTEMPT_SCALE,
            "total_simulated_exposure_seconds": total_exposure_seconds,
            "mean_exposure_seconds_per_path": total_exposure_seconds / base.PATHS,
            "phase_seconds": dict(phase_seconds),
            "phase_share": {k: v / total_exposure_seconds for k, v in phase_seconds.items()},
            "simulation": sim,
            "historical_prefight": _historical_reference(fight_id, side_to_id, names),
            "production_changed": False,
        }
        print("SCHNELL_COSTA_KO_V3_OPPORTUNITY_FUNNEL")
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        base.physiology_mod.resolve_empirical_ko_kd = original_empirical_resolver
        base.physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT = original_hurt_increment
        base.intent_mod._standing_rates = original_standing_rates
        shutil.move(base.BACKUP_PATH, base.FSR_V3_PREFIGHT_SNAPSHOTS_PATH)


if __name__ == "__main__":
    main()
