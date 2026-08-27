"""Schnell vs Costa pure-EWM 0.50 shadow using fitted KO hazard v2 model D.

Research only. Keeps Brain policy, strike generation, KD hazard, submissions,
judging, and all non-KO mechanics frozen. Replaces only the direct per-landed-
strike KO probability with coefficients fit chronologically in
run_event2_ko_hazard_v2_study.py model D (+ defender KD resistance + durability).
"""
from __future__ import annotations

from collections import Counter
import json
import shutil
from math import exp
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
import pipeline.research.fsr_recency_cohort_shadow as recency
from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION
from pipeline.simulation.event_clock_mc_v2.calibration.seeds import derive_path_seed
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineFunctions, run_causal_path
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_dynamic_pressure_shadow as pressure_mod
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_intent_rate_shadow as intent_mod
from pipeline.simulation.event_clock_mc_v2.mechanics import physiology as physiology_mod
from pipeline.simulation.event_clock_mc_v2.mechanics import ko_kd_empirical as ko_mod

# Manual workflow trigger after KD->hurt bridge commit 63674b2.
PATHS = 500
EWM_DECAY = 0.50
STANDING_ATTEMPT_SCALE = 0.25
BACKUP_PATH = Path("data/fsr_v3/fsr_v3_prefight_snapshots.canonical_backup.parquet")
MASTER_PATH = Path("data/master/ufc_master.parquet")
EVENT_DATE = pd.Timestamp("2026-06-06")
FIGHTER_A = "Matt Schnell"
FIGHTER_B = "Alessandro Costa"

# Chronological train/holdout grouped-significant-strike fit, model D.
KO_INTERCEPT = -5.39477498972103
KO_POWER_BETA = 0.01737017625766978
KO_ATTACKER_AGE_BETA = -0.019689536918568947
KO_DEFENDER_AGE_BETA = 0.05740492851521785
KO_PRIOR_KD_BETA = 0.08715532460013839
KO_DEFENDER_KDRES_BETA = -0.012066294431692607
KO_DEFENDER_DURABILITY_BETA = -0.016962796032197684


def _sigmoid(z: float) -> float:
    z = float(np.clip(z, -30.0, 30.0))
    return 1.0 / (1.0 + exp(-z))


def ko_v2_probability(attacker, defender, *, prior_defender_kds: int) -> float:
    z = KO_INTERCEPT
    z += KO_POWER_BETA * (attacker.striking_power - 50.0)
    z += KO_ATTACKER_AGE_BETA * (attacker.age_years - 30.0)
    z += KO_DEFENDER_AGE_BETA * (defender.age_years - 30.0)
    z += KO_PRIOR_KD_BETA * float(prior_defender_kds)
    z += KO_DEFENDER_KDRES_BETA * (defender.knockdown_resistance - 50.0)
    z += KO_DEFENDER_DURABILITY_BETA * (defender.damage_durability - 50.0)
    return _sigmoid(z)


def resolve_ko_v2_shadow(*, state, attacker_side, attacker, defender, rng):
    """Drop-in research resolver: KO-v2 first, production empirical KD second."""
    target = state.physiology.fighter(attacker_side.opponent)
    prior = target.knockdowns_suffered
    stamina = state.physiology.fighter(attacker_side).stamina
    p_ko = ko_v2_probability(attacker, defender, prior_defender_kds=prior)
    if bool(rng.random() < p_ko):
        return ko_mod.EmpiricalKOKDResult(p_ko, True, 0.0, False, prior)
    p_kd = ko_mod.kd_probability(
        attacker,
        defender,
        prior_defender_kds=prior,
        elapsed_seconds=state.fight_time_seconds,
        attacker_stamina=stamina,
    )
    return ko_mod.EmpiricalKOKDResult(p_ko, False, p_kd, bool(rng.random() < p_kd), prior)


def resolve_fight_id() -> str:
    master = pd.read_parquet(MASTER_PATH).copy()
    master["date"] = pd.to_datetime(master["date"]).dt.normalize()
    same_day = master.loc[master["date"].eq(EVENT_DATE)].copy()
    mask = (
        (same_day["r_name"].astype(str).eq(FIGHTER_A) & same_day["b_name"].astype(str).eq(FIGHTER_B))
        | (same_day["r_name"].astype(str).eq(FIGHTER_B) & same_day["b_name"].astype(str).eq(FIGHTER_A))
    )
    rows = same_day.loc[mask]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one Schnell-Costa row, found {len(rows)}")
    return str(rows.iloc[0]["fight_id"])


def main() -> None:
    fight_id = resolve_fight_id()
    canonical = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    canonical["event_date"] = pd.to_datetime(canonical["event_date"]).dt.normalize()
    canonical["fight_id"] = canonical["fight_id"].astype(str)
    canonical["fighter_id"] = canonical["fighter_id"].astype(str)

    recency.EWM_DECAY = EWM_DECAY
    recency.EWM_CANONICAL_BLEND = 0.0
    ewm = recency.build_variant(canonical, "ewm")

    shutil.copy2(FSR_V3_PREFIGHT_SNAPSHOTS_PATH, BACKUP_PATH)
    original_standing_rates = intent_mod._standing_rates
    original_empirical_resolver = physiology_mod.resolve_empirical_ko_kd
    try:
        ewm.to_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH, index=False)
        pressure_mod.FIGHT_ID = fight_id
        pressure_mod.PATHS = PATHS
        intent_mod.FIGHT_ID = fight_id
        intent_mod.PATHS = PATHS

        def calibrated_standing_rates(state, actor, capabilities, context, priors, config):
            rates, pressure = original_standing_rates(state, actor, capabilities, context, priors, config)
            rates = dict(rates)
            rates[ActionFamily.STAND_ATTACK] *= STANDING_ATTEMPT_SCALE
            return rates, pressure

        intent_mod._standing_rates = calibrated_standing_rates
        physiology_mod.resolve_empirical_ko_kd = resolve_ko_v2_shadow

        fight, inputs, priors, horizon, cfg = pressure_mod.build_setup()
        brain = intent_mod.IntentRateBrain(inputs, priors, horizon)
        funcs = EngineFunctions(timing_sampler=brain.timing_sampler, action_chooser=brain.action_chooser)

        names = {Side.RED: str(fight.r_name), Side.BLUE: str(fight.b_name)}
        red_mechanics = inputs.fighter(Side.RED).mechanics
        blue_mechanics = inputs.fighter(Side.BLUE).mechanics
        prefight_ko = {
            names[Side.RED]: ko_v2_probability(red_mechanics, blue_mechanics, prior_defender_kds=0),
            names[Side.BLUE]: ko_v2_probability(blue_mechanics, red_mechanics, prior_defender_kds=0),
        }
        mechanics_audit = {}
        for side in (Side.RED, Side.BLUE):
            f = inputs.fighter(side).mechanics
            mechanics_audit[names[side]] = {
                "striking_power": f.striking_power,
                "damage_durability": f.damage_durability,
                "knockdown_resistance": f.knockdown_resistance,
                "age_years": f.age_years,
                "ko_v2_probability_per_landed_strike_at_zero_prior_kds": prefight_ko[names[side]],
            }

        wins = Counter()
        sixway = Counter()
        for path_id in range(PATHS):
            seed = derive_path_seed(SEED_SET_VERSION, fight_id, path_id)
            out = run_causal_path(inputs, seed=seed, horizon_seconds=horizon, config=cfg, functions=funcs)
            if out.termination is None:
                continue
            winner = out.termination.winner
            method = out.termination.finish_method.value
            wins[winner] += 1
            sixway[(winner.value, method)] += 1

        fighter_methods = {}
        for side in (Side.RED, Side.BLUE):
            counts = {m: int(sixway[(side.value, m)]) for m in ("ko_tko", "submission", "decision")}
            fighter_methods[names[side]] = {
                "wins": int(wins[side]),
                "win_probability": wins[side] / PATHS,
                "ko_tko": counts["ko_tko"] / PATHS,
                "submission": counts["submission"] / PATHS,
                "decision": counts["decision"] / PATHS,
                "counts": counts,
            }

        payload = {
            "diagnostic": "Schnell-Costa pure EWM 0.50 + fitted KO hazard v2 model D",
            "fight_id": fight_id,
            "paths": PATHS,
            "ewm_decay": EWM_DECAY,
            "canonical_blend": 0.0,
            "standing_attempt_scale": STANDING_ATTEMPT_SCALE,
            "seed_set": SEED_SET_VERSION,
            "production_changed": False,
            "ko_changed_only": True,
            "kd_hazard_changed": False,
            "ko_v2_coefficients": {
                "intercept": KO_INTERCEPT,
                "attacker_power": KO_POWER_BETA,
                "attacker_age": KO_ATTACKER_AGE_BETA,
                "defender_age": KO_DEFENDER_AGE_BETA,
                "prior_defender_kds": KO_PRIOR_KD_BETA,
                "defender_kdres": KO_DEFENDER_KDRES_BETA,
                "defender_durability": KO_DEFENDER_DURABILITY_BETA,
            },
            "mechanics_audit": mechanics_audit,
            "fighter_methods": fighter_methods,
        }
        print("SCHNELL_COSTA_EWM05_KO_HAZARD_V2_SHADOW")
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        physiology_mod.resolve_empirical_ko_kd = original_empirical_resolver
        intent_mod._standing_rates = original_standing_rates
        shutil.move(BACKUP_PATH, FSR_V3_PREFIGHT_SNAPSHOTS_PATH)


if __name__ == "__main__":
    main()
