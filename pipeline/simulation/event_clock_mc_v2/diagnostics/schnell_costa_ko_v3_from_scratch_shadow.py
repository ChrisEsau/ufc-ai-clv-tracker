"""Schnell-Costa shadow integration for KO V3 from scratch.

Research only. The Brain, strike generation/landing, grappling, submissions,
judging, and non-KO mechanics are frozen to the existing pure-EWM50 intent-rate
benchmark. Only the landed-strike KO/KD consequence resolver is replaced.

KO V3 consequence flow on each landed strike:
  1) direct KO/TKO hazard;
  2) if survived, KD hazard;
  3) if KD occurs, post-KD finishing-sequence hazard;
  4) if the sequence fails, continue with the KD recorded but NO acute-hurt
     increment because hurt magnitude/decay is not identified by round data.

The resolver uses no FSR power, durability, or KD-resistance traits.
"""
from __future__ import annotations

from collections import Counter
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
import pipeline.research.fsr_recency_cohort_shadow as recency
from pipeline.research.ko_v3_from_scratch_shadow import fit_prefight_hazards
from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION
from pipeline.simulation.event_clock_mc_v2.calibration.seeds import derive_path_seed
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineFunctions, run_causal_path
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_dynamic_pressure_shadow as pressure_mod
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_intent_rate_shadow as intent_mod
from pipeline.simulation.event_clock_mc_v2.mechanics import physiology as physiology_mod
from pipeline.simulation.event_clock_mc_v2.mechanics import ko_kd_empirical as ko_mod

PATHS = 500
BASE_EWM_DECAY = 0.50
STANDING_ATTEMPT_SCALE = 0.25
BACKUP_PATH = Path("data/fsr_v3/fsr_v3_prefight_snapshots.ko_v3_shadow_backup.parquet")
MASTER_PATH = Path("data/master/ufc_master.parquet")
EVENT_DATE = pd.Timestamp("2026-06-06")
FIGHTER_A = "Matt Schnell"
FIGHTER_B = "Alessandro Costa"


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


def build_pure_ewm50_snapshot(canonical: pd.DataFrame) -> pd.DataFrame:
    recency.EWM_CANONICAL_BLEND = 0.0
    recency.EWM_DECAY = BASE_EWM_DECAY
    return recency.build_variant(canonical, "ewm")


class KOV3ShadowResolver:
    def __init__(self, hazards_by_side):
        self.hazards_by_side = hazards_by_side
        self.landed = Counter()
        self.direct_finishes = Counter()
        self.knockdowns = Counter()
        self.sequence_finishes = Counter()

    def __call__(self, *, state, attacker_side, attacker, defender, rng):
        del attacker, defender  # KO V3 deliberately ignores old FSR physical traits.
        h = self.hazards_by_side[attacker_side]
        target = state.physiology.fighter(attacker_side.opponent)
        prior = int(target.knockdowns_suffered)
        self.landed[attacker_side] += 1

        p_direct = float(h.direct_finish_per_landed)
        p_kd = float(h.kd_per_landed)
        p_seq = float(h.post_kd_sequence_per_kd)
        p_total_finish = float(p_direct + (1.0 - p_direct) * p_kd * p_seq)

        if bool(rng.random() < p_direct):
            self.direct_finishes[attacker_side] += 1
            return ko_mod.EmpiricalKOKDResult(
                p_total_finish,
                True,
                0.0,
                False,
                prior,
            )

        kd = bool(rng.random() < p_kd)
        if not kd:
            return ko_mod.EmpiricalKOKDResult(
                p_total_finish,
                False,
                p_kd,
                False,
                prior,
            )

        self.knockdowns[attacker_side] += 1
        sequence_finish = bool(rng.random() < p_seq)
        if sequence_finish:
            self.sequence_finishes[attacker_side] += 1
        return ko_mod.EmpiricalKOKDResult(
            p_total_finish,
            sequence_finish,
            p_kd,
            True,
            prior,
        )

    def summary(self, side: Side) -> dict:
        return {
            "landed_strike_resolutions": int(self.landed[side]),
            "direct_finishes": int(self.direct_finishes[side]),
            "knockdowns": int(self.knockdowns[side]),
            "post_kd_sequence_finishes": int(self.sequence_finishes[side]),
        }


def main() -> None:
    fight_id = resolve_fight_id()
    hazards_by_id = fit_prefight_hazards(fight_id=fight_id)

    canonical = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    canonical["event_date"] = pd.to_datetime(canonical["event_date"]).dt.normalize()
    canonical["fight_id"] = canonical["fight_id"].astype(str)
    canonical["fighter_id"] = canonical["fighter_id"].astype(str)
    ewm50 = build_pure_ewm50_snapshot(canonical)

    shutil.copy2(FSR_V3_PREFIGHT_SNAPSHOTS_PATH, BACKUP_PATH)
    original_standing_rates = intent_mod._standing_rates
    original_empirical_resolver = physiology_mod.resolve_empirical_ko_kd
    original_hurt_increment = physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT
    try:
        ewm50.to_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH, index=False)
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

        fight, inputs, priors, horizon, cfg = pressure_mod.build_setup()
        side_to_id = {
            Side.RED: str(fight.r_id),
            Side.BLUE: str(fight.b_id),
        }
        hazards_by_side = {side: hazards_by_id[fid] for side, fid in side_to_id.items()}
        resolver = KOV3ShadowResolver(hazards_by_side)
        physiology_mod.resolve_empirical_ko_kd = resolver
        # Deliberately disable the existing 0.5/30-second KD hurt bridge. Stage 2
        # did not identify a hurt magnitude or decay constant from aggregate data.
        physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT = 0.0

        brain = intent_mod.IntentRateBrain(inputs, priors, horizon)
        funcs = EngineFunctions(timing_sampler=brain.timing_sampler, action_chooser=brain.action_chooser)
        names = {Side.RED: str(fight.r_name), Side.BLUE: str(fight.b_name)}

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
        hazard_audit = {}
        for side in (Side.RED, Side.BLUE):
            h = hazards_by_side[side]
            counts = {m: int(sixway[(side.value, m)]) for m in ("ko_tko", "submission", "decision")}
            fighter_methods[names[side]] = {
                "wins": int(wins[side]),
                "win_probability": wins[side] / PATHS,
                "ko_tko": counts["ko_tko"] / PATHS,
                "submission": counts["submission"] / PATHS,
                "decision": counts["decision"] / PATHS,
                "counts": counts,
            }
            hazard_audit[names[side]] = {
                "fighter_id": h.fighter_id,
                "kd_per_landed": h.kd_per_landed,
                "direct_finish_per_landed": h.direct_finish_per_landed,
                "post_kd_sequence_per_kd": h.post_kd_sequence_per_kd,
                "total_finish_per_landed": h.total_finish_per_landed,
                "kd_population_hazard": h.kd_population_hazard,
                "direct_population_hazard": h.direct_population_hazard,
                "resolver_counts": resolver.summary(side),
            }

        payload = {
            "diagnostic": "Schnell-Costa KO V3 from scratch shadow",
            "fight_id": fight_id,
            "paths": PATHS,
            "base_ewm_decay_for_non_ko_brain_inputs": BASE_EWM_DECAY,
            "canonical_blend": 0.0,
            "standing_attempt_scale": STANDING_ATTEMPT_SCALE,
            "seed_set": SEED_SET_VERSION,
            "production_changed": False,
            "ko_v3_uses_fsr_physical_traits": False,
            "hurt_increment": 0.0,
            "hurt_decay_used_by_ko_v3": False,
            "hazard_audit": hazard_audit,
            "fighter_methods": fighter_methods,
        }
        print("SCHNELL_COSTA_KO_V3_FROM_SCRATCH_SHADOW")
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        physiology_mod.resolve_empirical_ko_kd = original_empirical_resolver
        physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT = original_hurt_increment
        intent_mod._standing_rates = original_standing_rates
        shutil.move(BACKUP_PATH, FSR_V3_PREFIGHT_SNAPSHOTS_PATH)


if __name__ == "__main__":
    main()
