"""Schnell-Costa research shadow using one total KO/TKO hazard per landed sig strike.

Research only. Production is unchanged.

Architecture on every landed modeled strike:
  1. sample ONE total KO/TKO hazard derived from all historical KO/TKO wins per
     sig landed plus opponent all KO/TKO losses per sig absorbed;
  2. if the strike does not finish, sample the independently validated KD hazard;
  3. a KD is recorded for state/judging only and creates NO additional KO/TKO
     probability, no finishing-sequence roll, and no acute-hurt bridge.

The total-KO hazard here is the literal unshrunk cumulative formulation requested
for the Schnell-Costa diagnostic:
    p_att = prior_all_ko_wins / prior_sig_landed
    p_def = opponent_prior_all_ko_losses / opponent_prior_sig_absorbed
    p_ko  = 1 - (1-p_att)*(1-p_def)

This diagnostic is intentionally not a production calibration decision. The
historical cohort study showed the raw formulation discriminates but overpredicts.
"""
from __future__ import annotations

from collections import Counter
import json
import shutil
from pathlib import Path

import pandas as pd

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
import pipeline.research.fsr_recency_cohort_shadow as recency
from pipeline.research import ko_v3_from_scratch_stage1 as s1
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
BACKUP_PATH = Path("data/fsr_v3/fsr_v3_prefight_snapshots.ko_v3_total_ko_shadow_backup.parquet")
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


def fit_total_ko_hazards(fight_id: str) -> dict[str, dict]:
    ff, _ = s1.load_raw_fighter_fights()
    frame = s1.build_matchup_frame(s1.build_prefight_states(ff)).copy()
    frame["fight_id"] = frame["fight_id"].astype(str)
    target = frame.loc[frame["fight_id"].eq(str(fight_id))].copy()
    if len(target) != 2:
        raise RuntimeError(f"Expected two target rows, found {len(target)}")

    out = {}
    for row in target.itertuples(index=False):
        att_n = float(row.prior_sig_landed)
        def_n = float(row.opp_prior_sig_absorbed)
        att_k = float(row.prior_ko_wins)
        def_k = float(row.opp_prior_ko_losses)
        p_att = att_k / att_n if att_n > 0 else 0.0
        p_def = def_k / def_n if def_n > 0 else 0.0
        p_total = 1.0 - (1.0 - p_att) * (1.0 - p_def)
        out[str(row.fighter_id)] = {
            "fighter_name": str(row.fighter_name),
            "attacker_ko_wins": att_k,
            "attacker_sig_landed": att_n,
            "attacker_ko_per_sig": p_att,
            "defender_ko_losses": def_k,
            "defender_sig_absorbed": def_n,
            "defender_ko_loss_per_sig": p_def,
            "total_ko_per_landed": p_total,
        }
    return out


class TotalKOOnlyResolver:
    def __init__(self, total_ko_by_side, kd_hazards_by_side):
        self.total_ko_by_side = total_ko_by_side
        self.kd_hazards_by_side = kd_hazards_by_side
        self.landed = Counter()
        self.ko_finishes = Counter()
        self.knockdowns = Counter()

    def __call__(self, *, state, attacker_side, attacker, defender, rng):
        del attacker, defender
        target = state.physiology.fighter(attacker_side.opponent)
        prior = int(target.knockdowns_suffered)
        self.landed[attacker_side] += 1

        p_ko = float(self.total_ko_by_side[attacker_side]["total_ko_per_landed"])
        if bool(rng.random() < p_ko):
            self.ko_finishes[attacker_side] += 1
            return ko_mod.EmpiricalKOKDResult(p_ko, True, 0.0, False, prior)

        p_kd = float(self.kd_hazards_by_side[attacker_side].kd_per_landed)
        kd = bool(rng.random() < p_kd)
        if kd:
            self.knockdowns[attacker_side] += 1
        return ko_mod.EmpiricalKOKDResult(p_ko, False, p_kd, kd, prior)

    def summary(self, side: Side) -> dict:
        return {
            "landed_strike_resolutions": int(self.landed[side]),
            "ko_finishes": int(self.ko_finishes[side]),
            "knockdowns": int(self.knockdowns[side]),
            "post_kd_finish_rolls": 0,
            "post_kd_finishes": 0,
        }


def main() -> None:
    fight_id = resolve_fight_id()
    total_ko_by_id = fit_total_ko_hazards(fight_id)
    kd_hazards_by_id = fit_prefight_hazards(fight_id=fight_id)

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
        side_to_id = {Side.RED: str(fight.r_id), Side.BLUE: str(fight.b_id)}
        total_ko_by_side = {side: total_ko_by_id[fid] for side, fid in side_to_id.items()}
        kd_hazards_by_side = {side: kd_hazards_by_id[fid] for side, fid in side_to_id.items()}
        resolver = TotalKOOnlyResolver(total_ko_by_side, kd_hazards_by_side)
        physiology_mod.resolve_empirical_ko_kd = resolver
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
        for side in Side:
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
                **total_ko_by_side[side],
                "kd_per_landed": float(kd_hazards_by_side[side].kd_per_landed),
                "resolver_counts": resolver.summary(side),
            }

        payload = {
            "diagnostic": "Schnell-Costa KO V3 total-KO hazard; KD scoring only",
            "fight_id": fight_id,
            "paths": PATHS,
            "production_changed": False,
            "total_ko_formula": "1-(1-attacker_all_KO_per_sig)*(1-defender_all_KO_loss_per_sig)",
            "uses_shrinkage_for_total_ko": False,
            "uses_fitted_logit_for_total_ko": False,
            "kd_can_finish": False,
            "post_kd_finish_loop": False,
            "kd_role": "state/judging only",
            "hurt_increment": 0.0,
            "standing_attempt_scale": STANDING_ATTEMPT_SCALE,
            "hazard_audit": hazard_audit,
            "fighter_methods": fighter_methods,
        }
        print("SCHNELL_COSTA_KO_V3_TOTAL_KO_SHADOW")
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        physiology_mod.resolve_empirical_ko_kd = original_empirical_resolver
        physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT = original_hurt_increment
        intent_mod._standing_rates = original_standing_rates
        shutil.move(BACKUP_PATH, FSR_V3_PREFIGHT_SNAPSHOTS_PATH)


if __name__ == "__main__":
    main()
