"""Schnell-Costa research shadow using one total KO/TKO hazard per landed sig strike.

Research only. Production is unchanged.

Architecture on every landed modeled strike:
  1. sample ONE total KO/TKO hazard derived from all historical KO/TKO wins per
     sig landed plus opponent all KO/TKO losses per sig absorbed;
  2. apply a chronological attacker/defender age adjustment to that total hazard;
  3. if the strike does not finish, sample the independently validated KD hazard;
  4. a KD is recorded for state/judging only and creates NO additional KO/TKO
     probability, no finishing-sequence roll, and no acute-hurt bridge.

Raw matchup hazard:
    p_att = prior_all_ko_wins / prior_sig_landed
    p_def = opponent_prior_all_ko_losses / opponent_prior_sig_absorbed
    p_raw = 1 - (1-p_att)*(1-p_def)

Age adjustment:
    logit(p_age) = logit(p_raw)
                   + beta_att_age * (attacker_age - 30)
                   + beta_def_age * (defender_age - 30)

The age coefficients are fit only on fighter-fights strictly before the target
event date, using aggregated KO-win / landed-significant-strike opportunities.
Only the age slopes are applied to the raw matchup hazard; the fitted intercept
is deliberately not used, so this step does not silently replace the raw hazard
with a separate calibrated model.

This diagnostic is intentionally not a production calibration decision. The
historical cohort study showed the raw formulation discriminates but overpredicts.
"""
from __future__ import annotations

from collections import Counter
import json
from math import exp, log
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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
AGE_CENTER = 30.0
BACKUP_PATH = Path("data/fsr_v3/fsr_v3_prefight_snapshots.ko_v3_total_ko_shadow_backup.parquet")
MASTER_PATH = Path("data/master/ufc_master.parquet")
EVENT_DATE = pd.Timestamp("2026-06-06")
FIGHTER_A = "Matt Schnell"
FIGHTER_B = "Alessandro Costa"


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-9, 1.0 - 1e-9))
    return log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + exp(-float(np.clip(z, -30.0, 30.0))))


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


def fit_age_slopes(frame: pd.DataFrame, target_date: pd.Timestamp) -> dict[str, float]:
    """Fit chronological per-landed-sig KO age slopes on pre-target rows only."""
    train = frame.loc[
        frame["event_date"].lt(target_date)
        & frame["sig_landed"].gt(0)
        & frame["attacker_age"].notna()
        & frame["defender_age"].notna()
    ].copy()
    if len(train) < 500:
        raise RuntimeError(f"Insufficient pre-target rows for KO age fit: {len(train)}")

    k = train["ko_win"].astype(float).to_numpy()
    n = train["sig_landed"].astype(float).to_numpy()
    att_age = train["attacker_age"].astype(float).to_numpy() - AGE_CENTER
    def_age = train["defender_age"].astype(float).to_numpy() - AGE_CENTER

    def objective(beta: np.ndarray) -> float:
        eta = beta[0] + beta[1] * att_age + beta[2] * def_age
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        ll = np.sum(k * np.log(p) + (n - k) * np.log1p(-p))
        # Tiny ridge only for numerical stability of the two age slopes.
        penalty = 0.5 * 1e-4 * float(np.sum(beta[1:] ** 2))
        return float(-ll + penalty)

    total_k = float(k.sum())
    total_n = float(n.sum())
    p0 = np.clip(total_k / max(total_n, 1.0), 1e-9, 1.0 - 1e-9)
    init = np.asarray([_logit(float(p0)), 0.0, 0.0], dtype=float)
    result = minimize(objective, init, method="L-BFGS-B", options={"maxiter": 1000})
    if not result.success:
        raise RuntimeError(f"KO age fit failed: {result.message}")
    beta = np.asarray(result.x, dtype=float)
    return {
        "fit_rows": int(len(train)),
        "fit_ko_wins": int(total_k),
        "fit_sig_landed": total_n,
        "fit_population_ko_per_sig": float(p0),
        "fitted_intercept_audit_only": float(beta[0]),
        "attacker_age_logodds_per_year": float(beta[1]),
        "defender_age_logodds_per_year": float(beta[2]),
        "age_center": AGE_CENTER,
    }


def fit_total_ko_hazards(fight_id: str) -> tuple[dict[str, dict], dict[str, float]]:
    ff, _ = s1.load_raw_fighter_fights()
    frame = s1.build_matchup_frame(s1.build_prefight_states(ff)).copy()
    frame["fight_id"] = frame["fight_id"].astype(str)
    target = frame.loc[frame["fight_id"].eq(str(fight_id))].copy()
    if len(target) != 2:
        raise RuntimeError(f"Expected two target rows, found {len(target)}")
    target_dates = pd.to_datetime(target["event_date"]).dt.normalize().unique()
    if len(target_dates) != 1:
        raise RuntimeError("Target fight has inconsistent event dates")
    target_date = pd.Timestamp(target_dates[0]).normalize()
    age_fit = fit_age_slopes(frame, target_date)

    out = {}
    for row in target.itertuples(index=False):
        att_n = float(row.prior_sig_landed)
        def_n = float(row.opp_prior_sig_absorbed)
        att_k = float(row.prior_ko_wins)
        def_k = float(row.opp_prior_ko_losses)
        p_att = att_k / att_n if att_n > 0 else 0.0
        p_def = def_k / def_n if def_n > 0 else 0.0
        p_raw = 1.0 - (1.0 - p_att) * (1.0 - p_def)
        attacker_age = float(row.attacker_age)
        defender_age = float(row.defender_age)
        age_logodds_delta = (
            age_fit["attacker_age_logodds_per_year"] * (attacker_age - AGE_CENTER)
            + age_fit["defender_age_logodds_per_year"] * (defender_age - AGE_CENTER)
        )
        p_age = _sigmoid(_logit(p_raw) + age_logodds_delta) if p_raw > 0.0 else 0.0
        out[str(row.fighter_id)] = {
            "fighter_name": str(row.fighter_name),
            "attacker_age": attacker_age,
            "defender_age": defender_age,
            "attacker_ko_wins": att_k,
            "attacker_sig_landed": att_n,
            "attacker_ko_per_sig": p_att,
            "defender_ko_losses": def_k,
            "defender_sig_absorbed": def_n,
            "defender_ko_loss_per_sig": p_def,
            "raw_total_ko_per_landed": p_raw,
            "age_logodds_delta": float(age_logodds_delta),
            "total_ko_per_landed": float(p_age),
        }
    return out, age_fit


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
    total_ko_by_id, age_fit = fit_total_ko_hazards(fight_id)
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
            "diagnostic": "Schnell-Costa KO V3 total-KO hazard with age; KD scoring only",
            "fight_id": fight_id,
            "paths": PATHS,
            "production_changed": False,
            "raw_total_ko_formula": "1-(1-attacker_all_KO_per_sig)*(1-defender_all_KO_loss_per_sig)",
            "age_adjustment_formula": "logit(p_age)=logit(p_raw)+beta_att*(att_age-30)+beta_def*(def_age-30)",
            "age_fit": age_fit,
            "uses_shrinkage_for_total_ko": False,
            "uses_fitted_logit_for_total_ko": False,
            "uses_chronological_age_slopes": True,
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
