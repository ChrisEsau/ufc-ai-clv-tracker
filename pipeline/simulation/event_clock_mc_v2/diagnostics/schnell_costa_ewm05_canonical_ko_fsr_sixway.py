"""Schnell vs Costa hybrid shadow: EWM 0.50 except canonical KO-facing FSR.

Research only. 500 matched seeds, 0.25 standing-attempt scale, dynamic pressure,
mechanics, judging, and event-clock behavior unchanged from the existing EWM05
Schnell-Costa research harness.

Intervention:
- build the normal pure EWM decay=0.50 / canonical_blend=0.0 snapshot;
- for the target fight only, restore canonical values for the active KO-facing
  fighter representation fields:
    * striking_power_v3
    * knockdown_resistance_v3
    * damage_durability
- all other EWM-overlaid and canonical/inherited fields remain exactly as the
  pure-EWM harness would provide them.

No production files are changed persistently.
"""
from __future__ import annotations

from collections import Counter
import json
import shutil
from pathlib import Path

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

PATHS = 500
EWM_DECAY = 0.50
STANDING_ATTEMPT_SCALE = 0.25
BACKUP_PATH = Path("data/fsr_v3/fsr_v3_prefight_snapshots.canonical_backup.parquet")
MASTER_PATH = Path("data/master/ufc_master.parquet")
EVENT_DATE = pd.Timestamp("2026-06-06")
FIGHTER_A = "Matt Schnell"
FIGHTER_B = "Alessandro Costa"
KO_FSR_COLUMNS = (
    "striking_power_v3",
    "knockdown_resistance_v3",
    "damage_durability",
)


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
        raise RuntimeError(
            f"Expected exactly one {FIGHTER_A} vs {FIGHTER_B} row on {EVENT_DATE.date()}, found {len(rows)}"
        )
    return str(rows.iloc[0]["fight_id"])


def _audit_rows(frame: pd.DataFrame, fight_id: str) -> list[dict]:
    rows = frame.loc[frame["fight_id"].astype(str).eq(fight_id)].copy()
    keep = ["fighter_id", *KO_FSR_COLUMNS]
    return rows[keep].to_dict(orient="records")


def main() -> None:
    fight_id = resolve_fight_id()
    canonical = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    canonical["event_date"] = pd.to_datetime(canonical["event_date"]).dt.normalize()
    canonical["fight_id"] = canonical["fight_id"].astype(str)
    canonical["fighter_id"] = canonical["fighter_id"].astype(str)

    recency.EWM_DECAY = EWM_DECAY
    recency.EWM_CANONICAL_BLEND = 0.0
    ewm = recency.build_variant(canonical, "ewm")
    ewm["fight_id"] = ewm["fight_id"].astype(str)
    ewm["fighter_id"] = ewm["fighter_id"].astype(str)

    target_canonical = canonical.loc[canonical["fight_id"].eq(fight_id), ["fighter_id", *KO_FSR_COLUMNS]].copy()
    if len(target_canonical) != 2:
        raise RuntimeError(f"Expected 2 canonical prefight rows for fight {fight_id}, found {len(target_canonical)}")

    before = _audit_rows(ewm, fight_id)
    canonical_audit = _audit_rows(canonical, fight_id)

    target_map = target_canonical.set_index("fighter_id")
    target_mask = ewm["fight_id"].eq(fight_id)
    for idx in ewm.index[target_mask]:
        fighter_id = str(ewm.at[idx, "fighter_id"])
        if fighter_id not in target_map.index:
            raise RuntimeError(f"Target fighter {fighter_id} missing from canonical target rows")
        for col in KO_FSR_COLUMNS:
            ewm.at[idx, col] = target_map.at[fighter_id, col]

    after = _audit_rows(ewm, fight_id)

    shutil.copy2(FSR_V3_PREFIGHT_SNAPSHOTS_PATH, BACKUP_PATH)
    original_standing_rates = intent_mod._standing_rates
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
        fight, inputs, priors, horizon, cfg = pressure_mod.build_setup()
        brain = intent_mod.IntentRateBrain(inputs, priors, horizon)
        funcs = EngineFunctions(timing_sampler=brain.timing_sampler, action_chooser=brain.action_chooser)

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

        names = {Side.RED: str(fight.r_name), Side.BLUE: str(fight.b_name)}
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
            "diagnostic": "Schnell-Costa EWM 0.50 with canonical KO-facing FSR six-way shadow",
            "fight_id": fight_id,
            "paths": PATHS,
            "ewm_decay": EWM_DECAY,
            "canonical_blend": 0.0,
            "standing_attempt_scale": STANDING_ATTEMPT_SCALE,
            "seed_set": SEED_SET_VERSION,
            "production_changed": False,
            "mechanics_changed": False,
            "judging_changed": False,
            "ko_fsr_columns_restored_to_canonical": list(KO_FSR_COLUMNS),
            "ko_fsr_audit": {
                "pure_ewm_before": before,
                "canonical": canonical_audit,
                "hybrid_after": after,
            },
            "fsr_effective_rates": {
                names[Side.RED]: {
                    "standing_rate_15m": priors[Side.RED].standing_attempt_rate_15m,
                    "takedown_rate_15m": priors[Side.RED].takedown_attempt_rate_15m,
                },
                names[Side.BLUE]: {
                    "standing_rate_15m": priors[Side.BLUE].standing_attempt_rate_15m,
                    "takedown_rate_15m": priors[Side.BLUE].takedown_attempt_rate_15m,
                },
            },
            "fighter_methods": fighter_methods,
        }
        print("SCHNELL_COSTA_EWM05_CANONICAL_KO_FSR_SIXWAY")
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        intent_mod._standing_rates = original_standing_rates
        shutil.move(BACKUP_PATH, FSR_V3_PREFIGHT_SNAPSHOTS_PATH)


if __name__ == "__main__":
    main()
