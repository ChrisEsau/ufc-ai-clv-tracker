"""Research-only pure-EWM grid for Brendan Allen vs Edmen Shahbazyan.

Compares EWM decays 0.50, 0.65, 0.75, 0.85 with NO canonical blend.
Each condition uses the same 500 matched seeds, 0.25 calibrated standing-attempt
scale, dynamic-pressure logic, mechanics, judging, and fight setup.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
import pipeline.research.fsr_recency_cohort_shadow as recency
from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Side
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_dynamic_pressure_shadow as pressure_mod
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_intent_rate_shadow as intent_mod

FIGHT_ID = "419fff06f338f5c6"
PATHS = 500
STANDING_ATTEMPT_SCALE = 0.25
DECAYS = (0.50, 0.65, 0.75, 0.85)
BACKUP_PATH = Path("data/fsr_v3/fsr_v3_prefight_snapshots.canonical_backup.parquet")


def run_condition(decay: float) -> dict:
    canonical = pd.read_parquet(BACKUP_PATH).copy()
    canonical["event_date"] = pd.to_datetime(canonical["event_date"]).dt.normalize()
    canonical["fight_id"] = canonical["fight_id"].astype(str)
    canonical["fighter_id"] = canonical["fighter_id"].astype(str)

    recency.EWM_DECAY = float(decay)
    recency.EWM_CANONICAL_BLEND = 0.0
    ewm = recency.build_variant(canonical, "ewm")
    ewm.to_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH, index=False)

    pressure_mod.FIGHT_ID = FIGHT_ID
    pressure_mod.PATHS = PATHS
    intent_mod.FIGHT_ID = FIGHT_ID
    intent_mod.PATHS = PATHS

    original_standing_rates = intent_mod._standing_rates

    def calibrated_standing_rates(state, actor, capabilities, context, priors, config):
        rates, pressure = original_standing_rates(
            state, actor, capabilities, context, priors, config
        )
        rates = dict(rates)
        rates[ActionFamily.STAND_ATTACK] *= STANDING_ATTEMPT_SCALE
        return rates, pressure

    intent_mod._standing_rates = calibrated_standing_rates
    try:
        fight, inputs, priors, horizon, cfg = pressure_mod.build_setup()
        result = intent_mod.run_intent_rate_condition(fight, inputs, priors, horizon, cfg)
    finally:
        intent_mod._standing_rates = original_standing_rates

    return {
        "decay": decay,
        "red_fighter": str(fight.r_name),
        "blue_fighter": str(fight.b_name),
        "fsr_effective_rates": {
            "red": {
                "standing_rate_15m": priors[Side.RED].standing_attempt_rate_15m,
                "takedown_rate_15m": priors[Side.RED].takedown_attempt_rate_15m,
            },
            "blue": {
                "standing_rate_15m": priors[Side.BLUE].standing_attempt_rate_15m,
                "takedown_rate_15m": priors[Side.BLUE].takedown_attempt_rate_15m,
            },
        },
        "result": result,
    }


def main() -> None:
    shutil.copy2(FSR_V3_PREFIGHT_SNAPSHOTS_PATH, BACKUP_PATH)
    try:
        results = [run_condition(decay) for decay in DECAYS]
    finally:
        shutil.move(BACKUP_PATH, FSR_V3_PREFIGHT_SNAPSHOTS_PATH)

    payload = {
        "diagnostic": "Allen-Shahbazyan pure EWM FSR grid",
        "fight_id": FIGHT_ID,
        "paths_per_decay": PATHS,
        "decays": list(DECAYS),
        "canonical_blend": 0.0,
        "standing_attempt_scale": STANDING_ATTEMPT_SCALE,
        "seed_set": SEED_SET_VERSION,
        "production_changed": False,
        "mechanics_changed": False,
        "judging_changed": False,
        "results": results,
    }
    print("ALLEN_SHAHBAZYAN_PURE_EWM_GRID")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
