"""Research-only Fares Ziam vs Tom Nolan last-5 FSR shadow with six-way methods."""
from __future__ import annotations

from collections import Counter
import json
import shutil
from pathlib import Path

import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
import pipeline.research.fsr_recency_cohort_shadow as recency
from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION
from pipeline.simulation.event_clock_mc_v2.calibration.seeds import derive_path_seed
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineFunctions, run_causal_path
from pipeline.simulation.event_clock_mc_v2.mechanics.resolution import FinishMethod
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_dynamic_pressure_shadow as pressure_mod
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_intent_rate_shadow as intent_mod

PATHS = 500
STANDING_ATTEMPT_SCALE = 0.25
BACKUP_PATH = Path("data/fsr_v3/fsr_v3_prefight_snapshots.canonical_backup.parquet")


def find_fight_id() -> str:
    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    names = master[["r_name", "b_name"]].astype(str)
    mask = (
        (names.r_name.str.contains("Fares Ziam", case=False, na=False) & names.b_name.str.contains("Tom Nolan", case=False, na=False))
        | (names.b_name.str.contains("Fares Ziam", case=False, na=False) & names.r_name.str.contains("Tom Nolan", case=False, na=False))
    )
    rows = master.loc[mask].sort_values("date")
    if rows.empty:
        raise RuntimeError("Fares Ziam vs Tom Nolan not found in master")
    return str(rows.iloc[-1].fight_id)


def run_with_methods(fight, inputs, priors, horizon, cfg, fight_id):
    brain = intent_mod.IntentRateBrain(inputs, priors, horizon)
    funcs = EngineFunctions(timing_sampler=brain.timing_sampler, action_chooser=brain.action_chooser)
    wins = Counter()
    six_way = Counter()
    methods = Counter()
    totals = {s: Counter() for s in Side}
    control = {s: 0.0 for s in Side}
    standing_exposure = 0.0

    for path_id in range(PATHS):
        seed = derive_path_seed(SEED_SET_VERSION, fight_id, path_id)
        out = run_causal_path(inputs, seed=seed, horizon_seconds=horizon, config=cfg, functions=funcs)
        for seg in out.timeline_segments:
            if seg.phase is Phase.STANDING:
                standing_exposure += seg.duration
            if seg.phase is Phase.GROUND and seg.controller in (Side.RED, Side.BLUE):
                control[seg.controller] += seg.duration
        for ev in out.events:
            side = ev.actor
            action = ev.selected_action
            if action in pressure_mod.TD:
                totals[side]["td_attempts"] += 1
                if ev.resulting_phase is Phase.GROUND and ev.resulting_controller is side:
                    totals[side]["td_success"] += 1
            if action in pressure_mod.STRIKES:
                totals[side]["strike_attempts"] += 1
                if ev.outcome.value == "landed":
                    totals[side]["strikes_landed"] += 1
            if action in pressure_mod.STAND:
                totals[side]["standing_attempts"] += 1
            if ev.submission_attempt:
                totals[side]["sub_attempts"] += 1
                if ev.submission_success:
                    totals[side]["sub_success"] += 1
            if ev.knockdown:
                totals[side]["kd"] += 1
        if out.termination is None:
            continue
        winner = out.termination.winner
        method = out.termination.finish_method
        wins[winner] += 1
        methods[method.value] += 1
        six_way[(winner.value, method.value)] += 1

    standing_seconds_per_path = standing_exposure / PATHS
    def summary(side, name):
        t = totals[side]
        key = side.value
        return {
            "fighter": name,
            "wins": wins[side],
            "win_probability": wins[side] / PATHS,
            "ko_tko_probability": six_way[(key, FinishMethod.KO_TKO.value)] / PATHS,
            "submission_probability": six_way[(key, FinishMethod.SUBMISSION.value)] / PATHS,
            "decision_probability": six_way[(key, FinishMethod.DECISION.value)] / PATHS,
            "standing_attempts_per_path": t["standing_attempts"] / PATHS,
            "standing_attempts_per_15m_standing_exposure": (t["standing_attempts"] / PATHS) * 900.0 / standing_seconds_per_path if standing_seconds_per_path else 0.0,
            "strike_attempts_per_path": t["strike_attempts"] / PATHS,
            "strikes_landed_per_path": t["strikes_landed"] / PATHS,
            "td_attempts_per_path": t["td_attempts"] / PATHS,
            "td_success_per_path": t["td_success"] / PATHS,
            "ground_control_seconds_per_path": control[side] / PATHS,
            "sub_attempts_per_path": t["sub_attempts"] / PATHS,
            "knockdowns_per_path": t["kd"] / PATHS,
            "brain_rate_diagnostics": brain.summary(side),
        }
    return {
        "red": summary(Side.RED, str(fight.r_name)),
        "blue": summary(Side.BLUE, str(fight.b_name)),
        "fight_methods": {k: v / PATHS for k, v in methods.items()},
    }


def main() -> None:
    fight_id = find_fight_id()
    canonical = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    canonical["event_date"] = pd.to_datetime(canonical["event_date"]).dt.normalize()
    canonical["fight_id"] = canonical["fight_id"].astype(str)
    canonical["fighter_id"] = canonical["fighter_id"].astype(str)
    recency.WINDOW = 5
    last5 = recency.build_variant(canonical, "last3")

    pressure_mod.FIGHT_ID = fight_id
    pressure_mod.PATHS = PATHS
    intent_mod.FIGHT_ID = fight_id
    intent_mod.PATHS = PATHS
    original_rates = intent_mod._standing_rates

    def calibrated_rates(state, actor, capabilities, context, priors, config):
        rates, p = original_rates(state, actor, capabilities, context, priors, config)
        rates = dict(rates)
        rates[ActionFamily.STAND_ATTACK] *= STANDING_ATTEMPT_SCALE
        return rates, p

    shutil.copy2(FSR_V3_PREFIGHT_SNAPSHOTS_PATH, BACKUP_PATH)
    intent_mod._standing_rates = calibrated_rates
    try:
        last5.to_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH, index=False)
        fight, inputs, priors, horizon, cfg = pressure_mod.build_setup()
        result = run_with_methods(fight, inputs, priors, horizon, cfg, fight_id)
    finally:
        intent_mod._standing_rates = original_rates
        shutil.move(BACKUP_PATH, FSR_V3_PREFIGHT_SNAPSHOTS_PATH)

    payload = {
        "diagnostic": "Ziam-Nolan last5 calibrated Brain shadow",
        "fight_id": fight_id,
        "paths": PATHS,
        "seed_set": SEED_SET_VERSION,
        "fsr_variant": "last5_all_v3_native",
        "fsr_window": 5,
        "standing_attempt_scale": STANDING_ATTEMPT_SCALE,
        "production_changed": False,
        "mechanics_changed": False,
        "judging_changed": False,
        "result": result,
    }
    print("ZIAM_NOLAN_LAST5_CALIBRATED_SHADOW")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
