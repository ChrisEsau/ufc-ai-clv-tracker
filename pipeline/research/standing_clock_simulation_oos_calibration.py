"""Simulation-based OOS calibration of the research standing action clock.

Unlike the earlier algebraic diagnostic, this calibrates against realized fight-level
UFCStats distance significant-strike attempts by actually running the causal simulator.
That lets simulated standing/clinch/ground exposure determine how much of the historical
fight horizon is available for standing actions.

Research standing architecture:
- STAND_ATTACK absolute rate = matchup-effective FSR standing rate * candidate scale
- no dynamic-pressure multiplier on standing strike rate
- no live strike-context multiplier on standing strike rate
- RESET_RANGE removed
- TD and clinch live-context rates preserved
Production is untouched.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.brain.intent_priors import BrainIntentPriors
from pipeline.simulation.event_clock_mc_v2.brain.policy import BrainDecisionContext
from pipeline.simulation.event_clock_mc_v2.brain.timing import BrainTimingContext
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineConfig, EngineFunctions, EngineInputs, FighterEngineInputs, run_causal_path
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import historical_fighter_rows, load_latest_profiles, load_prefight_snapshots
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1.capability_translation import CapabilityReference
from pipeline.simulation.event_clock_mc_v2.diagnostics.stage6_real_causal_path import _capabilities, _mechanics
from pipeline.simulation.event_clock_mc_v2.diagnostics.stage8_structural_population import MASTER, ROUND_STATS, actual_side_totals, pick_col, side_rows
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_intent_rate_shadow as intent_mod

OUTDIR = Path("data/research/standing_clock_simulation_oos")
CUTOFF = pd.Timestamp("2025-01-01")
TRAIN_FIGHTS = 40
HOLDOUT_FIGHTS = 40
PATHS = 8
SCALES = (0.50, 0.75, 1.00, 1.25)
SEED_BASE = 2026082701
ORIGINAL_STANDING_RATES = intent_mod._standing_rates


def elapsed(row: pd.Series) -> float:
    v = row.get("match_time_sec", np.nan)
    return float(v) if pd.notna(v) and float(v) > 0 else float("nan")


def choose_fights(master, rounds, snaps, *, before_cutoff: bool, n: int):
    fc = pick_col(rounds, "fight_id", "bout_id")
    available = set(rounds[fc].astype(str))
    m = master[(master._event_date < CUTOFF) if before_cutoff else (master._event_date >= CUTOFF)].copy()
    # For train use most recent pre-cutoff; holdout use earliest post-cutoff to preserve chronology.
    m = m.sort_values(["_event_date", "fight_id"], ascending=[not before_cutoff, True])
    out = []
    for _, row in m.iterrows():
        fid = str(row.fight_id)
        if fid not in available:
            continue
        h = elapsed(row)
        if not np.isfinite(h):
            continue
        try:
            red, blue = historical_fighter_rows(snaps, event_date=row._event_date, fight_id=fid, fighter_ids=(str(row.r_id), str(row.b_id)))
            side_rows(rounds, fid, str(row.r_id), "red")
            side_rows(rounds, fid, str(row.b_id), "blue")
        except Exception:
            continue
        out.append((row, red, blue))
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} complete fights; requested {n}")
    return out


def scaled_rate_function(scale: float):
    def _rates(state, actor, capabilities, context, priors, config):
        rates, _ = ORIGINAL_STANDING_RATES(state, actor, capabilities, context, priors, config)
        rates = dict(rates)
        rates[ActionFamily.STAND_ATTACK] = max(float(priors.standing_attempt_rate_15m) * float(scale), 1e-12)
        rates.pop(ActionFamily.RESET_RANGE, None)
        return rates, 0.0
    return _rates


def simulate_set(fights, scale, rounds_df, reference, split):
    intent_mod._standing_rates = scaled_rate_function(scale)
    rows = []
    actual_total = 0.0
    sim_total = 0.0
    for fi, (mr, red_fsr, blue_fsr) in enumerate(fights):
        fid = str(mr.fight_id)
        h = elapsed(mr)
        red_cap, red_runtime = _capabilities(red_fsr, blue_fsr, reference)
        blue_cap, blue_runtime = _capabilities(blue_fsr, red_fsr, reference)
        priors = {
            Side.RED: BrainIntentPriors(red_runtime.standing_rate_15m, red_runtime.takedown_rate_15m, 0.06, 3.0, 0.3),
            Side.BLUE: BrainIntentPriors(blue_runtime.standing_rate_15m, blue_runtime.takedown_rate_15m, 0.06, 3.0, 0.3),
        }
        inputs = EngineInputs(
            red=FighterEngineInputs(red_cap, BrainTimingContext(), BrainDecisionContext(), _mechanics(red_runtime)),
            blue=FighterEngineInputs(blue_cap, BrainTimingContext(), BrainDecisionContext(), _mechanics(blue_runtime)),
        )
        cfg = EngineConfig(number_of_rounds=max(1, int(math.ceil(h / 300.0))))
        actual_red = actual_side_totals(side_rows(rounds_df, fid, str(mr.r_id), "red"))["distance_att"]
        actual_blue = actual_side_totals(side_rows(rounds_df, fid, str(mr.b_id), "blue"))["distance_att"]
        actual = float(actual_red + actual_blue)
        path_counts = []
        path_standing_sec = []
        for pi in range(PATHS):
            brain = intent_mod.IntentRateBrain(inputs, priors, h)
            funcs = EngineFunctions(timing_sampler=brain.timing_sampler, action_chooser=brain.action_chooser)
            seed = SEED_BASE + fi * 1000 + pi
            result = run_causal_path(inputs, seed=seed, horizon_seconds=h, config=cfg, functions=funcs)
            c = sum(ev.selected_action is ActionFamily.STAND_ATTACK for ev in result.events)
            s = sum(seg.duration for seg in result.timeline_segments if seg.phase.value == "standing")
            path_counts.append(float(c))
            path_standing_sec.append(float(s))
        sim = float(np.mean(path_counts))
        actual_total += actual
        sim_total += sim
        rows.append({
            "split": split, "scale": float(scale), "fight_id": fid,
            "event_date": str(pd.Timestamp(mr._event_date).date()),
            "red_name": str(mr.get("r_name", mr.r_id)), "blue_name": str(mr.get("b_name", mr.b_id)),
            "horizon_seconds": h, "actual_distance_attempts_both": actual,
            "sim_distance_attempts_both_mean": sim,
            "sim_standing_seconds_mean": float(np.mean(path_standing_sec)),
            "paths": PATHS,
        })
    return rows, {"actual": actual_total, "sim": sim_total, "E_over_O": sim_total / actual_total}


def main():
    master = pd.read_parquet(MASTER).drop_duplicates("fight_id").copy()
    master["fight_id"] = master.fight_id.astype(str)
    dc = pick_col(master, "date", "event_date")
    master["_event_date"] = pd.to_datetime(master[dc], errors="coerce").dt.normalize()
    master = master.dropna(subset=["_event_date"])
    rounds = pd.read_parquet(ROUND_STATS).copy()
    snaps = load_prefight_snapshots()
    reference = CapabilityReference.from_latest(load_latest_profiles())

    train = choose_fights(master, rounds, snaps, before_cutoff=True, n=TRAIN_FIGHTS)
    holdout = choose_fights(master, rounds, snaps, before_cutoff=False, n=HOLDOUT_FIGHTS)

    all_rows = []
    grid = []
    for scale in SCALES:
        rows, summary = simulate_set(train, scale, rounds, reference, "train_grid")
        all_rows.extend(rows)
        grid.append({"scale": scale, **summary})
        print("TRAIN", scale, summary, flush=True)

    # Monotone interpolation on pooled simulated attempts to solve target scale.
    g = sorted(grid, key=lambda x: x["scale"])
    target = g[0]["actual"]
    fitted = min(g, key=lambda x: abs(x["sim"] - target))["scale"]
    for a, b in zip(g[:-1], g[1:]):
        if (a["sim"] - target) * (b["sim"] - target) <= 0 and b["sim"] != a["sim"]:
            fitted = a["scale"] + (target - a["sim"]) * (b["scale"] - a["scale"]) / (b["sim"] - a["sim"])
            break
    fitted = float(fitted)

    train_rows, train_fit = simulate_set(train, fitted, rounds, reference, "train_fitted")
    hold_rows, hold_fit = simulate_set(holdout, fitted, rounds, reference, "holdout")
    all_rows.extend(train_rows)
    all_rows.extend(hold_rows)
    intent_mod._standing_rates = ORIGINAL_STANDING_RATES

    result = {
        "study": "simulation-based standing clock OOS calibration",
        "production_changed": False,
        "research_standing_architecture": "raw matchup FSR standing rate * scale; pressure off; strike-context off; RESET removed; TD/clinch context retained",
        "cutoff": str(CUTOFF.date()), "train_fights": TRAIN_FIGHTS, "holdout_fights": HOLDOUT_FIGHTS,
        "paths_per_fight": PATHS, "candidate_grid": grid, "fitted_scale": fitted,
        "train_fitted": train_fit, "holdout": hold_fit,
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(OUTDIR / "fight_level_results.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
