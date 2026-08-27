"""Simulation-based diagnostic of clinch action timing.

Research only. Standing clock is fixed at 1.0x raw matchup FSR, with pressure and
standing strike-context inflation disabled and RESET removed. Clinch-entry intent
remains the existing unvalidated structural 0.06 x standing-rate prior with live
context. This study varies only the within-clinch action clock and compares realized
modeled CLINCH_STRIKE attempts with UFCStats clinch significant-strike attempts.

Important identification limit: because the same clinch clock governs strikes,
control, takedowns, and breaks, scaling it may change clinch duration much more than
strikes per entry. UFCStats does not provide clinch entry counts or clinch phase time,
so this is a diagnostic, not a claim that clinch timing is identified from strike
attempt counts alone.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.brain.intent_priors import BrainIntentPriors
from pipeline.simulation.event_clock_mc_v2.brain.policy import BrainDecisionContext
from pipeline.simulation.event_clock_mc_v2.brain.timing import BrainTimingContext
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineConfig, EngineFunctions, EngineInputs, FighterEngineInputs, run_causal_path
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import historical_fighter_rows, load_latest_profiles, load_prefight_snapshots
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1.capability_translation import CapabilityReference
from pipeline.simulation.event_clock_mc_v2.diagnostics.stage6_real_causal_path import _capabilities, _mechanics
from pipeline.simulation.event_clock_mc_v2.diagnostics.stage8_structural_population import MASTER, ROUND_STATS, actual_side_totals, pick_col, side_rows
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_intent_rate_shadow as intent_mod

OUTDIR = Path("data/research/clinch_timing_simulation_oos")
CUTOFF = pd.Timestamp("2025-01-01")
TRAIN_FIGHTS = 40
HOLDOUT_FIGHTS = 40
PATHS = 8
DELAY_SCALES = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
SEED_BASE = 2026082702
ORIGINAL_STANDING_RATES = intent_mod._standing_rates
ORIGINAL_SAMPLE_DELAY = intent_mod.sample_next_action_delay


def elapsed(row: pd.Series) -> float:
    v = row.get("match_time_sec", np.nan)
    return float(v) if pd.notna(v) and float(v) > 0 else float("nan")


def choose_fights(master, rounds, snaps, *, before_cutoff: bool, n: int):
    fc = pick_col(rounds, "fight_id", "bout_id")
    available = set(rounds[fc].astype(str))
    m = master[(master._event_date < CUTOFF) if before_cutoff else (master._event_date >= CUTOFF)].copy()
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


def standing_rates_one(state, actor, capabilities, context, priors, config):
    rates, _ = ORIGINAL_STANDING_RATES(state, actor, capabilities, context, priors, config)
    rates = dict(rates)
    rates[ActionFamily.STAND_ATTACK] = max(float(priors.standing_attempt_rate_15m), 1e-12)
    rates.pop(ActionFamily.RESET_RANGE, None)
    return rates, 0.0


def clinch_delay_sampler(delay_scale: float):
    def _sample(state, context, rng, config):
        if state.phase is Phase.CLINCH:
            config = replace(config, clinch_phase_factor=config.clinch_phase_factor * float(delay_scale))
        return ORIGINAL_SAMPLE_DELAY(state, context, rng, config)
    return _sample


def simulate_set(fights, delay_scale, rounds_df, reference, split):
    intent_mod._standing_rates = standing_rates_one
    intent_mod.sample_next_action_delay = clinch_delay_sampler(delay_scale)
    rows = []
    actual_total = 0.0
    sim_total = 0.0
    entry_total = 0.0
    clinch_sec_total = 0.0
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
        actual_red = actual_side_totals(side_rows(rounds_df, fid, str(mr.r_id), "red"))["clinch_att"]
        actual_blue = actual_side_totals(side_rows(rounds_df, fid, str(mr.b_id), "blue"))["clinch_att"]
        actual = float(actual_red + actual_blue)
        path_strikes, path_entries, path_secs = [], [], []
        for pi in range(PATHS):
            brain = intent_mod.IntentRateBrain(inputs, priors, h)
            funcs = EngineFunctions(timing_sampler=brain.timing_sampler, action_chooser=brain.action_chooser)
            seed = SEED_BASE + fi * 1000 + pi
            result = run_causal_path(inputs, seed=seed, horizon_seconds=h, config=cfg, functions=funcs)
            path_strikes.append(float(sum(ev.selected_action is ActionFamily.CLINCH_STRIKE for ev in result.events)))
            path_entries.append(float(sum(ev.selected_action is ActionFamily.CLINCH_ENTRY for ev in result.events)))
            path_secs.append(float(sum(seg.duration for seg in result.timeline_segments if seg.phase is Phase.CLINCH)))
        sim = float(np.mean(path_strikes))
        entries = float(np.mean(path_entries))
        secs = float(np.mean(path_secs))
        actual_total += actual
        sim_total += sim
        entry_total += entries
        clinch_sec_total += secs
        rows.append({
            "split": split,
            "clinch_delay_scale": float(delay_scale),
            "fight_id": fid,
            "event_date": str(pd.Timestamp(mr._event_date).date()),
            "red_name": str(mr.get("r_name", mr.r_id)),
            "blue_name": str(mr.get("b_name", mr.b_id)),
            "horizon_seconds": h,
            "actual_clinch_sig_attempts_both": actual,
            "sim_clinch_strike_attempts_both_mean": sim,
            "sim_clinch_entries_mean": entries,
            "sim_clinch_seconds_mean": secs,
            "sim_strikes_per_entry": sim / entries if entries > 0 else np.nan,
            "paths": PATHS,
        })
    return rows, {
        "actual_clinch_sig_attempts": actual_total,
        "sim_clinch_strike_attempts": sim_total,
        "E_over_O": sim_total / actual_total if actual_total > 0 else None,
        "sim_clinch_entries": entry_total,
        "sim_clinch_seconds": clinch_sec_total,
        "sim_strikes_per_entry": sim_total / entry_total if entry_total > 0 else None,
        "sim_seconds_per_entry": clinch_sec_total / entry_total if entry_total > 0 else None,
    }


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

    all_rows, train_grid = [], []
    for scale in DELAY_SCALES:
        rows, summary = simulate_set(train, scale, rounds, reference, "train_grid")
        all_rows.extend(rows)
        train_grid.append({"clinch_delay_scale": scale, **summary})
        print("TRAIN", scale, summary, flush=True)

    # Report holdout for baseline 1.0 only. We do not fit a clinch scale because
    # UFCStats does not identify clinch entries or phase-time separately.
    hold_rows, hold_summary = simulate_set(holdout, 1.0, rounds, reference, "holdout_baseline")
    all_rows.extend(hold_rows)
    intent_mod._standing_rates = ORIGINAL_STANDING_RATES
    intent_mod.sample_next_action_delay = ORIGINAL_SAMPLE_DELAY

    result = {
        "study": "simulation-based clinch timing diagnostic",
        "production_changed": False,
        "standing_clock_scale": 1.0,
        "clinch_entry_prior": "0.06 * matchup standing rate with existing live context; held fixed and unvalidated",
        "baseline_clinch_mean_delay_seconds_neutral": 3.6,
        "identification_warning": "UFCStats provides clinch significant-strike attempts, not clinch entries or clinch phase time. Delay scale is diagnostic only and is not fit/promoted from this target.",
        "cutoff": str(CUTOFF.date()),
        "train_fights": TRAIN_FIGHTS,
        "holdout_fights": HOLDOUT_FIGHTS,
        "paths_per_fight": PATHS,
        "train_delay_grid": train_grid,
        "holdout_baseline_1x": hold_summary,
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(OUTDIR / "fight_level_results.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
