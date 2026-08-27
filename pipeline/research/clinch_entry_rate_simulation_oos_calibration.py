"""Simulation-based OOS calibration of research clinch-entry intent.

Uses the clean-round clinch-equivalent proxy only as a RELATIVE fighter x opponent
propensity signal. The proxy is not interpreted as literal observed entry count.
Absolute clinch-entry rate is calibrated by simulation against UFCStats fight-level
clinch significant-strike attempts.

Frozen research architecture:
- standing strike clock = 1.0x raw matchup FSR
- pressure multiplier off for standing strike cadence
- standing strike-context multiplier off
- RESET_RANGE removed
- takedown intent/context unchanged
- inside-clinch timing unchanged (neutral mean 3.6 s)
- clinch-entry live policy context removed; absolute entry rate comes from PIT proxy
- production untouched
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.research.clinch_entry_proxy_oos_validation import _prepare
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

OUTDIR = Path("data/research/clinch_entry_rate_simulation_oos")
CUTOFF = pd.Timestamp("2025-01-01")
TRAIN_FIGHTS = 40
HOLDOUT_FIGHTS = 40
PATHS = 8
SCALES = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50)
SEED_BASE = 2026082703
SHRINK_N = 10.0
EPS = 1e-12
ORIGINAL_STANDING_RATES = intent_mod._standing_rates


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


def build_proxy_table(rounds: pd.DataFrame):
    clean = _prepare(rounds)
    train = clean[clean.event_date < CUTOFF].copy()
    positive = train[(train.clinch_att > 0) | (train.ctrl_sec > 0)]
    strike_scale = max(float(positive.loc[positive.clinch_att > 0, "clinch_att"].median()), 1.0)
    ctrl_scale = max(float(positive.loc[positive.ctrl_sec > 0, "ctrl_sec"].median()), 1.0)
    raw_proxy = np.maximum(clean.clinch_att / strike_scale, clean.ctrl_sec / ctrl_scale)
    clean["clinch_equiv"] = np.where(
        ((clean.clinch_att > 0) | (clean.ctrl_sec > 0)), np.maximum(1.0, raw_proxy), 0.0
    )
    train_proxy = clean.loc[clean.event_date < CUTOFF, "clinch_equiv"]
    global_equiv = float(train_proxy.mean())
    return clean, global_equiv, strike_scale, ctrl_scale


def pit_matchup_equiv(clean: pd.DataFrame, global_equiv: float, date, fighter: str, opponent: str) -> tuple[float, int, int]:
    hist = clean[clean.event_date < pd.Timestamp(date)]
    fvals = hist.loc[hist.fighter_name.astype(str) == str(fighter), "clinch_equiv"].to_numpy(float)
    ovals = hist.loc[hist.opponent_name.astype(str) == str(opponent), "clinch_equiv"].to_numpy(float)
    fn, on = len(fvals), len(ovals)
    fm = float(np.mean(fvals)) if fn else global_equiv
    om = float(np.mean(ovals)) if on else global_equiv
    fs = (fn * fm + SHRINK_N * global_equiv) / (fn + SHRINK_N)
    os = (on * om + SHRINK_N * global_equiv) / (on + SHRINK_N)
    return float(np.sqrt(max(fs, EPS) * max(os, EPS))), fn, on


def rate_function(clinch_rates):
    def _rates(state, actor, capabilities, context, priors, config):
        rates, _ = ORIGINAL_STANDING_RATES(state, actor, capabilities, context, priors, config)
        rates = dict(rates)
        rates[ActionFamily.STAND_ATTACK] = max(float(priors.standing_attempt_rate_15m), EPS)
        rates[ActionFamily.CLINCH_ENTRY] = max(float(clinch_rates[actor]), EPS)
        rates.pop(ActionFamily.RESET_RANGE, None)
        return rates, 0.0
    return _rates


def simulate_set(fights, scale, rounds_df, reference, clean, global_equiv, split):
    rows = []
    actual_total = sim_strike_total = sim_entry_total = sim_clinch_sec_total = 0.0
    for fi, (mr, red_fsr, blue_fsr) in enumerate(fights):
        fid = str(mr.fight_id)
        h = elapsed(mr)
        red_cap, red_runtime = _capabilities(red_fsr, blue_fsr, reference)
        blue_cap, blue_runtime = _capabilities(blue_fsr, red_fsr, reference)
        red_name = str(mr.get("r_name", mr.r_id))
        blue_name = str(mr.get("b_name", mr.b_id))
        red_eq, red_fn, red_on = pit_matchup_equiv(clean, global_equiv, mr._event_date, red_name, blue_name)
        blue_eq, blue_fn, blue_on = pit_matchup_equiv(clean, global_equiv, mr._event_date, blue_name, red_name)
        # Proxy is defined per 5-minute fighter-round. Convert relative level to per-15
        # units, then let the simulation fit the global absolute multiplier.
        clinch_rates = {
            Side.RED: 3.0 * red_eq * float(scale),
            Side.BLUE: 3.0 * blue_eq * float(scale),
        }
        intent_mod._standing_rates = rate_function(clinch_rates)
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
            result = run_causal_path(inputs, seed=SEED_BASE + fi * 1000 + pi, horizon_seconds=h, config=cfg, functions=funcs)
            path_strikes.append(float(sum(ev.selected_action is ActionFamily.CLINCH_STRIKE for ev in result.events)))
            path_entries.append(float(sum(ev.selected_action is ActionFamily.CLINCH_ENTRY for ev in result.events)))
            path_secs.append(float(sum(seg.duration for seg in result.timeline_segments if seg.phase is Phase.CLINCH)))
        sim_strikes = float(np.mean(path_strikes)); sim_entries = float(np.mean(path_entries)); sim_secs = float(np.mean(path_secs))
        actual_total += actual; sim_strike_total += sim_strikes; sim_entry_total += sim_entries; sim_clinch_sec_total += sim_secs
        rows.append({
            "split": split, "scale": float(scale), "fight_id": fid,
            "event_date": str(pd.Timestamp(mr._event_date).date()), "red_name": red_name, "blue_name": blue_name,
            "actual_clinch_sig_attempts_both": actual,
            "red_proxy_equiv_per_round": red_eq, "blue_proxy_equiv_per_round": blue_eq,
            "red_clinch_entry_rate_15m": clinch_rates[Side.RED], "blue_clinch_entry_rate_15m": clinch_rates[Side.BLUE],
            "red_fighter_prior_n": red_fn, "red_opponent_prior_n": red_on,
            "blue_fighter_prior_n": blue_fn, "blue_opponent_prior_n": blue_on,
            "sim_clinch_strikes_mean": sim_strikes, "sim_clinch_entries_mean": sim_entries,
            "sim_clinch_seconds_mean": sim_secs,
            "sim_strikes_per_entry": sim_strikes / sim_entries if sim_entries > 0 else np.nan,
            "paths": PATHS,
        })
    return rows, {
        "actual_clinch_sig_attempts": actual_total,
        "sim_clinch_strikes": sim_strike_total,
        "E_over_O": sim_strike_total / actual_total if actual_total > 0 else None,
        "sim_clinch_entries": sim_entry_total,
        "sim_clinch_seconds": sim_clinch_sec_total,
        "sim_strikes_per_entry": sim_strike_total / sim_entry_total if sim_entry_total > 0 else None,
        "sim_seconds_per_entry": sim_clinch_sec_total / sim_entry_total if sim_entry_total > 0 else None,
    }


def main():
    master = pd.read_parquet(MASTER).drop_duplicates("fight_id").copy()
    master["fight_id"] = master.fight_id.astype(str)
    dc = pick_col(master, "date", "event_date")
    master["_event_date"] = pd.to_datetime(master[dc], errors="coerce").dt.normalize()
    master = master.dropna(subset=["_event_date"])
    rounds = pd.read_parquet(ROUND_STATS).copy()
    clean, global_equiv, strike_norm, ctrl_norm = build_proxy_table(rounds)
    snaps = load_prefight_snapshots(); reference = CapabilityReference.from_latest(load_latest_profiles())
    train = choose_fights(master, rounds, snaps, before_cutoff=True, n=TRAIN_FIGHTS)
    holdout = choose_fights(master, rounds, snaps, before_cutoff=False, n=HOLDOUT_FIGHTS)

    all_rows, grid = [], []
    for scale in SCALES:
        rs, sm = simulate_set(train, scale, rounds, reference, clean, global_equiv, "train_grid")
        all_rows.extend(rs); grid.append({"scale": scale, **sm}); print("TRAIN", scale, sm, flush=True)

    g = sorted(grid, key=lambda x: x["scale"]); target = g[0]["actual_clinch_sig_attempts"]
    fitted = min(g, key=lambda x: abs(x["sim_clinch_strikes"] - target))["scale"]
    for a, b in zip(g[:-1], g[1:]):
        if (a["sim_clinch_strikes"]-target)*(b["sim_clinch_strikes"]-target) <= 0 and b["sim_clinch_strikes"] != a["sim_clinch_strikes"]:
            fitted = a["scale"] + (target-a["sim_clinch_strikes"])*(b["scale"]-a["scale"])/(b["sim_clinch_strikes"]-a["sim_clinch_strikes"])
            break
    fitted = float(fitted)
    tr, train_fit = simulate_set(train, fitted, rounds, reference, clean, global_equiv, "train_fitted")
    ho, hold_fit = simulate_set(holdout, fitted, rounds, reference, clean, global_equiv, "holdout")
    all_rows.extend(tr); all_rows.extend(ho); intent_mod._standing_rates = ORIGINAL_STANDING_RATES

    result = {
        "study": "simulation calibration of proxy-driven clinch-entry rate",
        "production_changed": False,
        "standing_clock_scale": 1.0,
        "inside_clinch_timing_scale": 1.0,
        "proxy_is_literal_entries": False,
        "relative_signal": "PIT shrunk geometric fighter clinch-equivalent tendency x opponent allowance on clean zero-ground rounds",
        "proxy_round_to_rate_mapping": "clinch_rate_15m = 3 * proxy_equiv_per_round * fitted_global_scale",
        "pre2025_proxy_normalization": {"clinch_att": strike_norm, "ctrl_sec": ctrl_norm, "global_equiv": global_equiv},
        "cutoff": str(CUTOFF.date()), "train_fights": TRAIN_FIGHTS, "holdout_fights": HOLDOUT_FIGHTS, "paths_per_fight": PATHS,
        "candidate_grid": grid, "fitted_scale": fitted, "train_fitted": train_fit, "holdout": hold_fit,
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(OUTDIR / "fight_level_results.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
