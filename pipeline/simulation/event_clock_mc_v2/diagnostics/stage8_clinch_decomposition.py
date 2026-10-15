"""Measurement-only clinch decomposition for the Stage 8 intent-prior shadow.

Uses the same historical cohort, FSR prefight inputs, timing, mechanics and
intent-prior chooser as stage8_intent_prior_shadow. No production policy changes.
"""
from __future__ import annotations

import argparse
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
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineConfig, EngineFunctions, EngineInputs, FighterEngineInputs, run_causal_path
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import historical_fighter_rows, load_latest_profiles, load_prefight_snapshots
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1.capability_translation import CapabilityReference
from .stage6_real_causal_path import _capabilities, _mechanics
from .stage8_intent_prior_shadow import IntentPriorChooser
from .stage8_structural_population import MASTER, ROUND_STATS, elapsed_seconds, pick_col, side_rows


def per15(value: float, seconds: float) -> float:
    return float(value * 900.0 / seconds) if seconds > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fights", type=int, default=25)
    ap.add_argument("--paths-per-fight", type=int, default=50)
    ap.add_argument("--seed-base", type=int, default=20260825)
    ap.add_argument("--output", type=Path, default=Path("data/diagnostics/event_clock_mc_v2/stage8_clinch_decomposition.json"))
    args = ap.parse_args()

    master = pd.read_parquet(MASTER).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    date_col = pick_col(master, "date", "event_date")
    master["_event_date"] = pd.to_datetime(master[date_col], errors="coerce").dt.normalize()
    master = master.dropna(subset=["_event_date"]).sort_values(["_event_date", "fight_id"], ascending=[False, False])
    rounds = pd.read_parquet(ROUND_STATS).copy()
    rs_fight_col = pick_col(rounds, "fight_id", "bout_id")
    available = set(rounds[rs_fight_col].astype(str))
    snapshots = load_prefight_snapshots()
    latest = load_latest_profiles()
    reference = CapabilityReference.from_latest(latest)
    neutral_timing = BrainTimingContext()
    neutral_decision = BrainDecisionContext()

    selected = []
    for _, fight in master.iterrows():
        if len(selected) >= args.fights:
            break
        fid = str(fight["fight_id"])
        if fid not in available:
            continue
        try:
            red_id, blue_id = str(fight["r_id"]), str(fight["b_id"])
            red_fsr, blue_fsr = historical_fighter_rows(
                snapshots, event_date=fight["_event_date"], fight_id=fid, fighter_ids=(red_id, blue_id)
            )
            side_rows(rounds, fid, red_id, "red")
            side_rows(rounds, fid, blue_id, "blue")
        except Exception:
            continue
        selected.append((fight, red_fsr, blue_fsr))
    if len(selected) < args.fights:
        raise RuntimeError(f"only {len(selected)} complete fights; requested {args.fights}")

    counts = Counter()
    clinch_durations: list[float] = []
    total_path_seconds = 0.0
    total_paths = 0
    fight_detail = []

    for fight_index, (fight, red_fsr, blue_fsr) in enumerate(selected):
        horizon = elapsed_seconds(fight)
        red_cap, red_runtime = _capabilities(red_fsr, blue_fsr, reference)
        blue_cap, blue_runtime = _capabilities(blue_fsr, red_fsr, reference)
        inputs = EngineInputs(
            FighterEngineInputs(red_cap, neutral_timing, neutral_decision, _mechanics(red_runtime)),
            FighterEngineInputs(blue_cap, neutral_timing, neutral_decision, _mechanics(blue_runtime)),
        )
        chooser = IntentPriorChooser({
            Side.RED: BrainIntentPriors(red_runtime.standing_rate_15m, red_runtime.takedown_rate_15m),
            Side.BLUE: BrainIntentPriors(blue_runtime.standing_rate_15m, blue_runtime.takedown_rate_15m),
        })
        functions = EngineFunctions(action_chooser=chooser)
        config = EngineConfig(number_of_rounds=max(1, int(math.ceil(horizon / 300.0))))
        fight_counts = Counter()

        for path_index in range(args.paths_per_fight):
            result = run_causal_path(
                inputs,
                seed=args.seed_base + fight_index * 10000 + path_index,
                horizon_seconds=horizon,
                config=config,
                functions=functions,
            )
            total_paths += 1
            total_path_seconds += result.reported_through_seconds
            for segment in result.timeline_segments:
                if segment.phase is Phase.CLINCH:
                    clinch_durations.append(segment.duration)
                    counts["clinch_segments"] += 1
                    fight_counts["clinch_segments"] += 1
                    counts["clinch_seconds"] += segment.duration
                    fight_counts["clinch_seconds"] += segment.duration
            for event in result.events:
                action = event.selected_action
                if action is ActionFamily.CLINCH_ENTRY:
                    counts["entry_attempts"] += 1; fight_counts["entry_attempts"] += 1
                    if event.resulting_phase is Phase.CLINCH:
                        counts["entry_successes"] += 1; fight_counts["entry_successes"] += 1
                elif action is ActionFamily.CLINCH_STRIKE:
                    counts["clinch_strikes"] += 1; fight_counts["clinch_strikes"] += 1
                elif action is ActionFamily.CLINCH_CONTROL:
                    counts["clinch_control"] += 1; fight_counts["clinch_control"] += 1
                elif action is ActionFamily.CLINCH_TAKEDOWN:
                    counts["clinch_td_attempts"] += 1; fight_counts["clinch_td_attempts"] += 1
                    if event.resulting_phase is Phase.GROUND:
                        counts["clinch_td_successes"] += 1; fight_counts["clinch_td_successes"] += 1
                elif action is ActionFamily.BREAK_CLINCH:
                    counts["break_attempts"] += 1; fight_counts["break_attempts"] += 1
                    if event.resulting_phase is Phase.STANDING:
                        counts["break_successes"] += 1; fight_counts["break_successes"] += 1

        fight_detail.append({
            "fight_id": str(fight["fight_id"]),
            "red_name": str(fight.get("r_name", fight["r_id"])),
            "blue_name": str(fight.get("b_name", fight["b_id"])),
            "mean_clinch_seconds_per_path": fight_counts["clinch_seconds"] / args.paths_per_fight,
            "mean_entry_attempts_per_path": fight_counts["entry_attempts"] / args.paths_per_fight,
            "mean_break_attempts_per_path": fight_counts["break_attempts"] / args.paths_per_fight,
            "mean_clinch_td_attempts_per_path": fight_counts["clinch_td_attempts"] / args.paths_per_fight,
        })

    durations = np.asarray(clinch_durations, dtype=float)
    payload = {
        "diagnostic": "Stage 8 clinch decomposition",
        "production_policy_changed": False,
        "fights": len(selected),
        "fighter_observations": 2 * len(selected),
        "paths_per_fight": args.paths_per_fight,
        "total_paths": total_paths,
        "rates_per15_per_fighter": {
            "clinch_entry_attempts": per15(counts["entry_attempts"] / 2.0, total_path_seconds),
            "clinch_strikes": per15(counts["clinch_strikes"] / 2.0, total_path_seconds),
            "clinch_control_actions": per15(counts["clinch_control"] / 2.0, total_path_seconds),
            "clinch_td_attempts": per15(counts["clinch_td_attempts"] / 2.0, total_path_seconds),
            "break_attempts": per15(counts["break_attempts"] / 2.0, total_path_seconds),
        },
        "success_rates": {
            "clinch_entry": counts["entry_successes"] / counts["entry_attempts"] if counts["entry_attempts"] else None,
            "clinch_takedown": counts["clinch_td_successes"] / counts["clinch_td_attempts"] if counts["clinch_td_attempts"] else None,
            "break_clinch": counts["break_successes"] / counts["break_attempts"] if counts["break_attempts"] else None,
        },
        "clinch_segments": {
            "count": int(counts["clinch_segments"]),
            "mean_seconds": float(np.mean(durations)) if durations.size else 0.0,
            "median_seconds": float(np.median(durations)) if durations.size else 0.0,
            "p90_seconds": float(np.quantile(durations, 0.90)) if durations.size else 0.0,
            "p99_seconds": float(np.quantile(durations, 0.99)) if durations.size else 0.0,
            "max_seconds": float(np.max(durations)) if durations.size else 0.0,
            "phase_share": counts["clinch_seconds"] / total_path_seconds if total_path_seconds else None,
        },
        "raw_counts": dict(counts),
        "fights_detail": fight_detail,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
