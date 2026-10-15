"""Stage 8 discovery-cohort sweep for the population clinch-entry prior.

The sweep changes only neutral CLINCH_ENTRY choice odds. FSR standing/TD intent,
brain timing, causal engine, legality and mechanics are frozen. The existing
25-fight cohort is a discovery cohort only; any selected prior must be validated
on a fresh holdout before promotion.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.brain.intent_priors import BrainIntentPriors
from pipeline.simulation.event_clock_mc_v2.brain.policy import BrainDecisionContext
from pipeline.simulation.event_clock_mc_v2.brain.timing import BrainTimingContext
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side
from pipeline.simulation.event_clock_mc_v2.engine import (
    EngineConfig,
    EngineFunctions,
    EngineInputs,
    FighterEngineInputs,
    run_causal_path,
)
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    historical_fighter_rows,
    load_latest_profiles,
    load_prefight_snapshots,
)
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1.capability_translation import CapabilityReference
from .stage6_real_causal_path import _capabilities, _mechanics
from .stage8_intent_prior_shadow import IntentPriorChooser
from .stage8_structural_population import (
    MASTER,
    ROUND_STATS,
    actual_side_totals,
    elapsed_seconds,
    pick_col,
    side_rows,
)

STANDING = {ActionFamily.STAND_ATTACK, ActionFamily.STAND_COUNTER}
GROUND = {ActionFamily.GROUND_STRIKE, ActionFamily.BOTTOM_STRIKE}
TD = {ActionFamily.TAKEDOWN_ENTRY, ActionFamily.CLINCH_TAKEDOWN}


def per15(value: float, seconds: float) -> float:
    return float(value * 900.0 / seconds) if seconds > 0 else float("nan")


def parse_ratios(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values or any((not math.isfinite(x) or x < 0.0) for x in values):
        raise ValueError("ratios must be comma-separated finite non-negative values")
    return values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fights", type=int, default=25)
    ap.add_argument("--paths-per-fight", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=20260825)
    ap.add_argument("--ratios", default="0.01,0.02,0.04,0.06,0.08,0.10")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/diagnostics/event_clock_mc_v2/stage8_clinch_intent_sweep.json"),
    )
    args = ap.parse_args()
    ratios = parse_ratios(args.ratios)

    master = pd.read_parquet(MASTER).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    date_col = pick_col(master, "date", "event_date")
    master["_event_date"] = pd.to_datetime(master[date_col], errors="coerce").dt.normalize()
    master = master.dropna(subset=["_event_date"]).sort_values(
        ["_event_date", "fight_id"], ascending=[False, False]
    )
    rounds = pd.read_parquet(ROUND_STATS).copy()
    rs_fight_col = pick_col(rounds, "fight_id", "bout_id")
    available = set(rounds[rs_fight_col].astype(str))
    snapshots = load_prefight_snapshots()
    reference = CapabilityReference.from_latest(load_latest_profiles())
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
                snapshots,
                event_date=fight["_event_date"],
                fight_id=fid,
                fighter_ids=(red_id, blue_id),
            )
            side_rows(rounds, fid, red_id, "red")
            side_rows(rounds, fid, blue_id, "blue")
        except Exception:
            continue
        selected.append((fight, red_fsr, blue_fsr))
    if len(selected) < args.fights:
        raise RuntimeError(f"only {len(selected)} complete fights; requested {args.fights}")

    actual = defaultdict(float)
    actual_seconds = 0.0
    prepared = []
    for fight, red_fsr, blue_fsr in selected:
        fid = str(fight["fight_id"])
        horizon = elapsed_seconds(fight)
        red_id, blue_id = str(fight["r_id"]), str(fight["b_id"])
        red_cap, red_runtime = _capabilities(red_fsr, blue_fsr, reference)
        blue_cap, blue_runtime = _capabilities(blue_fsr, red_fsr, reference)
        inputs = EngineInputs(
            FighterEngineInputs(red_cap, neutral_timing, neutral_decision, _mechanics(red_runtime)),
            FighterEngineInputs(blue_cap, neutral_timing, neutral_decision, _mechanics(blue_runtime)),
        )
        for side, fighter_id in (("red", red_id), ("blue", blue_id)):
            values = actual_side_totals(side_rows(rounds, fid, fighter_id, side))
            actual["standing_att"] += values["distance_att"]
            actual["clinch_att"] += values["clinch_att"]
            actual["ground_att"] += values["ground_att"]
            actual["td_att"] += values["td_att"]
            actual_seconds += horizon
        prepared.append((horizon, inputs, red_runtime, blue_runtime))

    actual_per15 = {key: per15(value, actual_seconds) for key, value in actual.items()}
    results = []

    for ratio_index, ratio in enumerate(ratios):
        counts = Counter()
        phase_seconds = defaultdict(float)
        sim_seconds = 0.0
        illegal = 0
        mismatches = 0
        total_paths = 0

        for fight_index, (horizon, inputs, red_runtime, blue_runtime) in enumerate(prepared):
            chooser = IntentPriorChooser(
                {
                    Side.RED: BrainIntentPriors(
                        red_runtime.standing_rate_15m,
                        red_runtime.takedown_rate_15m,
                        clinch_entry_to_standing_ratio=ratio,
                    ),
                    Side.BLUE: BrainIntentPriors(
                        blue_runtime.standing_rate_15m,
                        blue_runtime.takedown_rate_15m,
                        clinch_entry_to_standing_ratio=ratio,
                    ),
                }
            )
            functions = EngineFunctions(action_chooser=chooser)
            config = EngineConfig(number_of_rounds=max(1, int(math.ceil(horizon / 300.0))))
            for path_index in range(args.paths_per_fight):
                seed = args.seed_base + ratio_index * 1_000_000 + fight_index * 10_000 + path_index
                result = run_causal_path(
                    inputs,
                    seed=seed,
                    horizon_seconds=horizon,
                    config=config,
                    functions=functions,
                )
                total_paths += 1
                sim_seconds += result.reported_through_seconds
                exposure = 0.0
                for segment in result.timeline_segments:
                    phase_seconds[segment.phase.value] += segment.duration
                    exposure += segment.duration
                if not np.isclose(exposure, result.reported_through_seconds, atol=1e-9):
                    mismatches += 1
                for event in result.events:
                    action = event.selected_action
                    if event.source_phase is Phase.GROUND and action in STANDING:
                        illegal += 1
                    if action in STANDING:
                        counts["standing_att"] += 1
                    elif action is ActionFamily.CLINCH_ENTRY:
                        counts["clinch_entry"] += 1
                    elif action is ActionFamily.CLINCH_STRIKE:
                        counts["clinch_att"] += 1
                    elif action in GROUND:
                        counts["ground_att"] += 1
                    if action in TD:
                        counts["td_att"] += 1

        if illegal or mismatches:
            raise AssertionError({"ratio": ratio, "illegal": illegal, "mismatches": mismatches})
        phase_total = sum(phase_seconds.values())
        sim_per15 = {
            "standing_att": per15(counts["standing_att"] / 2.0, sim_seconds),
            "clinch_entry": per15(counts["clinch_entry"] / 2.0, sim_seconds),
            "clinch_att": per15(counts["clinch_att"] / 2.0, sim_seconds),
            "ground_att": per15(counts["ground_att"] / 2.0, sim_seconds),
            "td_att": per15(counts["td_att"] / 2.0, sim_seconds),
        }
        results.append(
            {
                "clinch_entry_to_standing_ratio": ratio,
                "paths": total_paths,
                "sim_per15_per_fighter": sim_per15,
                "phase_share": {
                    key: value / phase_total for key, value in phase_seconds.items()
                },
                "absolute_errors_vs_actual": {
                    key: abs(sim_per15[key] - actual_per15[key])
                    for key in ("standing_att", "clinch_att", "ground_att", "td_att")
                },
            }
        )

    for row in results:
        row["four_metric_mae"] = float(
            np.mean(list(row["absolute_errors_vs_actual"].values()))
        )
    best = min(results, key=lambda row: row["four_metric_mae"])

    payload = {
        "diagnostic": "Stage 8 clinch-entry population-prior sweep",
        "production_policy_changed": False,
        "discovery_cohort_only": True,
        "fights": len(selected),
        "fighter_observations": 2 * len(selected),
        "paths_per_fight_per_ratio": args.paths_per_fight,
        "ratios": ratios,
        "actual_per15_per_fighter": actual_per15,
        "results": results,
        "best_discovery_candidate": {
            "clinch_entry_to_standing_ratio": best["clinch_entry_to_standing_ratio"],
            "four_metric_mae": best["four_metric_mae"],
        },
        "next_step": "validate selected ratio on a fresh historical holdout before promotion",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
