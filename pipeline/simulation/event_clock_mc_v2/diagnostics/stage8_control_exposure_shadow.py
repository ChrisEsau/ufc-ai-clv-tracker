"""Compare historical UFCStats control time with TD-rate-anchored simulator control exposure.

Measurement only. Uses the raw round-stat control aliases already standardized by
pipeline.round_stats.build_round_fighter_state. No production behavior changes.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.round_stats.build_round_fighter_state import _parse_time_to_seconds
from pipeline.simulation.event_clock_mc_v2.brain.policy import BrainDecisionContext
from pipeline.simulation.event_clock_mc_v2.brain.timing import BrainTimingContext, expected_action_delay
from pipeline.simulation.event_clock_mc_v2.causal.state import FightState, Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineConfig, EngineFunctions, EngineInputs, FighterEngineInputs, run_causal_path
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import historical_fighter_rows, load_latest_profiles, load_prefight_snapshots
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1.capability_translation import CapabilityReference
from .stage6_real_causal_path import _capabilities, _mechanics
from .stage8_structural_population import MASTER, ROUND_STATS, elapsed_seconds, pick_col, side_rows
from .stage8_td_rate_anchor_shadow import RateAnchoredChooser

CONTROL_ALIASES = (
    "control_seconds", "ctrl_seconds", "ctrl_sec", "control_time_seconds",
    "control_time_sec", "ctrl", "control", "control_time",
)


def control_sum(frame: pd.DataFrame) -> float:
    col = pick_col(frame, *CONTROL_ALIASES, required=False)
    if col is None:
        raise RuntimeError(f"no control field; checked {CONTROL_ALIASES}")
    values = frame[col].map(_parse_time_to_seconds)
    return float(pd.to_numeric(values, errors="coerce").fillna(0.0).sum())


def per15(total: float, fighter_seconds: float) -> float:
    return float(total * 900.0 / fighter_seconds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fights", type=int, default=25)
    ap.add_argument("--paths-per-fight", type=int, default=50)
    args = ap.parse_args()

    master = pd.read_parquet(MASTER).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    date_col = pick_col(master, "date", "event_date")
    master["_event_date"] = pd.to_datetime(master[date_col], errors="coerce").dt.normalize()
    master = master.dropna(subset=["_event_date"]).sort_values(["_event_date", "fight_id"], ascending=[False, False])
    round_stats = pd.read_parquet(ROUND_STATS).copy()
    rs_fight_col = pick_col(round_stats, "fight_id", "bout_id")
    available = set(round_stats[rs_fight_col].astype(str))
    snapshots = load_prefight_snapshots()
    latest = load_latest_profiles()
    reference = CapabilityReference.from_latest(latest)
    neutral_timing = BrainTimingContext()
    neutral_decision = BrainDecisionContext()
    opportunities = 900.0 / expected_action_delay(FightState(), neutral_timing)

    selected = []
    for _, fight in master.iterrows():
        if len(selected) >= args.fights:
            break
        fight_id = str(fight["fight_id"])
        if fight_id not in available:
            continue
        try:
            red_id, blue_id = str(fight["r_id"]), str(fight["b_id"])
            red_fsr, blue_fsr = historical_fighter_rows(snapshots, event_date=fight["_event_date"], fight_id=fight_id, fighter_ids=(red_id, blue_id))
            side_rows(round_stats, fight_id, red_id, "red")
            side_rows(round_stats, fight_id, blue_id, "blue")
        except Exception:
            continue
        selected.append((fight, red_fsr, blue_fsr))

    actual_control = 0.0
    sim_control = 0.0
    fighter_seconds = 0.0
    actual_td_land = 0.0
    sim_ground_entries = 0.0
    sim_ground_segment_seconds = 0.0
    sim_ground_segments = 0

    for fight_index, (fight, red_fsr, blue_fsr) in enumerate(selected):
        fight_id = str(fight["fight_id"])
        red_id, blue_id = str(fight["r_id"]), str(fight["b_id"])
        horizon = elapsed_seconds(fight)
        fighter_seconds += 2.0 * horizon
        red_cap, red_runtime = _capabilities(red_fsr, blue_fsr, reference)
        blue_cap, blue_runtime = _capabilities(blue_fsr, red_fsr, reference)
        inputs = EngineInputs(
            FighterEngineInputs(red_cap, neutral_timing, neutral_decision, _mechanics(red_runtime)),
            FighterEngineInputs(blue_cap, neutral_timing, neutral_decision, _mechanics(blue_runtime)),
        )
        functions = EngineFunctions(action_chooser=RateAnchoredChooser(
            {Side.RED: red_runtime.takedown_rate_15m, Side.BLUE: blue_runtime.takedown_rate_15m}, opportunities
        ))
        config = EngineConfig(number_of_rounds=max(1, int(math.ceil(horizon / 300.0))))

        for side, fighter_id in (("red", red_id), ("blue", blue_id)):
            rows = side_rows(round_stats, fight_id, fighter_id, side)
            actual_control += control_sum(rows)
            td_col = pick_col(rows, "td_landed", "td_land", "takedowns_landed", required=False)
            if td_col is not None:
                actual_td_land += float(pd.to_numeric(rows[td_col], errors="coerce").fillna(0).sum())

        path_control = []
        path_entries = []
        for path_index in range(args.paths_per_fight):
            result = run_causal_path(inputs, seed=20260825 + fight_index * 10000 + path_index, horizon_seconds=horizon, config=config, functions=functions)
            controlled = {Side.RED: 0.0, Side.BLUE: 0.0}
            entries = 0
            for segment in result.timeline_segments:
                if segment.controller is not None:
                    controlled[segment.controller] += segment.duration
                if segment.phase.value == "ground":
                    sim_ground_segment_seconds += segment.duration
                    sim_ground_segments += 1
            for event in result.events:
                if event.transition_kind is not None and event.resulting_phase.value == "ground" and event.source_phase.value != "ground":
                    entries += 1
            path_control.append(controlled[Side.RED] + controlled[Side.BLUE])
            path_entries.append(entries)
        sim_control += float(np.mean(path_control))
        sim_ground_entries += float(np.mean(path_entries))

    print("=" * 100)
    print("STAGE 8 CONTROL EXPOSURE AUDIT — TD RATE ANCHOR")
    print("=" * 100)
    print(f"fights={len(selected)} paths_per_fight={args.paths_per_fight}")
    print(f"actual control sec/fighter15 : {per15(actual_control, fighter_seconds):.4f}")
    print(f"sim controller sec/fighter15: {per15(sim_control, fighter_seconds):.4f}")
    print(f"actual TD landed/fighter15  : {per15(actual_td_land, fighter_seconds):.4f}")
    print(f"sim ground entries/fighter15: {per15(sim_ground_entries, fighter_seconds):.4f}")
    print(f"sim mean ground segment sec : {sim_ground_segment_seconds / sim_ground_segments:.4f}")
    print(f"actual control sec / TD landed: {actual_control / actual_td_land if actual_td_land else float('nan'):.4f}")
    print(f"sim controller sec / ground entry: {sim_control / sim_ground_entries if sim_ground_entries else float('nan'):.4f}")


if __name__ == "__main__":
    main()
