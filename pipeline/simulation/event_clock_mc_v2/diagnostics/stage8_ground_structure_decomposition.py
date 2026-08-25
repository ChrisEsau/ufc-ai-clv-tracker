"""Stage 8 decomposition of why realistic TD frequency underproduces ground strikes.

Measurement only. Compares the existing Stage 5 policy against the shadow
FSR-TD-rate-anchored chooser on exactly the same historical fights and seeds.
No production policy, timing, mechanics, FSR, or causal engine is changed.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.brain.policy import BrainDecisionContext
from pipeline.simulation.event_clock_mc_v2.brain.timing import BrainTimingContext, expected_action_delay
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import FightState, Phase, Side
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
from .stage8_structural_population import MASTER, ROUND_STATS, actual_side_totals, elapsed_seconds, pick_col, side_rows
from .stage8_td_rate_anchor_shadow import RateAnchoredChooser

TD_DIRECT = ActionFamily.TAKEDOWN_ENTRY
TD_CLINCH = ActionFamily.CLINCH_TAKEDOWN
GROUND_STRIKES = {ActionFamily.GROUND_STRIKE, ActionFamily.BOTTOM_STRIKE}


def _per15(total: float, fighter_seconds: float) -> float:
    return float(total * 900.0 / fighter_seconds) if fighter_seconds > 0 else float("nan")


def _per_min(total: float, seconds: float) -> float:
    return float(total * 60.0 / seconds) if seconds > 0 else float("nan")


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _summarize(paths, fighter_seconds: float) -> dict[str, object]:
    counts = Counter()
    phase_seconds = defaultdict(float)
    segment_durations = defaultdict(list)
    ground_events = 0
    top_ground_events = 0
    bottom_ground_events = 0
    ground_entry_events = 0
    ground_exit_events = 0

    for result in paths:
        for segment in result.timeline_segments:
            phase_seconds[segment.phase.value] += segment.duration
            segment_durations[segment.phase.value].append(segment.duration)
        for event in result.events:
            action = event.selected_action
            counts[action.value] += 1
            counts[f"phase:{event.source_phase.value}"] += 1
            if action is TD_DIRECT:
                counts["direct_td_attempt"] += 1
                if event.resulting_phase is Phase.GROUND and event.transition_kind is not None:
                    counts["direct_td_land"] += 1
                    ground_entry_events += 1
            elif action is TD_CLINCH:
                counts["clinch_td_attempt"] += 1
                if event.resulting_phase is Phase.GROUND and event.transition_kind is not None:
                    counts["clinch_td_land"] += 1
                    ground_entry_events += 1
            if event.source_phase is Phase.GROUND:
                ground_events += 1
                if action in {ActionFamily.GROUND_STRIKE, ActionFamily.ADVANCE_POSITION, ActionFamily.CONTROL, ActionFamily.DISENGAGE}:
                    top_ground_events += 1
                else:
                    bottom_ground_events += 1
            if event.transition_kind is not None and event.source_phase is Phase.GROUND and event.resulting_phase is Phase.STANDING:
                ground_exit_events += 1

    path_count = len(paths)
    # phase_seconds is summed over paths. Divide count totals by summed phase exposure for phase-specific rates.
    ground_seconds = phase_seconds[Phase.GROUND.value]
    clinch_seconds = phase_seconds[Phase.CLINCH.value]
    standing_seconds = phase_seconds[Phase.STANDING.value]
    total_phase_seconds = sum(phase_seconds.values())

    ground_action_names = [
        ActionFamily.GROUND_STRIKE.value,
        ActionFamily.ADVANCE_POSITION.value,
        ActionFamily.SUBMISSION_ATTACK.value,
        ActionFamily.CONTROL.value,
        ActionFamily.DISENGAGE.value,
        ActionFamily.ESCAPE_STAND.value,
        ActionFamily.IMPROVE_POSITION.value,
        ActionFamily.REVERSAL.value,
        ActionFamily.BOTTOM_STRIKE.value,
    ]
    ground_mix_den = sum(counts[name] for name in ground_action_names)

    return {
        "paths": path_count,
        "phase_share": {k: v / total_phase_seconds for k, v in phase_seconds.items()},
        "mean_segment_duration_seconds": {k: _mean(v) for k, v in segment_durations.items()},
        "phase_action_opportunities_per_min": {
            "standing": _per_min(counts["phase:standing"], standing_seconds),
            "clinch": _per_min(counts["phase:clinch"], clinch_seconds),
            "ground": _per_min(counts["phase:ground"], ground_seconds),
        },
        "ground_action_mix": {
            name: (counts[name] / ground_mix_den if ground_mix_den else float("nan"))
            for name in ground_action_names
        },
        "ground_strike_attempts_per_ground_minute": _per_min(
            counts[ActionFamily.GROUND_STRIKE.value] + counts[ActionFamily.BOTTOM_STRIKE.value],
            ground_seconds,
        ),
        "submission_attempts_per_ground_minute": _per_min(counts[ActionFamily.SUBMISSION_ATTACK.value], ground_seconds),
        "ground_entries_per_fighter15": _per15(ground_entry_events / path_count, fighter_seconds),
        "ground_exits_per_fighter15": _per15(ground_exit_events / path_count, fighter_seconds),
        "direct_td_attempts_per_fighter15": _per15(counts["direct_td_attempt"] / path_count, fighter_seconds),
        "clinch_td_attempts_per_fighter15": _per15(counts["clinch_td_attempt"] / path_count, fighter_seconds),
        "direct_td_landed_per_fighter15": _per15(counts["direct_td_land"] / path_count, fighter_seconds),
        "clinch_td_landed_per_fighter15": _per15(counts["clinch_td_land"] / path_count, fighter_seconds),
        "clinch_entries_per_fighter15": _per15(counts[ActionFamily.CLINCH_ENTRY.value] / path_count, fighter_seconds),
        "break_clinch_per_fighter15": _per15(counts[ActionFamily.BREAK_CLINCH.value] / path_count, fighter_seconds),
        "ground_strike_attempts_per_fighter15": _per15(
            (counts[ActionFamily.GROUND_STRIKE.value] + counts[ActionFamily.BOTTOM_STRIKE.value]) / path_count,
            fighter_seconds,
        ),
        "submission_attempts_per_fighter15": _per15(counts[ActionFamily.SUBMISSION_ATTACK.value] / path_count, fighter_seconds),
        "top_ground_event_share": top_ground_events / ground_events if ground_events else float("nan"),
        "bottom_ground_event_share": bottom_ground_events / ground_events if ground_events else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fights", type=int, default=25)
    ap.add_argument("--paths-per-fight", type=int, default=50)
    ap.add_argument("--seed-base", type=int, default=20260825)
    ap.add_argument("--output", type=Path, default=Path("data/diagnostics/event_clock_mc_v2/stage8_ground_structure_decomposition.json"))
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
    opportunities_per15 = 900.0 / expected_action_delay(FightState(), neutral_timing)

    selected = []
    for _, fight in master.iterrows():
        if len(selected) >= args.fights:
            break
        fight_id = str(fight["fight_id"])
        if fight_id not in available:
            continue
        try:
            red_id, blue_id = str(fight["r_id"]), str(fight["b_id"])
            red_fsr, blue_fsr = historical_fighter_rows(
                snapshots,
                event_date=fight["_event_date"],
                fight_id=fight_id,
                fighter_ids=(red_id, blue_id),
            )
            side_rows(round_stats, fight_id, red_id, "red")
            side_rows(round_stats, fight_id, blue_id, "blue")
        except Exception:
            continue
        selected.append((fight, red_fsr, blue_fsr))
    if len(selected) < args.fights:
        raise RuntimeError(f"only {len(selected)} complete fights; requested {args.fights}")

    baseline_paths = []
    anchor_paths = []
    fighter_seconds = 0.0
    actual = defaultdict(float)

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
        anchored = EngineFunctions(
            action_chooser=RateAnchoredChooser(
                {Side.RED: red_runtime.takedown_rate_15m, Side.BLUE: blue_runtime.takedown_rate_15m},
                opportunities_per15,
            )
        )
        config = EngineConfig(number_of_rounds=max(1, int(math.ceil(horizon / 300.0))))

        for side, fighter_id in (("red", red_id), ("blue", blue_id)):
            values = actual_side_totals(side_rows(round_stats, fight_id, fighter_id, side))
            actual["ground_att"] += values["ground_att"]
            actual["td_att"] += values["td_att"]
            actual["td_land"] += values["td_land"]
            actual["sub_att"] += values["sub_att"]

        for path_index in range(args.paths_per_fight):
            seed = args.seed_base + fight_index * 10000 + path_index
            baseline_paths.append(run_causal_path(inputs, seed=seed, horizon_seconds=horizon, config=config))
            anchor_paths.append(run_causal_path(inputs, seed=seed, horizon_seconds=horizon, config=config, functions=anchored))

    baseline = _summarize(baseline_paths, fighter_seconds)
    anchor = _summarize(anchor_paths, fighter_seconds)
    payload = {
        "diagnostic": "Stage 8 baseline vs TD-rate-anchor ground structure decomposition",
        "production_policy_changed": False,
        "fights": len(selected),
        "paths_per_fight": args.paths_per_fight,
        "paths_each_variant": len(baseline_paths),
        "neutral_opportunities_per15": opportunities_per15,
        "historical_direct_comparators_per_fighter15": {
            "ground_strike_attempts": _per15(actual["ground_att"], fighter_seconds),
            "td_attempts": _per15(actual["td_att"], fighter_seconds),
            "td_landed": _per15(actual["td_land"], fighter_seconds),
            "submission_attempts": _per15(actual["sub_att"], fighter_seconds),
            "note": "No historical phase-time denominator exists, so simulator phase shares and per-phase action rates are not claimed as historical matches.",
        },
        "baseline": baseline,
        "td_rate_anchor_shadow": anchor,
        "anchor_minus_baseline": {
            "ground_share_pp": 100.0 * (anchor["phase_share"].get("ground", 0.0) - baseline["phase_share"].get("ground", 0.0)),
            "clinch_share_pp": 100.0 * (anchor["phase_share"].get("clinch", 0.0) - baseline["phase_share"].get("clinch", 0.0)),
            "standing_share_pp": 100.0 * (anchor["phase_share"].get("standing", 0.0) - baseline["phase_share"].get("standing", 0.0)),
            "mean_ground_segment_seconds": anchor["mean_segment_duration_seconds"].get("ground", float("nan")) - baseline["mean_segment_duration_seconds"].get("ground", float("nan")),
            "ground_actions_per_ground_minute": anchor["phase_action_opportunities_per_min"]["ground"] - baseline["phase_action_opportunities_per_min"]["ground"],
            "ground_strikes_per_ground_minute": anchor["ground_strike_attempts_per_ground_minute"] - baseline["ground_strike_attempts_per_ground_minute"],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print("=" * 100)
    print("STAGE 8 GROUND STRUCTURE DECOMPOSITION")
    print("=" * 100)
    print(json.dumps(payload, indent=2, allow_nan=True))
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
