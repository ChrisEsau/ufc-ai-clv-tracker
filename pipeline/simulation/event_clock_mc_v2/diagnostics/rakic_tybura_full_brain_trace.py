"""One-path Rakic-Tybura causal trace with full brain action probabilities.

Diagnostic only: no brain, timing, FSR, or mechanics behavior is changed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.brain.policy import (
    action_probabilities,
    choose_action,
)
from pipeline.simulation.event_clock_mc_v2.causal.state import Side
from pipeline.simulation.event_clock_mc_v2.diagnostics.stage6_real_causal_path import (
    _capabilities,
    _resolve_fighter_id,
)
from pipeline.simulation.event_clock_mc_v2.engine import (
    EngineFunctions,
    EngineInputs,
    FighterEngineInputs,
    run_causal_path,
)
from pipeline.simulation.event_clock_mc_v2.brain.policy import BrainDecisionContext
from pipeline.simulation.event_clock_mc_v2.brain.timing import BrainTimingContext
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import derive_runtime_inputs, load_latest_profiles
from pipeline.simulation.event_clock_mc_v2.physiology_adapter import fighter_mechanics_from_prefight
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1.capability_translation import (
    CapabilityReference,
    traits_from_row,
)


def _find_prefight_rows(red_id: str, blue_id: str) -> tuple[pd.Series, pd.Series]:
    path = Path("data/fsr_v3/fsr_v3_prefight_snapshots.parquet")
    frame = pd.read_parquet(path)
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    wanted = frame[frame["fighter_id"].isin({red_id, blue_id})].copy()
    if wanted.empty:
        raise RuntimeError("no prefight rows found for requested fighters")

    candidates = []
    for fight_id, group in wanted.groupby("fight_id", sort=False):
        ids = set(group["fighter_id"].astype(str))
        if {red_id, blue_id}.issubset(ids):
            date = pd.to_datetime(group["event_date"], errors="coerce").max()
            candidates.append((date, str(fight_id), group))
    if not candidates:
        raise RuntimeError("could not find a shared historical fight for Rakic and Tybura")
    candidates.sort(key=lambda row: (pd.Timestamp.min if pd.isna(row[0]) else row[0], row[1]))
    _, fight_id, group = candidates[-1]
    red = group[group["fighter_id"].eq(red_id)]
    blue = group[group["fighter_id"].eq(blue_id)]
    if len(red) != 1 or len(blue) != 1:
        raise RuntimeError(f"non-unique prefight rows for fight_id={fight_id}")
    return red.iloc[0], blue.iloc[0]


def _controller_snapshot(state):
    return {
        "phase": state.phase.value,
        "clinch_controller": None if state.clinch_controller is None else state.clinch_controller.value,
        "ground_controller": None if state.ground_controller is None else state.ground_controller.value,
        "round_number": state.round_number,
        "fight_time_seconds": state.fight_time_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--red", default="Aleksandar Rakic")
    parser.add_argument("--blue", default="Marcin Tybura")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--horizon", type=float, default=900.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/diagnostics/event_clock_mc_v2/rakic_tybura_full_brain_trace.json"),
    )
    args = parser.parse_args()

    latest = load_latest_profiles()
    latest["fighter_id"] = latest["fighter_id"].astype(str)
    red_id = _resolve_fighter_id(args.red, latest, Path("data"))
    blue_id = _resolve_fighter_id(args.blue, latest, Path("data"))
    red_row, blue_row = _find_prefight_rows(red_id, blue_id)

    reference = CapabilityReference.from_latest(latest)
    red_cap, red_runtime = _capabilities(red_row, blue_row, reference)
    blue_cap, blue_runtime = _capabilities(blue_row, red_row, reference)

    red_mechanics = fighter_mechanics_from_prefight(red_row, red_runtime)
    blue_mechanics = fighter_mechanics_from_prefight(blue_row, blue_runtime)
    neutral_timing = BrainTimingContext()
    neutral_decision = BrainDecisionContext()
    inputs = EngineInputs(
        red=FighterEngineInputs(red_cap, neutral_timing, neutral_decision, red_mechanics),
        blue=FighterEngineInputs(blue_cap, neutral_timing, neutral_decision, blue_mechanics),
    )

    decisions: list[dict] = []

    def traced_chooser(state, actor, capabilities, context, rng, config):
        distribution = action_probabilities(state, actor, capabilities, context, config)
        utilities = np.asarray([row.utility for row in distribution], dtype=float)
        max_utility = float(np.max(utilities))
        weights = np.exp((utilities - max_utility) / config.softmax_temperature)
        selected = choose_action(state, actor, capabilities, context, rng, config)
        decisions.append(
            {
                "actor": actor.value,
                "pre_state": _controller_snapshot(state),
                "context": asdict(context),
                "softmax_temperature": config.softmax_temperature,
                "distribution": [
                    {
                        "action": row.action_family.value,
                        "utility": row.utility,
                        "softmax_weight": float(weight),
                        "probability": row.probability,
                    }
                    for row, weight in zip(distribution, weights, strict=True)
                ],
                "selected_action": selected.value,
            }
        )
        return selected

    functions = EngineFunctions(action_chooser=traced_chooser)
    result = run_causal_path(
        inputs,
        seed=args.seed,
        horizon_seconds=args.horizon,
        functions=functions,
    )
    if len(decisions) != len(result.events):
        raise AssertionError((len(decisions), len(result.events)))

    event_trace = []
    for index, (decision, event) in enumerate(zip(decisions, result.events, strict=True), start=1):
        if decision["selected_action"] != event.selected_action.value:
            raise AssertionError((decision["selected_action"], event.selected_action.value))
        event_trace.append(
            {
                "event_index": index,
                "timestamp_seconds": event.timestamp_seconds,
                "brain": decision,
                "mechanics": {
                    "outcome": event.outcome.value,
                    "transition_kind": None if event.transition_kind is None else event.transition_kind.value,
                    "impact": event.impact,
                    "knockdown": event.knockdown,
                },
                "resulting_state": {
                    "phase": event.resulting_phase.value,
                    "controller": None if event.resulting_controller is None else event.resulting_controller.value,
                    "actor_memory": asdict(event.resulting_actor_memory),
                },
            }
        )

    final_phys = result.final_state.physiology
    payload = {
        "diagnostic": "Rakic-Tybura one-path full brain probability trace",
        "seed": args.seed,
        "horizon_seconds": args.horizon,
        "historical_identity": {
            "event_date": str(red_row["event_date"]),
            "fight_id": str(red_row["fight_id"]),
        },
        "fighters": {
            "red": {
                "name": args.red,
                "fighter_id": red_id,
                "capabilities": asdict(red_cap),
                "runtime": asdict(red_runtime),
                "mechanics": asdict(red_mechanics),
            },
            "blue": {
                "name": args.blue,
                "fighter_id": blue_id,
                "capabilities": asdict(blue_cap),
                "runtime": asdict(blue_runtime),
                "mechanics": asdict(blue_mechanics),
            },
        },
        "summary": {
            "event_count": len(result.events),
            "termination": None
            if result.termination is None
            else {
                "winner": result.termination.winner.value,
                "finish_method": result.termination.finish_method.value,
            },
            "reported_through_seconds": result.reported_through_seconds,
            "reached_horizon": result.reached_horizon,
            "final_physiology": {
                "red": asdict(final_phys.red),
                "blue": asdict(final_phys.blue),
            },
        },
        "events": event_trace,
        "timeline": [
            {
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "duration": segment.duration,
                "phase": segment.phase.value,
                "controller": None if segment.controller is None else segment.controller.value,
                "entry_reason": segment.entry_reason,
                "exit_reason": segment.exit_reason,
            }
            for segment in result.timeline_segments
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 100)
    print("RAKIC-TYBURA — ONE PATH FULL BRAIN TRACE")
    print("=" * 100)
    print(json.dumps(payload["historical_identity"], indent=2))
    print(json.dumps(payload["fighters"], indent=2))
    print(json.dumps(payload["summary"], indent=2))
    for row in event_trace:
        print(json.dumps(row, sort_keys=True))
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
