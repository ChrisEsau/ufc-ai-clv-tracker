"""Run one real Stage 6 causal path from canonical FSR V3 data.

Diagnostic only. This does not alter simulator mechanics or production entrypoints.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.brain.capabilities import capabilities_from_percentiles
from pipeline.simulation.event_clock_mc_v2.brain.policy import BrainDecisionContext
from pipeline.simulation.event_clock_mc_v2.brain.timing import BrainTimingContext
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineInputs, FighterEngineInputs, run_causal_path
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    derive_runtime_inputs,
    load_latest_profiles,
)
from pipeline.simulation.event_clock_mc_v2.mechanics.config import FighterMechanics
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1.capability_translation import (
    CapabilityReference,
    traits_from_row,
)


def _norm(value: object) -> str:
    return " ".join(str(value).strip().lower().replace("-", " ").split())


def _resolve_fighter_id(name: str, latest: pd.DataFrame, data_root: Path) -> str:
    target = _norm(name)
    name_columns = [c for c in latest.columns if "name" in c.lower()]
    for column in name_columns:
        match = latest[latest[column].astype(str).map(_norm).eq(target)]
        if len(match) == 1:
            return str(match.iloc[0]["fighter_id"])

    candidates: set[str] = set()
    for path in sorted(data_root.rglob("*")):
        if path.suffix.lower() not in {".parquet", ".csv"}:
            continue
        try:
            frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if "fighter_id" not in frame.columns:
            continue
        searchable = [c for c in frame.columns if "name" in c.lower() or "fighter" in c.lower()]
        for column in searchable:
            try:
                rows = frame[frame[column].astype(str).map(_norm).eq(target)]
            except Exception:
                continue
            for fighter_id in rows["fighter_id"].dropna().astype(str):
                candidates.add(fighter_id)
        if len(candidates) == 1:
            fighter_id = next(iter(candidates))
            if fighter_id in set(latest["fighter_id"].astype(str)):
                return fighter_id

    valid = [value for value in candidates if value in set(latest["fighter_id"].astype(str))]
    if len(valid) != 1:
        raise RuntimeError(f"could not uniquely resolve {name!r}; candidates={sorted(valid or candidates)}")
    return valid[0]


def _percentile(series: pd.Series, value: float) -> float:
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise RuntimeError("empty capability reference distribution")
    return float(np.mean(arr <= float(value)))


def _capabilities(attacker: pd.Series, defender: pd.Series, reference: CapabilityReference):
    runtime = derive_runtime_inputs(traits_from_row(attacker), traits_from_row(defender))
    pop = reference.runtime
    return capabilities_from_percentiles(
        standing_rate_percentile=_percentile(pop["standing_rate"], runtime.standing_rate_15m),
        standing_accuracy_percentile=_percentile(pop["standing_acc"], runtime.standing_accuracy),
        takedown_rate_percentile=_percentile(pop["td_rate"], runtime.takedown_rate_15m),
        takedown_completion_percentile=_percentile(pop["td_comp"], runtime.takedown_completion),
        ground_rate_percentile=_percentile(pop["ground_rate"], runtime.ground_slope_rate_15m_own_control),
        ground_accuracy_percentile=_percentile(pop["ground_acc"], runtime.ground_accuracy),
        submission_tendency_percentile=_percentile(
            pop["submission_tendency"], float(attacker["submission_tendency"])
        ),
    ), runtime


def _mechanics(runtime) -> FighterMechanics:
    # Standing, TD and ground probabilities are actual canonical FSR V3 matchup transforms.
    # Escape/reversal do not yet have an approved final Stage 6 runtime translation.
    return FighterMechanics(
        standing_strike_landing_probability=runtime.standing_accuracy,
        takedown_completion_probability=runtime.takedown_completion,
        ground_strike_landing_probability=runtime.ground_accuracy,
        submission_success_probability=0.0,
        ground_escape_probability=0.40,
        ground_reversal_probability=0.30,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--red", default="Aleksandar Rakic")
    parser.add_argument("--blue", default="Marcin Tybura")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--horizon", type=float, default=300.0)
    parser.add_argument("--output", type=Path, default=Path("data/diagnostics/event_clock_mc_v2/stage6_real_causal_path.json"))
    args = parser.parse_args()

    latest = load_latest_profiles()
    latest["fighter_id"] = latest["fighter_id"].astype(str)
    red_id = _resolve_fighter_id(args.red, latest, Path("data"))
    blue_id = _resolve_fighter_id(args.blue, latest, Path("data"))
    lookup = latest.set_index("fighter_id", drop=False)
    red_row = lookup.loc[red_id]
    blue_row = lookup.loc[blue_id]

    reference = CapabilityReference.from_latest(latest)
    red_cap, red_runtime = _capabilities(red_row, blue_row, reference)
    blue_cap, blue_runtime = _capabilities(blue_row, red_row, reference)

    neutral_timing = BrainTimingContext()
    neutral_decision = BrainDecisionContext()
    inputs = EngineInputs(
        red=FighterEngineInputs(red_cap, neutral_timing, neutral_decision, _mechanics(red_runtime)),
        blue=FighterEngineInputs(blue_cap, neutral_timing, neutral_decision, _mechanics(blue_runtime)),
    )
    result = run_causal_path(inputs, seed=args.seed, horizon_seconds=args.horizon)

    exposure = defaultdict(float)
    for segment in result.timeline_segments:
        exposure[segment.phase.value] += segment.duration

    action_counts = Counter((event.actor.value, event.selected_action.value) for event in result.events)
    illegal_ground_standing = [
        asdict(event)
        for event in result.events
        if event.source_phase is Phase.GROUND and event.selected_action.value.startswith("stand_")
    ]

    payload = {
        "diagnostic": "Stage 6 real causal path",
        "seed": args.seed,
        "horizon_seconds": args.horizon,
        "fighters": {
            "red": {
                "name": args.red,
                "fighter_id": red_id,
                "capabilities": asdict(red_cap),
                "runtime": asdict(red_runtime),
                "mechanics": asdict(inputs.red.mechanics),
            },
            "blue": {
                "name": args.blue,
                "fighter_id": blue_id,
                "capabilities": asdict(blue_cap),
                "runtime": asdict(blue_runtime),
                "mechanics": asdict(inputs.blue.mechanics),
            },
        },
        "unresolved_mechanics_placeholders": {
            "ground_escape_probability": 0.40,
            "ground_reversal_probability": 0.30,
        },
        "summary": {
            "event_count": len(result.events),
            "termination": None if result.termination is None else {
                "winner": result.termination.winner.value,
                "finish_method": result.termination.finish_method.value,
            },
            "reached_horizon": result.reached_horizon,
            "reported_through_seconds": result.reported_through_seconds,
            "phase_exposure_seconds": dict(exposure),
            "action_counts": {f"{side}:{action}": count for (side, action), count in sorted(action_counts.items())},
            "illegal_ground_standing_actions": len(illegal_ground_standing),
        },
        "events": [
            {
                "timestamp_seconds": event.timestamp_seconds,
                "actor": event.actor.value,
                "source_phase": event.source_phase.value,
                "selected_action": event.selected_action.value,
                "outcome": event.outcome.value,
                "transition_kind": None if event.transition_kind is None else event.transition_kind.value,
                "resulting_phase": event.resulting_phase.value,
                "resulting_controller": None if event.resulting_controller is None else event.resulting_controller.value,
            }
            for event in result.events
        ],
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

    if illegal_ground_standing:
        raise AssertionError(f"illegal standing actions while grounded: {illegal_ground_standing}")
    total_exposure = sum(exposure.values())
    if not np.isclose(total_exposure, result.reported_through_seconds, atol=1e-9):
        raise AssertionError((total_exposure, result.reported_through_seconds))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 100)
    print("STAGE 6 REAL CAUSAL PATH")
    print("=" * 100)
    print(f"RED : {args.red} [{red_id}]")
    print(f"BLUE: {args.blue} [{blue_id}]")
    print(f"seed={args.seed} horizon={args.horizon}")
    print("\nRED capabilities:", asdict(red_cap))
    print("RED runtime:", asdict(red_runtime))
    print("BLUE capabilities:", asdict(blue_cap))
    print("BLUE runtime:", asdict(blue_runtime))
    print("\nNOTE: submission policy now uses empirical prefight tendency; escape=0.40 and reversal=0.30 remain unresolved structural mechanics placeholders.")
    print("\nSUMMARY")
    print(json.dumps(payload["summary"], indent=2))
    print("\nEVENT TRACE")
    for event in payload["events"]:
        print(json.dumps(event, sort_keys=True))
    print("\nTIMELINE")
    for segment in payload["timeline"]:
        print(json.dumps(segment, sort_keys=True))
    print(f"\nWROTE {args.output}")


if __name__ == "__main__":
    main()
