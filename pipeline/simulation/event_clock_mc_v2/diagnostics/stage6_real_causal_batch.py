"""Run a real multi-path Stage 6 causal batch from canonical FSR V3 data.

Diagnostic only. This does not alter simulator mechanics or production entrypoints.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pipeline.simulation.event_clock_mc_v2.brain.policy import BrainDecisionContext
from pipeline.simulation.event_clock_mc_v2.brain.timing import BrainTimingContext
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineInputs, FighterEngineInputs, run_causal_path
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import load_latest_profiles
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1.capability_translation import CapabilityReference

from .stage6_real_causal_path import _capabilities, _mechanics, _resolve_fighter_id


STANDING_STRIKES = {ActionFamily.STAND_ATTACK, ActionFamily.STAND_COUNTER}
CLINCH_STRIKES = {ActionFamily.CLINCH_STRIKE}
GROUND_STRIKES = {ActionFamily.GROUND_STRIKE, ActionFamily.BOTTOM_STRIKE}
TD_ACTIONS = {ActionFamily.TAKEDOWN_ENTRY, ActionFamily.CLINCH_TAKEDOWN}


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--red", default="Aleksandar Rakic")
    parser.add_argument("--blue", default="Marcin Tybura")
    parser.add_argument("--paths", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=20260825)
    parser.add_argument("--horizon", type=float, default=900.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/diagnostics/event_clock_mc_v2/stage6_real_causal_batch_100.json"),
    )
    args = parser.parse_args()
    if args.paths < 1:
        raise ValueError("paths must be >= 1")

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

    path_rows: list[dict[str, object]] = []
    aggregate_actions: Counter[tuple[str, str]] = Counter()
    aggregate_outcomes: Counter[tuple[str, str, str]] = Counter()
    illegal_ground_standing = 0

    for index in range(args.paths):
        seed = args.seed_base + index
        result = run_causal_path(inputs, seed=seed, horizon_seconds=args.horizon)
        exposure = defaultdict(float)
        controller_seconds = defaultdict(float)
        for segment in result.timeline_segments:
            exposure[segment.phase.value] += segment.duration
            if segment.controller is not None:
                controller_seconds[segment.controller.value] += segment.duration

        action_counts: Counter[tuple[str, str]] = Counter()
        td_attempts = Counter()
        td_success = Counter()
        strike_attempts = Counter()
        strike_landed = Counter()
        standing_strike_attempts = Counter()
        standing_strike_landed = Counter()
        clinch_strike_attempts = Counter()
        clinch_strike_landed = Counter()
        ground_strike_attempts = Counter()
        ground_strike_landed = Counter()
        clinch_entries = Counter()
        clinch_entry_success = Counter()

        for event in result.events:
            side = event.actor.value
            action = event.selected_action
            action_counts[(side, action.value)] += 1
            aggregate_actions[(side, action.value)] += 1
            aggregate_outcomes[(side, action.value, event.outcome.value)] += 1

            if event.source_phase is Phase.GROUND and action in STANDING_STRIKES:
                illegal_ground_standing += 1

            if action in TD_ACTIONS:
                td_attempts[side] += 1
                if event.transition_kind is not None and event.resulting_phase is Phase.GROUND:
                    td_success[side] += 1

            if action is ActionFamily.CLINCH_ENTRY:
                clinch_entries[side] += 1
                if event.transition_kind is not None and event.resulting_phase is Phase.CLINCH:
                    clinch_entry_success[side] += 1

            if action in STANDING_STRIKES | CLINCH_STRIKES | GROUND_STRIKES:
                strike_attempts[side] += 1
                if event.outcome.value == "landed":
                    strike_landed[side] += 1
            if action in STANDING_STRIKES:
                standing_strike_attempts[side] += 1
                if event.outcome.value == "landed":
                    standing_strike_landed[side] += 1
            elif action in CLINCH_STRIKES:
                clinch_strike_attempts[side] += 1
                if event.outcome.value == "landed":
                    clinch_strike_landed[side] += 1
            elif action in GROUND_STRIKES:
                ground_strike_attempts[side] += 1
                if event.outcome.value == "landed":
                    ground_strike_landed[side] += 1

        total_exposure = sum(exposure.values())
        if not np.isclose(total_exposure, result.reported_through_seconds, atol=1e-9):
            raise AssertionError((seed, total_exposure, result.reported_through_seconds))

        row = {
            "path_index": index,
            "seed": seed,
            "events": len(result.events),
            "reported_seconds": result.reported_through_seconds,
            "termination": None if result.termination is None else {
                "winner": result.termination.winner.value,
                "finish_method": result.termination.finish_method.value,
            },
            "standing_seconds": exposure["standing"],
            "clinch_seconds": exposure["clinch"],
            "ground_seconds": exposure["ground"],
            "red_control_seconds": controller_seconds["red"],
            "blue_control_seconds": controller_seconds["blue"],
        }
        for side in ("red", "blue"):
            row.update({
                f"{side}_td_attempts": td_attempts[side],
                f"{side}_td_success": td_success[side],
                f"{side}_clinch_entries": clinch_entries[side],
                f"{side}_clinch_entry_success": clinch_entry_success[side],
                f"{side}_strike_attempts": strike_attempts[side],
                f"{side}_strike_landed": strike_landed[side],
                f"{side}_standing_strike_attempts": standing_strike_attempts[side],
                f"{side}_standing_strike_landed": standing_strike_landed[side],
                f"{side}_clinch_strike_attempts": clinch_strike_attempts[side],
                f"{side}_clinch_strike_landed": clinch_strike_landed[side],
                f"{side}_ground_strike_attempts": ground_strike_attempts[side],
                f"{side}_ground_strike_landed": ground_strike_landed[side],
            })
        path_rows.append(row)

    if illegal_ground_standing:
        raise AssertionError(f"illegal ground/standing actions across batch: {illegal_ground_standing}")

    numeric_keys = [key for key in path_rows[0] if key not in {"path_index", "seed", "termination"}]
    distributions = {
        key: _stats([float(row[key]) for row in path_rows])
        for key in numeric_keys
    }

    phase_means = {
        phase: distributions[f"{phase}_seconds"]["mean"]
        for phase in ("standing", "clinch", "ground")
    }
    phase_share = {phase: seconds / args.horizon for phase, seconds in phase_means.items()}

    payload = {
        "diagnostic": "Stage 6 real causal batch",
        "paths": args.paths,
        "seed_base": args.seed_base,
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
            "submission_success_probability": 0.0,
            "ground_escape_probability": 0.40,
            "ground_reversal_probability": 0.30,
        },
        "summary": {
            "illegal_ground_standing_actions": illegal_ground_standing,
            "phase_mean_seconds": phase_means,
            "phase_mean_share": phase_share,
            "distributions": distributions,
            "mean_action_counts_per_path": {
                f"{side}:{action}": count / args.paths
                for (side, action), count in sorted(aggregate_actions.items())
            },
            "aggregate_outcomes": {
                f"{side}:{action}:{outcome}": count
                for (side, action, outcome), count in sorted(aggregate_outcomes.items())
            },
        },
        "paths_detail": path_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 100)
    print("STAGE 6 REAL CAUSAL BATCH")
    print("=" * 100)
    print(f"RED : {args.red} [{red_id}]")
    print(f"BLUE: {args.blue} [{blue_id}]")
    print(f"paths={args.paths} seed_base={args.seed_base} horizon={args.horizon}")
    print("RED capabilities:", asdict(red_cap))
    print("RED runtime:", asdict(red_runtime))
    print("BLUE capabilities:", asdict(blue_cap))
    print("BLUE runtime:", asdict(blue_runtime))
    print("NOTE: submission=0.0, escape=0.40, reversal=0.30 are unresolved structural mechanics placeholders.")
    print("\nBATCH SUMMARY")
    print(json.dumps(payload["summary"], indent=2))
    print(f"\nWROTE {args.output}")


if __name__ == "__main__":
    main()
