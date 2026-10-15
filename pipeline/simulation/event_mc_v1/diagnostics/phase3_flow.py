"""Phase 3 nonterminal full-horizon mechanics diagnostics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from ..components.action_rates import FightFlowRateProvider
from ..components.formulas import clinch_td_interval_probability, interval_hazard_per_second, td_attempt_rate_per_second
from ..components.profiles import Side
from ..config import FightConfig
from ..contracts import NoOpTimeAdvanceModel
from ..engine import SimulationEngine
from ..flow_stats import FlowStatsSink
from ..rng import RNGManager
from .distance_parity import FSR_32_PATH, load_fixture_matchup

PHASE3_FIXTURES = (
    ("Rob Font", "Raul Rosas Jr.", "2026-03-07"),
    ("Merab Dvalishvili", "Petr Yan", "2023-03-11"),
    ("Max Holloway", "Calvin Kattar", "2021-01-16"),
    ("Derrick Lewis", "Chris Daukaus", "2021-12-18"),
    ("Charles Oliveira", "Dustin Poirier", "2021-12-11"),
)


def fixture_report(path: Path = FSR_32_PATH, *, paths: int = 100) -> dict[str, object]:
    frame = pd.read_parquet(path)
    started = time.perf_counter()
    report = {}
    for fixture_index, (red, blue, date) in enumerate(PHASE3_FIXTURES):
        profiles = load_fixture_matchup(frame, red, blue, date)
        runs = []
        for path_index in range(paths):
            result = SimulationEngine(
                FightConfig(), FightFlowRateProvider(profiles), NoOpTimeAdvanceModel(),
                RNGManager(20260813 + fixture_index * 100_000 + path_index), FlowStatsSink(),
            ).run()
            runs.append(result.sink_result)
            if result.state.finish_reason != "scheduled_horizon":
                raise RuntimeError("nonterminal Phase 3 path did not reach horizon")
        summary = _average(runs, paths)
        summary["td_rates_per_second"] = {
            side.value: {
                "distance_phase_2b": td_attempt_rate_per_second(profiles.fighter(side)),
                "clinch": interval_hazard_per_second(clinch_td_interval_probability(profiles.fighter(side))),
            }
            for side in Side
        }
        report[f"{red} vs {blue}"] = summary
    elapsed = time.perf_counter() - started
    report["performance"] = {"paths": paths * len(PHASE3_FIXTURES), "elapsed_seconds": elapsed, "paths_per_second": paths * len(PHASE3_FIXTURES) / elapsed}
    return report


def _average(runs, paths):
    phase = {name: sum(run["phase_seconds"][name] for run in runs) / paths for name in ("distance", "clinch", "ground")}
    attempts = {side: {} for side in ("red", "blue")}
    outcomes = {side: {} for side in ("red", "blue")}
    for side, target, key in ((s, attempts[s], "attempts") for s in attempts):
        for family in set().union(*(run[key][side] for run in runs)):
            target[family] = sum(run[key][side].get(family, 0) for run in runs) / paths
    for side, target, key in ((s, outcomes[s], "outcomes") for s in outcomes):
        for family in set().union(*(run[key][side] for run in runs)):
            target[family] = sum(run[key][side].get(family, 0) for run in runs) / paths
    return {
        "all_paths_reached_horizon": True,
        "average_phase_seconds": phase,
        "phase_share": {name: seconds / 900.0 for name, seconds in phase.items()},
        "average_attempts": attempts,
        "average_outcomes": outcomes,
        "average_clinch_control_seconds": {side: sum(run["clinch_control_seconds"][side] for run in runs) / paths for side in attempts},
        "average_ground_control_seconds": {side: sum(run["ground_control_seconds"][side] for run in runs) / paths for side in attempts},
        "average_transitions": sum(len(run["transitions"]) for run in runs) / paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=100)
    parser.add_argument("--fsr-path", type=Path, default=FSR_32_PATH)
    args = parser.parse_args()
    print(json.dumps(fixture_report(args.fsr_path, paths=args.paths), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
