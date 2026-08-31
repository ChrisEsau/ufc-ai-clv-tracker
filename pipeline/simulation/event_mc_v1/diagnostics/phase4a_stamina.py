"""Frozen-fixture Phase 3-neutral versus Phase 4A stamina diagnostics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from ..components.action_rates import FightFlowRateProvider
from ..components.profiles import Side
from ..config import FightConfig
from ..contracts import NoOpTimeAdvanceModel
from ..engine import SimulationEngine
from ..flow_stats import FlowStatsSink
from ..modifiers import DynamicModifierProvider
from ..rng import RNGManager
from ..stamina import StaminaModel, StaminaTimeAdvanceModel
from .distance_parity import FSR_32_PATH, load_fixture_matchup
from .phase3_flow import PHASE3_FIXTURES


def fixture_report(path: Path = FSR_32_PATH, *, paths: int = 100):
    frame = pd.read_parquet(path)
    started = time.perf_counter()
    report = {}
    modifiers = DynamicModifierProvider()
    for fixture_index, (red, blue, date) in enumerate(PHASE3_FIXTURES):
        profiles = load_fixture_matchup(frame, red, blue, date)
        arms = {"phase3_neutral": [], "phase4a_active": []}
        for path_index in range(paths):
            seed = 20260814 + fixture_index * 100_000 + path_index
            neutral = SimulationEngine(FightConfig(), FightFlowRateProvider(profiles), NoOpTimeAdvanceModel(), RNGManager(seed), FlowStatsSink()).run()
            stamina = StaminaModel(profiles)
            active = SimulationEngine(FightConfig(), FightFlowRateProvider(profiles, stamina, modifiers), StaminaTimeAdvanceModel(stamina), RNGManager(seed), FlowStatsSink(), round_recovery_model=stamina).run()
            arms["phase3_neutral"].append((neutral.state, neutral.sink_result))
            arms["phase4a_active"].append((active.state, active.sink_result))
        report[f"{red} vs {blue}"] = {name: _summarize(values, profiles, modifiers) for name, values in arms.items()}
    elapsed = time.perf_counter() - started
    report["performance"] = {"paths_per_arm": paths * len(PHASE3_FIXTURES), "elapsed_seconds": elapsed, "simulations_per_second": 2 * paths * len(PHASE3_FIXTURES) / elapsed}
    return report


def _summarize(values, profiles, modifier_provider):
    count = len(values)
    final = {side.value: sum(getattr(state, f"{side.value}_stamina") for state, _ in values) / count for side in Side}
    return {
        "all_paths_reached_horizon": all(state.finish_reason == "scheduled_horizon" for state, _ in values),
        "average_final_stamina": final,
        "average_phase_seconds": {phase: sum(stats["phase_seconds"][phase] for _, stats in values) / count for phase in ("distance", "clinch", "ground")},
        "average_attempts": {side.value: {family: sum(stats["attempts"][side.value].get(family, 0) for _, stats in values) / count for family in set().union(*(stats["attempts"][side.value] for _, stats in values))} for side in Side},
        "average_final_modifiers": {side.value: modifier_provider.modifiers(profiles.fighter(side), _state(final), side).__dict__ for side in Side},
        "sample_round_entries": values[0][1]["stamina_round_entries"],
    }


def _state(final):
    from ..state import FightState
    return FightState(red_stamina=final["red"], blue_stamina=final["blue"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=100)
    parser.add_argument("--fsr-path", type=Path, default=FSR_32_PATH)
    args = parser.parse_args()
    print(json.dumps(fixture_report(args.fsr_path, paths=args.paths), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
