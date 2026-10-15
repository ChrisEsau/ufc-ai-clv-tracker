"""Five-fixture nonterminal impact/trauma/KD mechanics diagnostics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from ..components.action_rates import FightFlowRateProvider
from ..components.profiles import Side
from ..config import FightConfig
from ..engine import SimulationEngine
from ..flow_stats import FlowStatsSink
from ..modifiers import DynamicModifierProvider
from ..physiology import ImpactTraumaKnockdownModel, PhysiologyTimeAdvanceModel
from ..rng import RNGManager
from ..stamina import StaminaModel
from .distance_parity import FSR_32_PATH, load_fixture_matchup
from .phase3_flow import PHASE3_FIXTURES


def fixture_report(path: Path = FSR_32_PATH, *, paths=50):
    frame = pd.read_parquet(path)
    started = time.perf_counter()
    report = {}
    for fixture_index, (red, blue, date) in enumerate(PHASE3_FIXTURES):
        profiles = load_fixture_matchup(frame, red, blue, date)
        rows = []
        for path_index in range(paths):
            stamina = StaminaModel(profiles)
            result = SimulationEngine(
                FightConfig(), FightFlowRateProvider(profiles, stamina, DynamicModifierProvider()),
                PhysiologyTimeAdvanceModel(stamina), RNGManager(20260815 + fixture_index * 100_000 + path_index),
                FlowStatsSink(), round_recovery_model=stamina,
                physiology_model=ImpactTraumaKnockdownModel(profiles),
            ).run()
            rows.append((result.state, result.sink_result))
        report[f"{red} vs {blue}"] = _summary(rows)
    elapsed = time.perf_counter() - started
    report["performance"] = {"paths": paths * len(PHASE3_FIXTURES), "elapsed_seconds": elapsed, "paths_per_second": paths * len(PHASE3_FIXTURES) / elapsed}
    return report


def _summary(rows):
    physiology = [item for _, stats in rows for item in stats["physiology"]]
    return {
        "all_paths_reached_horizon": all(state.finish_reason == "scheduled_horizon" for state, _ in rows),
        "landed_strikes_processed": len(physiology) / len(rows),
        "average_knockdowns": sum(item.knockdown for item in physiology) / len(rows),
        "average_final_trauma": {side.value: sum(getattr(state, f"{side.value}_cumulative_trauma") for state, _ in rows) / len(rows) for side in Side},
        "average_final_acute": {side.value: sum(getattr(state, f"{side.value}_acute_vulnerability") for state, _ in rows) / len(rows) for side in Side},
        "impact_mean": sum(item.impact for item in physiology) / max(len(physiology), 1),
        "impact_max": max((item.impact for item in physiology), default=0),
        "kd_probability_mean": sum(item.knockdown_probability for item in physiology) / max(len(physiology), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=50)
    parser.add_argument("--fsr-path", type=Path, default=FSR_32_PATH)
    args = parser.parse_args()
    print(json.dumps(fixture_report(args.fsr_path, paths=args.paths), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
