"""Five-fixture Phase 4B2 terminal KO/TKO mechanics diagnostics."""

from __future__ import annotations

import argparse, json, time
from pathlib import Path
import pandas as pd

from ..components.action_rates import FightFlowRateProvider
from ..config import FightConfig
from ..engine import SimulationEngine
from ..finishes import KOTKOFinishModel
from ..flow_stats import FlowStatsSink
from ..modifiers import DynamicModifierProvider
from ..physiology import ImpactTraumaKnockdownModel, PhysiologyTimeAdvanceModel
from ..rng import RNGManager
from ..stamina import StaminaModel
from .distance_parity import FSR_32_PATH, load_fixture_matchup
from .phase3_flow import PHASE3_FIXTURES


def fixture_report(path: Path = FSR_32_PATH, *, paths=50):
    frame = pd.read_parquet(path); started = time.perf_counter(); report = {}
    for fixture_index, (red, blue, date) in enumerate(PHASE3_FIXTURES):
        profiles = load_fixture_matchup(frame, red, blue, date); rows = []
        for path_index in range(paths):
            stamina = StaminaModel(profiles); sink = FlowStatsSink()
            result = SimulationEngine(FightConfig(), FightFlowRateProvider(profiles, stamina, DynamicModifierProvider()), PhysiologyTimeAdvanceModel(stamina), RNGManager(20260816 + fixture_index * 100000 + path_index), sink, round_recovery_model=stamina, physiology_model=ImpactTraumaKnockdownModel(profiles), finish_model=KOTKOFinishModel(profiles)).run()
            rows.append((result.state, result.sink_result))
        report[f"{red} vs {blue}"] = _summary(rows)
    elapsed = time.perf_counter() - started
    report["performance"] = {"paths": paths * len(PHASE3_FIXTURES), "elapsed_seconds": elapsed, "paths_per_second": paths * len(PHASE3_FIXTURES) / elapsed}
    return report


def _summary(rows):
    terminal = [(state, next((x for x in stats["finishes"] if x.finished), None), stats) for state, stats in rows]
    finished = [row for row in terminal if row[1] is not None]
    times = [state.fight_time_seconds for state, _, _ in finished]
    return {
        "ko_tko_finish_rate": len(finished) / len(rows),
        "scheduled_horizon_rate": sum(state.finish_reason == "scheduled_horizon" for state, _ in rows) / len(rows),
        "average_finish_time_seconds": sum(times) / max(len(times), 1),
        "finish_round_counts": {str(round_no): sum((int(time // 300) + 1) == round_no for time in times) for round_no in (1, 2, 3)},
        "finish_on_kd_fraction": sum(outcome.knockdown for _, outcome, _ in finished) / max(len(finished), 1),
        "direct_finish_fraction": sum(not outcome.knockdown for _, outcome, _ in finished) / max(len(finished), 1),
        "average_impact_ratio_at_finish": sum(outcome.impact_ratio for _, outcome, _ in finished) / max(len(finished), 1),
        "average_trauma_at_finish": sum(getattr(state, f"{outcome.defender.value}_cumulative_trauma") for state, outcome, _ in finished) / max(len(finished), 1),
        "average_acute_at_finish": sum(getattr(state, f"{outcome.defender.value}_acute_vulnerability") for state, outcome, _ in finished) / max(len(finished), 1),
        "average_knockdowns_before_termination": sum(sum(x.knockdown for x in stats["physiology"]) for _, _, stats in terminal) / len(rows),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--paths", type=int, default=50); parser.add_argument("--fsr-path", type=Path, default=FSR_32_PATH); args = parser.parse_args()
    print(json.dumps(fixture_report(args.fsr_path, paths=args.paths), indent=2, sort_keys=True))
