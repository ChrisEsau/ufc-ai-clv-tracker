"""Five-fixture Phase 4C submission-finish mechanics diagnostic."""

import argparse
import json
import time

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
from ..submission_finishes import SubmissionFinishModel
from .distance_parity import FSR_32_PATH, load_fixture_matchup
from .phase3_flow import PHASE3_FIXTURES


def fixture_report(paths=50):
    frame = pd.read_parquet(FSR_32_PATH)
    started = time.perf_counter()
    report = {}
    for fixture_index, (red, blue, date) in enumerate(PHASE3_FIXTURES):
        rows = []
        profiles = load_fixture_matchup(frame, red, blue, date)
        for path_index in range(paths):
            stamina = StaminaModel(profiles)
            result = SimulationEngine(
                FightConfig(), FightFlowRateProvider(profiles, stamina, DynamicModifierProvider()),
                PhysiologyTimeAdvanceModel(stamina), RNGManager(20260817 + fixture_index * 100000 + path_index), FlowStatsSink(),
                round_recovery_model=stamina, physiology_model=ImpactTraumaKnockdownModel(profiles),
                finish_model=KOTKOFinishModel(profiles), submission_finish_model=SubmissionFinishModel(profiles),
            ).run()
            rows.append(result)
        attempts = sum(sum(row.sink_result["attempts"][side].get("submission_attempt", 0) for side in ("red", "blue")) for row in rows)
        checks = [check for row in rows for check in row.sink_result["submission_checks"]]
        sub_finishes = sum(row.state.finish_method == "SUB" for row in rows)
        report[f"{red} vs {blue}"] = {
            "submission_attempts_per_path": attempts / paths,
            "submission_finish_rate": sub_finishes / paths,
            "p_submission_given_attempt": sub_finishes / attempts if attempts else None,
            "top_attempts": sum(check.position == "top" for check in checks),
            "bottom_attempts": sum(check.position == "bottom" for check in checks),
            "top_conversions": sum(check.position == "top" and check.finished for check in checks),
            "bottom_conversions": sum(check.position == "bottom" and check.finished for check in checks),
            "ko_tko": sum(row.state.finish_method == "KO_TKO" for row in rows),
            "submission": sub_finishes,
            "scheduled_horizon": sum(row.state.finish_reason == "scheduled_horizon" for row in rows),
            "average_submission_finish_time": sum(row.state.fight_time_seconds for row in rows if row.state.finish_method == "SUB") / max(sub_finishes, 1),
            "average_submission_finish_round": sum(int(max(row.state.fight_time_seconds - 1e-12, 0) // 300) + 1 for row in rows if row.state.finish_method == "SUB") / max(sub_finishes, 1),
        }
    elapsed = time.perf_counter() - started
    report["performance"] = {"paths": paths * len(PHASE3_FIXTURES), "elapsed_seconds": elapsed, "paths_per_second": paths * len(PHASE3_FIXTURES) / elapsed}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--paths", type=int, default=50); args = parser.parse_args()
    print(json.dumps(fixture_report(args.paths), indent=2, sort_keys=True))
