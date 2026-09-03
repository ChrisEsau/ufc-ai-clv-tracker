"""Five-fixture deterministic judging mechanics diagnostic."""

import argparse
import json

import pandas as pd

from ..components.action_rates import FightFlowRateProvider
from ..config import FightConfig
from ..engine import SimulationEngine
from ..finishes import KOTKOFinishModel
from ..flow_stats import FlowStatsSink
from ..judging import DeterministicJudgingModel
from ..modifiers import DynamicModifierProvider
from ..physiology import ImpactTraumaKnockdownModel, PhysiologyTimeAdvanceModel
from ..rng import RNGManager
from ..stamina import StaminaModel
from ..submission_finishes import SubmissionFinishModel
from .distance_parity import FSR_32_PATH, load_fixture_matchup
from .phase3_flow import PHASE3_FIXTURES


def fixture_report(paths=20):
    frame = pd.read_parquet(FSR_32_PATH); report = {}
    for fixture_index, (red, blue, date) in enumerate(PHASE3_FIXTURES):
        results = []
        profiles = load_fixture_matchup(frame, red, blue, date)
        for path_index in range(paths):
            stamina = StaminaModel(profiles)
            results.append(SimulationEngine(
                FightConfig(), FightFlowRateProvider(profiles, stamina, DynamicModifierProvider()), PhysiologyTimeAdvanceModel(stamina),
                RNGManager(20260818 + fixture_index * 100000 + path_index), FlowStatsSink(), round_recovery_model=stamina,
                physiology_model=ImpactTraumaKnockdownModel(profiles), finish_model=KOTKOFinishModel(profiles),
                submission_finish_model=SubmissionFinishModel(profiles), judging_model=DeterministicJudgingModel(),
            ).run())
        report[f"{red} vs {blue}"] = {
            "ko_tko": sum(r.state.finish_method == "KO_TKO" for r in results),
            "submission": sum(r.state.finish_method == "SUB" for r in results),
            "decision": sum(r.state.finish_method == "DEC" for r in results),
            "red_decision_wins": sum(r.state.finish_method == "DEC" and r.state.winner == "red" for r in results),
            "blue_decision_wins": sum(r.state.finish_method == "DEC" and r.state.winner == "blue" for r in results),
            "draws": sum(r.state.winner is None for r in results),
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--paths", type=int, default=20); args = parser.parse_args()
    print(json.dumps(fixture_report(args.paths), indent=2, sort_keys=True))
