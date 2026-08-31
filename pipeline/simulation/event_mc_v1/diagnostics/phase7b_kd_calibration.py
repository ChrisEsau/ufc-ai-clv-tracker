"""One-parameter temporal calibration of the EVENT MC knockdown midpoint."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml

from ..calibration import DEFAULT_CONFIG_PATH, EventMCCalibration
from ..components.action_rates import FightFlowRateProvider
from ..config import FightConfig
from ..engine import SimulationEngine
from ..finishes import KOTKOFinishModel
from ..judging import DeterministicJudgingModel
from ..modifiers import DynamicModifierProvider
from ..physiology import ImpactTraumaKnockdownModel, PhysiologyTimeAdvanceModel
from ..rng import RNGManager
from ..stamina import StaminaModel
from ..submission_finishes import SubmissionFinishModel
from .phase7a_decomposition import DecompositionSink
from .population_validation import _fight, build_cohort, normalize_method

TARGET_KD_PER_100_LANDED = 0.2800484408113836
TARGET_KD_PER_15MIN = 0.4398013629880078
COARSE_GRID = (8, 12, 16, 24, 32, 48, 64, 96, 128)


def calibration_for_midpoint(midpoint):
    values = deepcopy(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())["defaults"])
    values["knockdown"]["midpoint_impact_ratio"] = float(midpoint)
    return EventMCCalibration(values, f"in-memory midpoint={midpoint}")


def engine_for(fight, seed, sink, calibration):
    stamina = StaminaModel(fight.profiles, calibration=calibration)
    return SimulationEngine(
        FightConfig(fight.rounds), FightFlowRateProvider(fight.profiles, stamina, DynamicModifierProvider(calibration), calibration),
        PhysiologyTimeAdvanceModel(stamina, calibration), RNGManager(seed), sink,
        round_recovery_model=stamina, physiology_model=ImpactTraumaKnockdownModel(fight.profiles, calibration),
        finish_model=KOTKOFinishModel(fight.profiles, calibration), submission_finish_model=SubmissionFinishModel(fight.profiles, calibration),
        judging_model=DeterministicJudgingModel(calibration),
    )


def temporal_cohorts(train_limit=None, holdout_limit=None):
    cohort, fsr = build_cohort(2020)
    train = cohort[cohort.event_date.dt.year <= 2024]
    holdout = cohort[cohort.event_date.dt.year >= 2025]
    if train_limit: train = train.head(train_limit)
    if holdout_limit: holdout = holdout.head(holdout_limit)
    return train.reset_index(drop=True), holdout.reset_index(drop=True), fsr


def evaluate(cohort, fsr, midpoint, paths, seed=20260813):
    calibration = calibration_for_midpoint(midpoint); started=time.perf_counter(); rows=[]; kd_round=Counter(); kd_phase=Counter()
    for fight_index, (_, historical) in enumerate(cohort.iterrows()):
        fight=_fight(historical,fsr)
        for path_index in range(paths):
            sink=DecompositionSink(); result=engine_for(fight,seed+fight_index*100000+path_index,sink,calibration).run(); stats=result.sink_result; impacts=stats["impacts"]
            kds=sum(x["kd"] for x in impacts); kd_round.update(x["round"] for x in impacts if x["kd"]); kd_phase.update(x["phase"] for x in impacts if x["kd"])
            rows.append({"actual_red":int(historical.winner==historical.r_name),"red":int(result.state.winner=="red"),"method":result.state.finish_method,"seconds":result.state.fight_time_seconds,"attempts":sum(stats["attempts"].values()),"landed":len(impacts),"kds":kds})
    frame=pd.DataFrame(rows); exposure=frame.seconds.sum(); landed=frame.landed.sum(); kds=frame.kds.sum(); p=frame.groupby(frame.index//paths).red.mean().to_numpy(); y=cohort.actual_red.to_numpy() if "actual_red" in cohort else (cohort.winner==cohort.r_name).astype(int).to_numpy(); safe=np.clip(p,1e-12,1-1e-12); methods=frame.method.value_counts(normalize=True)
    kd100=kds/landed*100; kd15=kds/exposure*900
    return {"midpoint":float(midpoint),"fights":len(cohort),"paths_per_fight":paths,"attempts_per_15min":frame.attempts.sum()/exposure*900,"landed_per_path":landed/len(frame),"landed_per_15min":landed/exposure*900,"landing_rate":landed/frame.attempts.sum(),"kd_per_path":kds/len(frame),"kd_per_100_landed":kd100,"kd_per_15min":kd15,"zero_kd_share":float((frame.kds==0).mean()),"multi_kd_share":float((frame.kds>=2).mean()),"kd_by_round":dict(kd_round),"kd_by_phase":dict(kd_phase),"method_shares":{m:float(methods.get(m,0)) for m in ("KO_TKO","SUB","DEC")},"mean_fight_duration":float(frame.seconds.mean()),"mean_nondecision_finish_time":float(frame.loc[frame.method!="DEC","seconds"].mean()),"winner_accuracy":float(np.mean((p>=.5)==y)),"brier":float(np.mean((p-y)**2)),"log_loss":float(-np.mean(y*np.log(safe)+(1-y)*np.log(1-safe))),"normalized_error":abs(kd100/TARGET_KD_PER_100_LANDED-1)+abs(kd15/TARGET_KD_PER_15MIN-1),"runtime_seconds":time.perf_counter()-started}


def run(grid=COARSE_GRID, paths=2, train_limit=100, holdout_limit=50, seed=20260813, output=Path("data/diagnostics/event_mc_v1_phase7b.json")):
    train,holdout,fsr=temporal_cohorts(train_limit,holdout_limit); results=[]
    for midpoint in grid: results.append({"train":evaluate(train,fsr,midpoint,paths,seed),"holdout":evaluate(holdout,fsr,midpoint,paths,seed)})
    ranked=sorted(results,key=lambda x:x["train"]["normalized_error"]+x["holdout"]["normalized_error"]); report={"train_dates":[str(train.event_date.min().date()),str(train.event_date.max().date())],"holdout_dates":[str(holdout.event_date.min().date()),str(holdout.event_date.max().date())],"results":ranked}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,sort_keys=True));print(json.dumps(report,indent=2,sort_keys=True));return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--paths",type=int,default=2);p.add_argument("--train-limit",type=int,default=100);p.add_argument("--holdout-limit",type=int,default=50);p.add_argument("--grid",nargs="+",type=float,default=COARSE_GRID);p.add_argument("--seed",type=int,default=20260813);p.add_argument("--output",type=Path,default=Path("data/diagnostics/event_mc_v1_phase7b.json"));run(**vars(p.parse_args()))
