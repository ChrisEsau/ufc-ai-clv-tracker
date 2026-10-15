"""One-parameter temporal calibration of the EVENT MC finish midpoint."""

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
from .phase7a_decomposition import DecompositionSink
from .phase7b_kd_calibration import engine_for, temporal_cohorts
from .population_validation import _fight, normalize_method, observed_duration_seconds

COARSE_GRID = (10, 16, 24, 32, 48, 64, 96, 128, 192)


def calibration_for_finish_midpoint(midpoint):
    values = deepcopy(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())["defaults"])
    values["finish"]["midpoint_impact_ratio"] = float(midpoint)
    return EventMCCalibration(values, f"in-memory finish midpoint={midpoint}")


def historical_targets(cohort):
    methods = cohort["method"].map(normalize_method)
    finishes = cohort[methods != "DEC"]
    return {
        "ko_tko_share": float((methods == "KO_TKO").mean()),
        "mean_nondecision_finish_time": float(finishes.apply(observed_duration_seconds, axis=1).mean()),
        "finish_round_shares": {str(r): float((finishes.finish_round == r).mean()) for r in range(1, 6)},
    }


def evaluate(cohort, fsr, midpoint, paths, seed=20260813):
    calibration = calibration_for_finish_midpoint(midpoint); started=time.perf_counter(); rows=[]; impacts=[]
    for fight_index, (_, historical) in enumerate(cohort.iterrows()):
        fight = _fight(historical, fsr)
        for path_index in range(paths):
            sink=DecompositionSink(); result=engine_for(fight, seed+fight_index*100000+path_index, sink, calibration).run(); stats=result.sink_result; evidence=list(stats["impacts"]); impacts.extend(evidence)
            rows.append({"fight":fight_index,"actual_red":int(historical.winner==historical.r_name),"red":int(result.state.winner=="red"),"method":result.state.finish_method,"seconds":result.state.fight_time_seconds,"attempts":sum(stats["attempts"].values()),"landed":len(evidence),"kds":sum(x["kd"] for x in evidence),"prior_kds":sum(x["kd"] for x in evidence[:-1]) if result.state.finish_method=="KO_TKO" else 0})
    frame=pd.DataFrame(rows); impact=pd.DataFrame(impacts); exposure=frame.seconds.sum(); methods=frame.method.value_counts(normalize=True); ko=frame[frame.method=="KO_TKO"]; nondec=frame[frame.method!="DEC"]; p=frame.groupby("fight").red.mean().to_numpy(); y=cohort.apply(lambda x:int(x.winner==x.r_name),axis=1).to_numpy(); safe=np.clip(p,1e-12,1-1e-12); kd=impact[impact.kd]; nonkd=impact[~impact.kd]; ending=impact[impact.finished]; rounds=(np.maximum(nondec.seconds.to_numpy()-1e-12,0)//300+1).astype(int); historical=historical_targets(cohort)
    return {"midpoint":float(midpoint),"fights":len(cohort),"paths_per_fight":paths,"historical":historical,"ko_error":abs(float(methods.get("KO_TKO",0))-historical["ko_tko_share"]),"method_shares":{m:float(methods.get(m,0)) for m in ("KO_TKO","SUB","DEC")},"mean_nondecision_finish_time":float(nondec.seconds.mean()),"finish_round_shares":{str(r):float(np.mean(rounds==r)) for r in range(1,6)},"p_finish_given_kd":float(kd.finished.mean()) if len(kd) else None,"p_finish_given_non_kd":float(nonkd.finished.mean()),"non_kd_finishing_strike_share":float((~ending.kd).mean()),"ko_paths_zero_prior_kd_share":float((ko.prior_kds==0).mean()),"finish_checks_per_path":float(frame.landed.mean()),"finish_checks_per_15min":float(frame.landed.sum()/exposure*900),"kd_per_100_landed":float(frame.kds.sum()/frame.landed.sum()*100),"kd_per_15min":float(frame.kds.sum()/exposure*900),"zero_kd_share":float((frame.kds==0).mean()),"multi_kd_share":float((frame.kds>=2).mean()),"attempts_per_15min":float(frame.attempts.sum()/exposure*900),"landed_per_15min":float(frame.landed.sum()/exposure*900),"winner_accuracy":float(np.mean((p>=.5)==y)),"brier":float(np.mean((p-y)**2)),"log_loss":float(-np.mean(y*np.log(safe)+(1-y)*np.log(1-safe))),"runtime_seconds":time.perf_counter()-started}


def run(grid=COARSE_GRID, paths=2, train_limit=50, holdout_limit=25, seed=20260813, output=Path("data/diagnostics/event_mc_v1_phase7c.json")):
    train,holdout,fsr=temporal_cohorts(train_limit,holdout_limit); results=[]
    for midpoint in grid: results.append({"train":evaluate(train,fsr,midpoint,paths,seed),"holdout":evaluate(holdout,fsr,midpoint,paths,seed)})
    results.sort(key=lambda x:x["train"]["ko_error"]+x["holdout"]["ko_error"]); report={"train_dates":[str(train.event_date.min().date()),str(train.event_date.max().date())],"holdout_dates":[str(holdout.event_date.min().date()),str(holdout.event_date.max().date())],"results":results};output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,sort_keys=True));print(json.dumps(report,indent=2,sort_keys=True));return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--paths",type=int,default=2);p.add_argument("--train-limit",type=int,default=50);p.add_argument("--holdout-limit",type=int,default=25);p.add_argument("--grid",nargs="+",type=float,default=COARSE_GRID);p.add_argument("--seed",type=int,default=20260813);p.add_argument("--output",type=Path,default=Path("data/diagnostics/event_mc_v1_phase7c.json"));run(**vars(p.parse_args()))
