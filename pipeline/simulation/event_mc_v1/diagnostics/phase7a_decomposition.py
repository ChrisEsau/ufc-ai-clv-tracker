"""Measurement-only decomposition of EVENT MC strike, KD, and KO exposure."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ..components.actions import ActionAttempt
from ..events import ConsequenceEvent, PrimaryEvent
from ..finishes import FinishOutcome
from ..physiology import PhysiologyOutcome
from ..single_fight import build_engine
from .population_validation import build_cohort, _fight, observed_duration_seconds


STRIKE_FAMILIES = {"strike": "distance", "clinch_strike": "clinch", "ground_strike": "ground"}


@dataclass
class DecompositionSink:
    attempts: Counter = field(default_factory=Counter)
    landed: Counter = field(default_factory=Counter)
    impacts: list[dict] = field(default_factory=list)
    exposure_seconds: float = 0.0
    pending: tuple[PhysiologyOutcome, object] | None = None

    def on_time_advance(self, dt, before, after):
        self.exposure_seconds += dt

    def on_event(self, event, before, after):
        if isinstance(event, PrimaryEvent) and isinstance(event.payload, ActionAttempt):
            phase = STRIKE_FAMILIES.get(event.payload.action_family)
            if phase:
                self.attempts[phase] += 1
        if not isinstance(event, ConsequenceEvent):
            return
        if isinstance(event.payload, PhysiologyOutcome):
            self.pending = (event.payload, after)
            self.landed[event.payload.phase] += 1
        elif isinstance(event.payload, FinishOutcome):
            physiology, snapshot = self.pending
            defender = physiology.defender.value
            self.impacts.append({
                "impact": physiology.impact, "kd": physiology.knockdown,
                "finished": event.payload.finished, "phase": physiology.phase,
                "round": int(max(event.timestamp_seconds - 1e-12, 0) // 300) + 1,
                "trauma": getattr(snapshot, f"{defender}_cumulative_trauma"),
                "acute": getattr(snapshot, f"{defender}_acute_vulnerability"),
            })
            self.pending = None

    def finalize(self):
        return {"attempts": dict(self.attempts), "landed": dict(self.landed), "impacts": tuple(self.impacts), "exposure_seconds": self.exposure_seconds}


def _distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {key: None for key in ("count", "mean", "median", "p75", "p90", "p95", "p99", "max")}
    return {"count": len(values), "mean": float(values.mean()), "median": float(np.median(values)), **{f"p{q}": float(np.percentile(values, q)) for q in (75,90,95,99)}, "max": float(values.max())}


def _trauma_bins(impacts):
    frame = pd.DataFrame(impacts)
    if frame.empty: return []
    frame["trauma_bin"] = pd.cut(frame.trauma, [-np.inf,5,10,20,40,np.inf])
    return frame.groupby("trauma_bin", observed=False).agg(checks=("finished","size"), finish_rate=("finished","mean"), mean_acute=("acute","mean")).reset_index().assign(trauma_bin=lambda x:x.trauma_bin.astype(str)).to_dict("records")


def run(paths=10, start_year=2020, limit=100, seed=20260813, output_dir=Path("data/diagnostics/event_mc_v1_phase7a")):
    cohort, fsr = build_cohort(start_year, limit); started=time.perf_counter(); path_rows=[]; all_impacts=[]
    for fight_index, (_, row) in enumerate(cohort.iterrows()):
        fight = _fight(row, fsr)
        for path_index in range(paths):
            sink=DecompositionSink(); result=build_engine(fight, seed+fight_index*100000+path_index, sink)[0].run(); stats=result.sink_result
            impacts=list(stats["impacts"]); all_impacts.extend(impacts); kds=sum(x["kd"] for x in impacts)
            prior_kds = sum(x["kd"] for x in impacts[:-1]) if result.state.finish_method == "KO_TKO" else kds
            path_rows.append({"fight_id":fight.fight_id,"winner":result.state.winner,"method":result.state.finish_method,"exposure_seconds":stats["exposure_seconds"],"attempts":sum(stats["attempts"].values()),"landed":len(impacts),"kds":kds,"prior_kds_at_ko":prior_kds,"finish_checks":len(impacts), **{f"{phase}_attempts":stats["attempts"].get(phase,0) for phase in STRIKE_FAMILIES.values()}, **{f"{phase}_landed":stats["landed"].get(phase,0) for phase in STRIKE_FAMILIES.values()}})
    paths_frame=pd.DataFrame(path_rows); impact_frame=pd.DataFrame(all_impacts); exposure=paths_frame.exposure_seconds.sum(); landed=paths_frame.landed.sum(); kds=paths_frame.kds.sum()
    historical_exposure=cohort.apply(observed_duration_seconds,axis=1).sum()
    historical_attempts=(cohort.r_total_str_atmpted+cohort.b_total_str_atmpted).sum(); historical_landed=(cohort.r_total_str_landed+cohort.b_total_str_landed).sum(); historical_kd=(cohort.r_kd+cohort.b_kd).sum()
    finish_kd=impact_frame[impact_frame.kd]; finish_non=impact_frame[~impact_frame.kd]; ko_paths=paths_frame[paths_frame.method=="KO_TKO"]
    methods = paths_frame.method.value_counts(normalize=True)
    nondecision = paths_frame[paths_frame.method != "DEC"]
    finish_rounds = (np.maximum(nondecision.exposure_seconds.to_numpy() - 1e-12, 0) // 300 + 1).astype(int)
    summary={
        "fights":len(cohort),"paths":len(paths_frame),"runtime_seconds":time.perf_counter()-started,
        "comparability_note":"Historical UFCStats total strikes are the closest available comparator, but EVENT MC modeled offensive attempts are not guaranteed definition-identical; historical phase columns are significant-strike position fields and are not directly compared to EVENT MC total strikes.",
        "historical":{"attempts_per_15min":float(historical_attempts/historical_exposure*900),"landed_per_15min":float(historical_landed/historical_exposure*900),"landing_rate":float(historical_landed/historical_attempts),"kd_per_100_landed":float(historical_kd/historical_landed*100),"kd_per_15min":float(historical_kd/historical_exposure*900)},
        "simulated":{"attempts_per_15min":float(paths_frame.attempts.sum()/exposure*900),"landed_per_15min":float(landed/exposure*900),"landing_rate":float(landed/paths_frame.attempts.sum()),"kd_per_100_landed":float(kds/landed*100),"kd_per_15min":float(kds/exposure*900),"zero_kd_path_share":float((paths_frame.kds==0).mean()),"multi_kd_path_share":float((paths_frame.kds>=2).mean()),"finish_checks_per_path":float(paths_frame.finish_checks.mean()),"finish_checks_per_15min":float(paths_frame.finish_checks.sum()/exposure*900)},
        "phase":{phase:{"attempts":int(paths_frame[f"{phase}_attempts"].sum()),"landed":int(paths_frame[f"{phase}_landed"].sum()),"attempts_per_15min":float(paths_frame[f"{phase}_attempts"].sum()/exposure*900),"landed_per_15min":float(paths_frame[f"{phase}_landed"].sum()/exposure*900)} for phase in STRIKE_FAMILIES.values()},
        "impact":{"all":_distribution(impact_frame.impact),"non_kd":_distribution(impact_frame.loc[~impact_frame.kd,"impact"]),"kd":_distribution(impact_frame.loc[impact_frame.kd,"impact"]),"fight_ending":_distribution(impact_frame.loc[impact_frame.finished,"impact"])},
        "conversion":{"landed_finish_checks":len(impact_frame),"p_finish_given_kd":float(finish_kd.finished.mean()),"p_finish_given_non_kd":float(finish_non.finished.mean()),"non_kd_finishing_strike_share":float((impact_frame[impact_frame.finished].kd==False).mean()),"ko_paths_zero_prior_kd_share":float((ko_paths.prior_kds_at_ko==0).mean())},
        "outcomes":{"method_shares":{method:float(methods.get(method,0)) for method in ("KO_TKO","SUB","DEC")},"nondecision_finish_round_shares":{str(round_no):float(np.mean(finish_rounds==round_no)) for round_no in range(1,6)},"mean_nondecision_finish_time":float(nondecision.exposure_seconds.mean())},
        "kd_by_round":impact_frame[impact_frame.kd].groupby("round").size().astype(int).to_dict(),"kd_by_phase":impact_frame[impact_frame.kd].groupby("phase").size().astype(int).to_dict(),"trauma_bins":_trauma_bins(all_impacts),
    }
    summary["ratios"]={"attempt_exposure":summary["simulated"]["attempts_per_15min"]/summary["historical"]["attempts_per_15min"],"landed_exposure":summary["simulated"]["landed_per_15min"]/summary["historical"]["landed_per_15min"],"kd_per_landed":summary["simulated"]["kd_per_100_landed"]/summary["historical"]["kd_per_100_landed"],"kd_exposure":summary["simulated"]["kd_per_15min"]/summary["historical"]["kd_per_15min"]}
    output_dir.mkdir(parents=True,exist_ok=True); paths_frame.to_csv(output_dir/"path_sufficient_statistics.csv",index=False); (output_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)); print(json.dumps(summary,indent=2,sort_keys=True)); return paths_frame,summary


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--paths",type=int,default=10);p.add_argument("--start-year",type=int,default=2020);p.add_argument("--limit",type=int,default=100);p.add_argument("--seed",type=int,default=20260813);p.add_argument("--output-dir",type=Path,default=Path("data/diagnostics/event_mc_v1_phase7a"));run(**vars(p.parse_args()))
