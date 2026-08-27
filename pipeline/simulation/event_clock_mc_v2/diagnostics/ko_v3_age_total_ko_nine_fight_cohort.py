"""Run requested cohort fights through the age-adjusted total-KO V3 research shadow.

Research-only. Production unchanged. Set TARGET_INDEX=0..8 to run one fight.
"""
from __future__ import annotations

from collections import Counter
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
import pipeline.research.fsr_recency_cohort_shadow as recency
from pipeline.research import ko_v3_from_scratch_stage1 as s1
from pipeline.research.ko_v3_from_scratch_shadow import fit_prefight_hazards
from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION
from pipeline.simulation.event_clock_mc_v2.calibration.seeds import derive_path_seed
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineFunctions, run_causal_path
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_dynamic_pressure_shadow as pressure_mod
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_intent_rate_shadow as intent_mod
from pipeline.simulation.event_clock_mc_v2.mechanics import physiology as physiology_mod
from pipeline.simulation.event_clock_mc_v2.mechanics import ko_kd_empirical as ko_mod

PATHS = 500
BASE_EWM_DECAY = 0.50
STANDING_ATTEMPT_SCALE = 0.25
BACKUP_PATH = Path("data/fsr_v3/fsr_v3_prefight_snapshots.ko_v3_nine_backup.parquet")
MASTER_PATH = Path("data/master/ufc_master.parquet")
TARGETS = [
    ("Brendan Allen", "Edmen Shahbazyan"),
    ("Fares Ziam", "Tom Nolan"),
    ("Belal Muhammad", "Gabriel Bonfim"),
    ("Matt Schnell", "Alessandro Costa"),
    ("Bruno Silva", "Edgar Chairez"),
    ("Jordan Leavitt", "Joanderson Brito"),
    ("Ketlen Souza", "Ariane Carnelossi"),
    ("Karol Rosa", "Luana Santos"),
    ("Manel Kape", "Kyoji Horiguchi"),
]

def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-30.0,30.0)))
def logit(p):
    p=np.clip(float(p),1e-9,1.0-1e-9); return np.log(p/(1.0-p))

def build_pure_ewm50_snapshot(canonical):
    recency.EWM_CANONICAL_BLEND=0.0; recency.EWM_DECAY=BASE_EWM_DECAY
    return recency.build_variant(canonical,"ewm")

def resolve_target_rows(master):
    out=[]
    for a,b in TARGETS:
        m=master[((master.r_name.astype(str).eq(a))&(master.b_name.astype(str).eq(b)))|((master.r_name.astype(str).eq(b))&(master.b_name.astype(str).eq(a)))].copy()
        if len(m)!=1: raise RuntimeError(f"Expected one fight for {a} vs {b}, found {len(m)}")
        r=m.iloc[0]; out.append({"fight_id":str(r.fight_id),"date":pd.Timestamp(r.date).normalize(),"a":a,"b":b,"actual_method":str(r.method),"actual_winner_id":str(r.winner_id)})
    return out

def fit_age_slopes(frame,cutoff):
    tr=frame[(frame.event_date<cutoff)&(frame.sig_landed>0)&frame.attacker_age.notna()&frame.defender_age.notna()].copy()
    ids=[]; ys=[]; ws=[]
    for idx,row in tr.iterrows():
        k=float(row.ko_win); n=float(row.sig_landed)
        if k>0: ids.append(idx); ys.append(1); ws.append(k)
        if n-k>0: ids.append(idx); ys.append(0); ws.append(n-k)
    ex=tr.loc[ids].copy(); X=np.column_stack([ex.attacker_age.to_numpy(float)-30.0,ex.defender_age.to_numpy(float)-30.0])
    model=LogisticRegression(C=1.0,max_iter=5000,solver="lbfgs"); model.fit(X,np.asarray(ys,int),sample_weight=np.asarray(ws,float))
    return float(model.coef_[0][0]),float(model.coef_[0][1]),{"fit_rows":int(len(tr)),"fit_ko_wins":int(tr.ko_win.sum()),"fit_sig_landed":float(tr.sig_landed.sum()),"attacker_age_logodds_per_year":float(model.coef_[0][0]),"defender_age_logodds_per_year":float(model.coef_[0][1])}

def total_hazards_for_fight(frame,fight_id,beta_att,beta_def):
    target=frame[frame.fight_id.astype(str).eq(str(fight_id))].copy()
    if len(target)!=2: raise RuntimeError(f"Expected 2 rows for {fight_id}, got {len(target)}")
    out={}
    for row in target.itertuples(index=False):
        att_n=float(row.prior_sig_landed); def_n=float(row.opp_prior_sig_absorbed); att_k=float(row.prior_ko_wins); def_k=float(row.opp_prior_ko_losses)
        p_att=att_k/att_n if att_n>0 else 0.0; p_def=def_k/def_n if def_n>0 else 0.0; p_raw=1.0-(1.0-p_att)*(1.0-p_def)
        delta=beta_att*(float(row.attacker_age)-30.0)+beta_def*(float(row.defender_age)-30.0); p_age=float(sigmoid(logit(p_raw)+delta)) if p_raw>0 else 0.0
        out[str(row.fighter_id)]={"fighter_name":str(row.fighter_name),"attacker_age":float(row.attacker_age),"defender_age":float(row.defender_age),"attacker_ko_per_sig":p_att,"defender_ko_loss_per_sig":p_def,"raw_total_ko_per_landed":p_raw,"age_logodds_delta":delta,"total_ko_per_landed":p_age}
    return out

class Resolver:
    def __init__(self,total_by_side,kd_by_side): self.total_by_side=total_by_side; self.kd_by_side=kd_by_side; self.landed=Counter(); self.kos=Counter(); self.kds=Counter()
    def __call__(self,*,state,attacker_side,attacker,defender,rng):
        del attacker,defender
        target=state.physiology.fighter(attacker_side.opponent); prior=int(target.knockdowns_suffered); self.landed[attacker_side]+=1
        p_ko=float(self.total_by_side[attacker_side]["total_ko_per_landed"])
        if rng.random()<p_ko: self.kos[attacker_side]+=1; return ko_mod.EmpiricalKOKDResult(p_ko,True,0.0,False,prior)
        p_kd=float(self.kd_by_side[attacker_side].kd_per_landed); kd=bool(rng.random()<p_kd)
        if kd:self.kds[attacker_side]+=1
        return ko_mod.EmpiricalKOKDResult(p_ko,False,p_kd,kd,prior)

def main():
    master=pd.read_parquet(MASTER_PATH).copy(); master["date"]=pd.to_datetime(master["date"]).dt.normalize(); master["fight_id"]=master.fight_id.astype(str)
    targets=resolve_target_rows(master); target_index=os.getenv("TARGET_INDEX")
    if target_index is not None: targets=[targets[int(target_index)]]
    ff,_=s1.load_raw_fighter_fights(); frame=s1.build_matchup_frame(s1.build_prefight_states(ff)).copy(); frame["fight_id"]=frame.fight_id.astype(str)
    canonical=pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy(); canonical["event_date"]=pd.to_datetime(canonical.event_date).dt.normalize(); canonical["fight_id"]=canonical.fight_id.astype(str); canonical["fighter_id"]=canonical.fighter_id.astype(str); ewm50=build_pure_ewm50_snapshot(canonical)
    shutil.copy2(FSR_V3_PREFIGHT_SNAPSHOTS_PATH,BACKUP_PATH); original_standing=intent_mod._standing_rates; original_resolver=physiology_mod.resolve_empirical_ko_kd; original_hurt=physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT; results=[]
    try:
        ewm50.to_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH,index=False)
        def calibrated(state,actor,capabilities,context,priors,config):
            rates,pressure=original_standing(state,actor,capabilities,context,priors,config); rates=dict(rates); rates[ActionFamily.STAND_ATTACK]*=STANDING_ATTEMPT_SCALE; return rates,pressure
        intent_mod._standing_rates=calibrated; physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT=0.0
        for item in targets:
            fight_id=item["fight_id"]; cutoff=item["date"]; beta_att,beta_def,age_fit=fit_age_slopes(frame,cutoff); total_by_id=total_hazards_for_fight(frame,fight_id,beta_att,beta_def); kd_by_id=fit_prefight_hazards(fight_id=fight_id)
            pressure_mod.FIGHT_ID=fight_id; pressure_mod.PATHS=PATHS; intent_mod.FIGHT_ID=fight_id; intent_mod.PATHS=PATHS
            fight,inputs,priors,horizon,cfg=pressure_mod.build_setup(); side_to_id={Side.RED:str(fight.r_id),Side.BLUE:str(fight.b_id)}; total_by_side={s:total_by_id[fid] for s,fid in side_to_id.items()}; kd_by_side={s:kd_by_id[fid] for s,fid in side_to_id.items()}; resolver=Resolver(total_by_side,kd_by_side); physiology_mod.resolve_empirical_ko_kd=resolver
            brain=intent_mod.IntentRateBrain(inputs,priors,horizon); funcs=EngineFunctions(timing_sampler=brain.timing_sampler,action_chooser=brain.action_chooser); wins=Counter(); six=Counter()
            for path_id in range(PATHS):
                seed=derive_path_seed(SEED_SET_VERSION,fight_id,path_id); out=run_causal_path(inputs,seed=seed,horizon_seconds=horizon,config=cfg,functions=funcs)
                if out.termination is None: continue
                wins[out.termination.winner]+=1; six[(out.termination.winner.value,out.termination.finish_method.value)]+=1
            names={Side.RED:str(fight.r_name),Side.BLUE:str(fight.b_name)}; rows=[]
            for side in Side:
                rows.append({"fighter":names[side],"ml":wins[side]/PATHS,"ko_tko":six[(side.value,"ko_tko")]/PATHS,"submission":six[(side.value,"submission")]/PATHS,"decision":six[(side.value,"decision")]/PATHS,"total_ko_per_landed":total_by_side[side]["total_ko_per_landed"],"raw_total_ko_per_landed":total_by_side[side]["raw_total_ko_per_landed"],"kd_per_landed":float(kd_by_side[side].kd_per_landed),"landed_resolutions":int(resolver.landed[side]),"ko_finishes":int(resolver.kos[side]),"knockdowns":int(resolver.kds[side])})
            results.append({"fight_id":fight_id,"date":str(cutoff.date()),"matchup":f"{item['a']} vs {item['b']}","fighters":rows,"age_fit":age_fit,"actual_method":item["actual_method"],"production_changed":False,"kd_can_finish":False,"post_kd_finish_loop":False})
        print(json.dumps({"diagnostic":"KO V3 age-adjusted total-KO cohort","paths_per_fight":PATHS,"target_index":target_index,"results":results},indent=2,sort_keys=True))
    finally:
        physiology_mod.resolve_empirical_ko_kd=original_resolver; physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT=original_hurt; intent_mod._standing_rates=original_standing; shutil.move(BACKUP_PATH,FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
if __name__=="__main__": main()
