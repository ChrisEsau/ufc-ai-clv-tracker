#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.research.prefight_strength_fightmatrix_glicko import update, BASE, INIT_RD
from pipeline.research.elo_fsr_brain_one_fight_shadow import _target_indices, run_locked
from pipeline.research.locked_brain_bundle import DEFAULT_BUNDLE_DIR, FILES
from pipeline.research.domain_hazard_glicko_brain_10fight_shadow import (
    build_hazard_ratings, _set_clock_from_ratings,
    KO_PRIOR_EVENTS, KO_PRIOR_SECONDS, SUB_PRIOR_EVENTS, SUB_PRIOR_SECONDS,
)
from pipeline.research import ko_time_survival_oos as ko_surv
from pipeline.research import sub_time_survival_oos as sub_surv

FIGHT_IDS=["419fff06f338f5c6","58ffa2dac4f2e7d0","5c69b019e6deee41","5d2eedd05081ed23","20d74ed23d3e9b3a","44cfbb8c3c356c65","b0474597b2c60482","b23a1a5d35eb438a","33afdd7ad43a2756","7208e40818401e88"]
PATHS=500
ROOT=Path("data/diagnostics/trait_native_glicko_brain_10fight_shadow")
Q=math.log(10.0)/400.0

@dataclass
class S:
    r: float=BASE
    rd: float=INIT_RD

def h(path: Path)->str:
    x=hashlib.sha256(); x.update(path.read_bytes()); return x.hexdigest()

def upd_pair(a:S,b:S,score:float):
    score=float(np.clip(score,0.0,1.0))
    ar,ard=a.r,a.rd; br,brd=b.r,b.rd
    a.r,a.rd=update(ar,ard,br,brd,score)
    b.r,b.rd=update(br,brd,ar,ard,1.0-score)

def ratio_score(x:float,y:float,default=.5)->float:
    d=float(x)+float(y)
    return default if d<=0 else float(x)/d

def aggregate_fights()->pd.DataFrame:
    p=build_paired_rounds()
    # One fighter-side row per fight, using only information available in that fight.
    agg=p.groupby(["event_date","fight_id","fighter_id","fighter_name","opponent_id","opponent_name"],as_index=False).agg(
        sig_l=("sig_str_landed","sum"),sig_a=("sig_str_attempted","sum"),
        td_l=("td_landed","sum"),td_a=("td_attempted","sum"),
        g_l=("ground_landed","sum"),g_a=("ground_attempted","sum"),
        sub_a=("effective_submission_attempts","sum"),ctrl=("ctrl_sec","sum"),
        stand_sec=("standing_exposure_seconds","sum"),td_sec=("td_tendency_exposure_seconds","sum"),
        ground_sec=("modeled_ground_exposure_seconds","sum"),
    )
    return agg.sort_values(["event_date","fight_id","fighter_id"]).reset_index(drop=True)

def build_trait_ratings()->pd.DataFrame:
    ff=aggregate_fights()
    pools={k:defaultdict(S) for k in [
        "stand_off","stand_def","stand_tend","stand_supp",
        "td_off","td_def","td_tend","td_supp",
        "ground_off","ground_def","ground_tend","ground_supp",
        "sub_tend","sub_resist",
    ]}
    rows=[]
    for date,batch in ff.groupby("event_date",sort=True):
        # Snapshot same-date fights first, then update.
        staged=[]
        for fid,g in batch.groupby("fight_id",sort=True):
            if len(g)!=2: continue
            a,b=g.iloc[0],g.iloc[1]
            aid,bid=str(a.fighter_id),str(b.fighter_id)
            snap={"event_date":date,"fight_id":str(fid),
                  "a_id":aid,"a_name":str(a.fighter_name),"b_id":bid,"b_name":str(b.fighter_name)}
            for name,pool in pools.items():
                snap[f"a_{name}"]=pool[aid].r; snap[f"a_{name}_rd"]=pool[aid].rd
                snap[f"b_{name}"]=pool[bid].r; snap[f"b_{name}_rd"]=pool[bid].rd
            rows.append(snap)

            # Accuracy/completion contests: offense versus corresponding defense.
            staged.append(("cross","stand_off",aid,"stand_def",bid, (a.sig_l/a.sig_a) if a.sig_a>0 else None))
            staged.append(("cross","stand_off",bid,"stand_def",aid, (b.sig_l/b.sig_a) if b.sig_a>0 else None))
            staged.append(("cross","td_off",aid,"td_def",bid, (a.td_l/a.td_a) if a.td_a>0 else None))
            staged.append(("cross","td_off",bid,"td_def",aid, (b.td_l/b.td_a) if b.td_a>0 else None))
            staged.append(("cross","ground_off",aid,"ground_def",bid, (a.g_l/a.g_a) if a.g_a>0 else None))
            staged.append(("cross","ground_off",bid,"ground_def",aid, (b.g_l/b.g_a) if b.g_a>0 else None))

            # Pace/tendency contests use relative per-exposure rates.
            ar=a.sig_a/max(a.stand_sec,1.0); br=b.sig_a/max(b.stand_sec,1.0)
            staged.append(("cross","stand_tend",aid,"stand_supp",bid,ratio_score(ar,br)))
            staged.append(("cross","stand_tend",bid,"stand_supp",aid,ratio_score(br,ar)))
            ar=a.td_a/max(a.td_sec,1.0); br=b.td_a/max(b.td_sec,1.0)
            staged.append(("cross","td_tend",aid,"td_supp",bid,ratio_score(ar,br)))
            staged.append(("cross","td_tend",bid,"td_supp",aid,ratio_score(br,ar)))
            ar=a.g_a/max(a.ground_sec,1.0); br=b.g_a/max(b.ground_sec,1.0)
            staged.append(("cross","ground_tend",aid,"ground_supp",bid,ratio_score(ar,br)))
            staged.append(("cross","ground_tend",bid,"ground_supp",aid,ratio_score(br,ar)))
            ar=a.sub_a/max(a.ground_sec,1.0); br=b.sub_a/max(b.ground_sec,1.0)
            staged.append(("cross","sub_tend",aid,"sub_resist",bid,ratio_score(ar,br)))
            staged.append(("cross","sub_tend",bid,"sub_resist",aid,ratio_score(br,ar)))
        for typ,p1,id1,p2,id2,score in staged:
            if score is None: continue
            upd_pair(pools[p1][id1],pools[p2][id2],score)
    return pd.DataFrame(rows)

def target_snapshot(r:pd.DataFrame,fid:str,name:str)->dict:
    x=r[r.fight_id.eq(str(fid))]
    if len(x)!=1: raise RuntimeError(f"trait snapshot missing {fid}")
    z=x.iloc[0]
    if z.a_name==name: side="a"
    elif z.b_name==name: side="b"
    else: raise RuntimeError(f"fighter {name} not in {fid}")
    return {k[len(side)+1:]:float(v) for k,v in z.items() if k.startswith(side+"_") and k not in (side+"_id",side+"_name")}

def exp_rating(r:float)->float: return math.exp(Q*(r-BASE))
def logit(p:float)->float:
    p=float(np.clip(p,1e-6,1-1e-6)); return math.log(p/(1-p))
def sigmoid(x:float)->float: return 1/(1+math.exp(-x))

def set_field(df,idx,field,value,fighter,audit,kind):
    if field not in df.columns: return
    before=float(df.at[idx,field]); after=float(value)
    df.at[idx,field]=after
    audit.append({"fighter":fighter,"system":"trait_native_glicko","field":field,"transform":kind,"before":before,"after":after})

def apply_trait_snapshot(df,idx,fighter,snap,med,audit):
    # Positive pace traits are rebuilt from their own Glicko ratings around the historical population median.
    for field,key in [
        ("standing_striking_tendency","stand_tend"),("takedown_tendency","td_tend"),
        ("ground_striking_tendency","ground_tend"),("ground_striking_burst_baseline","ground_tend"),
        ("submission_tendency","sub_tend")]:
        if field in df.columns: set_field(df,idx,field,max(med[field]*exp_rating(snap[key]),0),fighter,audit,"population_median * exp(trait_glicko)")
    for field,key in [("standing_striking_suppression","stand_supp"),("takedown_suppression","td_supp"),("ground_striking_suppression","ground_supp")]:
        if field in df.columns: set_field(df,idx,field,max(med[field]*math.exp(-Q*(snap[key]-BASE)),0),fighter,audit,"population_median * exp(-trait_glicko)")
    # Latent matchup coordinates are now the trait Glicko log-odds coordinates themselves.
    for field,key in [
        ("standing_striking_offense","stand_off"),("standing_striking_defense","stand_def"),
        ("takedown_offense","td_off"),("takedown_defense","td_def"),
        ("ground_striking_offense","ground_off")]:
        if field in df.columns: set_field(df,idx,field,float(med[field])+Q*(snap[key]-BASE),fighter,audit,"population_median + trait_glicko_logodds")
    # Fighter-specific accuracy baselines are direct transforms of the corresponding offense rating.
    for field,key in [("standing_accuracy_baseline","stand_off"),("takedown_completion_baseline","td_off"),("ground_accuracy_baseline","ground_off")]:
        if field in df.columns:
            m=float(np.clip(med[field],1e-4,1-1e-4)); set_field(df,idx,field,sigmoid(logit(m)+Q*(snap[key]-BASE)),fighter,audit,"logit(population_median)+trait_glicko")

def target_hazard(ratings,fid,name):
    x=ratings[ratings.fight_id.astype(str).eq(str(fid)) & ratings.fighter_name.astype(str).eq(str(name))]
    if len(x)!=1: raise RuntimeError(f"hazard rating missing {fid}/{name}")
    return x.iloc[0]

def run_one(fid,trait_r,ko_r,sub_r):
    ko_raw=ko_surv.load_fighter_fights(); t=ko_raw[ko_raw.fight_id.astype(str).eq(fid)]
    if len(t)!=2: raise RuntimeError(f"target sides {fid}: {len(t)}")
    a,b=[str(x) for x in t.fighter_name.tolist()]
    winner=t[t.won]; actual=str(winner.fighter_name.iloc[0]) if len(winner)==1 else None
    out=ROOT/fid; out.mkdir(parents=True,exist_ok=True); bundle=out/"adjusted_bundle"
    if bundle.exists(): shutil.rmtree(bundle)
    shutil.copytree(Path(DEFAULT_BUNDLE_DIR),bundle); audit=[]

    ep=bundle/FILES["ewm_fsr"]; e=pd.read_parquet(ep); ia,ib=_target_indices(e,fid,a,b)
    med={c:float(pd.to_numeric(e[c],errors="coerce").median()) for c in e.columns if c in {
        "standing_striking_tendency","standing_striking_suppression","standing_striking_offense","standing_striking_defense","standing_accuracy_baseline",
        "takedown_tendency","takedown_suppression","takedown_offense","takedown_defense","takedown_completion_baseline",
        "ground_striking_tendency","ground_striking_suppression","ground_striking_offense","ground_accuracy_baseline","ground_striking_burst_baseline","submission_tendency"}}
    sa,sb=target_snapshot(trait_r,fid,a),target_snapshot(trait_r,fid,b)
    apply_trait_snapshot(e,ia,a,sa,med,audit); apply_trait_snapshot(e,ib,b,sb,med,audit)

    # Power / KD resistance are genuine KO hazard Glicko states rather than globally shifted FSR values.
    kra,krb=target_hazard(ko_r,fid,a),target_hazard(ko_r,fid,b)
    if "striking_power_v3" in e.columns:
        m=float(pd.to_numeric(e["striking_power_v3"],errors="coerce").median())
        set_field(e,ia,"striking_power_v3",m+Q*(float(kra.off_rating)-BASE),a,audit,"population_median + KO_OFF_glicko")
        set_field(e,ib,"striking_power_v3",m+Q*(float(krb.off_rating)-BASE),b,audit,"population_median + KO_OFF_glicko")
    if "knockdown_resistance_v3" in e.columns:
        m=float(pd.to_numeric(e["knockdown_resistance_v3"],errors="coerce").median())
        # Row A's opp_def is B; row B's opp_def is A.
        set_field(e,ia,"knockdown_resistance_v3",m+Q*(float(krb.opp_def_rating)-BASE),a,audit,"population_median + KO_DEF_glicko")
        set_field(e,ib,"knockdown_resistance_v3",m+Q*(float(kra.opp_def_rating)-BASE),b,audit,"population_median + KO_DEF_glicko")
    e.to_parquet(ep,index=False)

    kp=bundle/FILES["ko_prefight"]; k=pd.read_parquet(kp); kia,kib=_target_indices(k,fid,a,b)
    _set_clock_from_ratings(k,kia,off_rating=float(kra.off_rating),def_rating=float(kra.opp_def_rating),p0=float(kra.population_hazard),prior_events=2.0,audit=audit,fighter=a,label="ko_clock")
    _set_clock_from_ratings(k,kib,off_rating=float(krb.off_rating),def_rating=float(krb.opp_def_rating),p0=float(krb.population_hazard),prior_events=2.0,audit=audit,fighter=b,label="ko_clock")
    k.to_parquet(kp,index=False)

    sp=bundle/FILES["sub_prefight"]; s=pd.read_parquet(sp); sia,sib=_target_indices(s,fid,a,b)
    sra,srb=target_hazard(sub_r,fid,a),target_hazard(sub_r,fid,b)
    _set_clock_from_ratings(s,sia,off_rating=float(sra.off_rating),def_rating=float(sra.opp_def_rating),p0=float(sra.population_hazard),prior_events=1.0,audit=audit,fighter=a,label="sub_clock")
    _set_clock_from_ratings(s,sib,off_rating=float(srb.off_rating),def_rating=float(srb.opp_def_rating),p0=float(srb.population_hazard),prior_events=1.0,audit=audit,fighter=b,label="sub_clock")
    s.to_parquet(sp,index=False)

    mp=bundle/"manifest.json"; man=json.loads(mp.read_text())
    for key,p in (("ewm_fsr",ep),("ko_prefight",kp),("sub_prefight",sp)): man["files"][key]["sha256"]=h(p)
    man["research_shadow"]={"type":"trait_native_glicko_all_active_inputs","fight_id":fid,"rd_used_in_brain":False,"production_changed":False}
    mp.write_text(json.dumps(man,indent=2)+"\n")
    adjusted=run_locked(fid,bundle,PATHS,out/"adjusted_results.json")
    pd.DataFrame(audit).assign(fight_id=fid).to_csv(out/"trait_native_glicko_adjustments.csv",index=False)
    summary={"fight_id":fid,"fighter_a":a,"fighter_b":b,"actual_winner":actual,"paths":PATHS,"trait_a":sa,"trait_b":sb,"adjusted":adjusted}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=str)+"\n")
    shutil.rmtree(bundle); print(json.dumps(summary,indent=2,default=str))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--fight-id",required=True,choices=FIGHT_IDS); args=ap.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    tr=build_trait_ratings()
    ko_r=build_hazard_ratings(ko_surv.load_fighter_fights(),"ko_event",prior_events=KO_PRIOR_EVENTS,prior_seconds=KO_PRIOR_SECONDS)
    sub_r=build_hazard_ratings(sub_surv.load_fighter_fights(),"sub_event",prior_events=SUB_PRIOR_EVENTS,prior_seconds=SUB_PRIOR_SECONDS)
    run_one(args.fight_id,tr,ko_r,sub_r)
if __name__=="__main__": main()
