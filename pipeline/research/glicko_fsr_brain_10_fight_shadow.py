#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, shutil
from pathlib import Path
import pandas as pd
from pipeline.research.elo_fsr_brain_one_fight_shadow import _apply_fsr,_scale_clock_row,_target_indices,run_locked
from pipeline.research.locked_brain_bundle import DEFAULT_BUNDLE_DIR, FILES
from pipeline.research.prefight_strength_elo import build_bouts
from pipeline.research.prefight_strength_fightmatrix_glicko import run as run_glicko

FIGHT_IDS=["419fff06f338f5c6","58ffa2dac4f2e7d0","5c69b019e6deee41","5d2eedd05081ed23","20d74ed23d3e9b3a","44cfbb8c3c356c65","b0474597b2c60482","b23a1a5d35eb438a","33afdd7ad43a2756","7208e40818401e88"]
PATHS=500; DOMAIN_SHARE=0.20
ROOT=Path("data/diagnostics/glicko_fsr_brain_10_fight_shadow")

def sha256(path):
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--fight-id",required=True,choices=FIGHT_IDS); args=ap.parse_args(); fid=args.fight_id
    bouts=build_bouts(pd.read_parquet("data/master/ufc_master.parquet")); gf=run_glicko(bouts)
    t=gf[gf.bout_id.astype(str).eq(fid)]
    if len(t)!=1: raise RuntimeError(f"expected one Glicko row for {fid}, found {len(t)}")
    row=t.iloc[0]; a=str(row.red_fighter); b=str(row.blue_fighter)
    ra=float(row.red_pre_rating); rb=float(row.blue_pre_rating); rda=float(row.red_pre_rd); rdb=float(row.blue_pre_rd)
    delta=ra-rb; logodds=math.log(10.0)*delta/400.0; sa=0.5*logodds*DOMAIN_SHARE; sb=-0.5*logodds*DOMAIN_SHARE
    out=ROOT/fid; out.mkdir(parents=True,exist_ok=True); bundle=out/"adjusted_bundle"
    if bundle.exists(): shutil.rmtree(bundle)
    shutil.copytree(Path(DEFAULT_BUNDLE_DIR),bundle); audit=[]
    ep=bundle/FILES["ewm_fsr"]; e=pd.read_parquet(ep); ia,ib=_target_indices(e,fid,a,b); _apply_fsr(e,ia,a,sa,audit); _apply_fsr(e,ib,b,sb,audit); e.to_parquet(ep,index=False)
    kp=bundle/FILES["ko_prefight"]; k=pd.read_parquet(kp); ia,ib=_target_indices(k,fid,a,b); _scale_clock_row(k,ia,a,sa,2.0,"ko_clock",audit); _scale_clock_row(k,ib,b,sb,2.0,"ko_clock",audit); k.to_parquet(kp,index=False)
    sp=bundle/FILES["sub_prefight"]; s=pd.read_parquet(sp); ia,ib=_target_indices(s,fid,a,b); _scale_clock_row(s,ia,a,sa,1.0,"sub_clock",audit); _scale_clock_row(s,ib,b,sb,1.0,"sub_clock",audit); s.to_parquet(sp,index=False)
    mp=bundle/"manifest.json"; m=json.loads(mp.read_text())
    for key,p in (("ewm_fsr",ep),("ko_prefight",kp),("sub_prefight",sp)): m["files"][key]["sha256"]=sha256(p)
    m["research_shadow"]={"type":"glicko_central_rating_full_active_brain_state","fight_id":fid,"fighter_a":a,"fighter_b":b,"glicko_a":ra,"glicko_b":rb,"rd_a":rda,"rd_b":rdb,"rating_delta":delta,"domain_share":DOMAIN_SHARE,"fighter_a_domain_shift":sa,"fighter_b_domain_shift":sb,"rd_used_in_transform":False,"production_changed":False}; mp.write_text(json.dumps(m,indent=2)+"\n")
    adjusted=run_locked(fid,bundle,PATHS,out/"adjusted_results.json")
    pd.DataFrame(audit).assign(fight_id=fid).to_csv(out/"glicko_active_input_adjustments.csv",index=False)
    summary={"fight_id":fid,"fighter_a":a,"fighter_b":b,"actual_winner":row.winner,"glicko_a":ra,"glicko_b":rb,"rd_a":rda,"rd_b":rdb,"rating_delta":delta,"domain_share":DOMAIN_SHARE,"shift_a":sa,"shift_b":sb,"paths":PATHS,"adjusted":adjusted}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=str)+"\n"); shutil.rmtree(bundle); print(json.dumps(summary,indent=2,default=str))
if __name__=="__main__": main()
