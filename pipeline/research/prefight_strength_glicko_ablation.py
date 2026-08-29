#!/usr/bin/env python3
"""Research-only Glicko ablation: graded outcomes vs inactivity RD inflation."""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.research.prefight_strength_elo import build_bouts

Q=math.log(10.0)/400.0; BASE=1500.0; INIT_RD=350.0; MIN_RD=30.0; MAX_RD=350.0; GRACE=180.0

@dataclass
class State:
    rating: float=BASE
    rd: float=INIT_RD
    last_date: pd.Timestamp|None=None

def g(rd): return 1.0/math.sqrt(1.0+3.0*(Q*rd)**2/(math.pi**2))
def expected(r,or_,ord_): return 1.0/(1.0+10.0**(-g(ord_)*(r-or_)/400.0))

def update(r,rd,or_,ord_,score):
    gg=g(ord_); e=expected(r,or_,ord_); d2=1.0/max((Q*Q)*(gg*gg)*e*(1-e),1e-15)
    denom=(1.0/(rd*rd))+(1.0/d2)
    nr=r+(Q/denom)*gg*(score-e); nrd=math.sqrt(1.0/denom)
    return float(nr), float(max(MIN_RD,min(MAX_RD,nrd)))

def scores(method,winner_is_red,graded):
    if not graded: hi,lo=1.0,0.0
    else:
        m=(method or '').lower()
        if 'split' in m: hi,lo=.55,.45
        elif 'majority' in m: hi,lo=.61,.39
        elif 'decision' in m: hi,lo=.91,.09
        else: hi,lo=1.0,0.0
    return (hi,lo) if winner_is_red else (lo,hi)

def run(bouts,graded,inflation):
    states=defaultdict(State); rows=[]
    for b in bouts.itertuples(index=False):
        r,bl=b.red_fighter,b.blue_fighter; sr,sb=states[r],states[bl]
        for st in (sr,sb):
            if st.last_date is not None and inflation>0:
                extra=max(0.0,float((b.date-st.last_date).days)-GRACE)
                if extra>0: st.rd=min(MAX_RD,math.sqrt(st.rd*st.rd+inflation*extra))
        rp,bp,rr,br=sr.rating,sb.rating,sr.rd,sb.rd
        p=expected(rp,bp,br)
        rows.append({'date':b.date,'bout_id':b.bout_id,'red_fighter':r,'blue_fighter':bl,'winner':b.winner,'p_red':p})
        if b.winner is None: continue
        rs,bs=scores(getattr(b,'method',''),b.winner==r,graded)
        nr,nrr=update(rp,rr,bp,br,rs); nb,nbr=update(bp,br,rp,rr,bs)
        sr.rating,sr.rd,sr.last_date=nr,nrr,b.date; sb.rating,sb.rd,sb.last_date=nb,nbr,b.date
    return pd.DataFrame(rows)

def metrics(df):
    d=df[df.winner.notna()].copy(); p=d.p_red.clip(1e-9,1-1e-9).astype(float); y=(d.winner==d.red_fighter).astype(float)
    nt=np.abs(p-.5)>1e-12
    return {'n':int(len(d)),'non_ties':int(nt.sum()),'accuracy_non_ties':float((((p>.5)==(y>.5))[nt]).mean()),'brier':float(np.mean((p-y)**2)),'log_loss':float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('data/master/ufc_master.parquet')); ap.add_argument('--output-dir',type=Path,default=Path('data/diagnostics/prefight_strength_glicko_ablation')); ap.add_argument('--holdout-from',default='2025-01-01'); a=ap.parse_args()
    bouts=build_bouts(pd.read_parquet(a.input)); cutoff=pd.Timestamp(a.holdout_from)
    variants=[('full',True,4.0),('binary_outcomes',False,4.0),('no_inactivity_inflation',True,0.0),('rd_only_binary_no_inflation',False,0.0)]
    summary={}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for name,graded,infl in variants:
        df=run(bouts,graded,infl); df.to_csv(a.output_dir/f'{name}.csv',index=False)
        summary[name]={'graded_outcomes':graded,'rd_inflation_per_day':infl,'train':metrics(df[df.date<cutoff]),'holdout':metrics(df[df.date>=cutoff])}
    with open(a.output_dir/'summary.json','w') as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
