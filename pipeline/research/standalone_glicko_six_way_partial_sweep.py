#!/usr/bin/env python3
"""Partial negative-evidence sweep for direct joint Glicko-6.

Research-only. No Brain, FSR, or market inputs.

Tests non-observed method negative-evidence weights in {0.25, 0.50, 0.75, 1.00}
while preserving the pure direct joint six-class scoring and untouched holdout.
Observed class gets full positive evidence. The opposite-side class within the
observed method gets full negative evidence. Non-observed method classes get
partial negative evidence with weight w by blending the Glicko target toward
0.5: target = 0.5 * (1-w). Thus w=1.0 reproduces full negative evidence and
w=0.0 would be neutral/no directional evidence.
"""
from __future__ import annotations

import argparse, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.research.prefight_strength_elo import build_bouts
from pipeline.research.prefight_strength_fightmatrix_glicko import State, expected, inflate_rd, update

METHODS=("KO","SUB","DEC")
SIX_LABELS=("R_KO","R_SUB","R_DEC","B_KO","B_SUB","B_DEC")
SIX_COLS=("p_red_ko","p_red_sub","p_red_dec","p_blue_ko","p_blue_sub","p_blue_dec")
EPS=1e-12
WEIGHTS=(0.25,0.50,0.75,1.00)

def method_family(method:str)->str|None:
    m=str(method or "").lower()
    if "decision" in m: return "DEC"
    if "submission" in m or "sub" in m: return "SUB"
    if "ko" in m or "tko" in m: return "KO"
    return None

def softmax(scores):
    x=np.asarray(scores,float); x-=np.max(x); e=np.exp(x); return e/e.sum()

def logit(p):
    p=min(1-EPS,max(EPS,float(p))); return math.log(p)-math.log1p(-p)

def method_priors(counts):
    z=float(sum(counts.values())); return {m:counts[m]/z for m in METHODS}

def class_signal(offense, defense, now, prior):
    inflate_rd(offense,now); inflate_rd(defense,now)
    q=expected(offense.rating, defense.rating, defense.rd)
    return math.log(max(EPS,prior))+logit(q), q

def metrics(df):
    d=df[df.actual_six.notna()].copy(); P=d[list(SIX_COLS)].to_numpy(float)
    idx=np.array([SIX_LABELS.index(x) for x in d.actual_six]); y=np.zeros_like(P); y[np.arange(len(d)),idx]=1
    ptrue=np.clip(P[np.arange(len(d)),idx],EPS,1)
    pred=np.argmax(P,axis=1)
    out={
      "n":int(len(d)),
      "six_way_accuracy":float(np.mean(pred==idx)),
      "six_way_log_loss":float(-np.mean(np.log(ptrue))),
      "six_way_brier":float(np.mean(np.sum((P-y)**2,axis=1))),
      "mean_probability_actual_outcome":float(np.mean(ptrue)),
    }
    cols=["p_method_ko","p_method_sub","p_method_dec"]; labs=["KO","SUB","DEC"]
    M=d[cols].to_numpy(float); midx=np.array([labs.index(x) for x in d.actual_method]); mp=np.clip(M[np.arange(len(d)),midx],EPS,1)
    out.update({"method_accuracy":float(np.mean(np.argmax(M,axis=1)==midx)),"method_log_loss":float(-np.mean(np.log(mp)))})
    # per-class recall
    for lab in SIX_LABELS:
        mask=(d.actual_six==lab).to_numpy(); li=SIX_LABELS.index(lab)
        out[f"recall_{lab}"]=float(np.mean(pred[mask]==li)) if mask.any() else None
    out["sub_recall_combined"]=float((np.sum((pred==1)&(idx==1))+np.sum((pred==4)&(idx==4)))/np.sum(np.isin(idx,[1,4])))
    out["actual_method_shares"]={m:float(np.mean(d.actual_method==m)) for m in METHODS}
    out["predicted_method_shares"]={"KO":float(d.p_method_ko.mean()),"SUB":float(d.p_method_sub.mean()),"DEC":float(d.p_method_dec.mean())}
    return out

def run(bouts,w):
    offense={m:defaultdict(State) for m in METHODS}; defense={m:defaultdict(State) for m in METHODS}; counts={m:1.0 for m in METHODS}; rows=[]
    for b in bouts.itertuples(index=False):
        r,bl=b.red_fighter,b.blue_fighter; pri=method_priors(counts); scores=[]; qv={}
        for side,cw,cl in (("R",r,bl),("B",bl,r)):
            for meth in METHODS:
                s,q=class_signal(offense[meth][cw],defense[meth][cl],b.date,pri[meth]); scores.append(s); qv[f"q_{side.lower()}_{meth.lower()}"]=q
        p=softmax(scores); probs=dict(zip(SIX_COLS,p.tolist())); am=method_family(getattr(b,"method","")); actual=None
        if b.winner is not None and am is not None: actual=("R_" if b.winner==r else "B_")+am
        rows.append({"date":b.date,"bout_id":b.bout_id,"red_fighter":r,"blue_fighter":bl,"winner":b.winner,"method":getattr(b,"method",""),**probs,
                     "p_method_ko":probs["p_red_ko"]+probs["p_blue_ko"],"p_method_sub":probs["p_red_sub"]+probs["p_blue_sub"],"p_method_dec":probs["p_red_dec"]+probs["p_blue_dec"],
                     "actual_method":am,"actual_six":actual,**qv})
        if actual is None: continue
        pending=[]
        for side,cw,cl in (("R",r,bl),("B",bl,r)):
            for meth in METHODS:
                label=f"{side}_{meth}"
                if label==actual:
                    target=1.0
                elif meth==am:
                    target=0.0
                else:
                    target=0.5*(1.0-w)
                so=offense[meth][cw]; sd=defense[meth][cl]
                no,nro=update(so.rating,so.rd,sd.rating,sd.rd,target)
                nd,nrd=update(sd.rating,sd.rd,so.rating,so.rd,1.0-target)
                pending.append((so,no,nro,sd,nd,nrd))
        for so,no,nro,sd,nd,nrd in pending:
            so.rating,so.rd,so.last_date=no,nro,b.date; sd.rating,sd.rd,sd.last_date=nd,nrd,b.date
        counts[am]+=1.0
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,default=Path("data/master/ufc_master.parquet")); ap.add_argument("--holdout-from",default="2025-01-01"); ap.add_argument("--output-dir",type=Path,default=Path("data/diagnostics/standalone_glicko_six_way_partial_sweep")); args=ap.parse_args()
    bouts=build_bouts(pd.read_parquet(args.input)); cutoff=pd.Timestamp(args.holdout_from); args.output_dir.mkdir(parents=True,exist_ok=True)
    summary={"architecture":"pure direct joint Glicko-6 partial non-method negative-evidence sweep","holdout_from":args.holdout_from,"weights":{}}
    for w in WEIGHTS:
        pred=run(bouts,w); hold=pred[pred.date>=cutoff].copy(); key=f"{w:.2f}"; summary["weights"][key]=metrics(hold); hold.to_csv(args.output_dir/f"holdout_w_{key}.csv",index=False)
    with open(args.output_dir/"summary.json","w") as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
