#!/usr/bin/env python3
"""Fight-level submission propensity ablation for direct Glicko-6.

Research-only. No Brain, FSR, or market inputs.

Starts from the partial-evidence pure joint architecture. Adds one extra latent
fight-level SUB propensity track trained on whether ANY submission occurred.
The total SUB mass is determined by this fight-level propensity, while the
existing side-specific SUB matchup scores allocate that mass between red and blue.
KO and DEC class logits retain the existing pure-joint construction.
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
WEIGHTS=(0.25,0.50)

def method_family(method):
    m=str(method or "").lower()
    if "decision" in m: return "DEC"
    if "submission" in m or "sub" in m: return "SUB"
    if "ko" in m or "tko" in m: return "KO"
    return None

def softmax(xs):
    x=np.asarray(xs,float); x-=np.max(x); e=np.exp(x); return e/e.sum()

def logit(p):
    p=min(1-EPS,max(EPS,float(p))); return math.log(p)-math.log1p(-p)

def priors(counts):
    z=sum(counts.values()); return {m:counts[m]/z for m in METHODS}

def sig(off,defn,now,prior):
    inflate_rd(off,now); inflate_rd(defn,now)
    q=expected(off.rating,defn.rating,defn.rd)
    return math.log(max(EPS,prior))+logit(q),q

def metrics(d):
    d=d[d.actual_six.notna()].copy(); P=d[list(SIX_COLS)].to_numpy(float)
    idx=np.array([SIX_LABELS.index(x) for x in d.actual_six]); y=np.zeros_like(P); y[np.arange(len(d)),idx]=1
    ptrue=np.clip(P[np.arange(len(d)),idx],EPS,1); pred=np.argmax(P,axis=1)
    M=d[["p_method_ko","p_method_sub","p_method_dec"]].to_numpy(float); labs=["KO","SUB","DEC"]; midx=np.array([labs.index(x) for x in d.actual_method]); mp=np.clip(M[np.arange(len(d)),midx],EPS,1)
    out={"n":len(d),"six_way_accuracy":float(np.mean(pred==idx)),"six_way_log_loss":float(-np.mean(np.log(ptrue))),"six_way_brier":float(np.mean(np.sum((P-y)**2,axis=1))),"mean_probability_actual_outcome":float(np.mean(ptrue)),"method_accuracy":float(np.mean(np.argmax(M,axis=1)==midx)),"method_log_loss":float(-np.mean(np.log(mp))),"predicted_method_shares":{"KO":float(d.p_method_ko.mean()),"SUB":float(d.p_method_sub.mean()),"DEC":float(d.p_method_dec.mean())}}
    for lab in SIX_LABELS:
        mask=(d.actual_six==lab).to_numpy(); li=SIX_LABELS.index(lab); out[f"recall_{lab}"]=float(np.mean(pred[mask]==li)) if mask.any() else None
    out["sub_recall_combined"]=float((np.sum((pred==1)&(idx==1))+np.sum((pred==4)&(idx==4)))/np.sum(np.isin(idx,[1,4])))
    return out

def run(bouts,w):
    offense={m:defaultdict(State) for m in METHODS}; defense={m:defaultdict(State) for m in METHODS}
    subfight=defaultdict(State); counts={m:1.0 for m in METHODS}; rows=[]
    for b in bouts.itertuples(index=False):
        r,bl=b.red_fighter,b.blue_fighter; pr=priors(counts)
        base={}; qv={}
        for side,cw,cl in (("R",r,bl),("B",bl,r)):
            for m in METHODS:
                s,q=sig(offense[m][cw],defense[m][cl],b.date,pr[m]); base[(side,m)]=s; qv[f"q_{side.lower()}_{m.lower()}"]=q
        # fight-level SUB propensity from symmetric fighter matchup: average of each fighter's latent SUB-fight state against the other.
        sr,sb=subfight[r],subfight[bl]; inflate_rd(sr,b.date); inflate_rd(sb,b.date)
        q_r=expected(sr.rating,sb.rating,sb.rd); q_b=expected(sb.rating,sr.rating,sr.rd)
        # symmetric fight propensity: magnitude away from neutral pooled through mean logit; online prior anchors prevalence
        sub_prior=pr["SUB"]
        fight_sub_score=math.log(max(EPS,sub_prior))+0.5*(logit(q_r)+logit(q_b))
        # Existing SUB side scores allocate SUB mass; normalize side split only.
        sub_side=softmax([base[("R","SUB")],base[("B","SUB")]])
        # Compete total SUB against four KO/DEC classes.
        top_scores=[base[("R","KO")],base[("R","DEC")],base[("B","KO")],base[("B","DEC")],fight_sub_score]
        top=softmax(top_scores)
        probs={"p_red_ko":top[0],"p_red_dec":top[1],"p_blue_ko":top[2],"p_blue_dec":top[3],"p_red_sub":top[4]*sub_side[0],"p_blue_sub":top[4]*sub_side[1]}
        am=method_family(getattr(b,"method","")); actual=None
        if b.winner is not None and am is not None: actual=("R_" if b.winner==r else "B_")+am
        rows.append({"date":b.date,"bout_id":b.bout_id,"red_fighter":r,"blue_fighter":bl,"winner":b.winner,"method":getattr(b,"method",""),**probs,"p_method_ko":probs["p_red_ko"]+probs["p_blue_ko"],"p_method_sub":probs["p_red_sub"]+probs["p_blue_sub"],"p_method_dec":probs["p_red_dec"]+probs["p_blue_dec"],"actual_method":am,"actual_six":actual,"q_fight_sub_r":q_r,"q_fight_sub_b":q_b,**qv})
        if actual is None: continue
        pending=[]
        for side,cw,cl in (("R",r,bl),("B",bl,r)):
            for m in METHODS:
                label=f"{side}_{m}"
                if label==actual: target=1.0
                elif m==am: target=0.0
                else: target=0.5*(1-w)
                so=offense[m][cw]; sd=defense[m][cl]
                no,nro=update(so.rating,so.rd,sd.rating,sd.rd,target); nd,nrd=update(sd.rating,sd.rd,so.rating,so.rd,1-target)
                pending.append((so,no,nro,sd,nd,nrd))
        for so,no,nro,sd,nd,nrd in pending:
            so.rating,so.rd,so.last_date=no,nro,b.date; sd.rating,sd.rd,sd.last_date=nd,nrd,b.date
        # train fight-level SUB propensity as binary any-SUB. Symmetric paired update.
        ysub=1.0 if am=="SUB" else 0.0
        nr,nrr=update(sr.rating,sr.rd,sb.rating,sb.rd,ysub); nb,nbr=update(sb.rating,sb.rd,sr.rating,sr.rd,ysub)
        sr.rating,sr.rd,sr.last_date=nr,nrr,b.date; sb.rating,sb.rd,sb.last_date=nb,nbr,b.date
        counts[am]+=1.0
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,default=Path("data/master/ufc_master.parquet")); ap.add_argument("--holdout-from",default="2025-01-01"); ap.add_argument("--output-dir",type=Path,default=Path("data/diagnostics/standalone_glicko_six_way_sub_propensity")); args=ap.parse_args()
    bouts=build_bouts(pd.read_parquet(args.input)); cutoff=pd.Timestamp(args.holdout_from); args.output_dir.mkdir(parents=True,exist_ok=True)
    summary={"architecture":"fight-level SUB propensity + side-specific SUB allocation","holdout_from":args.holdout_from,"weights":{}}
    for w in WEIGHTS:
        pred=run(bouts,w); hold=pred[pred.date>=cutoff].copy(); key=f"{w:.2f}"; summary["weights"][key]=metrics(hold); hold.to_csv(args.output_dir/f"holdout_w_{key}.csv",index=False)
    with open(args.output_dir/"summary.json","w") as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
