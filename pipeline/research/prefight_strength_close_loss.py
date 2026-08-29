#!/usr/bin/env python3
"""Standalone Step 7: close-loss credit in chronological UFC Elo.

Research-only. Plain Elo remains the benchmark. For split/majority decisions only,
we test replacing the standard 1/0 result with a softer winner/loser score while
preserving zero-sum Elo updates. Candidate credit is selected strictly pre-2025
and then frozen for the 2025+ holdout.
"""
from __future__ import annotations

import argparse, json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.research.prefight_strength_elo import build_bouts, _expected

@dataclass
class State:
    rating: float = 1000.0

def is_close_decision(method: object) -> bool:
    s = str(method or "").lower()
    return "split" in s or "majority" in s

def metrics(df: pd.DataFrame, col: str) -> dict:
    d = df[df.winner.notna()].copy()
    p = np.clip(d[col].astype(float).to_numpy(), 1e-9, 1-1e-9)
    y = (d.winner == d.red_fighter).astype(float).to_numpy()
    nt = np.abs(p-0.5) > 1e-12
    return {
        "n": int(len(d)),
        "non_ties": int(nt.sum()),
        "accuracy_non_ties": float(np.mean((p[nt] > .5) == (y[nt] > .5))) if nt.any() else None,
        "brier": float(np.mean((p-y)**2)),
        "log_loss": float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
    }

def replay(bouts: pd.DataFrame, *, k: float, base: float, close_loss_credit: float) -> pd.DataFrame:
    states: dict[str, State] = {}
    rows=[]
    for bout in bouts.itertuples(index=False):
        r,b=bout.red_fighter,bout.blue_fighter
        sr=states.setdefault(r,State(base)); sb=states.setdefault(b,State(base))
        rr,br=sr.rating,sb.rating
        p=_expected(rr,br)
        winner=bout.winner
        method=getattr(bout,"method","")
        close=is_close_decision(method)
        rows.append({
            "date":pd.Timestamp(bout.date),"bout_id":bout.bout_id,"red_fighter":r,"blue_fighter":b,
            "winner":winner,"method":method,"is_close_decision":close,
            "red_pre_rating":rr,"blue_pre_rating":br,"red_win_prob":p,"blue_win_prob":1-p,
        })
        if winner not in (r,b):
            continue
        if close:
            # credit=0 is ordinary 1/0 Elo; credit=0.4 is 0.6/0.4.
            rscore = (1-close_loss_credit) if winner==r else close_loss_credit
        else:
            rscore = 1.0 if winner==r else 0.0
        delta=k*(rscore-p)
        sr.rating=rr+delta; sb.rating=br-delta
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,default=Path("data/master/ufc_master.parquet"))
    ap.add_argument("--output-dir",type=Path,default=Path("data/diagnostics/prefight_strength_close_loss"))
    ap.add_argument("--train-before",default="2025-01-01")
    ap.add_argument("--k-factor",type=float,default=170.0)
    ap.add_argument("--base-rating",type=float,default=1000.0)
    args=ap.parse_args()

    bouts=build_bouts(pd.read_parquet(args.input))
    cutoff=pd.Timestamp(args.train_before)
    credits=[0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45]
    cand=[]
    for c in credits:
        d=replay(bouts,k=args.k_factor,base=args.base_rating,close_loss_credit=c)
        m=metrics(d[d.date<cutoff],"red_win_prob")
        cand.append({"close_loss_credit":c,"train_log_loss":m["log_loss"],"train_brier":m["brier"],"train_accuracy_non_ties":m["accuracy_non_ties"]})
    grid=pd.DataFrame(cand).sort_values(["train_log_loss","train_brier"],kind="stable").reset_index(drop=True)
    selected=float(grid.iloc[0].close_loss_credit)

    baseline=replay(bouts,k=args.k_factor,base=args.base_rating,close_loss_credit=0.0)
    adjusted=replay(bouts,k=args.k_factor,base=args.base_rating,close_loss_credit=selected)
    out=adjusted.rename(columns={"red_win_prob":"close_loss_red_win_prob","blue_win_prob":"close_loss_blue_win_prob","red_pre_rating":"close_loss_red_pre_rating","blue_pre_rating":"close_loss_blue_pre_rating"})
    out["baseline_red_win_prob"]=baseline.red_win_prob.to_numpy()
    out["baseline_blue_win_prob"]=baseline.blue_win_prob.to_numpy()
    out["baseline_red_pre_rating"]=baseline.red_pre_rating.to_numpy()
    out["baseline_blue_pre_rating"]=baseline.blue_pre_rating.to_numpy()
    train=out[out.date<cutoff]; hold=out[out.date>=cutoff]
    summary={
        "source":str(args.input),"train_before":str(cutoff.date()),
        "mechanism":"split/majority decisions use winner score 1-credit and loser score credit; all other fights remain standard Elo",
        "candidate_credits":credits,"selected_close_loss_credit":selected,
        "selection_metric":"pre-2025 train log loss, then Brier",
        "train_baseline":metrics(train,"baseline_red_win_prob"),
        "train_close_loss":metrics(train,"close_loss_red_win_prob"),
        "holdout_baseline":metrics(hold,"baseline_red_win_prob"),
        "holdout_close_loss":metrics(hold,"close_loss_red_win_prob"),
        "scope":"research-only; plain Elo benchmark; prior rejected modifiers excluded",
        "leakage_rule":"credit selected only on fights before cutoff, then frozen for holdout",
    }
    args.output_dir.mkdir(parents=True,exist_ok=True)
    grid.to_csv(args.output_dir/"candidate_grid.csv",index=False)
    out.to_csv(args.output_dir/"fight_close_loss_elo.csv",index=False)
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
