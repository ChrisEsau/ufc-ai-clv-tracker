#!/usr/bin/env python3
"""Research-only UFC-data replication of FightMatrix's published Modified Elo rules.

Published rules reproduced where UFC master data supports them:
- starter rating 1000
- algorithm-of-400 Elo expectation
- K=275 for a fighter's first 3 observed fights, K=155 thereafter
- split decision score 0.667 winner / 0.333 loser
- majority decision score 0.833 winner / 0.167 loser
- other wins 1.0 / 0.0; draws 0.5 / 0.5
- DQ and one-round technical draws ignored

Known data-scope limitations:
1) FightMatrix uses first three PROFESSIONAL MMA fights; this UFC-only source can only
   count observed UFC fights, so early-career K is a UFC-fight proxy.
2) FightMatrix applies a 15-point continent-level home adjustment.  The current UFC
   master bout layer does not expose a validated fighter-home-continent/event-continent
   pair, so home advantage is intentionally omitted rather than inferred.

No Brain, FSR, market, QP, inactivity, schedule, or prior custom modifier dependency.
"""
from __future__ import annotations

import argparse, json, math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.research.prefight_strength_elo import build_bouts


def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def metric(df: pd.DataFrame, pcol: str) -> dict[str, Any]:
    d = df[df.winner.notna()].copy()
    y = (d.winner == d.red_fighter).astype(float).to_numpy()
    p = np.clip(d[pcol].astype(float).to_numpy(), 1e-12, 1-1e-12)
    picks = np.where(p > .5, d.red_fighter, np.where(p < .5, d.blue_fighter, ""))
    non = picks != ""
    correct = (picks == d.winner.to_numpy())
    return {
        "n": int(len(d)),
        "non_ties": int(non.sum()),
        "accuracy_non_ties": float(correct[non].mean()) if non.any() else None,
        "brier": float(np.mean((p-y)**2)),
        "log_loss": float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
    }


def decision_scores(method: str, winner_is_red: bool | None) -> tuple[float,float] | None:
    m = (method or "").lower()
    if "dq" in m or "disqual" in m:
        return None
    if winner_is_red is None:
        return (0.5, 0.5)
    if "split" in m:
        return (0.667,0.333) if winner_is_red else (0.333,0.667)
    if "majority" in m:
        return (0.833,0.167) if winner_is_red else (0.167,0.833)
    return (1.0,0.0) if winner_is_red else (0.0,1.0)


def run(bouts: pd.DataFrame) -> pd.DataFrame:
    base = defaultdict(lambda: 1000.0)
    mod = defaultdict(lambda: 1000.0)
    prior = defaultdict(int)
    rows=[]

    for b in bouts.itertuples(index=False):
        r,bn=b.red_fighter,b.blue_fighter
        rb,bb=float(base[r]),float(base[bn])
        rm,bm=float(mod[r]),float(mod[bn])
        pe_b=expected(rb,bb); pe_m=expected(rm,bm)
        winner=b.winner
        raw=str(getattr(b,"winner_raw","") or "").lower()
        method=str(getattr(b,"method","") or "")
        fr=getattr(b,"finish_round",None)
        is_draw=(winner is None and "draw" in raw and "no contest" not in raw)
        one_round_tech_draw=(is_draw and "technical" in raw and pd.notna(fr) and float(fr)==1.0)
        winner_red=True if winner==r else (False if winner==bn else None)
        score=decision_scores(method,winner_red) if (winner is not None or is_draw) else None
        if one_round_tech_draw:
            score=None

        rows.append({
            "date": b.date, "bout_id": b.bout_id,
            "red_fighter": r, "blue_fighter": bn, "winner": winner,
            "red_prior_ufc_fights": prior[r], "blue_prior_ufc_fights": prior[bn],
            "baseline_red_pre_rating": rb, "baseline_blue_pre_rating": bb,
            "baseline_red_win_prob": pe_b,
            "modified_red_pre_rating": rm, "modified_blue_pre_rating": bm,
            "modified_red_win_prob": pe_m,
            "red_k": 275.0 if prior[r] < 3 else 155.0,
            "blue_k": 275.0 if prior[bn] < 3 else 155.0,
            "method": method,
            "ignored_update": score is None,
        })
        if score is None:
            continue
        sr,sb=score
        # Baseline K-170, same graduated outcome scores only for direct apples-to-apples
        # probability benchmark remains separately available from prefight_strength_elo.py.
        if winner is not None:
            br=1.0 if winner==r else 0.0
            base[r]=rb+170.0*(br-pe_b)
            base[bn]=bb+170.0*((1.0-br)-(1.0-pe_b))
        elif is_draw:
            base[r]=rb+170.0*(0.5-pe_b)
            base[bn]=bb+170.0*(0.5-(1.0-pe_b))

        kr=275.0 if prior[r] < 3 else 155.0
        kb=275.0 if prior[bn] < 3 else 155.0
        mod[r]=rm+kr*(sr-pe_m)
        mod[bn]=bm+kb*(sb-(1.0-pe_m))
        prior[r]+=1; prior[bn]+=1
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,default=Path("data/master/ufc_master.parquet"))
    ap.add_argument("--output-dir",type=Path,default=Path("data/diagnostics/prefight_strength_fightmatrix_modified_elo"))
    ap.add_argument("--holdout-from",default="2025-01-01")
    args=ap.parse_args()
    src=pd.read_parquet(args.input)
    fights=run(build_bouts(src))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fights.to_csv(args.output_dir/"fight_modified_elo.csv",index=False)
    cut=pd.Timestamp(args.holdout_from)
    train=fights[fights.date<cut]; hold=fights[fights.date>=cut]
    summary={
        "source":str(args.input),
        "published_rules": {"starter":1000,"k_first3":275,"k_after3":155,"split":[0.667,0.333],"majority":[0.833,0.167],"other_win":[1.0,0.0],"draw":[0.5,0.5],"home_continent_points":15},
        "implementation_scope":"UFC-only approximation: first 3 observed UFC fights proxy first 3 pro fights; 15-point home-continent adjustment omitted because validated continent fields are unavailable in current master bout layer",
        "train_before":args.holdout_from,
        "train_modified":metric(train,"modified_red_win_prob"),
        "holdout_modified":metric(hold,"modified_red_win_prob"),
        "scope":"research-only; no Brain/FSR/market dependency",
    }
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
