#!/usr/bin/env python3
"""Standalone UFC-only FightMatrix-style Glicko-1 benchmark.

Research-only. No Brain/FSR/market dependency.

Implements published FightMatrix-style outcome scores:
- finish 1.00 / 0.00
- unanimous decision 0.91 / 0.09
- majority decision 0.61 / 0.39
- split decision 0.55 / 0.45

Uses Glicko-1 rating + rating deviation (RD). Inactivity does not decay rating;
RD inflates after 180 days. This is a UFC-only approximation because the master
parquet contains UFC bouts rather than full professional MMA history.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.research.prefight_strength_elo import build_bouts

Q = math.log(10.0) / 400.0
BASE = 1500.0
INIT_RD = 350.0
MIN_RD = 30.0
MAX_RD = 350.0
INACTIVITY_GRACE_DAYS = 180.0
# Research approximation for FightMatrix-style RD inflation. Chosen a priori;
# not optimized on the holdout. This yields gradual uncertainty growth.
RD_INFLATION_PER_DAY = 4.0

@dataclass
class State:
    rating: float = BASE
    rd: float = INIT_RD
    last_date: pd.Timestamp | None = None


def g(rd: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * (Q * rd) ** 2 / (math.pi ** 2))


def expected(r: float, opp_r: float, opp_rd: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-g(opp_rd) * (r - opp_r) / 400.0))


def inflate_rd(st: State, now: pd.Timestamp) -> None:
    if st.last_date is None:
        return
    days = float((now - st.last_date).days)
    extra = max(0.0, days - INACTIVITY_GRACE_DAYS)
    if extra <= 0:
        return
    st.rd = min(MAX_RD, math.sqrt(st.rd * st.rd + RD_INFLATION_PER_DAY * extra))


def update(r: float, rd: float, opp_r: float, opp_rd: float, score: float) -> tuple[float, float]:
    gg = g(opp_rd)
    e = expected(r, opp_r, opp_rd)
    d2_inv = (Q * Q) * (gg * gg) * e * (1.0 - e)
    d2 = 1.0 / max(d2_inv, 1e-15)
    denom = (1.0 / (rd * rd)) + (1.0 / d2)
    new_r = r + (Q / denom) * gg * (score - e)
    new_rd = math.sqrt(1.0 / denom)
    return float(new_r), float(max(MIN_RD, min(MAX_RD, new_rd)))


def outcome_scores(method: str, winner_is_red: bool) -> tuple[float, float]:
    m = (method or "").lower()
    if "split" in m:
        hi, lo = 0.55, 0.45
    elif "majority" in m:
        hi, lo = 0.61, 0.39
    elif "decision" in m:
        hi, lo = 0.91, 0.09
    else:
        hi, lo = 1.0, 0.0
    return (hi, lo) if winner_is_red else (lo, hi)


def metrics(df: pd.DataFrame, pcol: str) -> dict:
    d = df[df.winner.notna()].copy()
    p = d[pcol].clip(1e-9, 1-1e-9).astype(float)
    y = (d.winner == d.red_fighter).astype(float)
    non_tie = np.abs(p - 0.5) > 1e-12
    acc = (((p > .5) == (y > .5))[non_tie]).mean() if non_tie.any() else np.nan
    return {
        "n": int(len(d)),
        "non_ties": int(non_tie.sum()),
        "accuracy_non_ties": float(acc),
        "brier": float(np.mean((p-y)**2)),
        "log_loss": float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
    }


def run(bouts: pd.DataFrame) -> pd.DataFrame:
    states = defaultdict(State)
    rows=[]
    for b in bouts.itertuples(index=False):
        r,bn=b.red_fighter,b.blue_fighter
        sr,sb=states[r],states[bn]
        inflate_rd(sr,b.date); inflate_rd(sb,b.date)
        rp,bp=sr.rating,sb.rating
        rr,br=sr.rd,sb.rd
        pr=expected(rp,bp,br)
        row={
            "date": b.date, "bout_id": b.bout_id, "red_fighter": r, "blue_fighter": bn,
            "winner": b.winner, "method": getattr(b,"method", ""),
            "red_pre_rating": rp, "blue_pre_rating": bp, "red_pre_rd": rr, "blue_pre_rd": br,
            "glicko_red_win_prob": pr, "glicko_blue_win_prob": 1-pr,
        }
        if b.winner is None:
            rows.append(row); continue
        rs,bs=outcome_scores(getattr(b,"method", ""), b.winner==r)
        nr,nrr=update(rp,rr,bp,br,rs)
        nb,nbr=update(bp,br,rp,rr,bs)
        sr.rating,sr.rd,sr.last_date=nr,nrr,b.date
        sb.rating,sb.rd,sb.last_date=nb,nbr,b.date
        row.update({"red_result_score":rs,"blue_result_score":bs,"red_post_rating":nr,"blue_post_rating":nb,"red_post_rd":nrr,"blue_post_rd":nbr})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,default=Path("data/master/ufc_master.parquet"))
    ap.add_argument("--output-dir",type=Path,default=Path("data/diagnostics/prefight_strength_fightmatrix_glicko"))
    ap.add_argument("--holdout-from",default="2025-01-01")
    args=ap.parse_args()
    bouts=build_bouts(pd.read_parquet(args.input))
    fights=run(bouts)
    cutoff=pd.Timestamp(args.holdout_from)
    train=fights[fights.date < cutoff]
    hold=fights[fights.date >= cutoff]
    summary={
        "source":str(args.input),
        "published_style_rules":{
            "starter_rating":BASE,"starter_rd":INIT_RD,"finish":[1.0,0.0],"unanimous":[0.91,0.09],"majority":[0.61,0.39],"split":[0.55,0.45],"inactivity_rd_inflation_after_days":180
        },
        "implementation_scope":"UFC-only FightMatrix-style Glicko-1 approximation; rating is preserved during inactivity while RD inflates; no Brain/FSR/market dependency",
        "rd_inflation_per_day":RD_INFLATION_PER_DAY,
        "train_before":args.holdout_from,
        "train_glicko":metrics(train,"glicko_red_win_prob"),
        "holdout_glicko":metrics(hold,"glicko_red_win_prob"),
    }
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fights.to_csv(args.output_dir/"fight_glicko.csv",index=False)
    with open(args.output_dir/"summary.json","w") as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
