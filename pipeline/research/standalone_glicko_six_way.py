#!/usr/bin/env python3
"""Standalone hierarchical Glicko six-way UFC benchmark.

Research-only. No Brain, FSR, or market inputs.

Architecture:
1) Existing FightMatrix-style graded Glicko-1 predicts winner side.
2) Three method-specific Glicko tracks (KO/TKO, SUB, DEC) predict the
   conditional method distribution for each hypothetical winner-vs-loser side.
3) Six mutually-exclusive probabilities are side probability x normalized
   conditional method probability and therefore sum exactly to 1.

All predictions are captured pre-bout; updates happen only after the bout.
"""
from __future__ import annotations

import argparse, json, math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.research.prefight_strength_elo import build_bouts
from pipeline.research.prefight_strength_fightmatrix_glicko import (
    State, expected, inflate_rd, outcome_scores, update,
)

METHODS = ("KO", "SUB", "DEC")
EPS = 1e-12


def method_family(method: str) -> str | None:
    m = str(method or "").lower()
    if "decision" in m:
        return "DEC"
    if "submission" in m or "sub" in m:
        return "SUB"
    if "ko" in m or "tko" in m:
        return "KO"
    return None


def normalized_method_probs(winner: str, loser: str, states: dict[str, defaultdict[str, State]], now: pd.Timestamp) -> dict[str, float]:
    raw = {}
    for meth in METHODS:
        sw = states[meth][winner]
        sl = states[meth][loser]
        inflate_rd(sw, now)
        inflate_rd(sl, now)
        raw[meth] = max(EPS, expected(sw.rating, sl.rating, sl.rd))
    z = sum(raw.values())
    return {k: v / z for k, v in raw.items()}


def six_metrics(df: pd.DataFrame) -> dict:
    d = df[df.actual_six.notna()].copy()
    if d.empty:
        return {"n": 0}
    six_cols = ["p_red_ko","p_red_sub","p_red_dec","p_blue_ko","p_blue_sub","p_blue_dec"]
    labels = ["R_KO","R_SUB","R_DEC","B_KO","B_SUB","B_DEC"]
    P = d[six_cols].to_numpy(float)
    idx = np.array([labels.index(x) for x in d.actual_six])
    ptrue = np.clip(P[np.arange(len(d)), idx], 1e-12, 1.0)
    y = np.zeros_like(P); y[np.arange(len(d)), idx] = 1.0
    pred_idx = np.argmax(P, axis=1)
    return {
        "n": int(len(d)),
        "six_way_accuracy": float(np.mean(pred_idx == idx)),
        "six_way_log_loss": float(-np.mean(np.log(ptrue))),
        "six_way_brier": float(np.mean(np.sum((P-y)**2, axis=1))),
    }


def side_metrics(df: pd.DataFrame) -> dict:
    d = df[df.winner.notna()].copy()
    p = np.clip(d.p_red_win.to_numpy(float), 1e-12, 1-1e-12)
    y = (d.winner == d.red_fighter).to_numpy(float)
    return {
        "n": int(len(d)),
        "accuracy": float(np.mean((p > .5) == (y > .5))),
        "brier": float(np.mean((p-y)**2)),
        "log_loss": float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
    }


def method_metrics(df: pd.DataFrame) -> dict:
    d = df[df.actual_method.notna()].copy()
    if d.empty: return {"n": 0}
    cols = ["p_method_ko","p_method_sub","p_method_dec"]
    labels = ["KO","SUB","DEC"]
    P = d[cols].to_numpy(float)
    idx = np.array([labels.index(x) for x in d.actual_method])
    ptrue = np.clip(P[np.arange(len(d)), idx], 1e-12, 1.0)
    return {
        "n": int(len(d)),
        "accuracy": float(np.mean(np.argmax(P,axis=1)==idx)),
        "log_loss": float(-np.mean(np.log(ptrue))),
    }


def run(bouts: pd.DataFrame) -> pd.DataFrame:
    winner_states = defaultdict(State)
    method_states = {m: defaultdict(State) for m in METHODS}
    rows=[]
    for b in bouts.itertuples(index=False):
        r, bl = b.red_fighter, b.blue_fighter
        sr, sb = winner_states[r], winner_states[bl]
        inflate_rd(sr, b.date); inflate_rd(sb, b.date)
        p_r = expected(sr.rating, sb.rating, sb.rd)
        rm = normalized_method_probs(r, bl, method_states, b.date)
        bm = normalized_method_probs(bl, r, method_states, b.date)
        probs = {
            "p_red_ko": p_r*rm["KO"], "p_red_sub": p_r*rm["SUB"], "p_red_dec": p_r*rm["DEC"],
            "p_blue_ko": (1-p_r)*bm["KO"], "p_blue_sub": (1-p_r)*bm["SUB"], "p_blue_dec": (1-p_r)*bm["DEC"],
        }
        actual_m = method_family(getattr(b,"method", ""))
        actual_six = None
        if b.winner is not None and actual_m is not None:
            actual_six = ("R_" if b.winner == r else "B_") + actual_m
        row = {
            "date": b.date, "bout_id": b.bout_id, "red_fighter": r, "blue_fighter": bl,
            "winner": b.winner, "method": getattr(b,"method", ""),
            "p_red_win": p_r, "p_blue_win": 1-p_r,
            **probs,
            "p_method_ko": probs["p_red_ko"]+probs["p_blue_ko"],
            "p_method_sub": probs["p_red_sub"]+probs["p_blue_sub"],
            "p_method_dec": probs["p_red_dec"]+probs["p_blue_dec"],
            "actual_method": actual_m, "actual_six": actual_six,
            "red_pre_rating": sr.rating, "blue_pre_rating": sb.rating,
            "red_pre_rd": sr.rd, "blue_pre_rd": sb.rd,
        }
        rows.append(row)
        if b.winner is None:
            continue
        # Winner Glicko: exact same graded scoring as validated standalone benchmark.
        rs, bs = outcome_scores(getattr(b,"method", ""), b.winner == r)
        nr, nrr = update(sr.rating, sr.rd, sb.rating, sb.rd, rs)
        nb, nbr = update(sb.rating, sb.rd, sr.rating, sr.rd, bs)
        sr.rating, sr.rd, sr.last_date = nr, nrr, b.date
        sb.rating, sb.rd, sb.last_date = nb, nbr, b.date

        # Method tracks: conditional on the actual winner/loser orientation.
        if actual_m is not None:
            w, l = (r, bl) if b.winner == r else (bl, r)
            for meth in METHODS:
                sw, sl = method_states[meth][w], method_states[meth][l]
                # RD already inflated during prefight prediction for both orientations.
                score = 1.0 if meth == actual_m else 0.0
                nw, nrw = update(sw.rating, sw.rd, sl.rating, sl.rd, score)
                nl, nrl = update(sl.rating, sl.rd, sw.rating, sw.rd, 1.0-score)
                sw.rating, sw.rd, sw.last_date = nw, nrw, b.date
                sl.rating, sl.rd, sl.last_date = nl, nrl, b.date
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/master/ufc_master.parquet"))
    ap.add_argument("--holdout-from", default="2025-01-01")
    ap.add_argument("--output-dir", type=Path, default=Path("data/diagnostics/standalone_glicko_six_way"))
    args=ap.parse_args()
    bouts=build_bouts(pd.read_parquet(args.input))
    pred=run(bouts)
    cutoff=pd.Timestamp(args.holdout_from)
    train=pred[pred.date < cutoff].copy(); hold=pred[pred.date >= cutoff].copy()
    summary={
        "architecture":"hierarchical Glicko-1: graded winner Glicko x normalized KO/SUB/DEC Glicko tracks",
        "no_brain":True,"no_fsr":True,"no_market":True,
        "holdout_from":args.holdout_from,
        "train_side":side_metrics(train),"holdout_side":side_metrics(hold),
        "train_method":method_metrics(train),"holdout_method":method_metrics(hold),
        "train_six":six_metrics(train),"holdout_six":six_metrics(hold),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.output_dir/"fight_predictions.csv",index=False)
    hold.to_csv(args.output_dir/"holdout_predictions.csv",index=False)
    with open(args.output_dir/"summary.json","w") as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
