#!/usr/bin/env python3
"""Standalone Glicko-6 UFC outcome model.

Research-only. No Brain, FSR, or market inputs.

Architecture
------------
1. A validated FightMatrix-style Glicko-1 track estimates overall fighter strength.
2. Three additional Glicko-1 tracks estimate relative KO, SUB, and DEC outcome
   propensity. Each track is captured strictly prefight.
3. A multinomial logistic calibration layer maps the prefight Glicko coordinates
   to six mutually exclusive outcomes:
      RED_KO, RED_SUB, RED_DEC, BLUE_KO, BLUE_SUB, BLUE_DEC.
   The calibration layer is trained only before --holdout-from. Holdout fights
   are never used to fit coefficients.

The method tracks are intentionally simple for this first benchmark:
- if the bout ends by that method, the winner/loser receive the corresponding
  result score (finishes 1/0; decisions retain graded FightMatrix scoring);
- if it ends by another method, both fighters receive 0.5 in that method track.
This makes the tracks comparative method-propensity ratings while the softmax
intercepts learn absolute method prevalence from training data.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from pipeline.research.prefight_strength_elo import build_bouts
from pipeline.research.prefight_strength_fightmatrix_glicko import (
    BASE, INIT_RD, State, expected, inflate_rd, outcome_scores, update,
)

METHODS = ("KO", "SUB", "DEC")
CLASSES = ("RED_KO", "RED_SUB", "RED_DEC", "BLUE_KO", "BLUE_SUB", "BLUE_DEC")
OUT_DEFAULT = Path("data/research/standalone_glicko_sixway")


def normalize_method(value: object) -> str | None:
    s = str(value or "").strip().lower()
    if "decision" in s:
        return "DEC"
    if "submission" in s or s == "sub" or "technical submission" in s:
        return "SUB"
    if "ko/tko" in s or "tko" in s or s == "ko" or s.startswith("ko "):
        return "KO"
    return None


def sixway_class(red: str, blue: str, winner: str | None, method: str | None) -> str | None:
    if winner is None or method not in METHODS:
        return None
    if winner == red:
        return f"RED_{method}"
    if winner == blue:
        return f"BLUE_{method}"
    return None


def method_scores(method_track: str, actual_method: str, raw_method: str, winner_is_red: bool) -> tuple[float, float]:
    if method_track != actual_method:
        return 0.5, 0.5
    if method_track == "DEC":
        return outcome_scores(raw_method, winner_is_red)
    return (1.0, 0.0) if winner_is_red else (0.0, 1.0)


def _state_snapshot(st: State, now: pd.Timestamp) -> tuple[float, float]:
    inflate_rd(st, now)
    return float(st.rating), float(st.rd)


def build_prefight_features(bouts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    overall = defaultdict(State)
    method_states = {m: defaultdict(State) for m in METHODS}
    rows: list[dict] = []
    excluded = {"no_winner": 0, "unsupported_method": 0}

    for b in bouts.itertuples(index=False):
        red, blue = str(b.red_fighter), str(b.blue_fighter)
        date = pd.Timestamp(b.date)
        raw_method = str(getattr(b, "method", "") or "")
        method = normalize_method(raw_method)
        winner = b.winner

        sr, sb = overall[red], overall[blue]
        orating, ord_ = _state_snapshot(sr, date)
        brating, brd = _state_snapshot(sb, date)
        p_red = expected(orating, brating, brd)

        row = {
            "date": date,
            "bout_id": str(b.bout_id),
            "red_fighter": red,
            "blue_fighter": blue,
            "winner": winner,
            "raw_method": raw_method,
            "method": method,
            "overall_red_rating": orating,
            "overall_blue_rating": brating,
            "overall_red_rd": ord_,
            "overall_blue_rd": brd,
            "overall_red_win_prob": p_red,
            "overall_rating_diff": orating - brating,
            "overall_rd_sum": ord_ + brd,
            "overall_rd_diff": ord_ - brd,
        }

        for m in METHODS:
            mr, mb = method_states[m][red], method_states[m][blue]
            rr, rrd = _state_snapshot(mr, date)
            rb, rbd = _state_snapshot(mb, date)
            key = m.lower()
            row[f"{key}_red_rating"] = rr
            row[f"{key}_blue_rating"] = rb
            row[f"{key}_red_rd"] = rrd
            row[f"{key}_blue_rd"] = rbd
            row[f"{key}_rating_diff"] = rr - rb
            row[f"{key}_rd_sum"] = rrd + rbd
            row[f"{key}_red_edge_prob"] = expected(rr, rb, rbd)

        target = sixway_class(red, blue, winner, method)
        row["target"] = target
        rows.append(row)

        if winner is None:
            excluded["no_winner"] += 1
            continue
        if method is None:
            excluded["unsupported_method"] += 1

        # Overall Glicko updates on every resolved bout, preserving the validated
        # graded FightMatrix-style outcome semantics.
        rs, bs = outcome_scores(raw_method, winner == red)
        nr, nrd = update(orating, ord_, brating, brd, rs)
        nb, nbd = update(brating, brd, orating, ord_, bs)
        sr.rating, sr.rd, sr.last_date = nr, nrd, date
        sb.rating, sb.rd, sb.last_date = nb, nbd, date

        # Method tracks require one of the supported six-way outcome families.
        if method is None:
            continue
        for m in METHODS:
            mr, mb = method_states[m][red], method_states[m][blue]
            rr, rrd = float(row[f"{m.lower()}_red_rating"]), float(row[f"{m.lower()}_red_rd"])
            rb, rbd = float(row[f"{m.lower()}_blue_rating"]), float(row[f"{m.lower()}_blue_rd"])
            msr, msb = method_scores(m, method, raw_method, winner == red)
            nrr, nrrd = update(rr, rrd, rb, rbd, msr)
            nrb, nrbd = update(rb, rbd, rr, rrd, msb)
            mr.rating, mr.rd, mr.last_date = nrr, nrrd, date
            mb.rating, mb.rd, mb.last_date = nrb, nrbd, date

    return pd.DataFrame(rows), excluded


FEATURES = [
    "overall_rating_diff", "overall_rd_sum", "overall_rd_diff",
    "ko_rating_diff", "ko_rd_sum",
    "sub_rating_diff", "sub_rd_sum",
    "dec_rating_diff", "dec_rd_sum",
]


def multiclass_brier(y_idx: np.ndarray, p: np.ndarray) -> float:
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    non_tie = np.abs(p - 0.5) > 1e-12
    return {
        "n": int(len(y)),
        "accuracy_non_ties": float(np.mean((p[non_tie] > 0.5) == (y[non_tie] > 0.5))) if non_tie.any() else None,
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1-y) * np.log(1-p))),
    }


def evaluate(df: pd.DataFrame, prob_cols: list[str]) -> dict:
    class_to_i = {c: i for i, c in enumerate(CLASSES)}
    y_idx = np.array([class_to_i[x] for x in df.target], dtype=int)
    p = df[prob_cols].to_numpy(float)
    pred_idx = np.argmax(p, axis=1)
    exact_accuracy = float(np.mean(pred_idx == y_idx))
    six_ll = float(log_loss(y_idx, p, labels=np.arange(len(CLASSES))))
    six_brier = multiclass_brier(y_idx, p)

    red_p = p[:, 0:3].sum(axis=1)
    y_red = np.array([1.0 if str(x).startswith("RED_") else 0.0 for x in df.target])
    ml = binary_metrics(y_red, red_p)

    method_p = np.column_stack([p[:, 0] + p[:, 3], p[:, 1] + p[:, 4], p[:, 2] + p[:, 5]])
    method_to_i = {"KO": 0, "SUB": 1, "DEC": 2}
    y_method = np.array([method_to_i[str(x).split("_", 1)[1]] for x in df.target])
    method_accuracy = float(np.mean(np.argmax(method_p, axis=1) == y_method))
    method_ll = float(log_loss(y_method, method_p, labels=[0,1,2]))

    return {
        "n": int(len(df)),
        "sixway_exact_accuracy": exact_accuracy,
        "sixway_log_loss": six_ll,
        "sixway_brier": six_brier,
        "winner": ml,
        "method_3way_accuracy": method_accuracy,
        "method_3way_log_loss": method_ll,
        "actual_method_share": {
            "KO": float(np.mean(y_method == 0)),
            "SUB": float(np.mean(y_method == 1)),
            "DEC": float(np.mean(y_method == 2)),
        },
        "predicted_method_share": {
            "KO": float(method_p[:,0].mean()),
            "SUB": float(method_p[:,1].mean()),
            "DEC": float(method_p[:,2].mean()),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/master/ufc_master.parquet"))
    ap.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--holdout-from", default="2025-01-01")
    ap.add_argument("--c", type=float, default=0.25, help="Softmax inverse regularization; fixed a priori for v1")
    args = ap.parse_args()

    bouts = build_bouts(pd.read_parquet(args.input))
    all_rows, excluded = build_prefight_features(bouts)
    eligible = all_rows[all_rows.target.notna()].copy()
    cutoff = pd.Timestamp(args.holdout_from)
    train = eligible[eligible.date < cutoff].copy()
    hold = eligible[eligible.date >= cutoff].copy()
    if train.empty or hold.empty:
        raise RuntimeError("empty train or holdout after six-way filtering")

    # Scaling uses train only. Rating differences are naturally ~hundreds; RDs are
    # also hundreds. Standardization improves numerical conditioning without leakage.
    x_train_raw = train[FEATURES].to_numpy(float)
    mu = x_train_raw.mean(axis=0)
    sd = x_train_raw.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    x_train = (x_train_raw - mu) / sd
    x_hold = (hold[FEATURES].to_numpy(float) - mu) / sd

    model = LogisticRegression(
        C=float(args.c),
        solver="lbfgs",
        max_iter=2000,
        random_state=0,
    )
    model.fit(x_train, train.target.to_numpy())

    # sklearn class order can differ; reorder into canonical six-way order.
    def canonical_probs(x: np.ndarray) -> np.ndarray:
        raw = model.predict_proba(x)
        pos = {c: i for i, c in enumerate(model.classes_)}
        return np.column_stack([raw[:, pos[c]] for c in CLASSES])

    train_p = canonical_probs(x_train)
    hold_p = canonical_probs(x_hold)
    prob_cols = [f"p_{c.lower()}" for c in CLASSES]
    for j, c in enumerate(prob_cols):
        train[c] = train_p[:, j]
        hold[c] = hold_p[:, j]

    for d in (train, hold):
        d["p_red_ml"] = d[["p_red_ko","p_red_sub","p_red_dec"]].sum(axis=1)
        d["p_blue_ml"] = 1.0 - d["p_red_ml"]
        d["p_ko"] = d["p_red_ko"] + d["p_blue_ko"]
        d["p_sub"] = d["p_red_sub"] + d["p_blue_sub"]
        d["p_dec"] = d["p_red_dec"] + d["p_blue_dec"]
        d["predicted_sixway"] = d[prob_cols].idxmax(axis=1).str.removeprefix("p_").str.upper()

    summary = {
        "study": "standalone Glicko-6 six-way UFC outcome model",
        "brain_used": False,
        "fsr_used": False,
        "market_used": False,
        "holdout_from": args.holdout_from,
        "softmax_C_fixed_apriori": float(args.c),
        "features": FEATURES,
        "classes": list(CLASSES),
        "excluded": excluded,
        "train": evaluate(train, prob_cols),
        "holdout": evaluate(hold, prob_cols),
        "softmax_classes_internal": model.classes_.tolist(),
        "softmax_intercepts": {str(c): float(v) for c, v in zip(model.classes_, model.intercept_)},
        "softmax_coefficients": {
            str(c): {f: float(v) for f, v in zip(FEATURES, row)}
            for c, row in zip(model.classes_, model.coef_)
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(args.output_dir / "all_prefight_glicko_features.csv", index=False)
    train.to_csv(args.output_dir / "train_predictions.csv", index=False)
    hold.to_csv(args.output_dir / "holdout_predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
