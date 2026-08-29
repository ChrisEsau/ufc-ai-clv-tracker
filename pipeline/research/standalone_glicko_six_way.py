#!/usr/bin/env python3
"""Standalone direct six-way Glicko UFC benchmark.

Research-only. No Brain, FSR, or market inputs.

This is intentionally NOT winner x conditional method. It predicts the six
mutually-exclusive outcomes directly:

    red KO/TKO, red SUB, red DEC,
    blue KO/TKO, blue SUB, blue DEC.

Architecture
------------
For each method family (KO, SUB, DEC), every fighter has separate offensive
and defensive Glicko states. A class-specific prefight signal is produced by
matching the candidate winner's method offense against the candidate loser's
method defense. The six class signals are combined in one softmax, with a
strictly pre-bout online method-frequency intercept.

After the bout, each of the six class matchups receives a one-vs-rest binary
Glicko update (1 only for the observed class, 0 for the other five). Updates
are computed from the frozen prefight states; no same-bout update can leak
into another class prediction or update.

All predictions are captured pre-bout; updates happen only after the bout.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.research.prefight_strength_elo import build_bouts
from pipeline.research.prefight_strength_fightmatrix_glicko import (
    State,
    expected,
    inflate_rd,
    update,
)

METHODS = ("KO", "SUB", "DEC")
SIX_LABELS = ("R_KO", "R_SUB", "R_DEC", "B_KO", "B_SUB", "B_DEC")
SIX_COLS = (
    "p_red_ko", "p_red_sub", "p_red_dec",
    "p_blue_ko", "p_blue_sub", "p_blue_dec",
)
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


def softmax(scores: list[float]) -> np.ndarray:
    x = np.asarray(scores, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def logit(p: float) -> float:
    p = min(1.0 - EPS, max(EPS, float(p)))
    return math.log(p) - math.log1p(-p)


def method_priors(method_counts: dict[str, float]) -> dict[str, float]:
    total = float(sum(method_counts.values()))
    return {m: float(method_counts[m]) / total for m in METHODS}


def class_signal(
    offense: State,
    defense: State,
    now: pd.Timestamp,
    method_prior: float,
) -> tuple[float, float]:
    """Return direct class logit and underlying Glicko matchup probability."""
    inflate_rd(offense, now)
    inflate_rd(defense, now)
    q = expected(offense.rating, defense.rating, defense.rd)
    # Method prior is an online intercept. The fighter matchup contribution is
    # the Glicko log-odds. All six logits compete in one joint softmax.
    score = math.log(max(EPS, method_prior)) + logit(q)
    return score, q


def six_metrics(df: pd.DataFrame) -> dict:
    d = df[df.actual_six.notna()].copy()
    if d.empty:
        return {"n": 0}
    P = d[list(SIX_COLS)].to_numpy(float)
    idx = np.array([SIX_LABELS.index(x) for x in d.actual_six])
    ptrue = np.clip(P[np.arange(len(d)), idx], EPS, 1.0)
    y = np.zeros_like(P)
    y[np.arange(len(d)), idx] = 1.0
    pred_idx = np.argmax(P, axis=1)
    return {
        "n": int(len(d)),
        "six_way_accuracy": float(np.mean(pred_idx == idx)),
        "six_way_log_loss": float(-np.mean(np.log(ptrue))),
        "six_way_brier": float(np.mean(np.sum((P - y) ** 2, axis=1))),
        "mean_probability_actual_outcome": float(np.mean(ptrue)),
    }


def side_metrics(df: pd.DataFrame) -> dict:
    d = df[df.winner.notna()].copy()
    if d.empty:
        return {"n": 0}
    p = np.clip(d.p_red_win.to_numpy(float), EPS, 1.0 - EPS)
    y = (d.winner == d.red_fighter).to_numpy(float)
    return {
        "n": int(len(d)),
        "accuracy": float(np.mean((p > 0.5) == (y > 0.5))),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
    }


def method_metrics(df: pd.DataFrame) -> dict:
    d = df[df.actual_method.notna()].copy()
    if d.empty:
        return {"n": 0}
    cols = ["p_method_ko", "p_method_sub", "p_method_dec"]
    labels = ["KO", "SUB", "DEC"]
    P = d[cols].to_numpy(float)
    idx = np.array([labels.index(x) for x in d.actual_method])
    ptrue = np.clip(P[np.arange(len(d)), idx], EPS, 1.0)
    return {
        "n": int(len(d)),
        "accuracy": float(np.mean(np.argmax(P, axis=1) == idx)),
        "log_loss": float(-np.mean(np.log(ptrue))),
        "mean_probability_actual_method": float(np.mean(ptrue)),
    }


def share_metrics(df: pd.DataFrame) -> dict:
    d = df[df.actual_method.notna()].copy()
    if d.empty:
        return {"n": 0}
    return {
        "n": int(len(d)),
        "actual": {m: float(np.mean(d.actual_method == m)) for m in METHODS},
        "predicted_mean": {
            "KO": float(d.p_method_ko.mean()),
            "SUB": float(d.p_method_sub.mean()),
            "DEC": float(d.p_method_dec.mean()),
        },
    }


def run(bouts: pd.DataFrame) -> pd.DataFrame:
    # Separate method offense and method defense Glicko coordinates.
    offense = {m: defaultdict(State) for m in METHODS}
    defense = {m: defaultdict(State) for m in METHODS}

    # Laplace seed only; by the holdout this is dominated by historical bouts.
    # Counts are updated after each bout, so the intercept is always prefight.
    method_counts = {m: 1.0 for m in METHODS}

    rows: list[dict] = []

    for b in bouts.itertuples(index=False):
        r, bl = b.red_fighter, b.blue_fighter
        priors = method_priors(method_counts)

        scores: list[float] = []
        q_values: dict[str, float] = {}

        # Freeze all class signals before any update from this bout.
        for side, winner, loser in (("R", r, bl), ("B", bl, r)):
            for meth in METHODS:
                so = offense[meth][winner]
                sd = defense[meth][loser]
                score, q = class_signal(so, sd, b.date, priors[meth])
                scores.append(score)
                q_values[f"q_{side.lower()}_{meth.lower()}"] = q

        # scores were appended R KO/SUB/DEC then B KO/SUB/DEC.
        p = softmax(scores)
        probs = dict(zip(SIX_COLS, p.tolist()))

        actual_m = method_family(getattr(b, "method", ""))
        actual_six = None
        if b.winner is not None and actual_m is not None:
            actual_six = ("R_" if b.winner == r else "B_") + actual_m

        p_red_win = probs["p_red_ko"] + probs["p_red_sub"] + probs["p_red_dec"]
        p_blue_win = 1.0 - p_red_win

        row = {
            "date": b.date,
            "bout_id": b.bout_id,
            "red_fighter": r,
            "blue_fighter": bl,
            "winner": b.winner,
            "method": getattr(b, "method", ""),
            "p_red_win": p_red_win,
            "p_blue_win": p_blue_win,
            **probs,
            "p_method_ko": probs["p_red_ko"] + probs["p_blue_ko"],
            "p_method_sub": probs["p_red_sub"] + probs["p_blue_sub"],
            "p_method_dec": probs["p_red_dec"] + probs["p_blue_dec"],
            "actual_method": actual_m,
            "actual_six": actual_six,
            "prior_ko": priors["KO"],
            "prior_sub": priors["SUB"],
            "prior_dec": priors["DEC"],
            **q_values,
        }
        rows.append(row)

        if actual_six is None:
            continue

        # Compute every one-vs-rest class update from the same prefight states.
        # Each offense/defense coordinate appears in exactly one orientation per
        # method, so these six binary updates are independent within the bout.
        pending = []
        for side, winner, loser in (("R", r, bl), ("B", bl, r)):
            for meth in METHODS:
                label = f"{side}_{meth}"
                y = 1.0 if label == actual_six else 0.0
                so = offense[meth][winner]
                sd = defense[meth][loser]
                no, nro = update(so.rating, so.rd, sd.rating, sd.rd, y)
                nd, nrd = update(sd.rating, sd.rd, so.rating, so.rd, 1.0 - y)
                pending.append((so, no, nro, sd, nd, nrd))

        for so, no, nro, sd, nd, nrd in pending:
            so.rating, so.rd, so.last_date = no, nro, b.date
            sd.rating, sd.rd, sd.last_date = nd, nrd, b.date

        method_counts[actual_m] += 1.0

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/master/ufc_master.parquet"))
    ap.add_argument("--holdout-from", default="2025-01-01")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/diagnostics/standalone_glicko_six_way"),
    )
    args = ap.parse_args()

    bouts = build_bouts(pd.read_parquet(args.input))
    pred = run(bouts)
    cutoff = pd.Timestamp(args.holdout_from)
    train = pred[pred.date < cutoff].copy()
    hold = pred[pred.date >= cutoff].copy()

    summary = {
        "architecture": (
            "direct joint Glicko-6: method-specific offense/defense Glicko "
            "coordinates + online method intercept + single six-class softmax"
        ),
        "hierarchical_winner_x_method": False,
        "no_brain": True,
        "no_fsr": True,
        "no_market": True,
        "holdout_from": args.holdout_from,
        "train_side": side_metrics(train),
        "holdout_side": side_metrics(hold),
        "train_method": method_metrics(train),
        "holdout_method": method_metrics(hold),
        "train_six": six_metrics(train),
        "holdout_six": six_metrics(hold),
        "train_method_shares": share_metrics(train),
        "holdout_method_shares": share_metrics(hold),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.output_dir / "fight_predictions.csv", index=False)
    hold.to_csv(args.output_dir / "holdout_predictions.csv", index=False)
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
