#!/usr/bin/env python3
"""Standalone hybrid direct six-way Glicko UFC benchmark.

Research-only. No Brain, FSR, or market inputs.

This ablation preserves the validated FightMatrix-style graded winner Glicko
as a side-strength signal, while retaining method-specific offense/defense
Glicko coordinates. All six outcomes still compete in one joint softmax:

    red KO/TKO, red SUB, red DEC,
    blue KO/TKO, blue SUB, blue DEC.

For each class, the score is:

    log(prefight side win probability)
    + log(prefight method prior)
    + method offense-vs-defense Glicko log-odds

This is not P(win) x normalized P(method|win): method matchup partition sums
are not normalized within side before the six-way softmax. The graded winner
Glicko simply contributes a shared side-strength term to each of that side's
three class logits.

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
    outcome_scores,
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


def method_signal(
    offense: State,
    defense: State,
    now: pd.Timestamp,
) -> tuple[float, float]:
    """Return method matchup log-odds and underlying Glicko probability."""
    inflate_rd(offense, now)
    inflate_rd(defense, now)
    q = expected(offense.rating, defense.rating, defense.rd)
    return logit(q), q


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
    winner_states = defaultdict(State)
    offense = {m: defaultdict(State) for m in METHODS}
    defense = {m: defaultdict(State) for m in METHODS}
    method_counts = {m: 1.0 for m in METHODS}

    rows: list[dict] = []

    for b in bouts.itertuples(index=False):
        r, bl = b.red_fighter, b.blue_fighter
        priors = method_priors(method_counts)

        # Validated graded winner Glicko, frozen prefight.
        sr, sb = winner_states[r], winner_states[bl]
        inflate_rd(sr, b.date)
        inflate_rd(sb, b.date)
        p_r_base = expected(sr.rating, sb.rating, sb.rd)
        p_b_base = 1.0 - p_r_base

        scores: list[float] = []
        q_values: dict[str, float] = {}

        # Shared side term + method-specific matchup term. No within-side
        # normalization occurs before the single six-way softmax.
        for side, winner, loser, side_p in (
            ("R", r, bl, p_r_base),
            ("B", bl, r, p_b_base),
        ):
            for meth in METHODS:
                so = offense[meth][winner]
                sd = defense[meth][loser]
                matchup_logodds, q = method_signal(so, sd, b.date)
                score = (
                    math.log(max(EPS, side_p))
                    + math.log(max(EPS, priors[meth]))
                    + matchup_logodds
                )
                scores.append(score)
                q_values[f"q_{side.lower()}_{meth.lower()}"] = q

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
            "p_red_win_base_glicko": p_r_base,
            "p_blue_win_base_glicko": p_b_base,
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

        if b.winner is None:
            continue

        # Preserve exact graded winner update used by the validated benchmark.
        rs, bs = outcome_scores(getattr(b, "method", ""), b.winner == r)
        nr, nrr = update(sr.rating, sr.rd, sb.rating, sb.rd, rs)
        nb, nbr = update(sb.rating, sb.rd, sr.rating, sr.rd, bs)
        sr.rating, sr.rd, sr.last_date = nr, nrr, b.date
        sb.rating, sb.rd, sb.last_date = nb, nbr, b.date

        if actual_six is None:
            continue

        # Method offense/defense one-vs-rest updates from frozen prefight states.
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
            "hybrid direct joint Glicko-6: validated graded winner side-strength "
            "+ method offense/defense Glicko + online method intercept + single "
            "six-class softmax"
        ),
        "hierarchical_winner_x_method": False,
        "winner_signal": "validated FightMatrix-style graded Glicko",
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
