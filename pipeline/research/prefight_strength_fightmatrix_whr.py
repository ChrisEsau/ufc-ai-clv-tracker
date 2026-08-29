#!/usr/bin/env python3
"""Leakage-safe FightMatrix-style Whole-History Rating benchmark.

Research only. Uses the UFC master parquet and no Brain/FSR/market inputs.

This implements the published FightMatrix WHR ingredients that are available in
our UFC-only data:
- latent rating starts at 0 Elo-like points
- dynamic Bradley-Terry likelihood
- w^2 = 42 Elo^2/day random-walk prior
- iterate/optimize to convergence
- split decision: 0.55 / 0.45
- majority decision: 0.61 / 0.39
- unanimous decision: 0.91 / 0.09
- other wins: 1 / 0
- draws: 0.5 / 0.5
- DQ / no-contest ignored

Important: for leakage-safe predictive evaluation, each holdout event date is
predicted from a WHR fit using only bouts strictly before that date. Same-day
bouts are therefore simultaneous and cannot leak into one another.

The solver is a sparse-in-spirit MAP implementation using scipy L-BFGS-B over
one latent rating per fighter/date node. A weak zero-centered anchor is applied
to each fighter's first node to make the posterior identifiable. This is an
approximation of FightMatrix's implementation, not their proprietary code.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pipeline.research.prefight_strength_elo import build_bouts

LOG10_OVER_400 = math.log(10.0) / 400.0
W2_DEFAULT = 42.0
ANCHOR_SD = 350.0


def _method_score(method: Any, winner_is_red: bool) -> tuple[float, bool]:
    m = str(method or "").strip().lower()
    if "no contest" in m or m in {"nc", "no_contest"} or "disqualification" in m or m == "dq":
        return 0.5, False
    if "split" in m:
        w, l = 0.55, 0.45
    elif "majority" in m:
        w, l = 0.61, 0.39
    elif "unanimous" in m or "decision" in m:
        w, l = 0.91, 0.09
    else:
        w, l = 1.0, 0.0
    return (w if winner_is_red else l), True


def _prepare_games(bouts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in bouts.itertuples(index=False):
        if pd.isna(b.date):
            continue
        if b.winner is None:
            # unresolved fights are skipped; explicit draws are not reliably
            # distinguishable in the normalized winner field of this source.
            continue
        method = getattr(b, "method", "")
        s, valid = _method_score(method, b.winner == b.red_fighter)
        if not valid:
            continue
        rows.append({
            "date": pd.Timestamp(b.date).normalize(),
            "bout_id": str(b.bout_id),
            "red_fighter": b.red_fighter,
            "blue_fighter": b.blue_fighter,
            "winner": b.winner,
            "method": method,
            "red_score": float(s),
        })
    return pd.DataFrame(rows).sort_values(["date", "bout_id"], kind="stable").reset_index(drop=True)


def _build_problem(hist: pd.DataFrame, w2: float):
    # one node per fighter/date on which they fought
    node_index: dict[tuple[str, pd.Timestamp], int] = {}
    fighter_nodes: dict[str, list[tuple[pd.Timestamp, int]]] = {}

    def node(f: str, d: pd.Timestamp) -> int:
        k = (f, d)
        if k not in node_index:
            idx = len(node_index)
            node_index[k] = idx
            fighter_nodes.setdefault(f, []).append((d, idx))
        return node_index[k]

    red_idx, blue_idx, scores = [], [], []
    for r in hist.itertuples(index=False):
        red_idx.append(node(r.red_fighter, r.date))
        blue_idx.append(node(r.blue_fighter, r.date))
        scores.append(float(r.red_score))

    for f in fighter_nodes:
        fighter_nodes[f].sort(key=lambda x: x[0])

    transitions: list[tuple[int, int, float]] = []
    anchors: list[int] = []
    for nodes in fighter_nodes.values():
        if not nodes:
            continue
        anchors.append(nodes[0][1])
        for (d0, i0), (d1, i1) in zip(nodes[:-1], nodes[1:]):
            days = max(1.0, float((d1 - d0).days))
            transitions.append((i0, i1, w2 * days))

    return (
        node_index,
        fighter_nodes,
        np.asarray(red_idx, dtype=np.int32),
        np.asarray(blue_idx, dtype=np.int32),
        np.asarray(scores, dtype=float),
        transitions,
        np.asarray(anchors, dtype=np.int32),
    )


def _fit(hist: pd.DataFrame, w2: float = W2_DEFAULT, x0: np.ndarray | None = None):
    node_index, fighter_nodes, ri, bi, s, transitions, anchors = _build_problem(hist, w2)
    n = len(node_index)
    if n == 0:
        return node_index, fighter_nodes, np.zeros(0), None

    if x0 is None or len(x0) != n:
        x0 = np.zeros(n, dtype=float)

    trans_i0 = np.asarray([t[0] for t in transitions], dtype=np.int32)
    trans_i1 = np.asarray([t[1] for t in transitions], dtype=np.int32)
    trans_var = np.asarray([t[2] for t in transitions], dtype=float)

    def fun_grad(x: np.ndarray):
        z = (x[ri] - x[bi]) * LOG10_OVER_400
        z = np.clip(z, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-12
        nll = -np.sum(s * np.log(p + eps) + (1.0 - s) * np.log(1.0 - p + eps))
        g = np.zeros_like(x)
        # derivative of NLL wrt Elo-like rating
        d = (p - s) * LOG10_OVER_400
        np.add.at(g, ri, d)
        np.add.at(g, bi, -d)

        if len(trans_i0):
            diff = x[trans_i1] - x[trans_i0]
            nll += 0.5 * np.sum((diff * diff) / trans_var)
            td = diff / trans_var
            np.add.at(g, trans_i1, td)
            np.add.at(g, trans_i0, -td)

        if len(anchors):
            avar = ANCHOR_SD * ANCHOR_SD
            vals = x[anchors]
            nll += 0.5 * np.sum((vals * vals) / avar)
            np.add.at(g, anchors, vals / avar)

        return float(nll), g

    res = minimize(
        lambda x: fun_grad(x)[0],
        x0,
        jac=lambda x: fun_grad(x)[1],
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-5, "maxiter": 120, "maxls": 30},
    )
    return node_index, fighter_nodes, res.x, res


def _latest_rating(fighter_nodes, x, fighter: str) -> float:
    nodes = fighter_nodes.get(fighter)
    if not nodes:
        return 0.0
    return float(x[nodes[-1][1]])


def _prob(red_rating: float, blue_rating: float) -> float:
    z = np.clip((red_rating - blue_rating) * LOG10_OVER_400, -35.0, 35.0)
    return float(1.0 / (1.0 + np.exp(-z)))


def _metrics(df: pd.DataFrame, pcol: str) -> dict[str, float | int]:
    d = df[df["winner"].notna() & df[pcol].notna()].copy()
    y = (d["winner"] == d["red_fighter"]).astype(float).to_numpy()
    p = np.clip(d[pcol].to_numpy(float), 1e-9, 1 - 1e-9)
    picks = np.where(p > 0.5, d["red_fighter"], np.where(p < 0.5, d["blue_fighter"], ""))
    non = picks != ""
    acc = float(np.mean(picks[non] == d.loc[non, "winner"].to_numpy())) if non.any() else float("nan")
    return {
        "n": int(len(d)),
        "non_ties": int(non.sum()),
        "accuracy_non_ties": acc,
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
    }


def run(source: pd.DataFrame, holdout_from: str, w2: float):
    bouts = build_bouts(source)
    games = _prepare_games(bouts)
    cutoff = pd.Timestamp(holdout_from)

    # Retrospective pre-cutoff fit, used only for a descriptive training metric.
    train = games[games["date"] < cutoff].copy()
    ni, fn, x, res = _fit(train, w2=w2)
    train_rows = []
    # To avoid pretending retrospective smoothed ratings are strict predictions,
    # training metrics are omitted from model selection and reported as descriptive.

    holdout = games[games["date"] >= cutoff].copy()
    pred_rows = []
    unique_dates = list(pd.unique(holdout["date"]))
    for k, d in enumerate(unique_dates, 1):
        hist = games[games["date"] < d]
        ni_d, fn_d, x_d, res_d = _fit(hist, w2=w2)
        day = holdout[holdout["date"] == d]
        for r in day.itertuples(index=False):
            rr = _latest_rating(fn_d, x_d, r.red_fighter)
            br = _latest_rating(fn_d, x_d, r.blue_fighter)
            pred_rows.append({
                **r._asdict(),
                "whr_red_pre_rating": rr,
                "whr_blue_pre_rating": br,
                "whr_red_win_prob": _prob(rr, br),
                "history_bouts": int(len(hist)),
                "solver_success": bool(res_d.success) if res_d is not None else True,
                "solver_iterations": int(res_d.nit) if res_d is not None else 0,
            })
        print(f"[{k}/{len(unique_dates)}] {pd.Timestamp(d).date()} history={len(hist)} bouts={len(day)}")

    preds = pd.DataFrame(pred_rows)
    summary = {
        "source_rows": int(len(source)),
        "scored_games": int(len(games)),
        "holdout_from": str(cutoff.date()),
        "w2": float(w2),
        "convergence_target_gradient": 1e-5,
        "published_style_rules": {
            "starter_rating": 0.0,
            "w2": 42.0,
            "split": [0.55, 0.45],
            "majority": [0.61, 0.39],
            "unanimous": [0.91, 0.09],
            "other_win": [1.0, 0.0],
        },
        "implementation_scope": "UFC-only leakage-safe FightMatrix-style WHR approximation; each holdout event date fit on strictly prior bouts only; no Brain/FSR/market dependency",
        "holdout_whr": _metrics(preds, "whr_red_win_prob") if len(preds) else {},
    }
    return preds, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/master/ufc_master.parquet"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/diagnostics/prefight_strength_fightmatrix_whr"))
    ap.add_argument("--holdout-from", default="2025-01-01")
    ap.add_argument("--w2", type=float, default=W2_DEFAULT)
    args = ap.parse_args()

    src = pd.read_parquet(args.input)
    preds, summary = run(src, args.holdout_from, args.w2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.output_dir / "fight_whr_holdout.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
