#!/usr/bin/env python3
"""Canonical leakage-safe Whole-History Rating (WHR) research benchmark.

Research only. This module does not alter Brain/FSR/production mechanics.

Canonical model
---------------
Latent fighter strength r_i(t) is represented on the natural-log odds scale.
For a red/blue matchup at a rating state:

    P(red wins) = sigmoid(r_red - r_blue)

For consecutive fighter states separated by ``dt_days``:

    r_i(t+1) - r_i(t) ~ Normal(0, w^2 * dt_days)

The previously prediction-tuned setting was ``w = 2.75`` in the repo's Elo-like
rating units. Because this module represents r on the natural-log odds scale,
that drift must be transformed by ln(10)/400 before entering the temporal prior.

Leakage rule
------------
Every prediction on date d is fit using only fights with date < d. All bouts on
the same event/date are therefore predicted simultaneously and cannot leak into
one another. The target bout is never included in its own prefight fit.

Initial-state regularization
----------------------------
Fight likelihoods plus temporal differences alone are translation-invariant and
can also be unbounded for separated early histories. Following Coulom's original
WHR construction, each fighter's first state receives one virtual win and one
virtual loss against rating 0. This yields a finite, symmetric starter prior
centered at r=0 without changing the matchup likelihood or temporal equations.
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

ELO_TO_LOG_ODDS = math.log(10.0) / 400.0
W_SOURCE_ELO_DEFAULT = 2.75
W_DEFAULT = W_SOURCE_ELO_DEFAULT * ELO_TO_LOG_ODDS
W2_DEFAULT = W_DEFAULT * W_DEFAULT
INITIAL_PRIOR_WINS_DEFAULT = 1.0


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr)
    pos = z_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z_arr[pos]))
    ez = np.exp(z_arr[~pos])
    out[~pos] = ez / (1.0 + ez)
    if np.ndim(z) == 0:
        return float(out)
    return out


def _softplus(z: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, z)


def _prepare_games(bouts: pd.DataFrame) -> pd.DataFrame:
    """Normalize decisive bouts into binary red outcomes."""
    rows: list[dict[str, Any]] = []
    for b in bouts.itertuples(index=False):
        if pd.isna(b.date) or b.winner is None or pd.isna(b.winner):
            continue
        red = str(b.red_fighter)
        blue = str(b.blue_fighter)
        winner = str(b.winner)
        if winner not in {red, blue}:
            continue
        rows.append(
            {
                "date": pd.Timestamp(b.date).normalize(),
                "bout_id": str(b.bout_id),
                "red_fighter": red,
                "blue_fighter": blue,
                "winner": winner,
                "method": getattr(b, "method", ""),
                "red_win": 1.0 if winner == red else 0.0,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["date", "bout_id", "red_fighter", "blue_fighter", "winner", "method", "red_win"]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["date", "bout_id"], kind="stable")
        .reset_index(drop=True)
    )


def _build_problem(hist: pd.DataFrame, w2: float):
    if w2 <= 0:
        raise ValueError("w2 must be > 0")

    node_index: dict[tuple[str, pd.Timestamp], int] = {}
    fighter_nodes: dict[str, list[tuple[pd.Timestamp, int]]] = {}

    def node(fighter: str, date: pd.Timestamp) -> int:
        key = (fighter, date)
        if key not in node_index:
            idx = len(node_index)
            node_index[key] = idx
            fighter_nodes.setdefault(fighter, []).append((date, idx))
        return node_index[key]

    red_idx: list[int] = []
    blue_idx: list[int] = []
    outcomes: list[float] = []
    for r in hist.itertuples(index=False):
        red_idx.append(node(r.red_fighter, r.date))
        blue_idx.append(node(r.blue_fighter, r.date))
        outcomes.append(float(r.red_win))

    transitions: list[tuple[int, int, float]] = []
    first_nodes: list[int] = []
    for fighter in fighter_nodes:
        fighter_nodes[fighter].sort(key=lambda x: x[0])
        nodes = fighter_nodes[fighter]
        first_nodes.append(nodes[0][1])
        for (d0, i0), (d1, i1) in zip(nodes[:-1], nodes[1:]):
            dt_days = max(1.0, float((d1 - d0).days))
            transitions.append((i0, i1, w2 * dt_days))

    return (
        node_index,
        fighter_nodes,
        np.asarray(red_idx, dtype=np.int32),
        np.asarray(blue_idx, dtype=np.int32),
        np.asarray(outcomes, dtype=float),
        transitions,
        np.asarray(first_nodes, dtype=np.int32),
    )


def _fit(
    hist: pd.DataFrame,
    *,
    w2: float = W2_DEFAULT,
    initial_prior_wins: float = INITIAL_PRIOR_WINS_DEFAULT,
):
    if initial_prior_wins <= 0:
        raise ValueError("initial_prior_wins must be > 0")

    node_index, fighter_nodes, ri, bi, y, transitions, first_nodes = _build_problem(hist, w2)
    n = len(node_index)
    if n == 0:
        return node_index, fighter_nodes, np.zeros(0, dtype=float), None

    trans_i0 = np.asarray([t[0] for t in transitions], dtype=np.int32)
    trans_i1 = np.asarray([t[1] for t in transitions], dtype=np.int32)
    trans_var = np.asarray([t[2] for t in transitions], dtype=float)

    def fun_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        logits = x[ri] - x[bi]
        nll = float(np.sum(_softplus(logits) - y * logits))
        p = _sigmoid(logits)
        d = p - y
        g = np.zeros_like(x)
        np.add.at(g, ri, d)
        np.add.at(g, bi, -d)

        if len(trans_i0):
            diff = x[trans_i1] - x[trans_i0]
            nll += float(0.5 * np.sum((diff * diff) / trans_var))
            td = diff / trans_var
            np.add.at(g, trans_i1, td)
            np.add.at(g, trans_i0, -td)

        first_r = x[first_nodes]
        nll += float(initial_prior_wins * np.sum(_softplus(-first_r) + _softplus(first_r)))
        prior_grad = initial_prior_wins * (2.0 * _sigmoid(first_r) - 1.0)
        np.add.at(g, first_nodes, prior_grad)

        return nll, g

    res = minimize(
        lambda x: fun_grad(x)[0],
        np.zeros(n, dtype=float),
        jac=lambda x: fun_grad(x)[1],
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 250, "maxls": 40},
    )
    return node_index, fighter_nodes, res.x, res


def _latest_rating(fighter_nodes, x: np.ndarray, fighter: str) -> float:
    nodes = fighter_nodes.get(fighter)
    if not nodes:
        return 0.0
    return float(x[nodes[-1][1]])


def _prob(red_rating: float, blue_rating: float) -> float:
    return float(_sigmoid(red_rating - blue_rating))


def _metrics(df: pd.DataFrame, pcol: str) -> dict[str, float | int]:
    d = df[df["winner"].notna() & df[pcol].notna()].copy()
    if d.empty:
        return {"n": 0}
    y = (d["winner"] == d["red_fighter"]).astype(float).to_numpy()
    p = np.clip(d[pcol].to_numpy(float), 1e-12, 1.0 - 1e-12)
    picks = np.where(p > 0.5, d["red_fighter"], np.where(p < 0.5, d["blue_fighter"], ""))
    non_ties = picks != ""
    return {
        "n": int(len(d)),
        "non_ties": int(non_ties.sum()),
        "accuracy_non_ties": (
            float(np.mean(picks[non_ties] == d.loc[non_ties, "winner"].to_numpy()))
            if non_ties.any()
            else float("nan")
        ),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
    }


def causal_predict_games(
    games: pd.DataFrame,
    *,
    holdout_from: str | pd.Timestamp,
    w: float = W_DEFAULT,
    initial_prior_wins: float = INITIAL_PRIOR_WINS_DEFAULT,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate strict-prefight WHR predictions for each holdout event date."""
    if w <= 0:
        raise ValueError("w must be > 0")
    w2 = float(w * w)
    cutoff = pd.Timestamp(holdout_from).normalize()
    holdout = games[games["date"] >= cutoff].copy()
    pred_rows: list[dict[str, Any]] = []

    unique_dates = list(pd.unique(holdout["date"]))
    for k, d in enumerate(unique_dates, 1):
        d = pd.Timestamp(d).normalize()
        hist = games[games["date"] < d]
        _, fighter_nodes, x, res = _fit(hist, w2=w2, initial_prior_wins=initial_prior_wins)
        day = holdout[holdout["date"] == d]
        for r in day.itertuples(index=False):
            rr = _latest_rating(fighter_nodes, x, r.red_fighter)
            br = _latest_rating(fighter_nodes, x, r.blue_fighter)
            delta = rr - br
            p_red = _prob(rr, br)
            pred_rows.append(
                {
                    **r._asdict(),
                    "WHR_red": rr,
                    "WHR_blue": br,
                    "WHR_delta": delta,
                    "WHR_P_red": p_red,
                    "WHR_P_blue": 1.0 - p_red,
                    "WHR_w_log_odds": float(w),
                    "WHR_w2_log_odds": w2,
                    "WHR_source_w_elo": float(w / ELO_TO_LOG_ODDS),
                    "history_bouts": int(len(hist)),
                    "solver_success": bool(res.success) if res is not None else True,
                    "solver_iterations": int(res.nit) if res is not None else 0,
                    "solver_grad_inf": (
                        float(np.max(np.abs(res.jac))) if res is not None and res.jac is not None else 0.0
                    ),
                }
            )
        if verbose:
            print(f"[{k}/{len(unique_dates)}] {d.date()} history={len(hist)} bouts={len(day)}")

    return pd.DataFrame(pred_rows)


def run(source: pd.DataFrame, holdout_from: str, w: float = W_DEFAULT):
    bouts = build_bouts(source)
    games = _prepare_games(bouts)
    preds = causal_predict_games(games, holdout_from=holdout_from, w=w)
    cutoff = pd.Timestamp(holdout_from).normalize()

    summary = {
        "source_rows": int(len(source)),
        "binary_scored_games": int(len(games)),
        "holdout_from": str(cutoff.date()),
        "rating_scale": "natural_log_odds",
        "time_unit": "day",
        "source_w_elo": float(w / ELO_TO_LOG_ODDS),
        "w_log_odds": float(w),
        "w2_log_odds": float(w * w),
        "elo_to_log_odds": ELO_TO_LOG_ODDS,
        "initial_prior": {
            "type": "Coulom symmetric virtual games at first state",
            "virtual_wins_vs_zero": INITIAL_PRIOR_WINS_DEFAULT,
            "virtual_losses_vs_zero": INITIAL_PRIOR_WINS_DEFAULT,
        },
        "leakage_rule": "for date d, fit only bouts with date < d; same-day bouts are simultaneous",
        "outcome_rule": "binary decisive result only; rows without a normalized winner are omitted",
        "mechanics_scope": "research-only shadow rating; no Brain/FSR/market inputs or production mechanic changes",
        "holdout_whr": _metrics(preds, "WHR_P_red") if len(preds) else {"n": 0},
        "solver_failures": int((~preds["solver_success"]).sum()) if len(preds) else 0,
    }
    return preds, summary


def _self_test() -> None:
    dates = pd.to_datetime([
        "2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"
    ])
    games = pd.DataFrame(
        [
            {"date": dates[0], "bout_id": "1", "red_fighter": "A", "blue_fighter": "B", "winner": "A", "method": "", "red_win": 1.0},
            {"date": dates[1], "bout_id": "2", "red_fighter": "A", "blue_fighter": "C", "winner": "A", "method": "", "red_win": 1.0},
            {"date": dates[2], "bout_id": "3", "red_fighter": "B", "blue_fighter": "C", "winner": "B", "method": "", "red_win": 1.0},
            {"date": dates[3], "bout_id": "4", "red_fighter": "A", "blue_fighter": "B", "winner": "B", "method": "", "red_win": 0.0},
            {"date": dates[4], "bout_id": "5", "red_fighter": "C", "blue_fighter": "A", "winner": "C", "method": "", "red_win": 1.0},
        ]
    )
    pred = causal_predict_games(games, holdout_from="2024-04-01", verbose=False)
    assert len(pred) == 2
    assert np.allclose(pred["WHR_P_red"] + pred["WHR_P_blue"], 1.0, atol=1e-12)
    assert np.allclose(
        pred["WHR_P_red"].to_numpy(),
        _sigmoid(pred["WHR_delta"].to_numpy()),
        atol=1e-12,
    )

    mutated = games.copy()
    target = mutated["bout_id"] == "4"
    mutated.loc[target, "winner"] = "A"
    mutated.loc[target, "red_win"] = 1.0
    pred_mut = causal_predict_games(mutated, holdout_from="2024-04-01", verbose=False)
    p0 = pred.loc[pred["bout_id"] == "4", "WHR_P_red"].iloc[0]
    p1 = pred_mut.loc[pred_mut["bout_id"] == "4", "WHR_P_red"].iloc[0]
    assert abs(p0 - p1) < 1e-12, (p0, p1)

    extra = pd.concat(
        [
            games,
            pd.DataFrame([
                {"date": dates[3], "bout_id": "4b", "red_fighter": "C", "blue_fighter": "B", "winner": "C", "method": "", "red_win": 1.0}
            ]),
        ],
        ignore_index=True,
    )
    pred_extra = causal_predict_games(extra, holdout_from="2024-04-01", verbose=False)
    p2 = pred_extra.loc[pred_extra["bout_id"] == "4", "WHR_P_red"].iloc[0]
    assert abs(p0 - p2) < 1e-12, (p0, p2)

    cold = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-01"), "bout_id": "cold", "red_fighter": "X", "blue_fighter": "Y", "winner": "X", "method": "", "red_win": 1.0}
    ])
    cold_pred = causal_predict_games(cold, holdout_from="2024-01-01", verbose=False)
    assert abs(float(cold_pred.iloc[0]["WHR_P_red"]) - 0.5) < 1e-12

    print("PASS — canonical WHR equations, probability identity, cold start, target leakage, and same-date leakage")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/master/ufc_master.parquet"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/diagnostics/prefight_strength_whr_causal"))
    ap.add_argument("--holdout-from", default="2025-01-01")
    ap.add_argument(
        "--w",
        type=float,
        default=W_SOURCE_ELO_DEFAULT,
        help="Prediction-tuned drift in legacy Elo-like units; converted to natural-log odds internally.",
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return

    source = pd.read_parquet(args.input)
    w_log_odds = float(args.w) * ELO_TO_LOG_ODDS
    preds, summary = run(source, args.holdout_from, w_log_odds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.output_dir / "fight_whr_holdout.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
