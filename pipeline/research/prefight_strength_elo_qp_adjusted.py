#!/usr/bin/env python3
"""Standalone Step 3B: calibrate Elo with Quality Performance features.

Research-only. This script does not touch Brain, FSR, market inputs, or production
mechanics. It consumes the leakage-safe Quality Performance fight table, fits a
small logistic adjustment on fights strictly before a training cutoff, freezes
those coefficients, and evaluates untouched fights on/after the cutoff.

Model (red fighter perspective):
  logit(P(red wins)) = b0
                      + b1 * Elo logit
                      + b2 * QP-last5 edge
                      + b3 * QP-recency edge
                      + b4 * last5 residual edge

QP-recency score = exp(-days_since_qp / 365); missing history => 0.
The goal is to test whether QP adds incremental signal *conditional on Elo*.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _fit_logistic_irls(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, max_iter: int = 100) -> np.ndarray:
    """Small deterministic logistic regression via Newton/IRLS."""
    beta = np.zeros(X.shape[1], dtype=float)
    penalty = np.eye(X.shape[1]) * l2
    penalty[0, 0] = 0.0  # do not regularize intercept
    for _ in range(max_iter):
        eta = X @ beta
        p = _sigmoid(eta)
        w = np.clip(p * (1.0 - p), 1e-8, None)
        grad = X.T @ (y - p) - penalty @ beta
        h = X.T @ (X * w[:, None]) + penalty
        step = np.linalg.solve(h, grad)
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < 1e-9:
            beta = beta_new
            break
        beta = beta_new
    return beta


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    pick = p > 0.5
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(pick == (y > 0.5))),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
    }


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out[out["winner"].notna()].copy()
    out["red_win"] = (out["winner"] == out["red_fighter"]).astype(float)

    out["elo_logit"] = _logit(out["red_elo_win_prob"].astype(float).to_numpy())
    out["qp5_edge"] = out["red_qp_rate_last5"] - out["blue_qp_rate_last5"]

    rdays = out["red_days_since_qp"].astype(float)
    bdays = out["blue_days_since_qp"].astype(float)
    out["red_qp_recency"] = np.where(rdays.notna(), np.exp(-rdays.fillna(0.0) / 365.0), 0.0)
    out["blue_qp_recency"] = np.where(bdays.notna(), np.exp(-bdays.fillna(0.0) / 365.0), 0.0)
    out["qp_recency_edge"] = out["red_qp_recency"] - out["blue_qp_recency"]
    out["residual5_edge"] = out["red_last5_residual"] - out["blue_last5_residual"]

    # Require both fighters to have at least three prior UFC fights and complete
    # QP/residual history. This keeps the adjustment an established-fighter layer.
    eligible = (
        (out["red_prior_fights"] >= 3)
        & (out["blue_prior_fights"] >= 3)
        & out[["elo_logit", "qp5_edge", "qp_recency_edge", "residual5_edge"]].notna().all(axis=1)
    )
    return out.loc[eligible].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("data/diagnostics/prefight_strength_elo_qp"))
    ap.add_argument("--train-before", default="2025-01-01")
    ap.add_argument("--l2", type=float, default=1e-3)
    args = ap.parse_args()

    raw = pd.read_csv(args.quality_csv)
    df = _prepare(raw)
    cutoff = pd.Timestamp(args.train_before)
    train = df[df["date"] < cutoff].copy()
    holdout = df[df["date"] >= cutoff].copy()

    features = ["elo_logit", "qp5_edge", "qp_recency_edge", "residual5_edge"]
    X_train = np.column_stack([np.ones(len(train)), train[features].to_numpy(float)])
    y_train = train["red_win"].to_numpy(float)
    beta = _fit_logistic_irls(X_train, y_train, l2=args.l2)

    X_all = np.column_stack([np.ones(len(df)), df[features].to_numpy(float)])
    df["elo_qp_adjusted_red_prob"] = _sigmoid(X_all @ beta)
    df["elo_qp_adjusted_blue_prob"] = 1.0 - df["elo_qp_adjusted_red_prob"]
    df["elo_qp_pick"] = np.where(
        df["elo_qp_adjusted_red_prob"] > 0.5, df["red_fighter"],
        np.where(df["elo_qp_adjusted_red_prob"] < 0.5, df["blue_fighter"], None)
    )
    df["elo_qp_pick_correct"] = df["elo_qp_pick"] == df["winner"]

    # Compare adjusted model with raw Elo on the exact same eligible rows.
    train_mask = df["date"] < cutoff
    hold_mask = ~train_mask
    y_all = df["red_win"].to_numpy(float)
    elo_p = df["red_elo_win_prob"].to_numpy(float)
    adj_p = df["elo_qp_adjusted_red_prob"].to_numpy(float)

    names = ["intercept"] + features
    coefficients = {name: float(value) for name, value in zip(names, beta)}
    summary = {
        "source": str(args.quality_csv),
        "train_before": str(cutoff.date()),
        "eligible_fights": int(len(df)),
        "train_fights": int(train_mask.sum()),
        "holdout_fights": int(hold_mask.sum()),
        "features": features,
        "coefficients": coefficients,
        "train_raw_elo": _metrics(y_all[train_mask], elo_p[train_mask]),
        "train_elo_qp_adjusted": _metrics(y_all[train_mask], adj_p[train_mask]),
        "holdout_raw_elo": _metrics(y_all[hold_mask], elo_p[hold_mask]),
        "holdout_elo_qp_adjusted": _metrics(y_all[hold_mask], adj_p[hold_mask]),
        "leakage_rule": "coefficients fit only on fights before cutoff; holdout probabilities use frozen coefficients",
        "scope": "research-only established-fighter adjustment; no Brain/FSR/market dependency",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "fight_elo_qp_adjusted.csv", index=False)
    (args.output_dir / "coefficients.json").write_text(json.dumps(coefficients, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
