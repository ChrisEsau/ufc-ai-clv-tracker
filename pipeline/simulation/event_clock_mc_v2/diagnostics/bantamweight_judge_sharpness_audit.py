"""Held-out audit of Event Clock decision-judge probability sharpness.

Measurement only. Rebuilds the frozen total-fight judge training cohort, evaluates
on the existing fresh Stage-10 decision cohort, and applies post-hoc logit scaling
without refitting or changing ranking. Reports ALL, MEN, and bantamweight slices.

Purpose: determine whether the decision_sharp2 improvement seen in the
bantamweight market decomposition reflects a genuinely under-sharp historical
judge model or merely compensates for simulator-specific differential errors.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from pipeline.simulation.event_clock_mc_v1.prototype_stage3_correlation import prepare_direct_predictions
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage10d_total_fight_judge import (
    MASTER, STAGE10, VARIANTS, decision_mask, prepare_master, fit_model,
)

SCALES = (1.0, 1.5, 2.0, 2.5, 3.0)
FEATURES = VARIANTS["FULL_TOTAL"]


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def _normalize_division(v):
    s = str(v).strip().lower().replace("'", "").replace("’", "")
    s = " ".join(s.split())
    return s


def _is_womens_division(v):
    s = _normalize_division(v)
    return "women" in s or "womens" in s


def _is_bantam(v):
    s = _normalize_division(v)
    return ("bantam" in s) and (not _is_womens_division(s))


def _metrics(frame, p, scale, subset):
    if frame.empty:
        return {"subset": subset, "scale": scale, "fights": 0}
    y = frame["red_win"].astype(int).to_numpy()
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    return {
        "subset": subset,
        "scale": float(scale),
        "fights": int(len(frame)),
        "accuracy": float(accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "mean_confidence": float(np.maximum(p, 1-p).mean()),
        "actual_accuracy": float((pred == y).mean()),
        "mean_winner_probability": float(np.where(y == 1, p, 1-p).mean()),
    }


def _confidence_bins(frame, p, scale, subset):
    if frame.empty:
        return pd.DataFrame()
    y = frame["red_win"].astype(int).to_numpy()
    pred = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1-p)
    z = pd.DataFrame({"confidence": conf, "correct": (pred == y).astype(int)})
    z["bin"] = pd.cut(z["confidence"], [0.5,0.6,0.7,0.8,0.9,1.000001], labels=["50-60","60-70","70-80","80-90","90-100"], right=False)
    out = z.groupby("bin", observed=True).agg(fights=("correct","size"), mean_confidence=("confidence","mean"), actual_accuracy=("correct","mean")).reset_index()
    out["calibration_gap_pp"] = 100.0 * (out["mean_confidence"] - out["actual_accuracy"])
    out.insert(0, "scale", float(scale))
    out.insert(0, "subset", subset)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    train_direct, _ = prepare_direct_predictions()
    train_ids = set(train_direct["fight_id"].astype(str).unique())

    master = pd.read_parquet(MASTER)
    master = prepare_master(master)
    master["fight_id"] = master["fight_id"].astype(str)

    stage10 = pd.read_csv(STAGE10, low_memory=False)
    stage10["fight_id"] = stage10["fight_id"].astype(str)
    fresh_ids = set(stage10["fight_id"])

    train = master[master["fight_id"].isin(train_ids) & decision_mask(master) & master["red_win"].notna()].copy()
    test = master[master["fight_id"].isin(fresh_ids) & decision_mask(master) & master["red_win"].notna()].copy()

    order = {fid: i for i, fid in enumerate(stage10["fight_id"])}
    test["_order"] = test["fight_id"].map(order)
    test = test.sort_values("_order").reset_index(drop=True)

    model = fit_model(train, FEATURES)
    base_p = model.predict_proba(test[FEATURES])[:, 1]
    base_logit = _logit(base_p)

    div_col = next((c for c in ("weight_class", "division", "weight_class_name") if c in test.columns), None)
    if div_col is None:
        test["_audit_division"] = "unknown"
    else:
        test["_audit_division"] = test[div_col].astype(str)

    masks = {
        "all_decisions": np.ones(len(test), dtype=bool),
        "men_decisions": ~test["_audit_division"].map(_is_womens_division).to_numpy(),
        "mens_bantamweight": test["_audit_division"].map(_is_bantam).to_numpy(),
    }

    rows = []
    bins = []
    fight_rows = []
    for scale in SCALES:
        p_all = _sigmoid(base_logit * scale)
        for subset, mask in masks.items():
            f = test.loc[mask].copy()
            p = p_all[mask]
            rows.append(_metrics(f, p, scale, subset))
            b = _confidence_bins(f, p, scale, subset)
            if not b.empty:
                bins.append(b)
        fr = test[["fight_id", "r_name", "b_name", "red_win", "_audit_division"] + FEATURES].copy()
        fr["scale"] = scale
        fr["p_red"] = p_all
        fr["winner_probability"] = np.where(fr["red_win"].astype(int).eq(1), fr["p_red"], 1-fr["p_red"])
        fight_rows.append(fr)

    metrics = pd.DataFrame(rows)
    confidence = pd.concat(bins, ignore_index=True) if bins else pd.DataFrame()
    fights = pd.concat(fight_rows, ignore_index=True)

    # Coefficients in standardized feature space for audit context.
    lr = model.named_steps["logistic"]
    coef = pd.DataFrame({"feature": FEATURES, "standardized_coefficient": lr.coef_[0]})
    coef["abs_coefficient"] = coef["standardized_coefficient"].abs()
    coef = coef.sort_values("abs_coefficient", ascending=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir / "judge_sharpness_metrics.csv", index=False)
    confidence.to_csv(args.out_dir / "judge_confidence_bins.csv", index=False)
    fights.to_csv(args.out_dir / "judge_sharpness_fight_level.csv", index=False)
    coef.to_csv(args.out_dir / "judge_coefficients.csv", index=False)

    print("EVENT CLOCK V2 — HELD-OUT DECISION JUDGE SHARPNESS AUDIT")
    print(f"training decisions: {len(train)} | fresh decisions: {len(test)}")
    print(f"division column: {div_col}")
    print("\nMETRICS")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nSTANDARDIZED COEFFICIENTS")
    print(coef.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    print("\nCONFIDENCE BINS")
    print(confidence.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
