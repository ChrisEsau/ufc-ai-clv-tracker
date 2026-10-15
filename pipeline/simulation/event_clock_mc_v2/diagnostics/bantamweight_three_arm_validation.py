"""High-path men's bantamweight validation for the three shortlisted lethality arms.

Arms:
  - i10_b0: fresh-power intercept 10, KD sequence disabled
  - i5_b2: fresh-power intercept 5, KD sequence +2
  - i10_b1: fresh-power intercept 10, KD sequence +1

Uses the same frozen cohort/mechanics as the joint screen, but increases paths and
adds ML AUC/calibration plus method confusion/probability calibration outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline.simulation.event_clock_mc_v2.diagnostics import flyweight_joint_tuning_screen as base
from pipeline.simulation.event_clock_mc_v2.diagnostics import weight_class_audit as wc_audit

DIVISION = "bantamweight"
ARMS = [
    ("i10_b0", 10.0, None),
    ("i5_b2", 5.0, 2.0),
    ("i10_b1", 10.0, 1.0),
]


def _calibration_bins(p: np.ndarray, y: np.ndarray, width: float = 0.1) -> pd.DataFrame:
    edges = np.arange(0.0, 1.0 + width, width)
    idx = np.digitize(p, edges[1:-1], right=False)
    rows = []
    for i in range(len(edges) - 1):
        m = idx == i
        if not np.any(m):
            continue
        rows.append({
            "bin_lo": edges[i],
            "bin_hi": edges[i + 1],
            "n": int(m.sum()),
            "mean_pred": float(p[m].mean()),
            "actual_rate": float(y[m].mean()),
        })
    return pd.DataFrame(rows)


def _extra_metrics(summary: pd.DataFrame, arm: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_red = summary["actual_winner"].eq("red").astype(int).to_numpy()
    p_red = summary["p_red_win"].astype(float).to_numpy()
    auc = float(roc_auc_score(y_red, p_red)) if len(np.unique(y_red)) == 2 else np.nan

    ml_bins = _calibration_bins(p_red, y_red)
    ml_bins.insert(0, "arm", arm)

    classes = ["DEC", "KO_TKO", "SUB"]
    prob_cols = {"DEC": "p_fight_dec", "KO_TKO": "p_fight_ko_tko", "SUB": "p_fight_sub"}
    method_bins = []
    for cls in classes:
        p = summary[prob_cols[cls]].astype(float).to_numpy()
        y = summary["actual_method"].eq(cls).astype(int).to_numpy()
        b = _calibration_bins(p, y)
        b.insert(0, "method", cls)
        b.insert(0, "arm", arm)
        method_bins.append(b)
    method_bins_df = pd.concat(method_bins, ignore_index=True)

    pred_method = summary[[prob_cols[c] for c in classes]].to_numpy(float).argmax(axis=1)
    pred_method = np.array(classes, dtype=object)[pred_method]
    confusion = pd.crosstab(
        pd.Categorical(summary["actual_method"], categories=classes),
        pd.Categorical(pred_method, categories=classes),
        rownames=["actual_method"],
        colnames=["predicted_method"],
        dropna=False,
    ).reset_index()
    confusion.insert(0, "arm", arm)

    winner_p = np.where(y_red == 1, p_red, 1.0 - p_red)
    rec = {
        "arm": arm,
        "ml_auc": auc,
        "ml_mean_confidence": float(np.maximum(p_red, 1.0 - p_red).mean()),
        "mean_actual_winner_probability": float(winner_p.mean()),
        "high_conf_wrong_ge_065": int(((np.maximum(p_red, 1.0 - p_red) >= 0.65) & (summary["ml_correct"].astype(int).to_numpy() == 0)).sum()),
        "high_conf_wrong_ge_075": int(((np.maximum(p_red, 1.0 - p_red) >= 0.75) & (summary["ml_correct"].astype(int).to_numpy() == 0)).sum()),
    }
    return rec, ml_bins, method_bins_df, confusion


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-n", type=int, default=100)
    ap.add_argument("--paths", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--out-dir", type=Path, default=Path("data/diagnostics/event_clock_mc_v2/bantamweight_three_arm_validation"))
    args = ap.parse_args()

    base.DIVISION = DIVISION
    cohort, _ = wc_audit.select_cohort(DIVISION, args.target_n)
    cohort = cohort.reset_index(drop=True)
    hist = base._historical(cohort)
    base._install_round_wrapper()

    metrics = []
    summaries = []
    extras = []
    ml_bins_all = []
    method_bins_all = []
    confusion_all = []

    for arm, intercept, bonus in ARMS:
        summary = base._run_arm(cohort, args.paths, args.seed, arm, intercept, bonus)
        summary["arm"] = arm
        summary["intercept"] = intercept
        summary["sequence_bonus"] = 0.0 if bonus is None else bonus
        summaries.append(summary)
        metrics.append(base._metrics(summary, hist, arm, intercept, bonus, args.paths))
        rec, ml_bins, method_bins, confusion = _extra_metrics(summary, arm)
        extras.append(rec)
        ml_bins_all.append(ml_bins)
        method_bins_all.append(method_bins)
        confusion_all.append(confusion)

    m = pd.DataFrame(metrics).merge(pd.DataFrame(extras), on=["arm", "mean_actual_winner_probability"], how="left")
    m = m.sort_values(["ml_brier", "method_brier_multiclass"]).reset_index(drop=True)
    s = pd.concat(summaries, ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.out_dir / "arm_metrics.csv", index=False)
    s.to_csv(args.out_dir / "fight_summaries.csv", index=False)
    pd.concat(ml_bins_all, ignore_index=True).to_csv(args.out_dir / "ml_calibration_bins.csv", index=False)
    pd.concat(method_bins_all, ignore_index=True).to_csv(args.out_dir / "method_calibration_bins.csv", index=False)
    pd.concat(confusion_all, ignore_index=True).to_csv(args.out_dir / "method_confusion.csv", index=False)

    print("\nHISTORICAL BANTAMWEIGHT TARGETS")
    for k, v in hist.items():
        print(f"{k}: {v:.6f}")
    print("\nTHREE-ARM VALIDATION")
    cols = [
        "arm","ml_accuracy","ml_auc","ml_brier","ml_logloss","mean_actual_winner_probability",
        "method_accuracy","method_brier_multiclass","method_logloss","mean_actual_method_probability",
        "sim_dec","sim_ko","sim_sub","duration_bias","sim_nondec_by_300","sim_nondec_by_600",
        "sim_ko_r1","sim_ko_r2","sim_ko_r3","high_conf_wrong_ge_065","high_conf_wrong_ge_075",
    ]
    print(m[cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nOUTPUT: {args.out_dir}")


if __name__ == "__main__":
    main()
