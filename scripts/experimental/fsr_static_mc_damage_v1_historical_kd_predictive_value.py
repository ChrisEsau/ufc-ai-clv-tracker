"""Evaluate matchup-level predictive value of the KD=80 static MC.

Consumes the historical actual-vs-MC artifact produced by
``fsr_static_mc_damage_v1_historical_kd_actual_vs_mc.py``.

Goal
----
Measure whether the Monte Carlo assigns higher KD probabilities to historical
matchups that actually produced at least one UFCStats knockdown.

This is a predictive-value audit, not a calibration retune. KO/TKO remains out
of scope.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


INPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_historical_kd_actual_vs_mc.parquet"
)
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_historical_kd_predictive_value.parquet"
)

THRESHOLDS = (0.20, 0.30, 0.40, 0.50)


def _safe_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    return float(log_loss(y, clipped, labels=[0, 1]))


def _threshold_table(frame: pd.DataFrame) -> pd.DataFrame:
    y = frame["actual_any_kd"].astype(int).to_numpy()
    p = frame["mc_p_any_kd"].astype(float).to_numpy()
    rows: list[dict[str, float | int]] = []

    for threshold in THRESHOLDS:
        pred = (p >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())

        rows.append(
            {
                "threshold": threshold,
                "predicted_positive": int(pred.sum()),
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def _print_ranked(frame: pd.DataFrame, *, top_n: int = 20) -> None:
    cols = [
        c
        for c in (
            "bout_id",
            "event_date",
            "red_name",
            "blue_name",
            "mc_p_any_kd",
            "actual_any_kd",
            "actual_total_kd",
            "mc_expected_total_kd",
        )
        if c in frame.columns
    ]

    high = frame.sort_values("mc_p_any_kd", ascending=False).head(top_n)[cols]
    low = frame.sort_values("mc_p_any_kd", ascending=True).head(top_n)[cols]

    print(f"\nTOP {top_n} HIGHEST MC KD-PROBABILITY MATCHUPS")
    print(high.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nTOP {top_n} LOWEST MC KD-PROBABILITY MATCHUPS")
    print(low.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_actual_kd_capture(frame: pd.DataFrame) -> None:
    actual_kd = frame[frame["actual_any_kd"] == 1].copy()
    total = len(actual_kd)
    print("\nACTUAL-KD FIGHT CAPTURE")
    print(f"actual KD fights: {total}")
    for threshold in (0.10, 0.20, 0.30, 0.40, 0.50):
        caught = int((actual_kd["mc_p_any_kd"] >= threshold).sum())
        share = caught / total if total else float("nan")
        print(f"MC p(KD) >= {threshold:.0%}: {caught:3d} / {total} ({share:.2%})")


def _print_summary(frame: pd.DataFrame) -> None:
    y = frame["actual_any_kd"].astype(int).to_numpy()
    p = frame["mc_p_any_kd"].astype(float).to_numpy()

    roc_auc = float(roc_auc_score(y, p))
    pr_auc = float(average_precision_score(y, p))
    brier = float(brier_score_loss(y, p))
    ll = _safe_log_loss(y, p)
    prevalence = float(y.mean())

    print("\n" + "=" * 120)
    print("HISTORICAL MATCHUP-LEVEL KD PREDICTIVE VALUE — DAMAGE V1 / KD=80")
    print("=" * 120)
    print(f"historical bouts: {len(frame):,}")
    print(f"actual KD prevalence: {prevalence:.4f}")
    print("\nCORE PREDICTIVE METRICS")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"PR-AUC:  {pr_auc:.6f}")
    print(f"Brier:   {brier:.6f}")
    print(f"Log loss:{ll:.6f}")
    print(f"No-skill PR-AUC baseline (prevalence): {prevalence:.6f}")

    no_kd = frame.loc[frame["actual_any_kd"] == 0, "mc_p_any_kd"]
    yes_kd = frame.loc[frame["actual_any_kd"] == 1, "mc_p_any_kd"]
    print("\nSCORE SEPARATION")
    print(
        f"actual no-KD: mean={no_kd.mean():.4f}; median={no_kd.median():.4f}; "
        f"n={len(no_kd):,}"
    )
    print(
        f"actual KD:    mean={yes_kd.mean():.4f}; median={yes_kd.median():.4f}; "
        f"n={len(yes_kd):,}"
    )

    print("\nTHRESHOLD PERFORMANCE")
    thresholds = _threshold_table(frame)
    print(thresholds.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    _print_actual_kd_capture(frame)
    _print_ranked(frame)

    print("\nINTERPRETATION GUIDE")
    print("- ROC-AUC ~0.50 means no matchup ranking value; higher is better.")
    print("- PR-AUC should materially exceed the actual KD prevalence to show useful positive-case ranking.")
    print("- Threshold precision tells how often flagged matchups actually had a KD.")
    print("- Threshold recall tells how many historical KD matchups were captured.")
    print("- This audit measures matchup predictive value only; it does not retune KD or add KO/TKO logic.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure matchup-level predictive value of historical KD=80 MC probabilities"
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    print(f"[KD predictive value] loading {args.input}", flush=True)
    if not args.input.exists():
        raise FileNotFoundError(
            f"Input artifact not found: {args.input}. Run the historical actual-vs-MC KD audit first."
        )

    frame = pd.read_parquet(args.input)
    required = {"bout_id", "actual_any_kd", "mc_p_any_kd"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Predictive-value input missing required columns: {missing}")

    if frame["actual_any_kd"].nunique() < 2:
        raise ValueError("Historical sample must contain both KD and no-KD bouts for predictive metrics.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    _print_summary(frame)
    print(f"\n[KD predictive value] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
