"""Decompose TD-completion gain into FSR recalibration and attacker age.

Models:

1. Current FSR
   p = sigmoid(L)

2. Recalibrated FSR
   p = sigmoid(alpha + beta * L)

3. Exact age correction
   p = sigmoid(L + gamma * (age - 30))

4. Recalibrated FSR + age
   p = sigmoid(alpha + beta * L + gamma * (age - 30))

All learned parameters are fit only on historical pre-cutoff data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
)

from pipeline.fsr_v2.diagnostics.takedown_completion_xgb_audit import (
    build_attempt_frame,
)


SEED = 20260813


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def nll(y, z):
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    return float(
        np.mean(
            np.logaddexp(0.0, z)
            - y * z
        )
    )


def score(y, p):
    return {
        "auc": roc_auc_score(y, p),
        "logloss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
    }


def age_values(df, fill):
    return (
        df.att_age.fillna(fill).to_numpy(float)
        - 30.0
    )


def fit_recalibrated(df):
    L = df.baseline_logit.to_numpy(float)
    y = df.y.to_numpy(float)

    def objective(theta):
        alpha, beta = theta
        return nll(
            y,
            alpha + beta * L,
        )

    result = minimize(
        objective,
        x0=np.array([0.0, 1.0]),
        method="L-BFGS-B",
    )

    if not result.success:
        raise RuntimeError(result.message)

    return tuple(map(float, result.x))


def fit_exact_age(df, age_fill):
    L = df.baseline_logit.to_numpy(float)
    A = age_values(df, age_fill)
    y = df.y.to_numpy(float)

    def objective(gamma):
        return nll(
            y,
            L + gamma * A,
        )

    result = minimize_scalar(
        objective,
        bounds=(-0.20, 0.20),
        method="bounded",
    )

    return float(result.x)


def fit_recalibrated_age(df, age_fill):
    L = df.baseline_logit.to_numpy(float)
    A = age_values(df, age_fill)
    y = df.y.to_numpy(float)

    def objective(theta):
        alpha, beta, gamma = theta

        return nll(
            y,
            alpha
            + beta * L
            + gamma * A,
        )

    result = minimize(
        objective,
        x0=np.array([0.0, 1.0, -0.02]),
        method="L-BFGS-B",
    )

    if not result.success:
        raise RuntimeError(result.message)

    return tuple(map(float, result.x))


def predictions(df, age_fill, recal, exact_gamma, full):
    L = df.baseline_logit.to_numpy(float)
    A = age_values(df, age_fill)

    alpha_r, beta_r = recal

    alpha_f, beta_f, gamma_f = full

    return {
        "current":
            sigmoid(L),

        "recalibrated":
            sigmoid(
                alpha_r
                + beta_r * L
            ),

        "exact_age":
            sigmoid(
                L
                + exact_gamma * A
            ),

        "recalibrated_age":
            sigmoid(
                alpha_f
                + beta_f * L
                + gamma_f * A
            ),
    }


def fit_all(train):
    fill = float(train.att_age.median())

    recal = fit_recalibrated(train)

    exact_gamma = fit_exact_age(
        train,
        fill,
    )

    full = fit_recalibrated_age(
        train,
        fill,
    )

    return fill, recal, exact_gamma, full


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--holdout",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--cutoff",
        default="2025-03-22",
    )

    args = ap.parse_args()

    cutoff = pd.Timestamp(
        args.cutoff
    ).normalize()

    print(
        "Building leakage-safe TD-completion dataset..."
    )

    d = build_attempt_frame()

    train = d[
        d.event_date < cutoff
    ].copy()

    hold = pd.read_csv(
        args.holdout
    )

    holdout_ids = set(
        hold.bout_id.astype(str)
    )

    test = d[
        d.fight_id.astype(str).isin(
            holdout_ids
        )
    ].copy()

    # =========================================================
    # Temporal validation
    # =========================================================
    dates = np.array(
        sorted(train.event_date.unique())
    )

    starts = [
        int(len(dates) * 0.50),
        int(len(dates) * 0.625),
        int(len(dates) * 0.75),
        int(len(dates) * 0.875),
    ]

    fold_rows = []

    for i, start in enumerate(starts):
        end = (
            starts[i + 1]
            if i + 1 < len(starts)
            else len(dates)
        )

        tr = train[
            train.event_date.isin(
                dates[:start]
            )
        ]

        va = train[
            train.event_date.isin(
                dates[start:end]
            )
        ]

        fill, recal, exact_gamma, full = fit_all(tr)

        pred = predictions(
            va,
            fill,
            recal,
            exact_gamma,
            full,
        )

        base = score(
            va.y,
            pred["current"],
        )

        for model_name in [
            "recalibrated",
            "exact_age",
            "recalibrated_age",
        ]:
            m = score(
                va.y,
                pred[model_name],
            )

            fold_rows.append({
                "fold": i + 1,
                "model": model_name,
                "delta_auc":
                    m["auc"] - base["auc"],
                "delta_logloss":
                    m["logloss"] - base["logloss"],
                "delta_brier":
                    m["brier"] - base["brier"],
            })

    folds = pd.DataFrame(
        fold_rows
    )

    print("=" * 112)
    print(
        "TD COMPLETION — FSR RECALIBRATION + AGE DECOMPOSITION"
    )
    print("=" * 112)

    print(
        f"training attempts={len(train):,} | "
        f"holdout attempts={len(test):,}"
    )

    print(
        "\nPRE-CUTOFF TEMPORAL VALIDATION"
    )

    summary = (
        folds
        .groupby("model")
        .agg(
            mean_delta_auc=(
                "delta_auc",
                "mean",
            ),
            median_delta_auc=(
                "delta_auc",
                "median",
            ),
            positive_auc_fold_share=(
                "delta_auc",
                lambda x: np.mean(
                    np.asarray(x) > 0
                ),
            ),
            mean_delta_logloss=(
                "delta_logloss",
                "mean",
            ),
            mean_delta_brier=(
                "delta_brier",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False,
            formatters={
                "mean_delta_auc":
                    lambda x: f"{x:+.4f}",
                "median_delta_auc":
                    lambda x: f"{x:+.4f}",
                "positive_auc_fold_share":
                    lambda x: f"{x:.2f}",
                "mean_delta_logloss":
                    lambda x: f"{x:+.4f}",
                "mean_delta_brier":
                    lambda x: f"{x:+.4f}",
            },
        )
    )

    print(
        "\nFOLD-BY-FOLD AUC DELTAS"
    )

    pivot = folds.pivot(
        index="fold",
        columns="model",
        values="delta_auc",
    )

    print(
        pivot.to_string(
            float_format=lambda x: f"{x:+.4f}"
        )
    )

    # =========================================================
    # Fit all pre-cutoff data
    # =========================================================
    fill, recal, exact_gamma, full = fit_all(
        train
    )

    alpha_r, beta_r = recal

    alpha_f, beta_f, gamma_f = full

    print(
        "\nPARAMETERS FIT ON ALL PRE-CUTOFF DATA"
    )

    print("\nRecalibrated FSR:")
    print(
        f"  alpha = {alpha_r:+.6f}"
    )
    print(
        f"  beta  = {beta_r:+.6f}"
    )

    print("\nExact age equation:")
    print(
        f"  gamma = {exact_gamma:+.6f} per year"
    )

    print("\nRecalibrated FSR + age:")
    print(
        f"  alpha = {alpha_f:+.6f}"
    )
    print(
        f"  beta  = {beta_f:+.6f}"
    )
    print(
        f"  gamma = {gamma_f:+.6f} per year"
    )

    if beta_f != 0:
        gamma_effective = (
            gamma_f / beta_f
        )

        print(
            f"  effective gamma relative to "
            f"original FSR logit = "
            f"{gamma_effective:+.6f}"
        )

        print(
            f"  effective 5-year odds multiplier = "
            f"{np.exp(5 * gamma_effective):.4f}"
        )

        print(
            f"  effective 10-year odds multiplier = "
            f"{np.exp(10 * gamma_effective):.4f}"
        )

    # =========================================================
    # Frozen holdout
    # =========================================================
    pred = predictions(
        test,
        fill,
        recal,
        exact_gamma,
        full,
    )

    print(
        "\nFROZEN 500-FIGHT HOLDOUT"
    )

    print(
        f"{'model':30s}"
        f"{'AUC':>10s}"
        f"{'log loss':>12s}"
        f"{'Brier':>12s}"
    )

    holdout_metrics = {}

    names = [
        ("current", "Current FSR"),
        (
            "recalibrated",
            "Recalibrated FSR",
        ),
        (
            "exact_age",
            "Exact FSR + age",
        ),
        (
            "recalibrated_age",
            "Recalibrated FSR + age",
        ),
    ]

    for key, label in names:
        m = score(
            test.y,
            pred[key],
        )

        holdout_metrics[key] = m

        print(
            f"{label:30s}"
            f"{m['auc']:10.4f}"
            f"{m['logloss']:12.4f}"
            f"{m['brier']:12.4f}"
        )

    # =========================================================
    # Fight-cluster bootstrap
    # =========================================================
    z = test[
        [
            "fight_id",
            "y",
        ]
    ].copy()

    for key in pred:
        z[f"p_{key}"] = pred[key]

    groups = {
        fid: g
        for fid, g in z.groupby(
            "fight_id"
        )
    }

    fight_ids = np.array(
        list(groups)
    )

    rng = np.random.default_rng(
        SEED
    )

    comparisons = {
        "Exact age vs current": [],
        "Recal+age vs current": [],
        "Recal+age vs exact age": [],
    }

    for _ in range(2000):
        sampled = rng.choice(
            fight_ids,
            len(fight_ids),
            replace=True,
        )

        q = pd.concat(
            [
                groups[f]
                for f in sampled
            ],
            ignore_index=True,
        )

        auc_current = roc_auc_score(
            q.y,
            q.p_current,
        )

        auc_exact = roc_auc_score(
            q.y,
            q.p_exact_age,
        )

        auc_full = roc_auc_score(
            q.y,
            q.p_recalibrated_age,
        )

        comparisons[
            "Exact age vs current"
        ].append(
            auc_exact - auc_current
        )

        comparisons[
            "Recal+age vs current"
        ].append(
            auc_full - auc_current
        )

        comparisons[
            "Recal+age vs exact age"
        ].append(
            auc_full - auc_exact
        )

    print(
        "\nFIGHT-CLUSTER BOOTSTRAP — AUC DELTAS"
    )

    for label, values in comparisons.items():
        q = np.asarray(values)

        print(
            f"{label:28s} "
            f"mean={q.mean():+.4f}  "
            f"95% CI=["
            f"{np.quantile(q,.025):+.4f}, "
            f"{np.quantile(q,.975):+.4f}]  "
            f"P(>0)={np.mean(q > 0):.3f}"
        )


if __name__ == "__main__":
    main()
