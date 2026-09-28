"""TD completion audit: attacker age vs defender age.

Models:
1. Current FSR
2. Recalibrated FSR
3. Recalibrated FSR + attacker age
4. Recalibrated FSR + defender age
5. Recalibrated FSR + attacker age + defender age

All parameters fit on pre-cutoff data only.
Target = TD landed / TD attempt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v2.diagnostics.takedown_completion_prior_state_scan import (
    build_state_frame,
    attach_fsr,
    attach_context,
)


SEED = 20260813


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def nll(y, z):
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def metrics(y, p):
    return {
        "auc": roc_auc_score(y, p),
        "logloss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
    }


def attach_defender_age(x):
    master = pd.read_parquet(MASTER_PATH).copy()

    master["fight_id"] = master["fight_id"].astype(str)
    master["event_date"] = pd.to_datetime(
        master["date"]
    ).dt.normalize()

    red = master[
        ["fight_id", "event_date", "r_id", "r_dob"]
    ].rename(
        columns={
            "r_id": "opponent_id",
            "r_dob": "def_dob",
        }
    )

    blue = master[
        ["fight_id", "event_date", "b_id", "b_dob"]
    ].rename(
        columns={
            "b_id": "opponent_id",
            "b_dob": "def_dob",
        }
    )

    corners = pd.concat(
        [red, blue],
        ignore_index=True,
    )

    corners["opponent_id"] = corners["opponent_id"].astype(str)
    corners["def_dob"] = pd.to_datetime(
        corners["def_dob"],
        errors="coerce",
    )

    x["fight_id"] = x["fight_id"].astype(str)
    x["opponent_id"] = x["opponent_id"].astype(str)

    x = x.merge(
        corners,
        on=["fight_id", "event_date", "opponent_id"],
        how="left",
        validate="one_to_one",
    )

    x["def_age"] = (
        (x["event_date"] - x["def_dob"]).dt.days
        / 365.2425
    )

    return x


def build_attempts():
    x = build_state_frame()
    x = attach_fsr(x)
    x = attach_context(x)
    x = attach_defender_age(x)

    rows = []

    for r in x.itertuples():
        attempts = int(round(r.td_attempted))
        landed = int(round(r.td_landed))

        if attempts <= 0:
            continue

        labels = (
            [1] * landed
            + [0] * (attempts - landed)
        )

        for y in labels:
            rows.append({
                "fight_id": str(r.fight_id),
                "event_date": pd.Timestamp(r.event_date),
                "y": y,
                "baseline_probability":
                    float(r.baseline_probability),
                "baseline_logit":
                    float(r.baseline_logit),
                "att_age":
                    float(r.att_age)
                    if pd.notna(r.att_age)
                    else np.nan,
                "def_age":
                    float(r.def_age)
                    if pd.notna(r.def_age)
                    else np.nan,
            })

    return pd.DataFrame(rows)


def design(df, model_name, att_fill, def_fill):
    L = df.baseline_logit.to_numpy(float)

    A = (
        df.att_age.fillna(att_fill).to_numpy(float)
        - 30.0
    )

    D = (
        df.def_age.fillna(def_fill).to_numpy(float)
        - 30.0
    )

    if model_name == "recalibrated":
        return np.column_stack([
            np.ones(len(df)),
            L,
        ])

    if model_name == "attacker_age":
        return np.column_stack([
            np.ones(len(df)),
            L,
            A,
        ])

    if model_name == "defender_age":
        return np.column_stack([
            np.ones(len(df)),
            L,
            D,
        ])

    if model_name == "both_ages":
        return np.column_stack([
            np.ones(len(df)),
            L,
            A,
            D,
        ])

    raise ValueError(model_name)


def fit_model(df, model_name, att_fill, def_fill):
    X = design(
        df,
        model_name,
        att_fill,
        def_fill,
    )

    y = df.y.to_numpy(float)

    if model_name == "recalibrated":
        x0 = np.array([0.0, 1.0])

    elif model_name in {
        "attacker_age",
        "defender_age",
    }:
        x0 = np.array([0.0, 1.0, 0.0])

    else:
        x0 = np.array([0.0, 1.0, -0.02, 0.0])

    result = minimize(
        lambda theta: nll(y, X @ theta),
        x0=x0,
        method="L-BFGS-B",
    )

    if not result.success:
        raise RuntimeError(result.message)

    return np.asarray(result.x, float)


def predict(df, model_name, theta, att_fill, def_fill):
    X = design(
        df,
        model_name,
        att_fill,
        def_fill,
    )

    return sigmoid(X @ theta)


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

    print("Building leakage-safe TD-attempt dataset...")

    d = build_attempts()

    train = d[
        d.event_date < cutoff
    ].copy()

    hold = pd.read_csv(args.holdout)

    holdout_ids = set(
        hold.bout_id.astype(str)
    )

    test = d[
        d.fight_id.astype(str).isin(
            holdout_ids
        )
    ].copy()

    MODEL_NAMES = [
        "recalibrated",
        "attacker_age",
        "defender_age",
        "both_ages",
    ]

    # ==========================================================
    # Chronological validation
    # ==========================================================
    dates = np.array(
        sorted(train.event_date.unique())
    )

    starts = [
        int(len(dates) * .50),
        int(len(dates) * .625),
        int(len(dates) * .75),
        int(len(dates) * .875),
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
        ].copy()

        va = train[
            train.event_date.isin(
                dates[start:end]
            )
        ].copy()

        att_fill = float(
            tr.att_age.median()
        )

        def_fill = float(
            tr.def_age.median()
        )

        base = metrics(
            va.y,
            va.baseline_probability,
        )

        for model_name in MODEL_NAMES:
            theta = fit_model(
                tr,
                model_name,
                att_fill,
                def_fill,
            )

            p = predict(
                va,
                model_name,
                theta,
                att_fill,
                def_fill,
            )

            m = metrics(
                va.y,
                p,
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

    folds = pd.DataFrame(fold_rows)

    print("=" * 118)
    print(
        "TD COMPLETION — ATTACKER AGE VS DEFENDER AGE"
    )
    print("=" * 118)

    print(
        f"training attempts={len(train):,} | "
        f"holdout attempts={len(test):,}"
    )

    print(
        f"missing attacker ages: "
        f"train={train.att_age.isna().sum()} "
        f"holdout={test.att_age.isna().sum()}"
    )

    print(
        f"missing defender ages: "
        f"train={train.def_age.isna().sum()} "
        f"holdout={test.def_age.isna().sum()}"
    )

    print("\nPRE-CUTOFF TEMPORAL VALIDATION")

    summary = (
        folds
        .groupby("model")
        .agg(
            mean_delta_auc=("delta_auc", "mean"),
            median_delta_auc=("delta_auc", "median"),
            positive_auc_fold_share=(
                "delta_auc",
                lambda x: np.mean(
                    np.asarray(x) > 0
                ),
            ),
            mean_delta_logloss=("delta_logloss", "mean"),
            mean_delta_brier=("delta_brier", "mean"),
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

    # ==========================================================
    # Fit all training data.
    # ==========================================================
    att_fill = float(
        train.att_age.median()
    )

    def_fill = float(
        train.def_age.median()
    )

    fitted = {
        name: fit_model(
            train,
            name,
            att_fill,
            def_fill,
        )
        for name in MODEL_NAMES
    }

    print("\nPARAMETERS FIT ON ALL PRE-CUTOFF DATA")

    r = fitted["recalibrated"]
    a = fitted["attacker_age"]
    de = fitted["defender_age"]
    both = fitted["both_ages"]

    print(
        f"\nRecalibrated FSR:\n"
        f"  alpha={r[0]:+.6f}\n"
        f"  beta ={r[1]:+.6f}"
    )

    print(
        f"\nFSR + attacker age:\n"
        f"  alpha={a[0]:+.6f}\n"
        f"  beta ={a[1]:+.6f}\n"
        f"  attacker gamma={a[2]:+.6f}/year"
    )

    print(
        f"\nFSR + defender age:\n"
        f"  alpha={de[0]:+.6f}\n"
        f"  beta ={de[1]:+.6f}\n"
        f"  defender gamma={de[2]:+.6f}/year"
    )

    print(
        f"\nFSR + both ages:\n"
        f"  alpha={both[0]:+.6f}\n"
        f"  beta ={both[1]:+.6f}\n"
        f"  attacker gamma={both[2]:+.6f}/year\n"
        f"  defender gamma={both[3]:+.6f}/year"
    )

    print(
        "\nODDS MULTIPLIERS — BOTH-AGE MODEL"
    )

    print(
        f"attacker +5 years: "
        f"{np.exp(5 * both[2]):.4f}"
    )

    print(
        f"attacker +10 years: "
        f"{np.exp(10 * both[2]):.4f}"
    )

    print(
        f"defender +5 years: "
        f"{np.exp(5 * both[3]):.4f}"
    )

    print(
        f"defender +10 years: "
        f"{np.exp(10 * both[3]):.4f}"
    )

    # ==========================================================
    # Holdout.
    # ==========================================================
    preds = {
        "current":
            test.baseline_probability.to_numpy(float)
    }

    for name in MODEL_NAMES:
        preds[name] = predict(
            test,
            name,
            fitted[name],
            att_fill,
            def_fill,
        )

    print("\nFROZEN 500-FIGHT HOLDOUT")

    print(
        f"{'model':32s}"
        f"{'AUC':>10s}"
        f"{'log loss':>12s}"
        f"{'Brier':>12s}"
    )

    display_names = [
        ("current", "Current FSR"),
        ("recalibrated", "Recalibrated FSR"),
        ("attacker_age", "Recal FSR + attacker age"),
        ("defender_age", "Recal FSR + defender age"),
        ("both_ages", "Recal FSR + both ages"),
    ]

    for key, label in display_names:
        m = metrics(
            test.y,
            preds[key],
        )

        print(
            f"{label:32s}"
            f"{m['auc']:10.4f}"
            f"{m['logloss']:12.4f}"
            f"{m['brier']:12.4f}"
        )

    # ==========================================================
    # Fight-cluster bootstrap.
    # ==========================================================
    z = test[
        ["fight_id", "y"]
    ].copy()

    for key, p in preds.items():
        z[f"p_{key}"] = p

    groups = {
        fid: g
        for fid, g in z.groupby("fight_id")
    }

    fight_ids = np.array(
        list(groups)
    )

    rng = np.random.default_rng(SEED)

    comparisons = {
        "Attacker age vs current": [],
        "Defender age vs current": [],
        "Both ages vs current": [],
        "Both ages vs attacker age": [],
    }

    for _ in range(2000):
        sampled = rng.choice(
            fight_ids,
            len(fight_ids),
            replace=True,
        )

        q = pd.concat(
            [groups[f] for f in sampled],
            ignore_index=True,
        )

        aucs = {
            key: roc_auc_score(
                q.y,
                q[f"p_{key}"],
            )
            for key in [
                "current",
                "attacker_age",
                "defender_age",
                "both_ages",
            ]
        }

        comparisons[
            "Attacker age vs current"
        ].append(
            aucs["attacker_age"]
            - aucs["current"]
        )

        comparisons[
            "Defender age vs current"
        ].append(
            aucs["defender_age"]
            - aucs["current"]
        )

        comparisons[
            "Both ages vs current"
        ].append(
            aucs["both_ages"]
            - aucs["current"]
        )

        comparisons[
            "Both ages vs attacker age"
        ].append(
            aucs["both_ages"]
            - aucs["attacker_age"]
        )

    print("\nFIGHT-CLUSTER BOOTSTRAP — AUC DELTAS")

    for label, values in comparisons.items():
        q = np.asarray(values)

        print(
            f"{label:30s} "
            f"mean={q.mean():+.4f}  "
            f"95% CI=["
            f"{np.quantile(q,.025):+.4f}, "
            f"{np.quantile(q,.975):+.4f}]  "
            f"P(>0)={np.mean(q > 0):.3f}"
        )


if __name__ == "__main__":
    main()
