"""Test exact TD completion age adjustment:

P(TD lands) = sigmoid(
    current_fsr_logit + gamma * (attacker_age - 30)
)

Only gamma is learned.
Current FSR structure is otherwise frozen.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from pipeline.fsr_v2.diagnostics.takedown_completion_xgb_audit import (
    build_attempt_frame,
)


SEED = 20260813


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def metrics(y, p):
    return (
        roc_auc_score(y, p),
        log_loss(y, p),
        brier_score_loss(y, p),
    )


def fit_gamma(df, age_fill):
    age = df.att_age.fillna(age_fill).to_numpy(float)
    base = df.baseline_logit.to_numpy(float)
    y = df.y.to_numpy(float)

    x_age = age - 30.0

    def objective(gamma):
        p = sigmoid(base + gamma * x_age)

        # Bernoulli log loss.
        return log_loss(
            y,
            p,
            labels=[0, 1],
        )

    result = minimize_scalar(
        objective,
        bounds=(-0.20, 0.20),
        method="bounded",
        options={"xatol": 1e-10},
    )

    return float(result.x)


def predict(df, gamma, age_fill):
    age = df.att_age.fillna(age_fill).to_numpy(float)

    return sigmoid(
        df.baseline_logit.to_numpy(float)
        + gamma * (age - 30.0)
    )


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

    cutoff = pd.Timestamp(args.cutoff).normalize()

    d = build_attempt_frame()

    train = d[
        d.event_date < cutoff
    ].copy()

    hold = pd.read_csv(args.holdout)

    holdout_ids = set(
        hold.bout_id.astype(str)
    )

    test = d[
        d.fight_id.astype(str).isin(holdout_ids)
    ].copy()

    age_fill = float(
        train.att_age.median()
    )

    # ---------------------------------------------------------
    # Temporal validation.
    # ---------------------------------------------------------
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

        fold_fill = float(
            tr.att_age.median()
        )

        gamma = fit_gamma(
            tr,
            fold_fill,
        )

        p_base = va.baseline_probability.to_numpy(float)

        p_age = predict(
            va,
            gamma,
            fold_fill,
        )

        b_auc, b_ll, b_br = metrics(
            va.y,
            p_base,
        )

        a_auc, a_ll, a_br = metrics(
            va.y,
            p_age,
        )

        fold_rows.append({
            "fold": i + 1,
            "gamma_per_year": gamma,
            "delta_auc": a_auc - b_auc,
            "delta_logloss": a_ll - b_ll,
            "delta_brier": a_br - b_br,
        })

    folds = pd.DataFrame(fold_rows)

    # ---------------------------------------------------------
    # Fit ONE gamma on all pre-cutoff data.
    # ---------------------------------------------------------
    gamma = fit_gamma(
        train,
        age_fill,
    )

    p_base = test.baseline_probability.to_numpy(float)

    p_age = predict(
        test,
        gamma,
        age_fill,
    )

    base_auc, base_ll, base_br = metrics(
        test.y,
        p_base,
    )

    age_auc, age_ll, age_br = metrics(
        test.y,
        p_age,
    )

    print("=" * 100)
    print("TD COMPLETION — EXACT AGE EQUATION AUDIT")
    print("=" * 100)

    print(
        "\nEquation:\n"
        "P(TD lands) = sigmoid("
        "current_FSR_logit + gamma * (attacker_age - 30))"
    )

    print("\nPRE-CUTOFF TEMPORAL VALIDATION")

    print(
        folds.to_string(
            index=False,
            formatters={
                "gamma_per_year":
                    lambda x: f"{x:+.5f}",
                "delta_auc":
                    lambda x: f"{x:+.4f}",
                "delta_logloss":
                    lambda x: f"{x:+.4f}",
                "delta_brier":
                    lambda x: f"{x:+.4f}",
            },
        )
    )

    print("\nTEMPORAL SUMMARY")

    print(
        f"mean gamma: "
        f"{folds.gamma_per_year.mean():+.5f}"
    )

    print(
        f"negative gamma folds: "
        f"{(folds.gamma_per_year < 0).mean():.2f}"
    )

    print(
        f"mean delta AUC: "
        f"{folds.delta_auc.mean():+.4f}"
    )

    print(
        f"positive AUC folds: "
        f"{(folds.delta_auc > 0).mean():.2f}"
    )

    print(
        f"mean delta log loss: "
        f"{folds.delta_logloss.mean():+.4f}"
    )

    # ---------------------------------------------------------
    # Final training-only coefficient.
    # ---------------------------------------------------------
    print("\nFINAL GAMMA — FIT ON ALL PRE-CUTOFF DATA")

    print(
        f"gamma per year: {gamma:+.6f}"
    )

    print(
        f"odds multiplier per +1 year: "
        f"{np.exp(gamma):.4f}"
    )

    print(
        f"odds multiplier per +5 years: "
        f"{np.exp(5 * gamma):.4f}"
    )

    print(
        f"odds multiplier per +10 years: "
        f"{np.exp(10 * gamma):.4f}"
    )

    # ---------------------------------------------------------
    # Frozen holdout.
    # ---------------------------------------------------------
    print("\nFROZEN 500-FIGHT HOLDOUT")

    print(
        f"{'model':30s}"
        f"{'AUC':>10s}"
        f"{'log loss':>12s}"
        f"{'Brier':>12s}"
    )

    print(
        f"{'Current FSR':30s}"
        f"{base_auc:10.4f}"
        f"{base_ll:12.4f}"
        f"{base_br:12.4f}"
    )

    print(
        f"{'FSR + exact age penalty':30s}"
        f"{age_auc:10.4f}"
        f"{age_ll:12.4f}"
        f"{age_br:12.4f}"
    )

    print(
        f"{'Delta':30s}"
        f"{age_auc-base_auc:+10.4f}"
        f"{age_ll-base_ll:+12.4f}"
        f"{age_br-base_br:+12.4f}"
    )

    # ---------------------------------------------------------
    # Example effect at 35% starting FSR probability.
    # ---------------------------------------------------------
    example_base = 0.35
    example_logit = np.log(
        example_base / (1 - example_base)
    )

    print(
        "\nAGE EFFECT EXAMPLE "
        "(fighter whose unadjusted FSR completion = 35%)"
    )

    for age in [22, 25, 30, 32, 35, 37, 40, 42]:
        p = sigmoid(
            example_logit
            + gamma * (age - 30)
        )

        print(
            f"age {age:2d}: {p:.3f}"
        )

    # ---------------------------------------------------------
    # Fight-cluster bootstrap.
    # ---------------------------------------------------------
    z = test[
        [
            "fight_id",
            "y",
        ]
    ].copy()

    z["p_base"] = p_base
    z["p_age"] = p_age

    groups = {
        fid: g
        for fid, g in z.groupby("fight_id")
    }

    fight_ids = np.array(list(groups))

    rng = np.random.default_rng(SEED)

    deltas = []

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

        deltas.append(
            roc_auc_score(q.y, q.p_age)
            - roc_auc_score(q.y, q.p_base)
        )

    deltas = np.asarray(deltas)

    print("\nFIGHT-CLUSTER BOOTSTRAP — AUC")

    print(
        f"mean delta: {deltas.mean():+.4f}"
    )

    print(
        f"95% CI: "
        f"[{np.quantile(deltas,.025):+.4f}, "
        f"{np.quantile(deltas,.975):+.4f}]"
    )

    print(
        f"P(age equation improves): "
        f"{np.mean(deltas > 0):.3f}"
    )


if __name__ == "__main__":
    main()
