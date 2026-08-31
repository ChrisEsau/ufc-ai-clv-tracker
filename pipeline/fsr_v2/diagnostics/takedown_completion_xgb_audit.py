"""Small leakage-safe XGBoost model for TD completion.

Target:
    P(TD lands | TD attempted)

Purpose:
    Determine whether multiple pieces of PREFIGHT fighter state contain
    materially more TD-completion discrimination than the current FSR V2
    offense/defense completion matchup.

No production FSR or simulator changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from xgboost import XGBClassifier

from pipeline.fsr_v2.diagnostics.takedown_completion_prior_state_scan import (
    attach_context,
    attach_fsr,
    build_state_frame,
)


SEED = 20260813


FEATURES = [
    # Current FSR completion state.
    "baseline_logit",
    "takedown_offense",
    "opponent_takedown_defense",

    # Attacker state.
    "att_age",
    "att_layoff_days",
    "att_prior_fights",
    "att_prior_td_attempts",
    "att_career_td_success",
    "att_recent3_td_success",
    "att_recent5_td_success",
    "att_recent3_minus_career_td_success",
    "att_recent3_td_attempt_rate",

    # Defender state.
    "def_prior_fights",
    "def_career_td_stop_rate",
    "def_recent3_td_stop_rate",
    "def_career_td_faced",
    "def_recent3_control_suffered_per_entry",
]


GRID = [
    {
        "max_depth": 1,
        "min_child_weight": 10,
        "n_estimators": 100,
    },
    {
        "max_depth": 1,
        "min_child_weight": 30,
        "n_estimators": 100,
    },
    {
        "max_depth": 2,
        "min_child_weight": 10,
        "n_estimators": 100,
    },
    {
        "max_depth": 2,
        "min_child_weight": 30,
        "n_estimators": 100,
    },
    {
        "max_depth": 1,
        "min_child_weight": 10,
        "n_estimators": 200,
    },
    {
        "max_depth": 1,
        "min_child_weight": 30,
        "n_estimators": 200,
    },
    {
        "max_depth": 2,
        "min_child_weight": 10,
        "n_estimators": 200,
    },
    {
        "max_depth": 2,
        "min_child_weight": 30,
        "n_estimators": 200,
    },
]


def safe_logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1-p))


def build_attempt_frame():
    x = build_state_frame()
    x = attach_fsr(x)
    x = attach_context(x)

    # Explicitly expose baseline logit.
    x["baseline_logit"] = safe_logit(
        x["baseline_probability"]
    )

    rows = []

    for r in x.itertuples():
        attempts = int(round(r.td_attempted))
        landed = int(round(r.td_landed))

        if attempts <= 0:
            continue

        common = {
            "fight_id": str(r.fight_id),
            "event_date": pd.Timestamp(r.event_date),
            "baseline_probability": float(r.baseline_probability),
        }

        for feature in FEATURES:
            common[feature] = getattr(r, feature)

        # Attempt-weighted Bernoulli target.
        for y in (
            [1] * landed
            + [0] * (attempts-landed)
        ):
            rows.append({
                **common,
                "y": y,
            })

    return pd.DataFrame(rows)


def make_model(params):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",

        learning_rate=0.03,

        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        n_estimators=params["n_estimators"],

        subsample=0.80,
        colsample_bytree=0.80,

        reg_alpha=0.5,
        reg_lambda=5.0,

        random_state=SEED,
        n_jobs=1,
        tree_method="hist",
    )


def metrics(y, p):
    return {
        "auc": roc_auc_score(y, p),
        "logloss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
    }


def chronological_folds(train):
    dates = np.array(
        sorted(train.event_date.unique())
    )

    starts = [
        int(len(dates) * 0.50),
        int(len(dates) * 0.625),
        int(len(dates) * 0.75),
        int(len(dates) * 0.875),
    ]

    folds = []

    for i, start in enumerate(starts):
        end = (
            starts[i+1]
            if i+1 < len(starts)
            else len(dates)
        )

        train_dates = dates[:start]
        val_dates = dates[start:end]

        tr = train[
            train.event_date.isin(train_dates)
        ]

        va = train[
            train.event_date.isin(val_dates)
        ]

        if len(tr) >= 1000 and len(va) >= 200:
            folds.append((tr, va))

    return folds


def validate_grid(train):
    folds = chronological_folds(train)

    results = []

    for params in GRID:
        fold_rows = []

        for fold_number, (tr, va) in enumerate(
            folds,
            start=1,
        ):
            model = make_model(params)

            model.fit(
                tr[FEATURES],
                tr["y"],
            )

            p_xgb = model.predict_proba(
                va[FEATURES]
            )[:, 1]

            p_fsr = va[
                "baseline_probability"
            ].to_numpy()

            m_xgb = metrics(
                va.y,
                p_xgb,
            )

            m_fsr = metrics(
                va.y,
                p_fsr,
            )

            fold_rows.append({
                "fold": fold_number,

                "delta_auc":
                    m_xgb["auc"]
                    - m_fsr["auc"],

                "delta_logloss":
                    m_xgb["logloss"]
                    - m_fsr["logloss"],

                "delta_brier":
                    m_xgb["brier"]
                    - m_fsr["brier"],
            })

        f = pd.DataFrame(fold_rows)

        results.append({
            **params,

            "folds": len(f),

            "mean_delta_auc":
                f.delta_auc.mean(),

            "median_delta_auc":
                f.delta_auc.median(),

            "positive_auc_fold_share":
                (f.delta_auc > 0).mean(),

            # Negative is better for these.
            "mean_delta_logloss":
                f.delta_logloss.mean(),

            "mean_delta_brier":
                f.delta_brier.mean(),
        })

    return (
        pd.DataFrame(results)
        .sort_values(
            [
                "mean_delta_auc",
                "mean_delta_logloss",
            ],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def feature_importance(model):
    scores = model.get_booster().get_score(
        importance_type="gain"
    )

    rows = []

    for feature in FEATURES:
        rows.append({
            "feature": feature,
            "gain": scores.get(feature, 0.0),
        })

    z = pd.DataFrame(rows)

    total = z.gain.sum()

    if total > 0:
        z["gain_share"] = z.gain / total
    else:
        z["gain_share"] = 0.0

    return z.sort_values(
        "gain",
        ascending=False,
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

    cutoff = pd.Timestamp(
        args.cutoff
    ).normalize()

    print(
        "Building leakage-safe TD-attempt dataset..."
    )

    d = build_attempt_frame()

    train = d[
        d.event_date < cutoff
    ].copy()

    hold = pd.read_csv(
        args.holdout
    )

    holdout_ids = set(
        hold["bout_id"].astype(str)
    )

    test = d[
        d.fight_id.astype(str).isin(
            holdout_ids
        )
    ].copy()

    print()
    print("=" * 116)
    print(
        "FSR V2 — TD COMPLETION SMALL XGBOOST AUDIT"
    )
    print("=" * 116)

    print(
        f"training attempts: {len(train):,} | "
        f"holdout attempts: {len(test):,} | "
        f"holdout success: {test.y.mean():.3f}"
    )

    print(
        f"features: {len(FEATURES)}"
    )

    cv = validate_grid(train)

    print(
        "\nPRE-CUTOFF CHRONOLOGICAL HYPERPARAMETER VALIDATION"
    )

    print(
        cv.to_string(
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

    best = cv.iloc[0]

    params = {
        "max_depth":
            int(best.max_depth),

        "min_child_weight":
            int(best.min_child_weight),

        "n_estimators":
            int(best.n_estimators),
    }

    print(
        "\nSELECTED FROM TRAINING ONLY"
    )

    print(params)

    model = make_model(params)

    model.fit(
        train[FEATURES],
        train.y,
    )

    p_xgb = model.predict_proba(
        test[FEATURES]
    )[:, 1]

    p_fsr = test[
        "baseline_probability"
    ].to_numpy()

    m_xgb = metrics(
        test.y,
        p_xgb,
    )

    m_fsr = metrics(
        test.y,
        p_fsr,
    )

    print(
        "\nFROZEN 500-FIGHT HOLDOUT"
    )

    print(
        f"{'model':28s}"
        f"{'AUC':>10s}"
        f"{'log loss':>12s}"
        f"{'Brier':>12s}"
    )

    print(
        f"{'Current FSR V2':28s}"
        f"{m_fsr['auc']:10.4f}"
        f"{m_fsr['logloss']:12.4f}"
        f"{m_fsr['brier']:12.4f}"
    )

    print(
        f"{'Small XGBoost':28s}"
        f"{m_xgb['auc']:10.4f}"
        f"{m_xgb['logloss']:12.4f}"
        f"{m_xgb['brier']:12.4f}"
    )

    print(
        f"{'Delta (XGB - FSR)':28s}"
        f"{m_xgb['auc']-m_fsr['auc']:+10.4f}"
        f"{m_xgb['logloss']-m_fsr['logloss']:+12.4f}"
        f"{m_xgb['brier']-m_fsr['brier']:+12.4f}"
    )

    print(
        "\nFEATURE IMPORTANCE — GAIN"
    )

    importance = feature_importance(
        model
    )

    print(
        importance.head(15).to_string(
            index=False,
            formatters={
                "gain":
                    lambda x: f"{x:.3f}",
                "gain_share":
                    lambda x: f"{x:.3f}",
            },
        )
    )

    # Save attempt predictions for later bootstrap/calibration inspection.
    output = test[
        [
            "fight_id",
            "event_date",
            "y",
            "baseline_probability",
        ]
    ].copy()

    output["xgb_probability"] = p_xgb

    out_path = Path(
        "data/diagnostics/fsr_v2/"
        "td_completion_xgb_holdout_predictions.csv"
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        out_path,
        index=False,
    )

    print(
        f"\npredictions saved: {out_path}"
    )


if __name__ == "__main__":
    main()
