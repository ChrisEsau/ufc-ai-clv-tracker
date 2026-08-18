from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.simulation.event_clock_mc_v1.prototype_stage3_correlation import (
    prepare_direct_predictions,
)


MASTER = Path(
    "data/master/ufc_master.parquet"
)

STAGE10 = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "stage10_decision_judge_fresh.csv"
)

OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "stage10d_total_fight_judge.csv"
)


VARIANTS = {
    "SIG_ONLY": [
        "sig_diff",
    ],

    "SIG_KD": [
        "sig_diff",
        "kd_diff",
    ],

    "SIG_KD_TD": [
        "sig_diff",
        "kd_diff",
        "td_diff",
    ],

    "SIG_KD_TD_CTRL": [
        "sig_diff",
        "kd_diff",
        "td_diff",
        "ctrl_diff",
    ],

    "FULL_TOTAL": [
        "sig_diff",
        "kd_diff",
        "td_diff",
        "sub_diff",
        "ctrl_diff",
    ],
}


def decision_mask(frame):
    return (
        frame["method"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains("decision")
    )


def resolve_red_win(row):
    """
    Resolve historical winner from fighter IDs first,
    then fall back to fighter names.
    """

    winner_id = str(
        row.get("winner_id", "")
    ).strip()

    red_id = str(
        row.get("r_id", "")
    ).strip()

    blue_id = str(
        row.get("b_id", "")
    ).strip()

    if (
        winner_id
        and winner_id != "nan"
    ):
        if winner_id == red_id:
            return 1

        if winner_id == blue_id:
            return 0

    winner = str(
        row.get("winner", "")
    ).strip()

    red_name = str(
        row.get("r_name", "")
    ).strip()

    blue_name = str(
        row.get("b_name", "")
    ).strip()

    if winner == red_name:
        return 1

    if winner == blue_name:
        return 0

    return np.nan


def prepare_master(master):
    master = master.copy()

    master["fight_id"] = (
        master["fight_id"]
        .astype(str)
    )

    numeric_cols = [
        "r_sig_str_landed",
        "b_sig_str_landed",
        "r_kd",
        "b_kd",
        "r_td_landed",
        "b_td_landed",
        "r_sub_att",
        "b_sub_att",
        "r_ctrl",
        "b_ctrl",
    ]

    for col in numeric_cols:
        master[col] = pd.to_numeric(
            master[col],
            errors="coerce",
        ).fillna(0.0)

    master["red_win"] = master.apply(
        resolve_red_win,
        axis=1,
    )

    # ----------------------------------------------------------
    # Clean, non-redundant TOTAL-FIGHT observable differentials.
    #
    # Positive = Red did more.
    # ----------------------------------------------------------

    master["sig_diff"] = (
        master["r_sig_str_landed"]
        - master["b_sig_str_landed"]
    )

    master["kd_diff"] = (
        master["r_kd"]
        - master["b_kd"]
    )

    master["td_diff"] = (
        master["r_td_landed"]
        - master["b_td_landed"]
    )

    master["sub_diff"] = (
        master["r_sub_att"]
        - master["b_sub_att"]
    )

    master["ctrl_diff"] = (
        master["r_ctrl"]
        - master["b_ctrl"]
    )

    return master


def fit_model(
    train,
    features,
):
    model = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    max_iter=5000,
                ),
            ),
        ]
    )

    model.fit(
        train[features],
        train["red_win"].astype(int),
    )

    return model


def evaluate(
    model,
    frame,
    features,
):
    y = (
        frame["red_win"]
        .astype(int)
        .to_numpy()
    )

    p = model.predict_proba(
        frame[features]
    )[:, 1]

    pred = (
        p >= 0.5
    ).astype(int)

    return {
        "accuracy":
            accuracy_score(
                y,
                pred,
            ),

        "auc":
            roc_auc_score(
                y,
                p,
            ),

        "brier":
            brier_score_loss(
                y,
                p,
            ),

        "logloss":
            log_loss(
                y,
                p,
            ),

        "p":
            p,

        "pred":
            pred,
    }


def main():
    print("=" * 150)
    print(
        "EVENT CLOCK MC — STAGE 10D "
        "TOTAL-FIGHT LEARNED DECISION JUDGE"
    )
    print("=" * 150)

    # ----------------------------------------------------------
    # Rebuild the exact Stage-10 train cohort.
    # ----------------------------------------------------------

    train_direct, test_direct = (
        prepare_direct_predictions()
    )

    train_ids = set(
        train_direct[
            "fight_id"
        ]
        .astype(str)
        .unique()
    )

    print()
    print(
        f"Stage-10 training fights: "
        f"{len(train_ids)}"
    )

    # ----------------------------------------------------------
    # Load master historical fight totals.
    # ----------------------------------------------------------

    master = pd.read_parquet(
        MASTER
    )

    master = prepare_master(
        master
    )

    # ----------------------------------------------------------
    # Training:
    # exact Stage-10 historical training cohort,
    # decisions only.
    # ----------------------------------------------------------

    train = master[
        master["fight_id"].isin(
            train_ids
        )
        & decision_mask(master)
        & master["red_win"].notna()
    ].copy()

    # ----------------------------------------------------------
    # Fresh evaluation:
    # use the exact same 249 fight IDs already evaluated
    # by Stage 10.
    # ----------------------------------------------------------

    stage10 = pd.read_csv(
        STAGE10,
        low_memory=False,
    )

    stage10["fight_id"] = (
        stage10["fight_id"]
        .astype(str)
    )

    fresh_ids = set(
        stage10["fight_id"]
    )

    test = master[
        master["fight_id"].isin(
            fresh_ids
        )
        & decision_mask(master)
        & master["red_win"].notna()
    ].copy()

    # Preserve Stage-10 ordering.
    order = {
        fight_id: i
        for i, fight_id
        in enumerate(
            stage10["fight_id"]
        )
    }

    test["_order"] = (
        test["fight_id"]
        .map(order)
    )

    test = (
        test
        .sort_values("_order")
        .reset_index(drop=True)
    )

    print()
    print("=" * 150)
    print("COHORT")
    print("=" * 150)

    print(
        f"Training decisions: "
        f"{len(train)}"
    )

    print(
        f"Fresh decisions: "
        f"{len(test)}"
    )

    missing_fresh = (
        fresh_ids
        - set(
            test["fight_id"]
        )
    )

    print(
        f"Fresh IDs missing from master: "
        f"{len(missing_fresh)}"
    )

    if missing_fresh:
        print(
            sorted(
                missing_fresh
            )[:20]
        )

    # ----------------------------------------------------------
    # Fit/evaluate all nested variants.
    # ----------------------------------------------------------

    rows = []
    predictions = {}

    for name, features in (
        VARIANTS.items()
    ):

        model = fit_model(
            train,
            features,
        )

        metrics = evaluate(
            model,
            test,
            features,
        )

        predictions[name] = {
            "model": model,
            **metrics,
        }

        rows.append(
            {
                "variant":
                    name,

                "features":
                    len(features),

                "accuracy":
                    metrics[
                        "accuracy"
                    ],

                "auc":
                    metrics[
                        "auc"
                    ],

                "brier":
                    metrics[
                        "brier"
                    ],

                "logloss":
                    metrics[
                        "logloss"
                    ],
            }
        )

    summary = pd.DataFrame(
        rows
    )

    print()
    print("=" * 150)
    print("FRESH DECISION RESULTS")
    print("=" * 150)

    print(
        summary.to_string(
            index=False,
            formatters={
                "accuracy":
                    lambda x:
                        f"{x:.2%}",

                "auc":
                    lambda x:
                        f"{x:.4f}",

                "brier":
                    lambda x:
                        f"{x:.4f}",

                "logloss":
                    lambda x:
                        f"{x:.4f}",
            },
        )
    )

    # ----------------------------------------------------------
    # Full-model coefficients.
    # ----------------------------------------------------------

    full_features = (
        VARIANTS[
            "FULL_TOTAL"
        ]
    )

    full_model = (
        predictions[
            "FULL_TOTAL"
        ]["model"]
    )

    coefficients = (
        full_model
        .named_steps[
            "logistic"
        ]
        .coef_[0]
    )

    coef = pd.DataFrame(
        {
            "feature":
                full_features,

            "standardized_coefficient":
                coefficients,
        }
    ).sort_values(
        "standardized_coefficient",
        ascending=False,
    )

    print()
    print("=" * 150)
    print("FULL TOTAL-JUDGE COEFFICIENTS")
    print("=" * 150)

    print(
        coef.to_string(
            index=False,
            formatters={
                "standardized_coefficient":
                    lambda x:
                        f"{x:+.4f}"
            },
        )
    )

    # ----------------------------------------------------------
    # Compare directly with current Stage-10 oracle.
    # ----------------------------------------------------------

    out = test[
        [
            "fight_id",
            "r_name",
            "b_name",
            "winner",
            "method",
            "red_win",
            "sig_diff",
            "kd_diff",
            "td_diff",
            "sub_diff",
            "ctrl_diff",
        ]
    ].copy()

    for name in VARIANTS:

        p = (
            predictions[
                name
            ]["p"]
        )

        out[
            f"{name.lower()}_p_red"
        ] = p

        out[
            f"{name.lower()}_pred_red"
        ] = (
            p >= 0.5
        ).astype(int)

        out[
            f"{name.lower()}_correct"
        ] = (
            out[
                f"{name.lower()}_pred_red"
            ]
            == out[
                "red_win"
            ].astype(int)
        )

    out = out.merge(
        stage10[
            [
                "fight_id",
                "oracle_p_red",
                "expected_p_red",
                "mc_p_red",
            ]
        ],
        on="fight_id",
        how="left",
    )

    out[
        "old_oracle_pred_red"
    ] = (
        out[
            "oracle_p_red"
        ]
        >= 0.5
    ).astype(int)

    out[
        "old_oracle_correct"
    ] = (
        out[
            "old_oracle_pred_red"
        ]
        == out[
            "red_win"
        ].astype(int)
    )

    out[
        "full_total_correct"
    ] = (
        out[
            "full_total_correct"
        ]
        .astype(bool)
    )

    fixed = out[
        (
            ~out[
                "old_oracle_correct"
            ]
        )
        & (
            out[
                "full_total_correct"
            ]
        )
    ].copy()

    broken = out[
        (
            out[
                "old_oracle_correct"
            ]
        )
        & (
            ~out[
                "full_total_correct"
            ]
        )
    ].copy()

    print()
    print("=" * 150)
    print("DIRECT COMPARISON TO STAGE-10 ORACLE")
    print("=" * 150)

    print(
        f"Stage-10 oracle accuracy: "
        f"{out['old_oracle_correct'].mean():.2%}"
    )

    print(
        f"FULL_TOTAL accuracy:       "
        f"{out['full_total_correct'].mean():.2%}"
    )

    print(
        f"Old oracle misses fixed:   "
        f"{len(fixed)}"
    )

    print(
        f"Old oracle correct broken: "
        f"{len(broken)}"
    )

    # ----------------------------------------------------------
    # Confidence calibration.
    # ----------------------------------------------------------

    p = (
        out[
            "full_total_p_red"
        ]
    )

    confidence = np.maximum(
        p,
        1.0 - p,
    )

    print()
    print("=" * 150)
    print("FULL TOTAL-JUDGE CONFIDENCE")
    print("=" * 150)

    print(
        f"{'THRESHOLD':>10} "
        f"{'N':>8} "
        f"{'COVERAGE':>12} "
        f"{'ACCURACY':>12}"
    )

    for threshold in (
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ):

        mask = (
            confidence
            >= threshold
        )

        subset = out[
            mask
        ]

        accuracy = (
            subset[
                "full_total_correct"
            ].mean()
            if len(subset)
            else np.nan
        )

        print(
            f"{threshold:>9.0%} "
            f"{len(subset):>8} "
            f"{len(subset)/len(out):>11.1%} "
            f"{accuracy:>11.1%}"
        )

    # ----------------------------------------------------------
    # Show exactly what was fixed.
    # ----------------------------------------------------------

    show_cols = [
        "fight_id",
        "r_name",
        "b_name",
        "winner",
        "method",
        "sig_diff",
        "kd_diff",
        "td_diff",
        "sub_diff",
        "ctrl_diff",
        "oracle_p_red",
        "full_total_p_red",
    ]

    print()
    print("=" * 180)
    print("STAGE-10 ORACLE MISSES FIXED BY FULL TOTAL JUDGE")
    print("=" * 180)

    if len(fixed):
        print(
            fixed[
                show_cols
            ]
            .sort_values(
                "full_total_p_red"
            )
            .to_string(
                index=False,
                float_format=lambda x:
                    f"{x:+.3f}",
            )
        )
    else:
        print("None")

    print()
    print("=" * 180)
    print("STAGE-10 CORRECT PICKS BROKEN BY FULL TOTAL JUDGE")
    print("=" * 180)

    if len(broken):
        print(
            broken[
                show_cols
            ]
            .to_string(
                index=False,
                float_format=lambda x:
                    f"{x:+.3f}",
            )
        )
    else:
        print("None")

    # ----------------------------------------------------------
    # Most confident misses.
    # ----------------------------------------------------------

    out[
        "full_total_confidence"
    ] = np.maximum(
        out[
            "full_total_p_red"
        ],
        1.0
        - out[
            "full_total_p_red"
        ],
    )

    misses = out[
        ~out[
            "full_total_correct"
        ]
    ].copy()

    print()
    print("=" * 180)
    print("MOST CONFIDENT FULL-TOTAL JUDGE MISSES")
    print("=" * 180)

    print(
        misses[
            show_cols
            + [
                "full_total_confidence"
            ]
        ]
        .sort_values(
            "full_total_confidence",
            ascending=False,
        )
        .head(30)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:+.3f}",
        )
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUT,
        index=False,
    )

    print()
    print(
        f"wrote: {OUT}"
    )


if __name__ == "__main__":
    main()
