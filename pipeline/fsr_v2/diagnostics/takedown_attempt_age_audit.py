"""Does prefight age improve takedown-tendency discrimination?

Target:
    fighter TD attempts per 15 minutes

Baseline:
    leakage-safe prefight FSR V2 takedown_tendency

Method:
    - build historical fighter-side observations directly from round stats
    - compute exact prefight age from master DOB
    - train only before the supplied cutoff
    - choose age functional form with rolling temporal validation
    - evaluate selected form on frozen Stage 1 holdout

No simulator tuning.
No winner metrics.
No population calibration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import (
    MASTER_PATH,
    FSR_V2_PREFIGHT_SNAPSHOTS_PATH,
)
from pipeline.fsr_v2.sources.round_stats import (
    load_round_stats,
    build_paired_rounds,
)
from pipeline.fsr_v2.replay.engine import aggregate_fights


def corr(a, b, method="spearman"):
    z = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(z) < 3 or z.a.nunique() < 2 or z.b.nunique() < 2:
        return np.nan
    return z.a.corr(z.b, method=method)


def direction_accuracy(actual, pred):
    actual = np.asarray(actual, float)
    pred = np.asarray(pred, float)

    keep = np.isfinite(actual) & np.isfinite(pred) & (~np.isclose(actual, 0))

    if not keep.any():
        return np.nan

    return float(
        np.mean(
            np.sign(actual[keep])
            == np.sign(pred[keep])
        )
    )


def fighter_age(event_date, dob):
    return (
        (
            pd.to_datetime(event_date)
            - pd.to_datetime(dob)
        ).dt.days
        / 365.2425
    )


def age_transform(age, name):
    age = np.asarray(age, float)

    if name == "linear":
        return age - 30.0

    threshold = float(name.replace("over", ""))

    return np.maximum(
        age - threshold,
        0.0,
    )


CANDIDATES = [
    "linear",
    "over30",
    "over32",
    "over34",
    "over36",
]


def fit_model(df, candidate):
    age_edge = (
        age_transform(df.red_age, candidate)
        - age_transform(df.blue_age, candidate)
    )

    X = np.column_stack([
        df.baseline_edge.to_numpy(float),
        age_edge,
    ])

    y = df.actual_edge.to_numpy(float)

    keep = np.isfinite(X).all(axis=1) & np.isfinite(y)

    beta, *_ = np.linalg.lstsq(
        X[keep],
        y[keep],
        rcond=None,
    )

    return beta


def predict(df, candidate, beta):
    age_edge = (
        age_transform(df.red_age, candidate)
        - age_transform(df.blue_age, candidate)
    )

    X = np.column_stack([
        df.baseline_edge.to_numpy(float),
        age_edge,
    ])

    return X @ beta


def master_corner_frame():
    m = pd.read_parquet(MASTER_PATH).copy()

    m["fight_id"] = m["fight_id"].astype(str)
    m["event_date"] = (
        pd.to_datetime(m["date"])
        .dt.normalize()
    )

    red = m[
        [
            "fight_id",
            "event_date",
            "r_id",
            "r_dob",
        ]
    ].rename(
        columns={
            "r_id": "fighter_id",
            "r_dob": "dob",
        }
    )

    red["side"] = "red"

    blue = m[
        [
            "fight_id",
            "event_date",
            "b_id",
            "b_dob",
        ]
    ].rename(
        columns={
            "b_id": "fighter_id",
            "b_dob": "dob",
        }
    )

    blue["side"] = "blue"

    corners = pd.concat(
        [red, blue],
        ignore_index=True,
    )

    corners["fighter_id"] = (
        corners["fighter_id"].astype(str)
    )

    corners["dob"] = pd.to_datetime(
        corners["dob"],
        errors="coerce",
    )

    return corners


def historical_training_pairs(cutoff):
    hist = aggregate_fights(
        build_paired_rounds(
            rounds=load_round_stats()
        )
    ).copy()

    hist["fight_id"] = (
        hist["fight_id"].astype(str)
    )

    hist["fighter_id"] = (
        hist["fighter_id"].astype(str)
    )

    hist["event_date"] = (
        pd.to_datetime(hist["event_date"])
        .dt.normalize()
    )

    hist = hist[
        hist["fight_elapsed_seconds"] > 0
    ].copy()

    hist["actual_rate"] = (
        hist["td_attempted"]
        * 900.0
        / hist["fight_elapsed_seconds"]
    )

    fsr = pd.read_parquet(
        FSR_V2_PREFIGHT_SNAPSHOTS_PATH
    ).copy()

    fsr["fight_id"] = (
        fsr["fight_id"].astype(str)
    )

    fsr["fighter_id"] = (
        fsr["fighter_id"].astype(str)
    )

    fsr["event_date"] = (
        pd.to_datetime(fsr["event_date"])
        .dt.normalize()
    )

    x = hist.merge(
        fsr[
            [
                "fight_id",
                "event_date",
                "fighter_id",
                "takedown_tendency",
            ]
        ],
        on=[
            "fight_id",
            "event_date",
            "fighter_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    x = x.merge(
        master_corner_frame(),
        on=[
            "fight_id",
            "event_date",
            "fighter_id",
        ],
        how="left",
        validate="one_to_one",
    )

    x["age"] = fighter_age(
        x["event_date"],
        x["dob"],
    )

    x["baseline_rate"] = (
        x["takedown_tendency"]
        * 900.0
    )

    x = x[
        (x["event_date"] < cutoff)
        & x["age"].between(18, 50)
    ].copy()

    red = x[
        x.side.eq("red")
    ].copy()

    blue = x[
        x.side.eq("blue")
    ].copy()

    pairs = red.merge(
        blue,
        on=[
            "fight_id",
            "event_date",
        ],
        suffixes=("_red", "_blue"),
        validate="one_to_one",
    )

    out = pd.DataFrame({
        "fight_id": pairs.fight_id,
        "event_date": pairs.event_date,
        "red_age": pairs.age_red,
        "blue_age": pairs.age_blue,
        "actual_edge": (
            pairs.actual_rate_red
            - pairs.actual_rate_blue
        ),
        "baseline_edge": (
            pairs.baseline_rate_red
            - pairs.baseline_rate_blue
        ),
    })

    return out.sort_values(
        ["event_date", "fight_id"]
    ).reset_index(drop=True)


def holdout_pairs(path):
    x = pd.read_csv(path).copy()

    x["bout_id"] = (
        x["bout_id"].astype(str)
    )

    m = pd.read_parquet(MASTER_PATH).copy()

    m["fight_id"] = (
        m["fight_id"].astype(str)
    )

    m["event_date_master"] = (
        pd.to_datetime(m["date"])
        .dt.normalize()
    )

    m["r_dob"] = pd.to_datetime(
        m["r_dob"],
        errors="coerce",
    )

    m["b_dob"] = pd.to_datetime(
        m["b_dob"],
        errors="coerce",
    )

    x = x.merge(
        m[
            [
                "fight_id",
                "event_date_master",
                "r_dob",
                "b_dob",
            ]
        ],
        left_on="bout_id",
        right_on="fight_id",
        how="left",
        validate="one_to_one",
    )

    x["red_age"] = fighter_age(
        x["event_date_master"],
        x["r_dob"],
    )

    x["blue_age"] = fighter_age(
        x["event_date_master"],
        x["b_dob"],
    )

    exposure = (
        x["actual_elapsed_seconds"]
        .astype(float)
    )

    x["actual_edge"] = (
        (
            x["historical_red_td_attempts"]
            - x["historical_blue_td_attempts"]
        )
        * 900.0
        / exposure
    )

    x["baseline_edge"] = (
        (
            x["fsr_red_tendency_only"]
            - x["fsr_blue_tendency_only"]
        )
        * 900.0
    )

    return x[
        [
            "bout_id",
            "event_date_master",
            "red_age",
            "blue_age",
            "actual_edge",
            "baseline_edge",
        ]
    ].dropna().reset_index(drop=True)


def temporal_validation(train):
    dates = np.array(
        sorted(train.event_date.unique())
    )

    cutpoints = [
        int(len(dates) * 0.50),
        int(len(dates) * 0.625),
        int(len(dates) * 0.75),
        int(len(dates) * 0.875),
    ]

    results = []

    for candidate in CANDIDATES:
        deltas = []
        age_betas = []

        for i, start in enumerate(cutpoints):
            end = (
                cutpoints[i + 1]
                if i + 1 < len(cutpoints)
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

            if len(tr) < 100 or len(va) < 20:
                continue

            beta = fit_model(
                tr,
                candidate,
            )

            pred = predict(
                va,
                candidate,
                beta,
            )

            baseline_rho = corr(
                va.actual_edge,
                va.baseline_edge,
            )

            model_rho = corr(
                va.actual_edge,
                pred,
            )

            deltas.append(
                model_rho - baseline_rho
            )

            age_betas.append(beta[1])

        results.append({
            "candidate": candidate,
            "folds": len(deltas),
            "mean_delta_rho": (
                np.mean(deltas)
                if deltas else np.nan
            ),
            "median_delta_rho": (
                np.median(deltas)
                if deltas else np.nan
            ),
            "mean_age_beta": (
                np.mean(age_betas)
                if age_betas else np.nan
            ),
            "negative_age_beta_share": (
                np.mean(np.array(age_betas) < 0)
                if age_betas else np.nan
            ),
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--holdout",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--cutoff",
        default="2025-03-22",
    )

    args = parser.parse_args()

    cutoff = pd.Timestamp(
        args.cutoff
    ).normalize()

    train = historical_training_pairs(
        cutoff
    )

    holdout = holdout_pairs(
        args.holdout
    )

    print("=" * 100)
    print(
        "FSR V2 — TAKEDOWN ATTEMPT AGE AUDIT"
    )
    print("=" * 100)

    print(
        f"training fights: {len(train)}"
    )

    print(
        f"holdout fights:  {len(holdout)}"
    )

    print(
        "training age range: "
        f"{train[['red_age','blue_age']].min().min():.1f}"
        " - "
        f"{train[['red_age','blue_age']].max().max():.1f}"
    )

    cv = temporal_validation(train)

    print(
        "\nTEMPORAL TRAINING VALIDATION"
    )

    print(
        cv.to_string(
            index=False,
            formatters={
                "mean_delta_rho":
                    lambda x: f"{x:+.4f}",
                "median_delta_rho":
                    lambda x: f"{x:+.4f}",
                "mean_age_beta":
                    lambda x: f"{x:+.4f}",
                "negative_age_beta_share":
                    lambda x: f"{x:.2f}",
            },
        )
    )

    usable = cv.dropna(
        subset=["mean_delta_rho"]
    )

    if usable.empty:
        raise RuntimeError(
            "No usable temporal validation folds."
        )

    chosen = (
        usable.sort_values(
            "mean_delta_rho",
            ascending=False,
        )
        .iloc[0]
        .candidate
    )

    beta = fit_model(
        train,
        chosen,
    )

    pred = predict(
        holdout,
        chosen,
        beta,
    )

    base_rho = corr(
        holdout.actual_edge,
        holdout.baseline_edge,
    )

    age_rho = corr(
        holdout.actual_edge,
        pred,
    )

    base_r = corr(
        holdout.actual_edge,
        holdout.baseline_edge,
        "pearson",
    )

    age_r = corr(
        holdout.actual_edge,
        pred,
        "pearson",
    )

    base_dir = direction_accuracy(
        holdout.actual_edge,
        holdout.baseline_edge,
    )

    age_dir = direction_accuracy(
        holdout.actual_edge,
        pred,
    )

    print(
        "\nSELECTED AGE FORM FROM TRAINING ONLY"
    )

    print(f"candidate: {chosen}")
    print(
        f"baseline coefficient: {beta[0]:+.6f}"
    )
    print(
        f"age coefficient:      {beta[1]:+.6f}"
    )

    print(
        "\nFROZEN 500-FIGHT HOLDOUT"
    )

    print(
        f"{'signal':28s}"
        f"{'edge r':>10s}"
        f"{'edge rho':>12s}"
        f"{'direction':>12s}"
    )

    print(
        f"{'FSR tendency only':28s}"
        f"{base_r:>10.3f}"
        f"{base_rho:>12.3f}"
        f"{base_dir:>12.3f}"
    )

    print(
        f"{'FSR tendency + age':28s}"
        f"{age_r:>10.3f}"
        f"{age_rho:>12.3f}"
        f"{age_dir:>12.3f}"
    )

    print(
        "\nINCREMENTAL AGE VALUE"
    )

    print(
        f"delta edge rho: "
        f"{age_rho - base_rho:+.4f}"
    )

    print(
        f"delta direction: "
        f"{age_dir - base_dir:+.4f}"
    )


if __name__ == "__main__":
    main()
