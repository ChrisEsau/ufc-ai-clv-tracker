"""Leakage-safe prior-state signal scan for next-fight TD completion.

Question:
    Conditional on a TD being attempted in the next fight, which information
    available before that fight improves prediction of whether it lands?

Baseline:
    Current FSR V2 completion probability:
        population baseline + attacker offense - defender defense

Candidate prior-state features are tested ONE AT A TIME on top of baseline.

Selection/scoring:
    - chronological pre-cutoff validation
    - frozen holdout reported separately
    - attempt-weighted AUC
    - same-event updates delayed to prevent leakage

This is measurement only. It changes no FSR traits.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.common.paths import MASTER_PATH, FSR_V2_PREFIGHT_SNAPSHOTS_PATH
from pipeline.fsr_v2.sources.round_stats import load_round_stats, build_paired_rounds
from pipeline.fsr_v2.replay.engine import aggregate_fights


def logistic(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1-p))


def auc(y, score):
    y = np.asarray(y, int)
    score = np.asarray(score, float)

    pos = y == 1
    neg = y == 0

    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan

    ranks = pd.Series(score).rank(method="average").to_numpy()

    return (
        ranks[pos].sum()
        - pos.sum() * (pos.sum() + 1) / 2
    ) / (pos.sum() * neg.sum())


def pooled_ratio(history, num, den, last_n=None):
    h = history[-last_n:] if last_n else history

    numerator = sum(x[num] for x in h)
    denominator = sum(x[den] for x in h)

    return numerator / denominator if denominator > 0 else np.nan


def pooled_rate(history, num, sec="fight_elapsed_seconds", last_n=None):
    h = history[-last_n:] if last_n else history

    numerator = sum(x[num] for x in h)
    seconds = sum(x[sec] for x in h)

    return numerator * 900.0 / seconds if seconds > 0 else np.nan


def most_recent_with_attempt(history):
    for x in reversed(history):
        if x["td_attempted"] > 0:
            return x
    return None


def state_features(history, prefix):
    if not history:
        return {
            f"{prefix}_prior_fights": 0.0,
            f"{prefix}_prior_td_attempts": 0.0,
            f"{prefix}_last_td_success": np.nan,
            f"{prefix}_career_td_success": np.nan,
            f"{prefix}_recent3_td_success": np.nan,
            f"{prefix}_recent5_td_success": np.nan,
            f"{prefix}_recent3_minus_career_td_success": np.nan,
            f"{prefix}_last_td_attempt_rate": np.nan,
            f"{prefix}_recent3_td_attempt_rate": np.nan,
            f"{prefix}_career_td_attempt_rate": np.nan,
            f"{prefix}_last_ground_entry_rate": np.nan,
            f"{prefix}_recent3_ground_entry_rate": np.nan,
            f"{prefix}_career_ground_entry_rate": np.nan,
            f"{prefix}_last_control_per_entry": np.nan,
            f"{prefix}_recent3_control_per_entry": np.nan,
            f"{prefix}_career_control_per_entry": np.nan,
            f"{prefix}_career_td_stop_rate": np.nan,
            f"{prefix}_recent3_td_stop_rate": np.nan,
            f"{prefix}_last_td_stop_rate": np.nan,
            f"{prefix}_career_td_faced": 0.0,
            f"{prefix}_recent3_control_suffered_per_entry": np.nan,
            f"{prefix}_career_control_suffered_per_entry": np.nan,
        }

    last = history[-1]
    recent_td = most_recent_with_attempt(history)

    career_success = pooled_ratio(
        history,
        "td_landed",
        "td_attempted",
    )

    recent3_success = pooled_ratio(
        history,
        "td_landed",
        "td_attempted",
        3,
    )

    recent5_success = pooled_ratio(
        history,
        "td_landed",
        "td_attempted",
        5,
    )

    last_td_success = (
        recent_td["td_landed"] / recent_td["td_attempted"]
        if recent_td is not None
        else np.nan
    )

    def control_per_entry(h, last_n=None):
        q = h[-last_n:] if last_n else h
        ctrl = sum(x["qualified_control_inflicted_seconds"] for x in q)
        entries = sum(x["ground_entries"] for x in q)
        return ctrl / entries if entries > 0 else np.nan

    def suffered_control_per_entry(h, last_n=None):
        q = h[-last_n:] if last_n else h
        ctrl = sum(x["qualified_control_suffered_seconds"] for x in q)
        entries = sum(x["opponent_ground_entries"] for x in q)
        return ctrl / entries if entries > 0 else np.nan

    def stop_rate(h, last_n=None):
        q = h[-last_n:] if last_n else h
        faced = sum(x["opponent_td_attempted"] for x in q)
        landed = sum(x["opponent_td_landed"] for x in q)
        return (faced-landed) / faced if faced > 0 else np.nan

    last_faced = last["opponent_td_attempted"]
    last_landed_allowed = last["opponent_td_landed"]

    last_stop = (
        (last_faced-last_landed_allowed) / last_faced
        if last_faced > 0 else np.nan
    )

    return {
        f"{prefix}_prior_fights":
            float(len(history)),

        f"{prefix}_prior_td_attempts":
            float(sum(x["td_attempted"] for x in history)),

        f"{prefix}_last_td_success":
            last_td_success,

        f"{prefix}_career_td_success":
            career_success,

        f"{prefix}_recent3_td_success":
            recent3_success,

        f"{prefix}_recent5_td_success":
            recent5_success,

        f"{prefix}_recent3_minus_career_td_success":
            (
                recent3_success-career_success
                if np.isfinite(recent3_success)
                and np.isfinite(career_success)
                else np.nan
            ),

        f"{prefix}_last_td_attempt_rate":
            (
                last["td_attempted"] * 900.0
                / last["fight_elapsed_seconds"]
                if last["fight_elapsed_seconds"] > 0
                else np.nan
            ),

        f"{prefix}_recent3_td_attempt_rate":
            pooled_rate(history, "td_attempted", last_n=3),

        f"{prefix}_career_td_attempt_rate":
            pooled_rate(history, "td_attempted"),

        f"{prefix}_last_ground_entry_rate":
            (
                last["ground_entries"] * 900.0
                / last["fight_elapsed_seconds"]
                if last["fight_elapsed_seconds"] > 0
                else np.nan
            ),

        f"{prefix}_recent3_ground_entry_rate":
            pooled_rate(history, "ground_entries", last_n=3),

        f"{prefix}_career_ground_entry_rate":
            pooled_rate(history, "ground_entries"),

        f"{prefix}_last_control_per_entry":
            (
                last["qualified_control_inflicted_seconds"]
                / last["ground_entries"]
                if last["ground_entries"] > 0
                else np.nan
            ),

        f"{prefix}_recent3_control_per_entry":
            control_per_entry(history, 3),

        f"{prefix}_career_control_per_entry":
            control_per_entry(history),

        f"{prefix}_career_td_stop_rate":
            stop_rate(history),

        f"{prefix}_recent3_td_stop_rate":
            stop_rate(history, 3),

        f"{prefix}_last_td_stop_rate":
            last_stop,

        f"{prefix}_career_td_faced":
            float(
                sum(
                    x["opponent_td_attempted"]
                    for x in history
                )
            ),

        f"{prefix}_recent3_control_suffered_per_entry":
            suffered_control_per_entry(history, 3),

        f"{prefix}_career_control_suffered_per_entry":
            suffered_control_per_entry(history),
    }


def build_state_frame():
    fights = aggregate_fights(
        build_paired_rounds(
            rounds=load_round_stats()
        )
    ).copy()

    fights["fight_id"] = fights["fight_id"].astype(str)
    fights["fighter_id"] = fights["fighter_id"].astype(str)
    fights["opponent_id"] = fights["opponent_id"].astype(str)

    fights["event_date"] = pd.to_datetime(
        fights["event_date"]
    ).dt.normalize()

    histories = defaultdict(list)
    rows = []

    for date, batch in fights.groupby("event_date", sort=True):
        pending = []

        for r in batch.itertuples():
            rec = r._asdict()

            attacker_history = histories[str(r.fighter_id)]
            defender_history = histories[str(r.opponent_id)]

            features = {
                **state_features(attacker_history, "att"),
                **state_features(defender_history, "def"),
            }

            features.update({
                "fight_id": str(r.fight_id),
                "event_date": date,
                "fighter_id": str(r.fighter_id),
                "fighter_name": r.fighter_name,
                "opponent_id": str(r.opponent_id),
                "opponent_name": r.opponent_name,
                "td_landed": float(r.td_landed),
                "td_attempted": float(r.td_attempted),
            })

            rows.append(features)
            pending.append(rec)

        # Same-event delayed update.
        for rec in pending:
            histories[str(rec["fighter_id"])].append(rec)

    return pd.DataFrame(rows)


def attach_fsr(x):
    fsr = pd.read_parquet(
        FSR_V2_PREFIGHT_SNAPSHOTS_PATH
    ).copy()

    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    fsr["event_date"] = pd.to_datetime(
        fsr["event_date"]
    ).dt.normalize()

    attacker = fsr[
        [
            "fight_id",
            "event_date",
            "fighter_id",
            "takedown_offense",
            "takedown_completion_baseline",
        ]
    ]

    x = x.merge(
        attacker,
        on=["fight_id", "event_date", "fighter_id"],
        how="inner",
        validate="one_to_one",
    )

    defender = fsr[
        [
            "fight_id",
            "event_date",
            "fighter_id",
            "takedown_defense",
        ]
    ].rename(
        columns={
            "fighter_id": "opponent_id",
            "takedown_defense": "opponent_takedown_defense",
        }
    )

    x = x.merge(
        defender,
        on=["fight_id", "event_date", "opponent_id"],
        how="left",
        validate="one_to_one",
    )

    x["baseline_probability"] = logistic(
        logit(x["takedown_completion_baseline"])
        + x["takedown_offense"]
        - x["opponent_takedown_defense"]
    )

    x["baseline_logit"] = logit(
        x["baseline_probability"]
    )

    return x


def attach_context(x):
    master = pd.read_parquet(MASTER_PATH).copy()

    master["fight_id"] = master["fight_id"].astype(str)
    master["event_date"] = pd.to_datetime(
        master["date"]
    ).dt.normalize()

    red = master[
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

    blue = master[
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

    corners = pd.concat(
        [red, blue],
        ignore_index=True,
    )

    corners["fighter_id"] = corners["fighter_id"].astype(str)
    corners["dob"] = pd.to_datetime(corners["dob"], errors="coerce")

    x = x.merge(
        corners,
        on=["fight_id", "event_date", "fighter_id"],
        how="left",
        validate="one_to_one",
    )

    x["att_age"] = (
        (x["event_date"]-x["dob"]).dt.days
        / 365.2425
    )

    # Layoff from prior event.
    prev = (
        x.sort_values(["fighter_id", "event_date"])
        .groupby("fighter_id")["event_date"]
        .shift(1)
    )

    x["att_layoff_days"] = (
        x["event_date"] - prev
    ).dt.days.astype(float)

    return x


def expand_attempts(x):
    rows = []

    feature_cols = [
        c for c in x.columns
        if c.startswith("att_") or c.startswith("def_")
    ]

    for r in x.itertuples():
        attempts = int(round(r.td_attempted))
        landed = int(round(r.td_landed))

        if attempts <= 0:
            continue

        labels = [1]*landed + [0]*(attempts-landed)

        common = {
            "fight_id": r.fight_id,
            "event_date": r.event_date,
            "baseline_probability": r.baseline_probability,
            "baseline_logit": r.baseline_logit,
        }

        for c in feature_cols:
            common[c] = getattr(r, c)

        for y in labels:
            rows.append({
                **common,
                "y": y,
            })

    return pd.DataFrame(rows)


def fit_predict(train, test, feature):
    baseline_model = LogisticRegression(
        C=1e6,
        max_iter=1000,
    )

    baseline_model.fit(
        train[["baseline_logit"]],
        train["y"],
    )

    p0 = baseline_model.predict_proba(
        test[["baseline_logit"]]
    )[:, 1]

    candidate_model = Pipeline([
        (
            "impute",
            SimpleImputer(strategy="median"),
        ),
        (
            "scale",
            StandardScaler(),
        ),
        (
            "model",
            LogisticRegression(
                C=1.0,
                max_iter=1000,
            ),
        ),
    ])

    candidate_model.fit(
        train[
            [
                "baseline_logit",
                feature,
            ]
        ],
        train["y"],
    )

    p1 = candidate_model.predict_proba(
        test[
            [
                "baseline_logit",
                feature,
            ]
        ]
    )[:, 1]

    return (
        auc(test.y, p0),
        auc(test.y, p1),
    )


def temporal_scan(d, cutoff):
    train = d[
        d.event_date < cutoff
    ].copy()

    dates = np.array(
        sorted(train.event_date.unique())
    )

    cutpoints = [
        int(len(dates)*.50),
        int(len(dates)*.625),
        int(len(dates)*.75),
        int(len(dates)*.875),
    ]

    excluded = {
        "att_age",
        "att_layoff_days",
    }

    features = [
        c for c in d.columns
        if (
            c.startswith("att_")
            or c.startswith("def_")
        )
    ]

    results = []

    for feature in features:
        deltas = []

        for i, start in enumerate(cutpoints):
            end = (
                cutpoints[i+1]
                if i+1 < len(cutpoints)
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

            if len(tr) < 500 or len(va) < 100:
                continue

            a0, a1 = fit_predict(
                tr,
                va,
                feature,
            )

            deltas.append(a1-a0)

        results.append({
            "feature": feature,
            "folds": len(deltas),
            "mean_delta_auc": np.mean(deltas),
            "median_delta_auc": np.median(deltas),
            "positive_fold_share": np.mean(
                np.asarray(deltas) > 0
            ),
        })

    return (
        pd.DataFrame(results)
        .sort_values(
            "mean_delta_auc",
            ascending=False,
        )
        .reset_index(drop=True)
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

    print("building leakage-safe prior-state frame...")

    x = build_state_frame()
    x = attach_fsr(x)
    x = attach_context(x)

    attempts = expand_attempts(x)

    cv = temporal_scan(
        attempts,
        cutoff,
    )

    print("=" * 112)
    print(
        "TD COMPLETION — PRIOR-STATE SIGNAL SCAN"
    )
    print("=" * 112)

    print("\nPRE-CUTOFF TEMPORAL VALIDATION — TOP SIGNALS")

    print(
        cv.head(20).to_string(
            index=False,
            formatters={
                "mean_delta_auc":
                    lambda v: f"{v:+.4f}",
                "median_delta_auc":
                    lambda v: f"{v:+.4f}",
                "positive_fold_share":
                    lambda v: f"{v:.2f}",
            },
        )
    )

    hold = pd.read_csv(args.holdout)
    holdout_ids = set(
        hold["bout_id"].astype(str)
    )

    test = attempts[
        attempts.fight_id.astype(str).isin(
            holdout_ids
        )
    ].copy()

    train = attempts[
        attempts.event_date < cutoff
    ].copy()

    print("\nFROZEN 500-FIGHT HOLDOUT — TOP 10 TRAINING-SELECTED SIGNALS")

    print(
        f"{'feature':48s}"
        f"{'baseline':>11s}"
        f"{'+feature':>11s}"
        f"{'delta':>10s}"
    )

    for feature in cv.head(10).feature:

        a0, a1 = fit_predict(
            train,
            test,
            feature,
        )

        print(
            f"{feature:48s}"
            f"{a0:11.4f}"
            f"{a1:11.4f}"
            f"{a1-a0:+10.4f}"
        )


if __name__ == "__main__":
    main()
