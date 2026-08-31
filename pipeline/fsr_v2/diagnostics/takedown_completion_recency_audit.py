"""Leakage-safe recency audit for offensive TD completion.

Keeps current FSR V2 takedown defense unchanged.

Candidate offensive rate:
    exponentially weighted prior TD landed / attempted
    + current 10-attempt population prior.

Candidate half-life is selected on pre-cutoff temporal validation only,
then evaluated once on the frozen holdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import FSR_V2_PREFIGHT_SNAPSHOTS_PATH
from pipeline.fsr_v2.sources.round_stats import load_round_stats, build_paired_rounds
from pipeline.fsr_v2.replay.engine import aggregate_fights


HALF_LIVES = [1, 2, 3, 5, 8, 12, 20]
PRIOR_ATTEMPTS = 10.0


def logistic(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1-p))


def auc(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)

    pos = y == 1
    neg = y == 0

    if not pos.any() or not neg.any():
        return np.nan

    ranks = pd.Series(p).rank(method="average").to_numpy()

    return (
        ranks[pos].sum()
        - pos.sum() * (pos.sum() + 1) / 2
    ) / (pos.sum() * neg.sum())


def weighted_prior(history, half_life):
    """Most recent prior fight has lag=0."""
    if not history:
        return 0.0, 0.0

    landed = np.array(
        [x[0] for x in history[::-1]],
        float,
    )
    attempted = np.array(
        [x[1] for x in history[::-1]],
        float,
    )

    lag = np.arange(len(history), dtype=float)

    weights = 0.5 ** (lag / half_life)

    return (
        float(np.sum(weights * landed)),
        float(np.sum(weights * attempted)),
    )


def build_frame():
    fights = aggregate_fights(
        build_paired_rounds(rounds=load_round_stats())
    ).copy()

    fights["fight_id"] = fights["fight_id"].astype(str)
    fights["fighter_id"] = fights["fighter_id"].astype(str)
    fights["opponent_id"] = fights["opponent_id"].astype(str)
    fights["event_date"] = pd.to_datetime(
        fights["event_date"]
    ).dt.normalize()

    fsr = pd.read_parquet(
        FSR_V2_PREFIGHT_SNAPSHOTS_PATH
    ).copy()

    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    fsr["event_date"] = pd.to_datetime(
        fsr["event_date"]
    ).dt.normalize()

    cols = [
        "fight_id",
        "event_date",
        "fighter_id",
        "takedown_offense",
        "takedown_defense",
        "takedown_completion_baseline",
    ]

    x = fights.merge(
        fsr[cols],
        on=["fight_id", "event_date", "fighter_id"],
        how="inner",
        validate="one_to_one",
    )

    defender = fsr[cols].rename(
        columns={
            "fighter_id": "opponent_id",
            "takedown_defense": "opponent_takedown_defense",
        }
    )[
        [
            "fight_id",
            "event_date",
            "opponent_id",
            "opponent_takedown_defense",
        ]
    ]

    x = x.merge(
        defender,
        on=["fight_id", "event_date", "opponent_id"],
        how="left",
        validate="one_to_one",
    )

    return x.sort_values(
        ["event_date", "fight_id", "fighter_id"]
    ).reset_index(drop=True)


def add_recency_predictions(x):
    histories = {}
    rows = []

    # Same-date delayed updates.
    for event_date, batch in x.groupby("event_date", sort=True):
        pending = []

        for r in batch.itertuples():
            fighter = str(r.fighter_id)

            history = histories.get(fighter, [])

            base = float(r.takedown_completion_baseline)
            base_logit = logit(base)

            current_full = logistic(
                base_logit
                + float(r.takedown_offense)
                - float(r.opponent_takedown_defense)
            )

            row = {
                "fight_id": str(r.fight_id),
                "event_date": event_date,
                "fighter_id": fighter,
                "td_landed": float(r.td_landed),
                "td_attempted": float(r.td_attempted),
                "current_full": float(current_full),
            }

            for half_life in HALF_LIVES:
                wl, wa = weighted_prior(
                    history,
                    half_life,
                )

                rate = (
                    wl + base * PRIOR_ATTEMPTS
                ) / (
                    wa + PRIOR_ATTEMPTS
                )

                offense_delta = (
                    logit(rate)
                    - base_logit
                )

                row[f"hl_{half_life}"] = float(
                    logistic(
                        base_logit
                        + offense_delta
                        - float(r.opponent_takedown_defense)
                    )
                )

            rows.append(row)

            pending.append(
                (
                    fighter,
                    float(r.td_landed),
                    float(r.td_attempted),
                )
            )

        for fighter, landed, attempted in pending:
            histories.setdefault(
                fighter,
                []
            ).append(
                (landed, attempted)
            )

    return pd.DataFrame(rows)


def expand_attempts(x):
    rows = []

    prediction_cols = [
        "current_full",
        *[f"hl_{h}" for h in HALF_LIVES],
    ]

    for r in x.itertuples():
        attempts = int(round(r.td_attempted))
        landed = int(round(r.td_landed))

        if attempts <= 0:
            continue

        labels = (
            [1] * landed
            + [0] * (attempts - landed)
        )

        common = {
            c: getattr(r, c)
            for c in prediction_cols
        }

        for y in labels:
            rows.append({
                "fight_id": r.fight_id,
                "event_date": r.event_date,
                "y": y,
                **common,
            })

    return pd.DataFrame(rows)


def temporal_validation(d, cutoff):
    train = d[
        d.event_date < cutoff
    ].copy()

    dates = np.array(
        sorted(train.event_date.unique())
    )

    cutpoints = [
        int(len(dates) * .50),
        int(len(dates) * .625),
        int(len(dates) * .75),
        int(len(dates) * .875),
    ]

    results = []

    for half_life in HALF_LIVES:
        deltas = []

        for i, start in enumerate(cutpoints):
            end = (
                cutpoints[i+1]
                if i+1 < len(cutpoints)
                else len(dates)
            )

            val_dates = dates[start:end]

            z = train[
                train.event_date.isin(val_dates)
            ]

            if len(z) < 100:
                continue

            baseline = auc(
                z.y,
                z.current_full,
            )

            candidate = auc(
                z.y,
                z[f"hl_{half_life}"],
            )

            deltas.append(
                candidate - baseline
            )

        results.append({
            "half_life_fights": half_life,
            "folds": len(deltas),
            "mean_delta_auc": np.mean(deltas),
            "median_delta_auc": np.median(deltas),
            "positive_fold_share": np.mean(
                np.asarray(deltas) > 0
            ),
        })

    return pd.DataFrame(results)


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

    full = add_recency_predictions(
        build_frame()
    )

    attempts = expand_attempts(full)

    cv = temporal_validation(
        attempts,
        cutoff,
    )

    print("=" * 100)
    print(
        "FSR V2 — TD OFFENSIVE COMPLETION RECENCY AUDIT"
    )
    print("=" * 100)

    print("\nPRE-CUTOFF TEMPORAL VALIDATION")

    print(
        cv.to_string(
            index=False,
            formatters={
                "mean_delta_auc":
                    lambda x: f"{x:+.4f}",
                "median_delta_auc":
                    lambda x: f"{x:+.4f}",
                "positive_fold_share":
                    lambda x: f"{x:.2f}",
            },
        )
    )

    chosen = int(
        cv.sort_values(
            "mean_delta_auc",
            ascending=False,
        ).iloc[0].half_life_fights
    )

    print(
        f"\nSelected half-life from training only: "
        f"{chosen} fights"
    )

    # Frozen fight IDs.
    holdout = pd.read_csv(args.holdout)
    holdout_ids = set(
        holdout["bout_id"].astype(str)
    )

    test = attempts[
        attempts.fight_id.isin(
            holdout_ids
        )
    ].copy()

    baseline_auc = auc(
        test.y,
        test.current_full,
    )

    candidate_auc = auc(
        test.y,
        test[f"hl_{chosen}"],
    )

    print("\nFROZEN 500-FIGHT HOLDOUT")

    print(
        f"Current offense + defense: "
        f"AUC={baseline_auc:.4f}"
    )

    print(
        f"Recency offense + current defense: "
        f"AUC={candidate_auc:.4f}"
    )

    print(
        f"Delta AUC: "
        f"{candidate_auc-baseline_auc:+.4f}"
    )


if __name__ == "__main__":
    main()
