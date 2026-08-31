"""Compare current TD effectiveness traits with opponent-adjusted ratings.

Uses the existing leakage-safe generic paired replay machinery as an
experimental opponent-adjusted TD completion model.

Frozen holdout only for final comparison.
No simulator changes.
No global calibration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.fsr_v2.sources.round_stats import (
    load_round_stats,
    build_paired_rounds,
)
from pipeline.fsr_v2.replay.engine import (
    ReplayEngine,
    aggregate_fights,
)
from pipeline.fsr_v2.traits.registry import TraitGroup


def logistic(x):
    return 1.0 / (
        1.0 + np.exp(-np.clip(x, -40, 40))
    )


def logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p / (1-p))


def auc(y, score):
    y = np.asarray(y, int)
    score = np.asarray(score, float)

    pos = y == 1
    neg = y == 0

    n_pos = pos.sum()
    n_neg = neg.sum()

    ranks = (
        pd.Series(score)
        .rank(method="average")
        .to_numpy()
    )

    return (
        ranks[pos].sum()
        - n_pos * (n_pos + 1) / 2
    ) / (n_pos * n_neg)


def quintile_lift(y, p):
    z = pd.DataFrame({
        "y": y,
        "p": p,
    })

    z["bucket"] = pd.qcut(
        z.p.rank(method="first"),
        5,
        labels=False,
    )

    rates = z.groupby("bucket").y.mean()

    return (
        float(rates.iloc[-1]),
        float(rates.iloc[0]),
        float(
            rates.iloc[-1]
            - rates.iloc[0]
        ),
    )


def opponent_adjusted_history():
    fights = aggregate_fights(
        build_paired_rounds(
            rounds=load_round_stats()
        )
    )

    group = TraitGroup(
        name="td_completion_opponent_adjusted_audit",
        kind="paired",
        traits=(
            "oa_takedown_offense",
            "oa_takedown_defense",
        ),
        numerator="td_landed",
        denominator="td_attempted",
    )

    result = ReplayEngine().replay(
        group,
        fights,
    )

    h = result.history

    # Offensive rows already contain the defender's prefight
    # opponent-adjusted rating in opponent_pre_rating.
    h = h[
        h.trait.eq(
            "oa_takedown_offense"
        )
    ].copy()

    return h[
        [
            "fight_id",
            "fighter_name",
            "pre_rating",
            "opponent_pre_rating",
            "population_baseline",
            "expected",
        ]
    ].rename(
        columns={
            "pre_rating":
                "oa_offense",
            "opponent_pre_rating":
                "oa_opponent_defense",
            "population_baseline":
                "oa_baseline",
            "expected":
                "oa_full_probability",
        }
    )


def attach_oa(x):
    oa = opponent_adjusted_history()

    oa["fight_id"] = (
        oa["fight_id"].astype(str)
    )

    x["bout_id"] = (
        x["bout_id"].astype(str)
    )

    for side in ("red", "blue"):
        q = oa.rename(
            columns={
                "fight_id":
                    "bout_id",
                "fighter_name":
                    f"{side}_fighter",
                "oa_offense":
                    f"{side}_oa_offense",
                "oa_opponent_defense":
                    f"{side}_oa_opponent_defense",
                "oa_baseline":
                    f"{side}_oa_baseline",
                "oa_full_probability":
                    f"{side}_oa_full_probability",
            }
        )

        x = x.merge(
            q,
            on=[
                "bout_id",
                f"{side}_fighter",
            ],
            how="left",
            validate="one_to_one",
        )

    return x


def expand_attempts(x):
    rows = []

    for r in x.itertuples():

        for side in ("red", "blue"):

            opp = (
                "blue"
                if side == "red"
                else "red"
            )

            attempts = int(round(
                getattr(
                    r,
                    f"historical_{side}_td_attempts",
                )
            ))

            landed = int(round(
                getattr(
                    r,
                    f"historical_{side}_td_landed",
                )
            ))

            if attempts <= 0:
                continue

            baseline = float(
                getattr(
                    r,
                    f"{side}_takedown_completion_baseline",
                )
            )

            base = logit(baseline)

            current_off = float(
                getattr(
                    r,
                    f"{side}_takedown_offense",
                )
            )

            current_def = float(
                getattr(
                    r,
                    f"{opp}_takedown_defense",
                )
            )

            oa_off = float(
                getattr(
                    r,
                    f"{side}_oa_offense",
                )
            )

            oa_def = float(
                getattr(
                    r,
                    f"{side}_oa_opponent_defense",
                )
            )

            predictions = {
                "current_current":
                    logistic(
                        base
                        + current_off
                        - current_def
                    ),

                "oa_off_current_def":
                    logistic(
                        base
                        + oa_off
                        - current_def
                    ),

                "current_off_oa_def":
                    logistic(
                        base
                        + current_off
                        - oa_def
                    ),

                "oa_off_oa_def":
                    logistic(
                        base
                        + oa_off
                        - oa_def
                    ),
            }

            labels = (
                [1] * landed
                + [0] * (
                    attempts-landed
                )
            )

            for y in labels:
                rows.append({
                    "outcome": y,
                    **predictions,
                })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    x = pd.read_csv(args.csv)

    x = attach_oa(x)

    needed = [
        "red_oa_offense",
        "blue_oa_offense",
        "red_oa_opponent_defense",
        "blue_oa_opponent_defense",
    ]

    if x[needed].isna().any().any():
        raise RuntimeError(
            "Failed to resolve opponent-adjusted "
            "ratings for all holdout fights."
        )

    attempts = expand_attempts(x)

    print("=" * 100)
    print(
        "FSR V2 — TD COMPLETION OPPONENT-ADJUSTMENT AUDIT"
    )
    print("=" * 100)

    print(
        f"historical attempts={len(attempts)} | "
        f"landed={attempts.outcome.sum()} | "
        f"success={attempts.outcome.mean():.3f}"
    )

    print(
        f"\n{'model':30s}"
        f"{'AUC':>10s}"
        f"{'top20':>10s}"
        f"{'bottom20':>10s}"
        f"{'lift':>10s}"
    )

    models = [
        (
            "current_current",
            "Current offense + defense",
        ),
        (
            "oa_off_current_def",
            "Adjusted offense + current D",
        ),
        (
            "current_off_oa_def",
            "Current offense + adjusted D",
        ),
        (
            "oa_off_oa_def",
            "Adjusted offense + defense",
        ),
    ]

    for col, label in models:
        y = attempts.outcome
        p = attempts[col]

        top, bottom, lift = (
            quintile_lift(y, p)
        )

        print(
            f"{label:30s}"
            f"{auc(y,p):>10.3f}"
            f"{top:>10.3f}"
            f"{bottom:>10.3f}"
            f"{lift:>10.3f}"
        )


if __name__ == "__main__":
    main()
