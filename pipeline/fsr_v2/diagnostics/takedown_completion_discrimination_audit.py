"""Audit FSR V2 takedown-completion discrimination on the frozen holdout.

Primary question:
Do successful historical TD attempts occur in matchups assigned higher
completion probability than failed attempts?

Compares:
1. population baseline only
2. attacker offense only
3. defender defense only
4. full FSR offense + defense matchup

AUC is the primary discrimination metric.
No simulator tuning. No winner metrics.
"""

from __future__ import annotations

import argparse
from math import exp, log
from pathlib import Path

import numpy as np
import pandas as pd


def logistic(x):
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def auc(y, score):
    """AUC via average ranks; handles tied predictions."""
    y = np.asarray(y, int)
    score = np.asarray(score, float)

    pos = y == 1
    neg = y == 0

    n_pos = pos.sum()
    n_neg = neg.sum()

    if not n_pos or not n_neg:
        return np.nan

    ranks = pd.Series(score).rank(method="average").to_numpy()

    rank_sum_pos = ranks[pos].sum()

    return (
        rank_sum_pos
        - n_pos * (n_pos + 1) / 2
    ) / (n_pos * n_neg)


def log_loss(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)

    return float(
        -np.mean(
            y * np.log(p)
            + (1-y) * np.log(1-p)
        )
    )


def brier(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)

    return float(np.mean((p-y) ** 2))


def expand_attempts(x):
    rows = []

    for r in x.itertuples():
        for side in ("red", "blue"):

            attempts = int(
                round(
                    getattr(
                        r,
                        f"historical_{side}_td_attempts",
                    )
                )
            )

            landed = int(
                round(
                    getattr(
                        r,
                        f"historical_{side}_td_landed",
                    )
                )
            )

            if attempts <= 0:
                continue

            landed = min(
                max(landed, 0),
                attempts,
            )

            opponent = (
                "blue"
                if side == "red"
                else "red"
            )

            baseline = float(
                getattr(
                    r,
                    f"{side}_takedown_completion_baseline",
                )
            )

            offense = float(
                getattr(
                    r,
                    f"{side}_takedown_offense",
                )
            )

            defense = float(
                getattr(
                    r,
                    f"{opponent}_takedown_defense",
                )
            )

            base_logit = logit(baseline)

            preds = {
                "population_baseline":
                    baseline,

                "attacker_offense_only":
                    float(
                        logistic(
                            base_logit
                            + offense
                        )
                    ),

                "defender_defense_only":
                    float(
                        logistic(
                            base_logit
                            - defense
                        )
                    ),

                "full_matchup":
                    float(
                        logistic(
                            base_logit
                            + offense
                            - defense
                        )
                    ),
            }

            labels = (
                [1] * landed
                + [0] * (attempts-landed)
            )

            for y in labels:
                rows.append({
                    "bout_id": r.bout_id,
                    "side": side,
                    "outcome": y,
                    **preds,
                })

    return pd.DataFrame(rows)


def quintile_lift(y, p):
    z = pd.DataFrame({
        "y": y,
        "p": p,
    })

    z["bucket"] = pd.qcut(
        z["p"].rank(method="first"),
        5,
        labels=False,
    )

    rates = (
        z.groupby("bucket")
        .y.mean()
    )

    return (
        float(rates.iloc[-1]),
        float(rates.iloc[0]),
        float(
            rates.iloc[-1]
            - rates.iloc[0]
        ),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    x = pd.read_csv(args.csv)

    attempts = expand_attempts(x)

    print("=" * 108)
    print(
        "FSR V2 — TAKEDOWN COMPLETION DISCRIMINATION AUDIT"
    )
    print("=" * 108)

    print(
        f"fights={len(x)} | "
        f"historical TD attempts={len(attempts)} | "
        f"landed={attempts.outcome.sum()} | "
        f"success={attempts.outcome.mean():.3f}"
    )

    print(
        "\nATTEMPT-WEIGHTED DISCRIMINATION"
    )

    print(
        f"{'signal':28s}"
        f"{'AUC':>10s}"
        f"{'log loss':>12s}"
        f"{'Brier':>10s}"
        f"{'top20':>10s}"
        f"{'bottom20':>10s}"
        f"{'lift':>10s}"
    )

    for col, label in [
        (
            "population_baseline",
            "Population baseline",
        ),
        (
            "attacker_offense_only",
            "Attacker offense only",
        ),
        (
            "defender_defense_only",
            "Defender defense only",
        ),
        (
            "full_matchup",
            "Full offense + defense",
        ),
    ]:

        p = attempts[col]
        y = attempts["outcome"]

        top, bottom, lift = (
            quintile_lift(y, p)
        )

        print(
            f"{label:28s}"
            f"{auc(y,p):>10.3f}"
            f"{log_loss(y,p):>12.4f}"
            f"{brier(y,p):>10.4f}"
            f"{top:>10.3f}"
            f"{bottom:>10.3f}"
            f"{lift:>10.3f}"
        )

    print(
        "\nINTERPRETATION"
    )

    print(
        "AUC 0.500 = no completion discrimination."
    )

    print(
        "If offense-only or defense-only materially beats the full matchup, "
        "the current combination is hurting signal."
    )

    print(
        "If full matchup clearly beats both components, the structure is useful "
        "even if fighter-fight success-rate correlation looks weak."
    )


if __name__ == "__main__":
    main()
