from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.fsr_v2.sources.round_stats import load_round_stats, build_paired_rounds
from pipeline.fsr_v2.replay.engine import ReplayEngine, aggregate_fights
from pipeline.fsr_v2.traits.registry import GROUPS


def safe_div(n, d):
    return np.where(d > 0, n / d, np.nan)


def corr(a, b):
    return (
        a.corr(b),
        a.corr(b, method="spearman"),
    )


def main():
    # ---------------------------------------------------------------
    # Historical fighter-fight observations
    # ---------------------------------------------------------------
    fights = aggregate_fights(
        build_paired_rounds(rounds=load_round_stats())
    ).copy()

    fights["event_date"] = pd.to_datetime(fights["event_date"])
    fights["fight_id"] = fights["fight_id"].astype(str)
    fights["fighter_id"] = fights["fighter_id"].astype(str)
    fights["opponent_id"] = fights["opponent_id"].astype(str)

    # ---------------------------------------------------------------
    # Leakage-safe prefight escape ratings
    # ---------------------------------------------------------------
    replay = ReplayEngine().replay(
        GROUPS["escape_effectiveness"],
        fights,
    ).history.copy()

    replay["event_date"] = pd.to_datetime(replay["event_date"])
    replay["fight_id"] = replay["fight_id"].astype(str)
    replay["fighter_id"] = replay["fighter_id"].astype(str)
    replay["opponent_id"] = replay["opponent_id"].astype(str)

    offense = (
        replay[replay["trait"] == "escape_offense"][
            [
                "event_date",
                "fight_id",
                "fighter_id",
                "opponent_id",
                "fighter_name",
                "opponent_name",
                "pre_rating",
                "population_duration_baseline_seconds",
            ]
        ]
        .rename(columns={
            "pre_rating": "escape_offense",
            "population_duration_baseline_seconds": "population_mean_seconds",
        })
    )

    defense = (
        replay[replay["trait"] == "escape_defense"][
            [
                "event_date",
                "fight_id",
                "fighter_id",
                "pre_rating",
            ]
        ]
        .rename(columns={"pre_rating": "escape_defense"})
    )

    ratings = offense.merge(
        defense,
        on=["event_date", "fight_id", "fighter_id"],
        how="left",
        validate="one_to_one",
    )

    # Opponent ratings
    opp = ratings[
        [
            "event_date",
            "fight_id",
            "fighter_id",
            "escape_offense",
            "escape_defense",
        ]
    ].rename(columns={
        "fighter_id": "opponent_id",
        "escape_offense": "opponent_escape_offense",
        "escape_defense": "opponent_escape_defense",
    })

    ratings = ratings.merge(
        opp,
        on=["event_date", "fight_id", "opponent_id"],
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------------
    # Realized ground/submission outcomes
    # ---------------------------------------------------------------
    obs = fights[
        [
            "event_date",
            "fight_id",
            "fighter_id",
            "ground_entries",
            "effective_submission_attempts",
            "ground_attempted",
            "qualified_control_inflicted_seconds",
        ]
    ].copy()

    x = ratings.merge(
        obs,
        on=["event_date", "fight_id", "fighter_id"],
        how="left",
        validate="one_to_one",
    )

    # Only evaluate fights where this fighter actually reached ground.
    x = x[x["ground_entries"] > 0].copy()

    x["subs_per_entry"] = safe_div(
        x["effective_submission_attempts"],
        x["ground_entries"],
    )

    x["ground_strikes_per_entry"] = safe_div(
        x["ground_attempted"],
        x["ground_entries"],
    )

    # ---------------------------------------------------------------
    # Exact residence mean implied by current MC transform:
    #
    # bottom opponent escape offense
    # versus
    # top fighter escape defense
    # ---------------------------------------------------------------
    x["predicted_residence_seconds"] = (
        x["population_mean_seconds"]
        * np.exp(
            -x["opponent_escape_offense"]
            + x["escape_defense"]
        )
    )

    print("=" * 110)
    print("ESCAPE / GROUND RETENTION vs SUBMISSION ATTEMPTS")
    print("=" * 110)

    print(f"\nfighter-fights with ground entry: {len(x)}")

    # ---------------------------------------------------------------
    # Direct correlations
    # ---------------------------------------------------------------
    print("\nCORRELATION WITH SUBMISSION ATTEMPTS")

    tests = [
        ("escape defense", "escape_defense"),
        ("opponent escape offense", "opponent_escape_offense"),
        ("predicted residence sec", "predicted_residence_seconds"),
    ]

    for label, col in tests:
        p1, s1 = corr(
            x[col],
            x["effective_submission_attempts"],
        )

        p2, s2 = corr(
            x[col],
            x["subs_per_entry"],
        )

        print(f"\n{label}")
        print(
            f"  total SUB attempts : "
            f"Pearson={p1: .3f}  Spearman={s1: .3f}"
        )
        print(
            f"  SUB attempts/entry : "
            f"Pearson={p2: .3f}  Spearman={s2: .3f}"
        )

    # ---------------------------------------------------------------
    # Escape-defense deciles
    # ---------------------------------------------------------------
    print("\n" + "=" * 110)
    print("ESCAPE DEFENSE DECILES")
    print("=" * 110)

    x["defense_decile"] = pd.qcut(
        x["escape_defense"],
        10,
        duplicates="drop",
    )

    defense_summary = (
        x.groupby("defense_decile", observed=True)
        .agg(
            rows=("fighter_id", "size"),
            rating=("escape_defense", "mean"),
            ground_entries=("ground_entries", "mean"),
            submission_attempts=("effective_submission_attempts", "mean"),
            subs_per_entry=("subs_per_entry", "mean"),
            ground_strikes_per_entry=("ground_strikes_per_entry", "mean"),
            predicted_residence=("predicted_residence_seconds", "mean"),
        )
    )

    print(
        defense_summary.to_string(
            float_format=lambda v: f"{v:.3f}"
        )
    )

    # ---------------------------------------------------------------
    # Predicted-residence deciles
    # ---------------------------------------------------------------
    print("\n" + "=" * 110)
    print("MATCHUP-PREDICTED GROUND RESIDENCE DECILES")
    print("=" * 110)

    x["residence_decile"] = pd.qcut(
        x["predicted_residence_seconds"],
        10,
        duplicates="drop",
    )

    residence_summary = (
        x.groupby("residence_decile", observed=True)
        .agg(
            rows=("fighter_id", "size"),
            predicted_seconds=("predicted_residence_seconds", "mean"),
            ground_entries=("ground_entries", "mean"),
            submission_attempts=("effective_submission_attempts", "mean"),
            subs_per_entry=("subs_per_entry", "mean"),
            ground_strikes_per_entry=("ground_strikes_per_entry", "mean"),
        )
    )

    print(
        residence_summary.to_string(
            float_format=lambda v: f"{v:.3f}"
        )
    )

    # ---------------------------------------------------------------
    # Submission-active versus no-submission observations
    # ---------------------------------------------------------------
    print("\n" + "=" * 110)
    print("SUBMISSION-ACTIVE vs NO-SUBMISSION GROUND SEQUENCES")
    print("=" * 110)

    x["had_submission_attempt"] = (
        x["effective_submission_attempts"] > 0
    )

    comparison = (
        x.groupby("had_submission_attempt")
        .agg(
            rows=("fighter_id", "size"),
            escape_defense=("escape_defense", "mean"),
            opponent_escape_offense=("opponent_escape_offense", "mean"),
            predicted_residence=("predicted_residence_seconds", "mean"),
            ground_entries=("ground_entries", "mean"),
            ground_strikes=("ground_attempted", "mean"),
        )
    )

    print(
        comparison.to_string(
            float_format=lambda v: f"{v:.3f}"
        )
    )

    x.to_csv(
        "/tmp/escape_vs_submission_attempts.csv",
        index=False,
    )

    print(
        "\nwrote: /tmp/escape_vs_submission_attempts.csv"
    )


if __name__ == "__main__":
    main()
