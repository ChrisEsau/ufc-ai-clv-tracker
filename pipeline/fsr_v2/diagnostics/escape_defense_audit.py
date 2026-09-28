from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.fsr_v2.sources.round_stats import load_round_stats, build_paired_rounds
from pipeline.fsr_v2.replay.engine import ReplayEngine, aggregate_fights
from pipeline.fsr_v2.traits.registry import GROUPS


def safe_div(n, d):
    return np.where(d > 0, n / d, np.nan)


def main():
    fights = aggregate_fights(
        build_paired_rounds(rounds=load_round_stats())
    ).copy()

    fights["event_date"] = pd.to_datetime(fights["event_date"])
    fights["fight_id"] = fights["fight_id"].astype(str)
    fights["fighter_id"] = fights["fighter_id"].astype(str)

    replay = ReplayEngine().replay(
        GROUPS["escape_effectiveness"],
        fights,
    ).history.copy()

    replay["event_date"] = pd.to_datetime(replay["event_date"])
    replay["fight_id"] = replay["fight_id"].astype(str)
    replay["fighter_id"] = replay["fighter_id"].astype(str)

    defense = replay[
        replay["trait"] == "escape_defense"
    ][
        [
            "event_date",
            "fight_id",
            "fighter_id",
            "fighter_name",
            "opponent_name",
            "pre_rating",
            "population_duration_baseline_seconds",
        ]
    ].rename(
        columns={
            "pre_rating": "escape_defense",
            "population_duration_baseline_seconds": "population_mean_seconds",
        }
    )

    obs = fights[
        [
            "event_date",
            "fight_id",
            "fighter_id",
            "ground_entries",
            "ground_attempted",
            "effective_submission_attempts",
            "qualified_control_inflicted_seconds",
        ]
    ].copy()

    x = defense.merge(
        obs,
        on=["event_date", "fight_id", "fighter_id"],
        how="left",
        validate="one_to_one",
    )

    # Only fights where fighter actually generated a ground entry.
    x = x[x["ground_entries"] > 0].copy()

    x["ground_strikes_per_entry"] = safe_div(
        x["ground_attempted"],
        x["ground_entries"],
    )

    x["subs_per_entry"] = safe_div(
        x["effective_submission_attempts"],
        x["ground_entries"],
    )

    x["ground_activity"] = (
        x["ground_attempted"]
        + x["effective_submission_attempts"]
    )

    x["ground_activity_per_entry"] = safe_div(
        x["ground_activity"],
        x["ground_entries"],
    )

    x["control_per_entry"] = safe_div(
        x["qualified_control_inflicted_seconds"],
        x["ground_entries"],
    )

    # Simple activity-based proxies for sustained ground sequences.
    x["entry_5plus_actions"] = (
        x["ground_activity_per_entry"] >= 5
    ).astype(int)

    x["entry_8plus_actions"] = (
        x["ground_activity_per_entry"] >= 8
    ).astype(int)

    print("=" * 105)
    print("ESCAPE DEFENSE AUDIT")
    print("=" * 105)

    print(f"\nobservations: {len(x)}")

    print("\nCORRELATION WITH FUTURE OUTCOMES")

    cols = [
        "control_per_entry",
        "ground_strikes_per_entry",
        "subs_per_entry",
        "ground_activity_per_entry",
        "entry_5plus_actions",
        "entry_8plus_actions",
    ]

    for col in cols:
        pearson = x["escape_defense"].corr(x[col])
        spearman = x["escape_defense"].corr(
            x[col],
            method="spearman",
        )

        print(
            f"{col:30s} "
            f"Pearson={pearson: .3f} "
            f"Spearman={spearman: .3f}"
        )

    print("\n" + "=" * 105)
    print("ESCAPE DEFENSE DECILES")
    print("=" * 105)

    x["decile"] = pd.qcut(
        x["escape_defense"],
        10,
        duplicates="drop",
    )

    summary = x.groupby(
        "decile",
        observed=True,
    ).agg(
        rows=("fighter_id", "size"),
        rating=("escape_defense", "mean"),
        entries=("ground_entries", "mean"),
        control_per_entry=("control_per_entry", "mean"),
        ground_strikes_per_entry=("ground_strikes_per_entry", "mean"),
        subs_per_entry=("subs_per_entry", "mean"),
        activity_per_entry=("ground_activity_per_entry", "mean"),
        pct_5plus_actions=("entry_5plus_actions", "mean"),
        pct_8plus_actions=("entry_8plus_actions", "mean"),
    )

    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 105)
    print("LATEST BEST ESCAPE DEFENSE")
    print("=" * 105)

    latest = replay[
        (replay["trait"] == "escape_defense")
        & replay["latest_rating"].notna()
    ].copy()

    print(
        latest.nlargest(30, "latest_rating")[
            ["fighter_name", "latest_rating"]
        ].to_string(index=False)
    )

    print("\n" + "=" * 105)
    print("LATEST WORST ESCAPE DEFENSE")
    print("=" * 105)

    print(
        latest.nsmallest(30, "latest_rating")[
            ["fighter_name", "latest_rating"]
        ].to_string(index=False)
    )

    x.to_csv(
        "/tmp/escape_defense_audit.csv",
        index=False,
    )

    print("\nwrote: /tmp/escape_defense_audit.csv")


if __name__ == "__main__":
    main()
