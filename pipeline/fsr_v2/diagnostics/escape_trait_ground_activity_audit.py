from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.fsr_v2.sources.round_stats import load_round_stats, build_paired_rounds
from pipeline.fsr_v2.replay.engine import ReplayEngine, aggregate_fights
from pipeline.fsr_v2.traits.registry import GROUPS


def safe_div(n, d):
    return np.where(d > 0, n / d, np.nan)


def main():
    # ------------------------------------------------------------------
    # Historical fighter-fight observations
    # ------------------------------------------------------------------
    fights = aggregate_fights(
        build_paired_rounds(rounds=load_round_stats())
    ).copy()

    fights["event_date"] = pd.to_datetime(fights["event_date"])
    fights["fight_id"] = fights["fight_id"].astype(str)
    fights["fighter_id"] = fights["fighter_id"].astype(str)
    fights["opponent_id"] = fights["opponent_id"].astype(str)

    # ------------------------------------------------------------------
    # Leakage-safe prefight escape ratings from the canonical replay.
    # Each historical row's pre_rating is what was known BEFORE that fight.
    # ------------------------------------------------------------------
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

    # Opponent escape traits for matchup calculations.
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

    # ------------------------------------------------------------------
    # Join realized fight observations.
    #
    # Ground activity is our best direct evidence of meaningful ground
    # residence:
    #
    #   ground strike attempts + submission attempts
    #
    # We examine it per ground entry where possible.
    # ------------------------------------------------------------------
    obs = fights[
        [
            "event_date",
            "fight_id",
            "fighter_id",
            "ground_entries",
            "ground_attempted",
            "effective_submission_attempts",
            "opponent_ground_entries",
            "opponent_ground_attempted",
            "opponent_effective_submission_attempts",
            "qualified_control_inflicted_seconds",
            "qualified_control_suffered_seconds",
        ]
    ].copy()

    x = ratings.merge(
        obs,
        on=["event_date", "fight_id", "fighter_id"],
        how="left",
        validate="one_to_one",
    )

    # Fighter-generated ground activity.
    x["ground_activity_inflicted"] = (
        x["ground_attempted"]
        + x["effective_submission_attempts"]
    )

    # Opponent-generated activity while this fighter is potentially bottom.
    x["ground_activity_suffered"] = (
        x["opponent_ground_attempted"]
        + x["opponent_effective_submission_attempts"]
    )

    x["activity_inflicted_per_entry"] = safe_div(
        x["ground_activity_inflicted"],
        x["ground_entries"],
    )

    x["activity_suffered_per_entry"] = safe_div(
        x["ground_activity_suffered"],
        x["opponent_ground_entries"],
    )

    # ------------------------------------------------------------------
    # Matchup-implied residence duration.
    #
    # If THIS fighter is on top:
    #   bottom = opponent
    #   top defense = fighter
    #
    # expected mean =
    # population * exp(-bottom escape offense + top escape defense)
    # ------------------------------------------------------------------
    x["predicted_top_mean_seconds"] = (
        x["population_mean_seconds"]
        * np.exp(
            -x["opponent_escape_offense"]
            + x["escape_defense"]
        )
    )

    # If THIS fighter is on bottom:
    x["predicted_bottom_mean_seconds"] = (
        x["population_mean_seconds"]
        * np.exp(
            -x["escape_offense"]
            + x["opponent_escape_defense"]
        )
    )

    # ------------------------------------------------------------------
    # Core correlations
    # ------------------------------------------------------------------
    print("=" * 110)
    print("ESCAPE OFFENSE / DEFENSE — FUTURE GROUND-ACTIVITY AUDIT")
    print("=" * 110)

    # Only observations where the relevant fighter had ground-entry evidence.
    top = x[x["ground_entries"] > 0].copy()
    bottom = x[x["opponent_ground_entries"] > 0].copy()

    print("\nSAMPLE")
    print(f"fighter-fights total                 : {len(x)}")
    print(f"fighter-fights with own ground entry : {len(top)}")
    print(f"fighter-fights suffering entry       : {len(bottom)}")

    print("\nESCAPE DEFENSE")
    print("Higher should correspond to MORE opponent-trapping / ground activity.")

    for col in [
        "ground_activity_inflicted",
        "activity_inflicted_per_entry",
        "qualified_control_inflicted_seconds",
    ]:
        print(
            f"{col:38s} "
            f"Pearson={top.escape_defense.corr(top[col]): .3f}  "
            f"Spearman={top.escape_defense.corr(top[col], method='spearman'): .3f}"
        )

    print("\nESCAPE OFFENSE")
    print("Higher should correspond to LESS opponent ground activity suffered.")

    for col in [
        "ground_activity_suffered",
        "activity_suffered_per_entry",
        "qualified_control_suffered_seconds",
    ]:
        print(
            f"{col:38s} "
            f"Pearson={bottom.escape_offense.corr(bottom[col]): .3f}  "
            f"Spearman={bottom.escape_offense.corr(bottom[col], method='spearman'): .3f}"
        )

    print("\nMATCHUP-IMPLIED TOP RESIDENCE")
    print("Higher predicted seconds should correspond to MORE realized ground activity.")

    for col in [
        "ground_activity_inflicted",
        "activity_inflicted_per_entry",
        "qualified_control_inflicted_seconds",
    ]:
        print(
            f"{col:38s} "
            f"Pearson={top.predicted_top_mean_seconds.corr(top[col]): .3f}  "
            f"Spearman={top.predicted_top_mean_seconds.corr(top[col], method='spearman'): .3f}"
        )

    # ------------------------------------------------------------------
    # Bucket calibration
    # ------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("ESCAPE DEFENSE QUINTILES — REALIZED GROUND ACTIVITY")
    print("=" * 110)

    if len(top) >= 5:
        top["defense_quintile"] = pd.qcut(
            top["escape_defense"],
            5,
            labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
            duplicates="drop",
        )

        summary = top.groupby(
            "defense_quintile",
            observed=True,
        ).agg(
            rows=("fighter_id", "size"),
            mean_escape_defense=("escape_defense", "mean"),
            ground_activity=("ground_activity_inflicted", "mean"),
            activity_per_entry=("activity_inflicted_per_entry", "mean"),
            control_seconds=("qualified_control_inflicted_seconds", "mean"),
            predicted_mean_seconds=("predicted_top_mean_seconds", "mean"),
        )

        print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 110)
    print("ESCAPE OFFENSE QUINTILES — GROUND ACTIVITY SUFFERED")
    print("=" * 110)

    if len(bottom) >= 5:
        bottom["offense_quintile"] = pd.qcut(
            bottom["escape_offense"],
            5,
            labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
            duplicates="drop",
        )

        summary = bottom.groupby(
            "offense_quintile",
            observed=True,
        ).agg(
            rows=("fighter_id", "size"),
            mean_escape_offense=("escape_offense", "mean"),
            ground_activity_suffered=("ground_activity_suffered", "mean"),
            activity_suffered_per_entry=("activity_suffered_per_entry", "mean"),
            control_suffered_seconds=("qualified_control_suffered_seconds", "mean"),
            predicted_bottom_seconds=("predicted_bottom_mean_seconds", "mean"),
        )

        print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    # ------------------------------------------------------------------
    # Current latest fighter rankings.
    # ------------------------------------------------------------------
    latest = replay[
        replay["latest_rating"].notna()
    ].copy()

    print("\n" + "=" * 110)
    print("LATEST ESCAPE OFFENSE — TOP / BOTTOM")
    print("=" * 110)

    eo = latest[latest.trait == "escape_offense"].copy()

    print("\nBEST ESCAPE OFFENSE")
    print(
        eo.nlargest(20, "latest_rating")[
            ["fighter_name", "latest_rating"]
        ].to_string(index=False)
    )

    print("\nWORST ESCAPE OFFENSE")
    print(
        eo.nsmallest(20, "latest_rating")[
            ["fighter_name", "latest_rating"]
        ].to_string(index=False)
    )

    print("\n" + "=" * 110)
    print("LATEST ESCAPE DEFENSE — TOP / BOTTOM")
    print("=" * 110)

    ed = latest[latest.trait == "escape_defense"].copy()

    print("\nBEST ESCAPE DEFENSE")
    print(
        ed.nlargest(20, "latest_rating")[
            ["fighter_name", "latest_rating"]
        ].to_string(index=False)
    )

    print("\nWORST ESCAPE DEFENSE")
    print(
        ed.nsmallest(20, "latest_rating")[
            ["fighter_name", "latest_rating"]
        ].to_string(index=False)
    )

    x.to_csv(
        "/tmp/escape_trait_ground_activity_audit.csv",
        index=False,
    )

    print(
        "\nwrote: /tmp/escape_trait_ground_activity_audit.csv"
    )


if __name__ == "__main__":
    main()
