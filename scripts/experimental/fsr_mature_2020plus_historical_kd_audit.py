"""Historical KD audit for the exact mature 2020+ cohort used by the MC.

This script does not run the simulator. It rebuilds the same aligned FSR-32
mature 2020+ cohort, filters the authoritative UFC round-stats dataset to those
fight IDs only, and reports total knockdowns plus Round 1 knockdowns.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32


ROUND_STATS_PATH = Path("data/fight_details/ufc_round_stats.parquet")


def main() -> None:
    cohort, _pairs = cohort32.build_aligned_cohort()
    cohort_ids = set(cohort["bout_id"].astype(str))

    if not ROUND_STATS_PATH.exists():
        raise FileNotFoundError(f"Round stats dataset not found: {ROUND_STATS_PATH}")

    rounds = pd.read_parquet(ROUND_STATS_PATH).copy()
    required = {"fight_id", "round", "kd"}
    missing = sorted(required - set(rounds.columns))
    if missing:
        raise RuntimeError(f"Round stats missing required columns: {missing}")

    rounds["fight_id"] = rounds["fight_id"].astype(str)
    rounds["round"] = pd.to_numeric(rounds["round"], errors="coerce")
    rounds["kd"] = pd.to_numeric(rounds["kd"], errors="coerce")

    sample = rounds[rounds["fight_id"].isin(cohort_ids)].copy()
    if sample["round"].isna().any() or sample["kd"].isna().any():
        raise RuntimeError("Selected cohort contains non-numeric round or KD values")

    covered_ids = set(sample["fight_id"].unique())
    missing_ids = cohort_ids - covered_ids

    fight_kd = (
        sample.groupby("fight_id", as_index=False)["kd"]
        .sum()
        .rename(columns={"kd": "total_kd"})
    )
    r1_kd = (
        sample.loc[sample["round"].eq(1)]
        .groupby("fight_id", as_index=False)["kd"]
        .sum()
        .rename(columns={"kd": "r1_kd"})
    )

    fight_level = (
        cohort[["bout_id"]]
        .rename(columns={"bout_id": "fight_id"})
        .assign(fight_id=lambda x: x["fight_id"].astype(str))
        .merge(fight_kd, on="fight_id", how="left", validate="one_to_one")
        .merge(r1_kd, on="fight_id", how="left", validate="one_to_one")
    )

    fight_level["total_kd"] = fight_level["total_kd"].fillna(0.0)
    fight_level["r1_kd"] = fight_level["r1_kd"].fillna(0.0)

    n = len(fight_level)
    total_kd = float(fight_level["total_kd"].sum())
    total_r1_kd = float(fight_level["r1_kd"].sum())

    print("=" * 88)
    print("HISTORICAL KD AUDIT — EXACT MATURE 2020+ MC COHORT")
    print("=" * 88)
    print(f"cohort fights: {n:,}")
    print(f"round-stats fights covered: {len(covered_ids):,}")
    print(f"missing cohort fights in round stats: {len(missing_ids):,}")
    print()
    print("TOTAL KNOCKDOWNS")
    print(f"historical total KDs: {total_kd:,.0f}")
    print(f"historical mean KDs per fight: {fight_level['total_kd'].mean():.4f}")
    print(f"fights with >=1 KD: {fight_level['total_kd'].gt(0).sum():,} ({fight_level['total_kd'].gt(0).mean():.2%})")
    print()
    print("ROUND 1 KNOCKDOWNS")
    print(f"historical R1 KDs: {total_r1_kd:,.0f}")
    print(f"historical mean R1 KDs per fight: {fight_level['r1_kd'].mean():.4f}")
    print(f"fights with >=1 R1 KD: {fight_level['r1_kd'].gt(0).sum():,} ({fight_level['r1_kd'].gt(0).mean():.2%})")
    print(f"share of all KDs occurring in R1: {(total_r1_kd / total_kd if total_kd > 0 else 0.0):.2%}")

    if missing_ids:
        print()
        print("WARNING: not every simulated cohort fight has round-stat coverage.")
        print("First missing fight IDs:")
        for fight_id in sorted(missing_ids)[:10]:
            print(f"  {fight_id}")


if __name__ == "__main__":
    main()
