"""Research audit for V7 KO-only fresh striking power.

Answers two questions:
1. Which fighters lost the most apparent power when KD-inclusive V5 evidence was
   replaced by KO-only V7 evidence?
2. Among fighters rated exactly 50 in V7, how many actually own UFC KO/TKO wins
   in Round 1 or in any round according to the master fight table?

This audit does not modify any FSR artifact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.experimental import fsr_striking_power_evidence_v2_sweep as v2
from scripts.experimental import fsr_striking_power_evidence_v5_kd_efficiency_sweep as v5
from scripts.experimental import fsr_striking_power_evidence_v7_ko_only_softened_penalties_sweep as v7


def _winner_ko_table(master: pd.DataFrame) -> pd.DataFrame:
    required = {"fight_id", "winner_id", "method", "finish_round"}
    missing = sorted(required - set(master.columns))
    if missing:
        raise RuntimeError(f"Master missing required columns: {missing}")

    fights = master[list(required)].copy()
    fights["fight_id"] = fights["fight_id"].astype(str)
    fights["winner_id"] = fights["winner_id"].astype(str)
    fights["finish_round"] = pd.to_numeric(fights["finish_round"], errors="coerce")
    fights["is_ko_tko"] = fights["method"].map(v2.is_ko_tko)
    fights = fights.loc[fights["is_ko_tko"] & fights["winner_id"].notna()].copy()
    fights["is_r1_ko_tko"] = fights["finish_round"].eq(1)
    fights["is_later_ko_tko"] = fights["finish_round"].gt(1)
    return fights


def _one_row_per_fighter(rankings: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    """Collapse name-variant duplicates to one row per fighter_id.

    The evidence builders group by (fighter_id, fighter_name), so historical name
    variants can yield more than one row for the same fighter_id. For this audit
    the comparison grain is fighter_id. Keep the highest-ranked row as the
    representative record; this avoids double-counting IDs and makes the merge
    contract explicit.
    """
    return (
        rankings.sort_values(rank_col, ascending=True)
        .drop_duplicates(subset=["fighter_id"], keep="first")
        .reset_index(drop=True)
    )


def main() -> None:
    master = pd.read_parquet(v2.MASTER_PATH)
    rounds = pd.read_parquet(v2.ROUND_STATS_PATH)
    fighter_fights = v2.fighter_fight_frame(master, rounds)

    v5_rankings, _ = v5.build_v5_rankings(fighter_fights)
    v7_rankings, _ = v7.build_v7_rankings(fighter_fights)

    v5_rankings = _one_row_per_fighter(v5_rankings, "rank_v5")
    v7_rankings = _one_row_per_fighter(v7_rankings, "rank_v7")

    if v5_rankings["fighter_id"].duplicated().any() or v7_rankings["fighter_id"].duplicated().any():
        raise RuntimeError("fighter_id is still duplicated after normalization")

    compare = v7_rankings[[
        "fighter_id", "fighter_name", "rank_v7", "power_evidence_rating_v7",
        "r1_ko_tko_wins", "r1_kds", "ufc_r1_fights",
    ]].merge(
        v5_rankings[[
            "fighter_id", "rank_v5", "power_evidence_rating_v5",
            "kd_evidence_v5", "power_event_fights",
        ]].rename(columns={"power_event_fights": "power_event_fights_v5"}),
        on="fighter_id",
        how="left",
        validate="one_to_one",
    )
    compare["rating_change_v5_to_v7"] = (
        compare["power_evidence_rating_v7"] - compare["power_evidence_rating_v5"]
    )
    compare["rank_change_v5_to_v7"] = compare["rank_v5"] - compare["rank_v7"]

    neutral = compare.loc[np.isclose(compare["power_evidence_rating_v7"], 50.0)].copy()

    ko = _winner_ko_table(master)
    ko_by_fighter = ko.groupby("winner_id", as_index=False).agg(
        career_ko_tko_wins=("fight_id", "nunique"),
        career_r1_ko_tko_wins=("is_r1_ko_tko", "sum"),
        career_later_ko_tko_wins=("is_later_ko_tko", "sum"),
    ).rename(columns={"winner_id": "fighter_id"})

    neutral = neutral.merge(ko_by_fighter, on="fighter_id", how="left", validate="one_to_one")
    for c in ("career_ko_tko_wins", "career_r1_ko_tko_wins", "career_later_ko_tko_wins"):
        neutral[c] = neutral[c].fillna(0).astype(int)

    neutral_with_r1_kd = neutral.loc[neutral["r1_kds"].gt(0)].copy()
    neutral_with_any_ko = neutral.loc[neutral["career_ko_tko_wins"].gt(0)].copy()
    neutral_with_master_r1_ko = neutral.loc[neutral["career_r1_ko_tko_wins"].gt(0)].copy()
    neutral_with_later_ko_only = neutral.loc[
        neutral["career_r1_ko_tko_wins"].eq(0)
        & neutral["career_later_ko_tko_wins"].gt(0)
    ].copy()

    print("\n" + "=" * 125)
    print("V7 KD-REMOVAL + KO-COVERAGE AUDIT")
    print("=" * 125)
    print(f"fighters ranked after fighter_id normalization: {len(compare):,}")
    print(f"V7 neutral-50 fighters: {len(neutral):,}")

    print("\nQUESTION 1 — DID REMOVING KDs REMOVE POWER EVIDENCE?")
    print(
        f"V7=50 fighters who nevertheless scored >=1 recorded R1 KD: "
        f"{len(neutral_with_r1_kd):,} / {len(neutral):,} "
        f"({len(neutral_with_r1_kd)/max(len(neutral),1)*100:.2f}%)"
    )
    print(f"Total recorded R1 KDs among those neutral-50 fighters: {neutral_with_r1_kd['r1_kds'].sum():.0f}")

    lost = compare.sort_values(
        ["rating_change_v5_to_v7", "rank_change_v5_to_v7"], ascending=[True, True]
    ).head(30)
    print("\nBIGGEST RATING LOSSES: KD-INCLUSIVE V5 -> KO-ONLY V7")
    cols = [
        "fighter_name", "power_evidence_rating_v5", "power_evidence_rating_v7",
        "rating_change_v5_to_v7", "rank_v5", "rank_v7", "rank_change_v5_to_v7",
        "r1_kds", "r1_ko_tko_wins", "kd_evidence_v5", "ufc_r1_fights",
    ]
    print(lost[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if len(neutral_with_r1_kd):
        print("\nHIGHEST-KD FIGHTERS WHO BECAME V7=50")
        ex = neutral_with_r1_kd.sort_values(
            ["r1_kds", "power_evidence_rating_v5"], ascending=[False, False]
        ).head(30)
        print(ex[[
            "fighter_name", "r1_kds", "power_evidence_rating_v5", "rank_v5",
            "ufc_r1_fights", "career_ko_tko_wins", "career_r1_ko_tko_wins",
            "career_later_ko_tko_wins",
        ]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nQUESTION 2 — KO/TKO HISTORY AMONG V7=50 FIGHTERS (MASTER TABLE)")
    print(
        f"Have >=1 KO/TKO win in ANY round: {len(neutral_with_any_ko):,} / {len(neutral):,} "
        f"({len(neutral_with_any_ko)/max(len(neutral),1)*100:.2f}%)"
    )
    print(
        f"Have >=1 ROUND-1 KO/TKO win in master: {len(neutral_with_master_r1_ko):,} / {len(neutral):,} "
        f"({len(neutral_with_master_r1_ko)/max(len(neutral),1)*100:.2f}%)"
    )
    print(
        f"Have KO/TKO win(s), but ONLY in Round 2+: {len(neutral_with_later_ko_only):,} / {len(neutral):,} "
        f"({len(neutral_with_later_ko_only)/max(len(neutral),1)*100:.2f}%)"
    )
    print(f"Total KO/TKO wins owned by V7=50 fighters: {neutral['career_ko_tko_wins'].sum():,}")
    print(f"  Round 1: {neutral['career_r1_ko_tko_wins'].sum():,}")
    print(f"  Round 2+: {neutral['career_later_ko_tko_wins'].sum():,}")

    if len(neutral_with_master_r1_ko):
        print("\nIMPORTANT: V7=50 FIGHTERS WITH MASTER-TABLE R1 KO/TKO WINS")
        print("These indicate R1-stat/evidence coverage mismatch and should be investigated.")
        ex = neutral_with_master_r1_ko.sort_values(
            ["career_r1_ko_tko_wins", "career_ko_tko_wins"], ascending=[False, False]
        ).head(50)
        print(ex[[
            "fighter_name", "career_r1_ko_tko_wins", "career_later_ko_tko_wins",
            "career_ko_tko_wins", "r1_kds", "ufc_r1_fights",
        ]].to_string(index=False))

    if len(neutral_with_later_ko_only):
        print("\nV7=50 FIGHTERS WITH THE MOST LATER-ROUND KO/TKO WINS")
        ex = neutral_with_later_ko_only.sort_values(
            ["career_later_ko_tko_wins", "career_ko_tko_wins"], ascending=[False, False]
        ).head(30)
        print(ex[[
            "fighter_name", "career_later_ko_tko_wins", "career_ko_tko_wins",
            "r1_kds", "ufc_r1_fights",
        ]].to_string(index=False))

    print("\nResearch only: no FSR artifact modified.")


if __name__ == "__main__":
    main()
