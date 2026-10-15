"""Recompute KO/TKO winner-direction accuracy from an existing full-cohort MC output.

This is a post-hoc audit only. It does NOT rerun Monte Carlo paths.

Root cause addressed
--------------------
The full-cohort validation cohort already carried ``winner_id`` from the existing
population metadata merge, then merged another metadata table containing
``winner_id``. Pandas therefore created suffixed winner columns. The validation
later requested plain ``winner_id`` and fell back to an empty string, causing every
non-tie KO-side call to be scored incorrect.

This script rejoins canonical winner/corner metadata from ufc_master.parquet by
bout_id and recomputes directional metrics from the already-saved p_r_ko / p_b_ko.
Only historical KO/TKO fights within R1-R3 require winner metadata; unrelated
cohort rows with missing canonical metadata are retained but do not block the audit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature as modern

INPUT_PATH = Path(
    "data/experimental/full_cohort_ko_validation_r3_d60_s0/bout_level.csv"
)
OUTPUT_PATH = Path(
    "data/experimental/full_cohort_ko_validation_r3_d60_s0/ko_winner_audit_corrected.csv"
)


def _canonical_master() -> pd.DataFrame:
    raw = pd.read_parquet(modern.MASTER_PATH).copy()
    date_col = modern._resolve_date_column(raw)
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col]).copy()
    raw["fight_id"] = raw["fight_id"].astype(str)
    for col in ("r_id", "b_id"):
        raw[col] = raw[col].astype(str)
    # Keep winner_id nullable until after the merge so missing values remain
    # distinguishable from the literal string "nan".
    raw = (
        raw.sort_values([date_col, "fight_id"])
        .drop_duplicates("fight_id", keep="last")
        .reset_index(drop=True)
    )
    keep = ["fight_id", "r_id", "b_id", "winner_id"]
    for col in ("r_name", "b_name", "method", "finish_round"):
        if col in raw.columns:
            keep.append(col)
    return raw[keep].rename(
        columns={
            "fight_id": "bout_id",
            "r_id": "master_r_id",
            "b_id": "master_b_id",
            "winner_id": "master_winner_id",
        }
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Existing MC bout output not found: {INPUT_PATH}")

    bouts = pd.read_csv(INPUT_PATH, dtype={"bout_id": str, "r_id": str, "b_id": str})
    master = _canonical_master()
    work = bouts.merge(master, on="bout_id", how="left", validate="one_to_one")

    # Winner-direction accuracy only concerns historical KO/TKO finishes that
    # occurred inside the simulator's three-round horizon. Do not abort because
    # some unrelated cohort rows lack canonical winner metadata.
    actual_round = pd.to_numeric(work["actual_finish_round"], errors="coerce")
    actual_ko_within_r3 = work["actual_ko_tko"].eq(1) & actual_round.le(3)

    missing_all = work["master_winner_id"].isna()
    if missing_all.any():
        print(
            f"note: {int(missing_all.sum())} cohort bouts lack canonical winner metadata; "
            "only KO/TKO fights within R1-R3 are required for this audit"
        )

    missing_target = actual_ko_within_r3 & work["master_winner_id"].isna()
    if missing_target.any():
        sample = work.loc[
            missing_target,
            ["bout_id", "event_date", "red_name", "blue_name", "actual_method", "actual_finish_round"],
        ].head(10)
        raise ValueError(
            "Missing canonical winner metadata for historical KO/TKO fights inside R1-R3. Examples:\n"
            + sample.to_string(index=False)
        )

    # Corner validation likewise only needs rows that actually matched master.
    matched_master = work["master_r_id"].notna() & work["master_b_id"].notna()
    corner_mismatch = matched_master & (
        work["r_id"].astype(str).ne(work["master_r_id"].astype(str))
        | work["b_id"].astype(str).ne(work["master_b_id"].astype(str))
    )
    if corner_mismatch.any():
        sample = work.loc[
            corner_mismatch,
            ["bout_id", "r_id", "b_id", "master_r_id", "master_b_id"],
        ].head(10)
        raise ValueError(
            "Saved MC corners do not match canonical master corners. Examples:\n"
            + sample.to_string(index=False)
        )

    work["corrected_predicted_ko_winner_side"] = np.where(
        work["p_r_ko"] > work["p_b_ko"],
        "red",
        np.where(work["p_b_ko"] > work["p_r_ko"], "blue", "tie"),
    )
    work["corrected_predicted_ko_winner_id"] = np.where(
        work["corrected_predicted_ko_winner_side"].eq("red"),
        work["r_id"].astype(str),
        np.where(
            work["corrected_predicted_ko_winner_side"].eq("blue"),
            work["b_id"].astype(str),
            "",
        ),
    )

    ko = work.loc[actual_ko_within_r3].copy()
    ko["master_winner_id"] = ko["master_winner_id"].astype(str)

    ko["actual_ko_winner_side_corrected"] = np.where(
        ko["master_winner_id"].eq(ko["r_id"].astype(str)),
        "red",
        np.where(
            ko["master_winner_id"].eq(ko["b_id"].astype(str)),
            "blue",
            "unknown",
        ),
    )

    unknown = ko["actual_ko_winner_side_corrected"].eq("unknown")
    if unknown.any():
        sample = ko.loc[
            unknown,
            ["bout_id", "r_id", "b_id", "master_winner_id"],
        ].head(10)
        raise ValueError(
            "Canonical winner does not match either corner for historical KO bouts. Examples:\n"
            + sample.to_string(index=False)
        )

    ko["corrected_direction_tie"] = ko["corrected_predicted_ko_winner_side"].eq("tie")
    ko["corrected_direction_hit"] = np.where(
        ko["corrected_direction_tie"],
        np.nan,
        ko["corrected_predicted_ko_winner_id"].astype(str).eq(
            ko["master_winner_id"]
        ).astype(float),
    )

    ko["p_actual_ko_winner"] = np.where(
        ko["actual_ko_winner_side_corrected"].eq("red"),
        ko["p_r_ko"],
        ko["p_b_ko"],
    )
    ko["p_actual_ko_loser"] = np.where(
        ko["actual_ko_winner_side_corrected"].eq("red"),
        ko["p_b_ko"],
        ko["p_r_ko"],
    )
    ko["ko_probability_edge_for_actual_winner"] = (
        ko["p_actual_ko_winner"] - ko["p_actual_ko_loser"]
    )

    non_tie = ko.loc[ko["corrected_direction_hit"].notna()]

    print("\n" + "=" * 116)
    print("CORRECTED FULL-COHORT KO/TKO WINNER-DIRECTION AUDIT — NO MC RERUN")
    print("=" * 116)
    print(f"historical KO/TKO fights within R1-R3: {len(ko):,}")
    print(f"non-tie KO-side calls:                  {len(non_tie):,}")
    print(f"tie KO-side calls:                      {int(ko['corrected_direction_tie'].sum()):,} ({ko['corrected_direction_tie'].mean():.2%})")
    if len(non_tie):
        print(f"KO winner direction accuracy:           {non_tie['corrected_direction_hit'].mean():.2%}")
    print(f"mean P(actual KO winner scores KO):     {ko['p_actual_ko_winner'].mean():.2%}")
    print(f"mean P(actual KO loser scores KO):      {ko['p_actual_ko_loser'].mean():.2%}")
    print(f"mean KO probability edge, true winner:  {ko['ko_probability_edge_for_actual_winner'].mean():+.4f}")

    print("\nDIRECTION COUNTS")
    counts = pd.crosstab(
        ko["actual_ko_winner_side_corrected"],
        ko["corrected_predicted_ko_winner_side"],
        margins=True,
    )
    print(counts.to_string())

    cols = [
        "bout_id",
        "event_date",
        "red_name",
        "blue_name",
        "r_id",
        "b_id",
        "master_winner_id",
        "actual_method",
        "actual_finish_round",
        "actual_ko_winner_side_corrected",
        "p_any_ko",
        "p_r_ko",
        "p_b_ko",
        "corrected_predicted_ko_winner_side",
        "corrected_predicted_ko_winner_id",
        "corrected_direction_tie",
        "corrected_direction_hit",
        "p_actual_ko_winner",
        "p_actual_ko_loser",
        "ko_probability_edge_for_actual_winner",
    ]
    cols = [c for c in cols if c in ko.columns]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ko[cols].sort_values(
        ["corrected_direction_hit", "ko_probability_edge_for_actual_winner"],
        ascending=[True, True],
        na_position="last",
    ).to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved corrected fight-level KO audit: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
