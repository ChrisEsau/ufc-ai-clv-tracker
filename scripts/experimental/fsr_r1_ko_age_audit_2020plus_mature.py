"""Audit fighter age in the 2020+ mature-fighter Round-1 KO/TKO cohort.

Purpose
-------
Test whether age is associated with Round-1 KO/TKO occurrence and direction
before considering any age decay in FSR traits.

Cohort contract
---------------
- UFC bouts dated 2020-01-01 or later.
- Both fighters had at least 3 completed prior UFC fights.
- Same outcome cohort helper used by the existing mature R1-KO research.
- Ages are taken from canonical master age fields when available, otherwise
  computed from canonical DOB fields.

Outputs
-------
1. Every actual R1 KO/TKO bout with winner/loser ages.
2. Winner vs loser age summary for those 220 bouts (subject to age coverage).
3. R1 KO/TKO occurrence rate by oldest-fighter age band and mean-age band.
4. Side-level probability of being the R1 KO winner by fighter age band.
5. Age-gap direction summary.

This is diagnostic-only. It changes no FSR values or simulator constants.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.research.run_post_ko_next_fight_study import build_corner_age_series
from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature as modern


OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/r1_ko_age_audit_2020plus_mature.parquet"
)
DETAIL_CSV_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/r1_ko_age_audit_2020plus_mature.csv"
)

AGE_BINS = [-np.inf, 27.999, 30.999, 33.999, 36.999, 39.999, np.inf]
AGE_LABELS = ["<=27", "28-30", "31-33", "34-36", "37-39", "40+"]


def _prepare(master_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = modern._load_master(master_path)
    master = master.copy()

    # The shared age helper expects fight_date.
    master["fight_date"] = pd.to_datetime(master["event_date"], errors="coerce")
    master["r_age"] = build_corner_age_series(master, "r")
    master["b_age"] = build_corner_age_series(master, "b")

    cohort = modern._build_outcome_cohort(master)

    keep_cols = [
        "fight_id", "event_date", "r_id", "b_id", "winner_id",
        "r_age", "b_age",
    ]
    for optional in ("r_name", "b_name", "method", "finish_round"):
        if optional in master.columns:
            keep_cols.append(optional)

    meta = master[keep_cols].rename(columns={"fight_id": "bout_id"}).copy()
    meta["bout_id"] = meta["bout_id"].astype(str)
    cohort["bout_id"] = cohort["bout_id"].astype(str)
    frame = cohort.merge(meta, on=["bout_id", "event_date", "r_id", "b_id"], how="left", validate="one_to_one")

    frame["r_age"] = pd.to_numeric(frame["r_age"], errors="coerce")
    frame["b_age"] = pd.to_numeric(frame["b_age"], errors="coerce")
    frame["winner_id"] = frame["winner_id"].astype(str)

    frame["r_is_winner"] = frame["r_id"].astype(str).eq(frame["winner_id"])
    frame["winner_age"] = np.where(frame["r_is_winner"], frame["r_age"], frame["b_age"])
    frame["loser_age"] = np.where(frame["r_is_winner"], frame["b_age"], frame["r_age"])
    frame["winner_name"] = np.where(
        frame["r_is_winner"],
        frame.get("r_name", pd.Series("RED", index=frame.index)),
        frame.get("b_name", pd.Series("BLUE", index=frame.index)),
    )
    frame["loser_name"] = np.where(
        frame["r_is_winner"],
        frame.get("b_name", pd.Series("BLUE", index=frame.index)),
        frame.get("r_name", pd.Series("RED", index=frame.index)),
    )
    frame["age_gap_winner_minus_loser"] = frame["winner_age"] - frame["loser_age"]
    frame["oldest_age"] = frame[["r_age", "b_age"]].max(axis=1)
    frame["mean_age"] = frame[["r_age", "b_age"]].mean(axis=1)
    frame["age_gap_abs"] = (frame["r_age"] - frame["b_age"]).abs()

    r1 = frame.loc[frame["actual_r1_ko"].eq(1)].copy()
    return frame, r1


def _age_band(series: pd.Series) -> pd.Series:
    return pd.cut(series, bins=AGE_BINS, labels=AGE_LABELS, right=True)


def _print_r1_detail(r1: pd.DataFrame) -> None:
    print("\n" + "=" * 120)
    print("ACTUAL ROUND-1 KO/TKO BOUTS — FIGHTER AGES")
    print("=" * 120)
    cols = [
        "event_date", "bout_id", "winner_name", "winner_age",
        "loser_name", "loser_age", "age_gap_winner_minus_loser",
    ]
    detail = r1[cols].sort_values(["event_date", "bout_id"]).copy()
    print(detail.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def _print_summary(frame: pd.DataFrame, r1: pd.DataFrame) -> None:
    valid = r1.dropna(subset=["winner_age", "loser_age"]).copy()
    print("\n" + "=" * 120)
    print("R1 KO/TKO AGE SUMMARY")
    print("=" * 120)
    print(f"eligible mature 2020+ bouts: {len(frame):,}")
    print(f"actual R1 KO/TKO bouts: {len(r1):,}")
    print(f"R1 KO bouts with both ages available: {len(valid):,} ({len(valid)/max(1,len(r1)):.1%})")

    if valid.empty:
        print("No usable age coverage in canonical master.")
        return

    print(f"winner age mean/median: {valid['winner_age'].mean():.2f} / {valid['winner_age'].median():.2f}")
    print(f"loser age mean/median:  {valid['loser_age'].mean():.2f} / {valid['loser_age'].median():.2f}")
    print(f"winner-minus-loser age gap mean: {valid['age_gap_winner_minus_loser'].mean():+.2f} years")
    print(f"younger fighter won R1 KO: {(valid['age_gap_winner_minus_loser'] < 0).mean():.2%}")
    print(f"older fighter won R1 KO:   {(valid['age_gap_winner_minus_loser'] > 0).mean():.2%}")
    print(f"same-age (within exact calculated age equality): {(valid['age_gap_winner_minus_loser'] == 0).mean():.2%}")

    # Bout-level occurrence: does having older fighters make an R1 KO more likely?
    occurrence = frame.dropna(subset=["oldest_age", "mean_age"]).copy()
    occurrence["oldest_age_band"] = _age_band(occurrence["oldest_age"])
    occurrence["mean_age_band"] = _age_band(occurrence["mean_age"])

    print("\nR1 KO/TKO OCCURRENCE BY OLDEST FIGHTER AGE")
    out = (
        occurrence.groupby("oldest_age_band", observed=False)
        .agg(bouts=("bout_id", "size"), r1_kos=("actual_r1_ko", "sum"), r1_ko_rate=("actual_r1_ko", "mean"))
        .reset_index()
    )
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nR1 KO/TKO OCCURRENCE BY BOUT MEAN AGE")
    out2 = (
        occurrence.groupby("mean_age_band", observed=False)
        .agg(bouts=("bout_id", "size"), r1_kos=("actual_r1_ko", "sum"), r1_ko_rate=("actual_r1_ko", "mean"))
        .reset_index()
    )
    print(out2.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Side-level direction: among all fighter-sides in eligible bouts, who becomes the R1-KO winner?
    sides = []
    for corner in ("r", "b"):
        is_winner = frame[f"{corner}_id"].astype(str).eq(frame["winner_id"].astype(str))
        sides.append(pd.DataFrame({
            "bout_id": frame["bout_id"],
            "fighter_age": frame[f"{corner}_age"],
            "r1_ko_winner": (frame["actual_r1_ko"].eq(1) & is_winner).astype(int),
            "in_r1_ko_bout": frame["actual_r1_ko"].astype(int),
        }))
    side = pd.concat(sides, ignore_index=True).dropna(subset=["fighter_age"])
    side["age_band"] = _age_band(side["fighter_age"])

    print("\nSIDE-LEVEL R1 KO WINNER RATE BY FIGHTER AGE")
    side_summary = (
        side.groupby("age_band", observed=False)
        .agg(
            fighter_sides=("bout_id", "size"),
            r1_ko_wins=("r1_ko_winner", "sum"),
            r1_ko_win_rate=("r1_ko_winner", "mean"),
            r1_ko_bout_exposure=("in_r1_ko_bout", "mean"),
        )
        .reset_index()
    )
    print(side_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Within actual R1 KO bouts, compare direction by material age gaps.
    valid["gap_bucket"] = pd.cut(
        valid["age_gap_winner_minus_loser"],
        bins=[-np.inf, -5, -2, 2, 5, np.inf],
        labels=["winner 5+ younger", "winner 2-5 younger", "within 2 years", "winner 2-5 older", "winner 5+ older"],
        include_lowest=True,
    )
    print("\nR1 KO WINNER/LOSER AGE-GAP DISTRIBUTION")
    gap = valid["gap_bucket"].value_counts(sort=False).rename_axis("gap_bucket").reset_index(name="bouts")
    gap["share"] = gap["bouts"] / len(valid)
    print(gap.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=modern.MASTER_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--csv", type=Path, default=DETAIL_CSV_PATH)
    args = parser.parse_args()

    frame, r1 = _prepare(args.master)
    _print_r1_detail(r1)
    _print_summary(frame, r1)

    detail_cols = [
        "bout_id", "event_date", "r_id", "b_id", "winner_id",
        "r_name", "b_name", "r_age", "b_age", "winner_name", "loser_name",
        "winner_age", "loser_age", "age_gap_winner_minus_loser",
        "oldest_age", "mean_age", "actual_r1_ko",
    ]
    detail_cols = [c for c in detail_cols if c in frame.columns]
    out = frame[detail_cols].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    out.to_csv(args.csv, index=False)
    print(f"\nWrote {len(out):,} eligible bout rows to {args.output}")
    print(f"Wrote CSV detail to {args.csv}")
    print("No FSR values or simulator constants were changed.")


if __name__ == "__main__":
    main()
