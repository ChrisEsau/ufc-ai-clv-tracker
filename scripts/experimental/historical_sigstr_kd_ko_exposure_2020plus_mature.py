"""Historical significant-strike exposure vs knockdown / KO-TKO audit.

Research-only diagnostic for calibrating the simulator contact-quality distribution.

Cohort
------
Uses the exact aligned 2020+ mature FSR-32 cohort used by the current Monte Carlo
calibration:
- event date >= 2020-01-01
- both fighters >= 3 prior UFC fights
- leakage-safe FSR-32 pair available

Unit of analysis
----------------
One fight-round. The two fighter rows in ``ufc_round_stats.parquet`` are combined:
- sig_str_landed = red + blue significant strikes landed in the round
- kd = red + blue recorded knockdowns in the round
- ko_tko = 1 only when the bout ended by KO/TKO in that round

Outputs
-------
1. Detailed fight-round CSV.
2. Summary by round number.
3. Summary by significant-strike exposure bin.
4. Summary by round number x exposure bin.

No simulator constants, FSR values, or production artifacts are modified.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32


ROUND_STATS_PATH = Path("data/fight_details/ufc_round_stats.parquet")
OUTPUT_DIR = Path("data/experimental")
OUTPUT_PREFIX = "historical_sigstr_kd_ko_2020plus_mature"

EXPOSURE_BINS = [-1, 5, 10, 15, 20, 30, 40, 50, np.inf]
EXPOSURE_LABELS = ["0-5", "6-10", "11-15", "16-20", "21-30", "31-40", "41-50", "51+"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit historical significant-strike exposure versus KD and KO/TKO outcomes."
    )
    p.add_argument("--round-stats", type=Path, default=ROUND_STATS_PATH)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return p.parse_args()


def _load_round_stats(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Round stats dataset not found: {path}")

    df = pd.read_parquet(path).copy()
    required = {"fight_id", "round", "fighter_id", "corner", "sig_str_landed", "kd"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Round stats missing required columns: {missing}")

    df["fight_id"] = df["fight_id"].astype(str)
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df["sig_str_landed"] = pd.to_numeric(df["sig_str_landed"], errors="coerce")
    df["kd"] = pd.to_numeric(df["kd"], errors="coerce")

    bad = df[["round", "sig_str_landed", "kd"]].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"Round stats contain {int(bad.sum())} rows with invalid round/SIG STR/KD values")

    df["round"] = df["round"].astype(int)
    df["sig_str_landed"] = df["sig_str_landed"].astype(int)
    df["kd"] = df["kd"].astype(int)
    return df


def _build_fight_rounds(round_stats: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    eligible = cohort.copy()
    eligible["bout_id"] = eligible["bout_id"].astype(str)
    ids = set(eligible["bout_id"])

    rs = round_stats[round_stats["fight_id"].isin(ids)].copy()
    if rs.empty:
        raise ValueError("No round-stat rows matched the aligned 2020+ mature FSR cohort")

    # The canonical round-stat grain is one row per fighter per fight-round.
    counts = rs.groupby(["fight_id", "round"]).size()
    bad_counts = counts[counts.ne(2)]
    if not bad_counts.empty:
        preview = bad_counts.head(10).to_dict()
        raise ValueError(
            f"Expected exactly 2 fighter rows per fight-round; found {len(bad_counts)} bad keys. "
            f"Examples: {preview}"
        )

    rounds = (
        rs.groupby(["fight_id", "round"], as_index=False)
        .agg(
            sig_str_landed=("sig_str_landed", "sum"),
            kd=("kd", "sum"),
        )
        .rename(columns={"fight_id": "bout_id"})
    )

    meta = eligible[
        ["bout_id", "event_date", "actual_ko_tko", "actual_finish_round"]
    ].drop_duplicates("bout_id")
    rounds = rounds.merge(meta, on="bout_id", how="left", validate="many_to_one")

    rounds["actual_finish_round"] = pd.to_numeric(rounds["actual_finish_round"], errors="coerce")
    rounds["ko_tko"] = (
        rounds["actual_ko_tko"].eq(1)
        & rounds["actual_finish_round"].eq(rounds["round"])
    ).astype(int)
    rounds["any_kd"] = rounds["kd"].gt(0).astype(int)
    rounds["multi_kd"] = rounds["kd"].ge(2).astype(int)
    rounds["kd_in_ko_round"] = (rounds["ko_tko"].eq(1) & rounds["any_kd"].eq(1)).astype(int)

    rounds["sig_str_bin"] = pd.cut(
        rounds["sig_str_landed"],
        bins=EXPOSURE_BINS,
        labels=EXPOSURE_LABELS,
        ordered=True,
    )

    return rounds.sort_values(["event_date", "bout_id", "round"]).reset_index(drop=True)


def _summarize(group: pd.DataFrame) -> dict[str, float | int]:
    n_rounds = len(group)
    sig = int(group["sig_str_landed"].sum())
    kd = int(group["kd"].sum())
    ko = int(group["ko_tko"].sum())

    return {
        "fight_rounds": n_rounds,
        "sig_str_landed": sig,
        "knockdowns": kd,
        "ko_tko_finishes": ko,
        "mean_sig_str_landed": float(group["sig_str_landed"].mean()),
        "mean_kd_per_round": float(group["kd"].mean()),
        "p_any_kd": float(group["any_kd"].mean()),
        "p_multi_kd": float(group["multi_kd"].mean()),
        "p_ko_tko": float(group["ko_tko"].mean()),
        "kd_per_100_sig_landed": 100.0 * kd / sig if sig else np.nan,
        "ko_per_1000_sig_landed": 1000.0 * ko / sig if sig else np.nan,
    }


def _summary_table(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if not group_cols:
        return pd.DataFrame([_summarize(frame)])

    rows: list[dict[str, object]] = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    for key, group in frame.groupby(grouper, observed=False, sort=True):
        keys = (key,) if len(group_cols) == 1 else tuple(key)
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update(_summarize(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _print_table(title: str, df: pd.DataFrame) -> None:
    print("\n" + title)
    print("-" * len(title))
    if df.empty:
        print("<empty>")
        return
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    args = parse_args()

    cohort, _pairs = cohort32.build_aligned_cohort()
    round_stats = _load_round_stats(args.round_stats)
    rounds = _build_fight_rounds(round_stats, cohort)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / f"{OUTPUT_PREFIX}_rounds.csv"
    by_round_path = args.output_dir / f"{OUTPUT_PREFIX}_by_round.csv"
    by_exposure_path = args.output_dir / f"{OUTPUT_PREFIX}_by_exposure.csv"
    by_round_exposure_path = args.output_dir / f"{OUTPUT_PREFIX}_by_round_exposure.csv"

    by_round = _summary_table(rounds, ["round"])
    by_exposure = _summary_table(rounds, ["sig_str_bin"])
    by_round_exposure = _summary_table(rounds, ["round", "sig_str_bin"])
    overall = _summary_table(rounds, [])

    rounds.to_csv(detail_path, index=False)
    by_round.to_csv(by_round_path, index=False)
    by_exposure.to_csv(by_exposure_path, index=False)
    by_round_exposure.to_csv(by_round_exposure_path, index=False)

    print("\n" + "=" * 118)
    print("HISTORICAL SIGNIFICANT-STRIKE EXPOSURE -> KNOCKDOWN / KO-TKO AUDIT")
    print("=" * 118)
    print(f"Aligned mature cohort bouts: {len(cohort):,}")
    print(f"Bouts with round stats:       {rounds['bout_id'].nunique():,}")
    print(f"Fight-rounds analyzed:        {len(rounds):,}")
    print(f"Date range:                   {rounds['event_date'].min()} -> {rounds['event_date'].max()}")

    _print_table("OVERALL", overall)
    _print_table("BY ROUND", by_round)
    _print_table("BY SIGNIFICANT-STRIKE EXPOSURE", by_exposure)
    _print_table("BY ROUND x SIGNIFICANT-STRIKE EXPOSURE", by_round_exposure)

    ko_rounds = rounds[rounds["ko_tko"].eq(1)]
    if not ko_rounds.empty:
        ko_with_kd = int(ko_rounds["any_kd"].sum())
        ko_without_kd = int(len(ko_rounds) - ko_with_kd)
        print("\nKO/TKO ROUND KD RELATIONSHIP")
        print("----------------------------")
        print(f"KO/TKO rounds:              {len(ko_rounds):,}")
        print(f"KO/TKO rounds with KD:      {ko_with_kd:,} ({ko_with_kd / len(ko_rounds):.2%})")
        print(f"KO/TKO rounds without KD:   {ko_without_kd:,} ({ko_without_kd / len(ko_rounds):.2%})")
        print(f"Recorded KDs in KO rounds:  {int(ko_rounds['kd'].sum()):,}")

    print("\nOUTPUTS")
    print("-------")
    print(detail_path)
    print(by_round_path)
    print(by_exposure_path)
    print(by_round_exposure_path)
    print("\nResearch only: no simulator constants or FSR artifacts modified.")


if __name__ == "__main__":
    main()
