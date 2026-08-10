"""Estimate historical round-to-round performance retention for stamina calibration.

This is an empirical calibration audit, not a new FSR builder.  It uses complete
three-round UFC decisions so each compared round represents a full five minutes,
then joins the leakage-safe pre-fight FSR-32 cardio traits.

The source data is round-level, so this script does NOT claim to observe literal
10-second physiological stamina.  Its purpose is to establish the historical
R1->R2 and R1->R3 performance-retention targets that a 10-second MC fatigue
curve should reproduce in aggregate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH
from scripts.experimental import build_fsr_32_database as fsr32
from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature as modern


DEFAULT_START_DATE = "2020-01-01"
DEFAULT_MIN_PRIOR_FIGHTS = 3
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/"
    "historical_fatigue_baseline_fighter_fights.csv"
)
SUMMARY_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/"
    "historical_fatigue_baseline_summary.csv"
)

ROUND_METRICS = (
    "sig_str_attempted",
    "sig_str_landed",
    "total_str_attempted",
    "total_str_landed",
    "td_attempted",
    "td_landed",
    "ctrl_sec",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--min-prior-fights", type=int, default=DEFAULT_MIN_PRIOR_FIGHTS)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--summary-output", type=Path, default=SUMMARY_PATH)
    return p.parse_args()


def _decision_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.contains("decision", na=False)


def _load_complete_three_round_decisions(
    start_date: pd.Timestamp,
    min_prior_fights: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = modern._load_master(MASTER_PATH)
    master = modern._attach_prior_ufc_fight_counts(master)
    decisions = master[
        master["event_date"].ge(start_date)
        & _decision_mask(master["method"])
        & pd.to_numeric(master["finish_round"], errors="coerce").eq(3)
        & master["r_prior_ufc_fights"].ge(min_prior_fights)
        & master["b_prior_ufc_fights"].ge(min_prior_fights)
    ].copy()

    round_stats = pd.read_parquet(ROUND_STATS_PATH).copy()
    required = {
        "fight_id",
        "fighter_id",
        "round",
        *ROUND_METRICS,
    }
    missing = sorted(required - set(round_stats.columns))
    if missing:
        raise ValueError(f"round stats missing required columns: {missing}")

    round_stats["fight_id"] = round_stats["fight_id"].astype(str)
    round_stats["fighter_id"] = round_stats["fighter_id"].astype(str)
    round_stats["round"] = pd.to_numeric(round_stats["round"], errors="coerce")
    for metric in ROUND_METRICS:
        round_stats[metric] = pd.to_numeric(round_stats[metric], errors="coerce")

    wanted = set(decisions["fight_id"].astype(str))
    rounds = round_stats[
        round_stats["fight_id"].isin(wanted)
        & round_stats["round"].isin([1, 2, 3])
    ].copy()

    complete_keys: list[tuple[str, str]] = []
    for key, group in rounds.groupby(["fight_id", "fighter_id"], sort=False):
        observed = set(group["round"].dropna().astype(int))
        if observed == {1, 2, 3} and len(group) == 3:
            complete_keys.append((str(key[0]), str(key[1])))

    complete_index = pd.MultiIndex.from_tuples(
        complete_keys, names=["fight_id", "fighter_id"]
    )
    indexed = rounds.set_index(["fight_id", "fighter_id"])
    rounds = indexed.loc[indexed.index.isin(complete_index)].reset_index()
    return decisions, rounds


def _wide_fighter_fights(rounds: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for metric in ROUND_METRICS:
        pivot = rounds.pivot(
            index=["fight_id", "fighter_id"],
            columns="round",
            values=metric,
        ).rename(columns={1.0: f"r1_{metric}", 2.0: f"r2_{metric}", 3.0: f"r3_{metric}",
                          1: f"r1_{metric}", 2: f"r2_{metric}", 3: f"r3_{metric}"})
        parts.append(pivot)

    wide = pd.concat(parts, axis=1).reset_index()

    for round_no in (1, 2, 3):
        attempted = wide[f"r{round_no}_sig_str_attempted"]
        landed = wide[f"r{round_no}_sig_str_landed"]
        wide[f"r{round_no}_sig_accuracy"] = np.where(
            attempted > 0,
            landed / attempted,
            np.nan,
        )

    for metric in ROUND_METRICS:
        r1 = wide[f"r1_{metric}"]
        for later in (2, 3):
            value = wide[f"r{later}_{metric}"]
            wide[f"r{later}_vs_r1_{metric}_ratio"] = np.where(
                r1 > 0,
                value / r1,
                np.nan,
            )
            wide[f"r{later}_minus_r1_{metric}"] = value - r1

    for later in (2, 3):
        wide[f"r{later}_minus_r1_sig_accuracy"] = (
            wide[f"r{later}_sig_accuracy"] - wide["r1_sig_accuracy"]
        )

    return wide


def _attach_prefight_stamina_contract(wide: pd.DataFrame) -> pd.DataFrame:
    fsr = pd.read_parquet(fsr32.OUTPUT_PATH).copy()
    required = {
        "fight_id",
        "fighter_id",
        fsr32.STAMINA_DEPLETION_RESISTANCE,
        fsr32.STAMINA_PERFORMANCE_RESILIENCE,
        fsr32.STAMINA_RECOVERY_ABILITY,
    }
    missing = sorted(required - set(fsr.columns))
    if missing:
        raise ValueError(f"FSR-32 missing historical join columns: {missing}")

    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    contract = fsr[list(required)].drop_duplicates(["fight_id", "fighter_id"])
    return wide.merge(
        contract,
        on=["fight_id", "fighter_id"],
        how="inner",
        validate="one_to_one",
    )


def _population_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_group(label: str, group: pd.DataFrame) -> None:
        row: dict[str, object] = {"group": label, "n_fighter_fights": len(group)}
        for metric in ("sig_str_attempted", "sig_str_landed", "total_str_attempted", "total_str_landed"):
            r1_mean = float(group[f"r1_{metric}"].mean())
            r2_mean = float(group[f"r2_{metric}"].mean())
            r3_mean = float(group[f"r3_{metric}"].mean())
            row[f"mean_r1_{metric}"] = r1_mean
            row[f"mean_r2_{metric}"] = r2_mean
            row[f"mean_r3_{metric}"] = r3_mean
            row[f"population_r2_vs_r1_{metric}_retention"] = r2_mean / r1_mean if r1_mean > 0 else np.nan
            row[f"population_r3_vs_r1_{metric}_retention"] = r3_mean / r1_mean if r1_mean > 0 else np.nan
            row[f"median_fighter_r2_vs_r1_{metric}_ratio"] = float(
                group[f"r2_vs_r1_{metric}_ratio"].median()
            )
            row[f"median_fighter_r3_vs_r1_{metric}_ratio"] = float(
                group[f"r3_vs_r1_{metric}_ratio"].median()
            )

        for round_no in (1, 2, 3):
            row[f"mean_r{round_no}_sig_accuracy"] = float(
                group[f"r{round_no}_sig_accuracy"].mean()
            )
        row["mean_r2_minus_r1_sig_accuracy"] = float(
            group["r2_minus_r1_sig_accuracy"].mean()
        )
        row["mean_r3_minus_r1_sig_accuracy"] = float(
            group["r3_minus_r1_sig_accuracy"].mean()
        )
        row["mean_stamina_depletion_resistance"] = float(
            group[fsr32.STAMINA_DEPLETION_RESISTANCE].mean()
        )
        row["mean_stamina_performance_resilience"] = float(
            group[fsr32.STAMINA_PERFORMANCE_RESILIENCE].mean()
        )
        rows.append(row)

    add_group("all", frame)

    for trait, short in [
        (fsr32.STAMINA_DEPLETION_RESISTANCE, "depletion"),
        (fsr32.STAMINA_PERFORMANCE_RESILIENCE, "resilience"),
    ]:
        ranked = frame[trait].rank(method="first")
        bins = pd.qcut(ranked, 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
        for label in bins.cat.categories:
            add_group(f"{short}_{label}", frame.loc[bins.eq(label)])

    return pd.DataFrame(rows)


def _print_summary(summary: pd.DataFrame, n_fights: int) -> None:
    print("\n" + "=" * 150)
    print("HISTORICAL FATIGUE BASELINE — COMPLETE 3-ROUND UFC DECISIONS")
    print("=" * 150)
    print(f"complete fights represented: {n_fights:,}")
    print(f"fighter-fights with matched leakage-safe FSR-32: {int(summary.loc[summary['group'].eq('all'), 'n_fighter_fights'].iloc[0]):,}")
    print("Source resolution: one UFCStats row per fighter per completed round (5 minutes).")
    print("Interpretation: retention targets are empirical performance retention, not literal measured stamina.")

    cols = [
        "group",
        "n_fighter_fights",
        "population_r2_vs_r1_sig_str_attempted_retention",
        "population_r3_vs_r1_sig_str_attempted_retention",
        "population_r2_vs_r1_sig_str_landed_retention",
        "population_r3_vs_r1_sig_str_landed_retention",
        "mean_r1_sig_accuracy",
        "mean_r2_sig_accuracy",
        "mean_r3_sig_accuracy",
        "mean_stamina_depletion_resistance",
        "mean_stamina_performance_resilience",
    ]
    print("\nCORE RETENTION TARGETS")
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    args = _parse_args()
    start_date = pd.Timestamp(args.start_date)
    decisions, rounds = _load_complete_three_round_decisions(
        start_date=start_date,
        min_prior_fights=args.min_prior_fights,
    )
    wide = _wide_fighter_fights(rounds)
    wide = _attach_prefight_stamina_contract(wide)
    summary = _population_summary(wide)

    _print_summary(summary, wide["fight_id"].nunique())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(args.output, index=False)
    summary.to_csv(args.summary_output, index=False)
    print(f"\nWrote fighter-fight baseline rows to {args.output}")
    print(f"Wrote aggregate/cardio-quartile summary to {args.summary_output}")


if __name__ == "__main__":
    main()
