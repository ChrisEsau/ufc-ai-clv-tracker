"""Rebuild only wrestling_entry and merge it into the existing FSR-32 prefight parquet.

This script intentionally avoids the full 25-family canonical replay. It replays
only the leakage-safe wrestling_entry rating chronologically from the existing
RFS history, then replaces just that one column in the existing FSR-32 pre-fight
snapshot database. The other 24 learned ratings are preserved byte-for-value in
the resulting dataframe.

Current wrestling_entry contract
--------------------------------
- Observation: percentile of takedown attempts per round.
- Zero-attempt fights are valid low-entry observations.
- Evidence quality: q_exp(rounds / 2), so confidence comes from fight exposure.
- Expected observation: leakage-safe population baseline plus own rating edge
  versus the neutral 50 prior (no opponent td_defense adjustment).
- Ratings remain on the canonical 10-90 scale with 50 neutral prior.

The script writes a shadow parquet by default. Use --in-place only after checking
its diagnostics; in-place mode creates a .bak copy first.
"""

from __future__ import annotations

import argparse
import shutil
from bisect import bisect_right, insort
from collections import defaultdict
from math import exp, log, sqrt
from pathlib import Path

import pandas as pd


RFS_PATH = Path("data/features/round_fighter_state_history.parquet")
BASE_FSR_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/"
    "fsr_32_prefight_snapshots.parquet"
)
DEFAULT_OUTPUT_PATH = BASE_FSR_PATH.with_name(
    "fsr_32_prefight_snapshots_wrestling_entry_v2.parquet"
)

BASE_RATING = 50.0
MIN_RATING = 10.0
MAX_RATING = 90.0
RATING_SCALE = 12.0
BASE_K = 7.0
LOGIT_EPSILON = 1e-4

TD_RATE_COL = "rfs_phase_base_fight_td_attempts_per_round"
ROUNDS_COL = "rfs_finish_state_fight_rounds_observed"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def logit(probability: float) -> float:
    p = clamp(float(probability), LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    return log(p / (1.0 - p))


def q_exp(units: float) -> float:
    return 1.0 - exp(-max(0.0, float(units)))


def k_factor(update_count: int) -> float:
    return BASE_K / sqrt(1.0 + float(update_count) / 6.0)


def percentile(pool: list[float], value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    if not pool:
        return 0.5
    return bisect_right(pool, float(value)) / len(pool)


def population_baseline(weighted_sum: float, quality_sum: float) -> float:
    if quality_sum <= 0.0:
        return 0.5
    return clamp(weighted_sum / quality_sum, 0.0, 1.0)


def expected_probability(rating: float, baseline: float) -> float:
    return sigmoid(logit(baseline) + (float(rating) - BASE_RATING) / RATING_SCALE)


def _date_column(df: pd.DataFrame) -> str:
    if "date" in df.columns:
        return "date"
    if "event_date" in df.columns:
        return "event_date"
    raise RuntimeError("RFS history has neither date nor event_date")


def replay_wrestling_entry(rfs: pd.DataFrame) -> pd.DataFrame:
    required = {
        "fight_id",
        "fighter_id",
        "fighter_name",
        TD_RATE_COL,
        ROUNDS_COL,
    }
    missing = sorted(required - set(rfs.columns))
    if missing:
        raise RuntimeError("RFS history missing required columns: " + ", ".join(missing))

    df = rfs.copy()
    date_col = _date_column(df)
    df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    df["fight_id"] = df["fight_id"].astype(str)
    df["fighter_id"] = df["fighter_id"].astype(str)

    if df.duplicated(["fight_id", "fighter_id"]).any():
        raise RuntimeError("RFS history violates fighter-fight grain")

    df = df.sort_values([date_col, "fight_id", "fighter_id"]).reset_index(drop=True)

    ratings: dict[str, float] = defaultdict(lambda: BASE_RATING)
    update_counts: dict[str, int] = defaultdict(int)
    td_rate_pool: list[float] = []
    weighted_obs_sum = 0.0
    quality_sum = 0.0
    snapshots: list[dict[str, object]] = []

    for fight_date, date_rows in df.groupby(date_col, sort=True):
        # Snapshot every fighter before any same-date results are incorporated.
        for fight_id, fight in date_rows.groupby("fight_id", sort=False):
            if len(fight) != 2:
                continue
            for row in fight.itertuples(index=False):
                fighter_id = str(row.fighter_id)
                snapshots.append(
                    {
                        "fight_id": str(fight_id),
                        "fighter_id": fighter_id,
                        "fighter_name": str(row.fighter_name),
                        "date": pd.Timestamp(fight_date),
                        "wrestling_entry": float(ratings[fighter_id]),
                    }
                )

        date_deltas: dict[str, float] = defaultdict(float)
        date_updates: dict[str, int] = defaultdict(int)
        date_weighted_obs_sum = 0.0
        date_quality_sum = 0.0

        for _, fight in date_rows.groupby("fight_id", sort=False):
            if len(fight) != 2:
                continue

            for _, row in fight.iterrows():
                fighter_id = str(row["fighter_id"])
                td_rate_raw = row.get(TD_RATE_COL)
                rounds_raw = row.get(ROUNDS_COL)

                td_rate = None if pd.isna(td_rate_raw) else float(td_rate_raw)
                rounds = 0.0 if pd.isna(rounds_raw) else float(rounds_raw)

                observation = percentile(td_rate_pool, td_rate)
                quality = (
                    q_exp(rounds / 2.0)
                    if observation is not None and rounds > 0.0
                    else 0.0
                )

                baseline = population_baseline(weighted_obs_sum, quality_sum)
                expected = expected_probability(ratings[fighter_id], baseline)

                if observation is not None and quality > 0.0:
                    delta = (
                        k_factor(update_counts[fighter_id])
                        * quality
                        * (float(observation) - expected)
                    )
                    date_updates[fighter_id] += 1
                    date_weighted_obs_sum += quality * float(observation)
                    date_quality_sum += quality
                else:
                    delta = 0.0

                date_deltas[fighter_id] += delta

        # Same-date updates are simultaneous.
        for fighter_id, delta in date_deltas.items():
            ratings[fighter_id] = clamp(
                ratings[fighter_id] + delta,
                MIN_RATING,
                MAX_RATING,
            )
            update_counts[fighter_id] += date_updates[fighter_id]

        weighted_obs_sum += date_weighted_obs_sum
        quality_sum += date_quality_sum

        # Current-date rate observations enter the percentile pool only after
        # every fight on the date is updated, preserving leakage safety.
        for value in date_rows[TD_RATE_COL]:
            if not pd.isna(value):
                insort(td_rate_pool, float(value))

    result = pd.DataFrame(snapshots)
    if result.empty:
        raise RuntimeError("wrestling_entry replay produced no snapshots")
    if result.duplicated(["fight_id", "fighter_id"]).any():
        raise RuntimeError("wrestling_entry replay produced duplicate keys")
    return result


def merge_into_base_fsr(
    base_fsr: pd.DataFrame,
    wrestling: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["fight_id", "fighter_id"]
    base = base_fsr.copy()
    base["fight_id"] = base["fight_id"].astype(str)
    base["fighter_id"] = base["fighter_id"].astype(str)

    if base.duplicated(keys).any():
        raise RuntimeError("FSR-32 prefight database violates fighter-fight grain")
    if "wrestling_entry" not in base.columns:
        raise RuntimeError("FSR-32 prefight database has no wrestling_entry column")

    new_trait = wrestling[keys + ["wrestling_entry"]].rename(
        columns={"wrestling_entry": "wrestling_entry_new"}
    )

    merged = base.merge(new_trait, on=keys, how="left", validate="one_to_one")
    missing = merged["wrestling_entry_new"].isna()
    if missing.any():
        sample = merged.loc[missing, keys].head(10).to_dict("records")
        raise RuntimeError(
            f"{int(missing.sum())} FSR-32 rows lack rebuilt wrestling_entry; sample={sample}"
        )

    audit_cols = [*keys]
    if "fighter_name" in merged.columns:
        audit_cols.append("fighter_name")
    audit_cols += ["wrestling_entry", "wrestling_entry_new"]
    audit = merged[audit_cols].copy()
    audit["delta"] = audit["wrestling_entry_new"] - audit["wrestling_entry"]

    merged["wrestling_entry"] = merged.pop("wrestling_entry_new")
    return merged, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace FSR-32 prefight parquet after creating a .bak copy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Shadow output path when not using --in-place.",
    )
    parser.add_argument(
        "--fight-id",
        help="Optional fight ID to print before/after wrestling_entry values.",
    )
    args = parser.parse_args()

    for path in (RFS_PATH, BASE_FSR_PATH):
        if not path.exists():
            raise RuntimeError(f"required input not found: {path}")

    print(f"[wrestling_entry] loading RFS: {RFS_PATH}", flush=True)
    rfs = pd.read_parquet(RFS_PATH)
    print(f"[wrestling_entry] loading FSR-32: {BASE_FSR_PATH}", flush=True)
    base_fsr = pd.read_parquet(BASE_FSR_PATH)

    print(
        f"[wrestling_entry] replaying ONE trait across {len(rfs):,} fighter-fight rows...",
        flush=True,
    )
    rebuilt = replay_wrestling_entry(rfs)
    updated, audit = merge_into_base_fsr(base_fsr, rebuilt)

    print("\nWRESTLING_ENTRY REBUILD SUMMARY")
    print(f"FSR-32 rows:    {len(updated):,}")
    print(f"mean old:       {audit['wrestling_entry'].mean():.3f}")
    print(f"mean new:       {audit['wrestling_entry_new'].mean():.3f}")
    print(f"median old:     {audit['wrestling_entry'].median():.3f}")
    print(f"median new:     {audit['wrestling_entry_new'].median():.3f}")
    print(f"min new:        {audit['wrestling_entry_new'].min():.3f}")
    print(f"max new:        {audit['wrestling_entry_new'].max():.3f}")
    print(f"mean |delta|:   {audit['delta'].abs().mean():.3f}")

    if args.fight_id:
        selected = audit.loc[audit["fight_id"].eq(str(args.fight_id))].copy()
        print(f"\nTARGET FIGHT: {args.fight_id}")
        if selected.empty:
            print("No matching FSR-32 rows.")
        else:
            display_cols = [
                col
                for col in (
                    "fighter_name",
                    "wrestling_entry",
                    "wrestling_entry_new",
                    "delta",
                )
                if col in selected.columns
            ]
            print(
                selected[display_cols].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.3f}",
                )
            )

    if args.in_place:
        backup_path = BASE_FSR_PATH.with_suffix(BASE_FSR_PATH.suffix + ".bak")
        shutil.copy2(BASE_FSR_PATH, backup_path)
        updated.to_parquet(BASE_FSR_PATH, index=False)
        print(f"\n[wrestling_entry] backup: {backup_path}")
        print(f"[wrestling_entry] updated FSR-32: {BASE_FSR_PATH}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        updated.to_parquet(args.output, index=False)
        audit_path = args.output.with_name(args.output.stem + "_audit.csv")
        audit.to_csv(audit_path, index=False)
        print(f"\n[wrestling_entry] shadow output: {args.output}")
        print(f"[wrestling_entry] audit: {audit_path}")


if __name__ == "__main__":
    main()
