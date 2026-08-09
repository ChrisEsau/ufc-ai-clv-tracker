"""Shadow Elo-style FSR rating for distance striking pressure.

Adds one genuinely new persistent rating:

- distance_striking_pressure

The existing locked `distance_precision` and `distance_defense` ratings are not
replayed here. FSR-26 promotes them to canonical compatibility aliases named
`distance_striking_precision` and `distance_striking_defense`.

Pressure mirrors the locked clinch/ground pressure semantics:

- 60% percentile(distance attempts per observed round)
- 40% percentile(distance attempt share)

All population pools/baselines are prior-date only and same-date updates are
simultaneous. Shadow/research only.
"""

from __future__ import annotations

from bisect import bisect_right, insort
from collections import defaultdict
from math import exp, isfinite, log, sqrt
from pathlib import Path

import pandas as pd

RFS_PATH = Path("data/features/round_fighter_state_history.parquet")
OUTPUT_DIR = Path("data/simulation/rfs_mc_v2_shared_state/fsr_26_shadow")

BASE_RATING = 50.0
MIN_RATING = 10.0
MAX_RATING = 90.0
RATING_SCALE = 12.0
BASE_K = 7.0
LOGIT_EPSILON = 1e-4
SKILL = "distance_striking_pressure"

C = {
    "distance_attempts_per_round": "rfs_phase_base_fight_distance_attempts_per_round",
    "distance_attempt_share": "rfs_phase_base_fight_distance_attempt_share",
    "distance_attempts": "rfs_phase_interact_fight_distance_attempts",
}

POOL_KEYS = (
    "distance_attempts_per_round",
    "distance_attempt_share",
)


def finite(value):
    if value is None or pd.isna(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def clamp(value, low, high):
    return max(low, min(high, value))


def sigmoid(value):
    return 1.0 / (1.0 + exp(-value))


def logit(probability):
    selected = clamp(float(probability), LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    return log(selected / (1.0 - selected))


def k_factor(update_count):
    return BASE_K / sqrt(1.0 + float(update_count) / 6.0)


def q_exp(units):
    return 1.0 - exp(-max(0.0, float(units)))


def percentile(pool, value):
    if value is None:
        return None
    if not pool:
        return 0.5
    return bisect_right(pool, float(value)) / len(pool)


def row_value(row, key):
    return finite(row.get(C[key]))


def weighted_available(parts):
    available = [(weight, value) for weight, value in parts if value is not None]
    if not available:
        return None
    total = sum(weight for weight, _ in available)
    if total <= 0.0:
        return None
    return sum(weight * float(value) for weight, value in available) / total


def population_baseline(weighted_sum, quality_sum):
    if quality_sum <= 0.0:
        return 0.50
    return clamp(float(weighted_sum) / float(quality_sum), 0.0, 1.0)


def expected_intrinsic(rating, baseline):
    return sigmoid(logit(baseline) + (float(rating) - BASE_RATING) / RATING_SCALE)


def observation(row, pools):
    attempts = row_value(row, "distance_attempts") or 0.0
    if attempts <= 0.0:
        return None, 0.0

    obs = weighted_available((
        (
            0.60,
            percentile(
                pools["distance_attempts_per_round"],
                row_value(row, "distance_attempts_per_round"),
            ),
        ),
        (
            0.40,
            percentile(
                pools["distance_attempt_share"],
                row_value(row, "distance_attempt_share"),
            ),
        ),
    ))
    return obs, q_exp(attempts / 10.0)


def append_date_to_pools(date_rows, pools):
    for _, row in date_rows.iterrows():
        for key in POOL_KEYS:
            value = row_value(row, key)
            if value is not None:
                insort(pools[key], value)


def validate_columns(df):
    required = {"fight_id", "fighter_id", "fighter_name", *C.values()}
    if "date" not in df.columns and "event_date" not in df.columns:
        required.add("date")
    missing = sorted(column for column in required if column not in df.columns)
    if missing:
        raise ValueError(
            f"RFS history missing required distance-pressure FSR columns: {missing}"
        )


def build_prefight_snapshots(rfs: pd.DataFrame, *, progress: bool = False) -> pd.DataFrame:
    validate_columns(rfs)
    df = rfs.copy()
    date_col = "date" if "date" in df.columns else "event_date"
    df["date"] = pd.to_datetime(df[date_col], errors="raise")
    df["fight_id"] = df["fight_id"].astype(str)
    df["fighter_id"] = df["fighter_id"].astype(str)
    df = df.sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)

    ratings = defaultdict(lambda: BASE_RATING)
    updates = defaultdict(int)
    fights = defaultdict(int)
    pools = {key: [] for key in POOL_KEYS}
    weighted_sum = 0.0
    quality_sum = 0.0
    snapshots = []

    grouped_dates = list(df.groupby("date", sort=True))
    total_dates = len(grouped_dates)

    for date_index, (fight_date, date_rows) in enumerate(grouped_dates, start=1):
        if progress and (date_index == 1 or date_index % 100 == 0 or date_index == total_dates):
            print(
                f"[distance FSR] event dates {date_index}/{total_dates} through "
                f"{pd.Timestamp(fight_date).date()} | snapshots {len(snapshots):,}",
                flush=True,
            )

        for fight_id, fight in date_rows.groupby("fight_id", sort=False):
            if len(fight) != 2:
                continue
            for row in fight.itertuples(index=False):
                fighter_id = str(row.fighter_id)
                snapshots.append({
                    "fight_id": str(fight_id),
                    "date": pd.Timestamp(fight_date),
                    "fighter_id": fighter_id,
                    "fighter_name": str(row.fighter_name),
                    "prior_ufc_fights": int(fights[fighter_id]),
                    SKILL: float(ratings[fighter_id]),
                    f"{SKILL}_updates": int(updates[fighter_id]),
                })

        date_deltas = defaultdict(float)
        date_updates = defaultdict(int)
        date_fights = defaultdict(int)
        date_weighted_sum = 0.0
        date_quality_sum = 0.0

        for _, fight in date_rows.groupby("fight_id", sort=False):
            if len(fight) != 2:
                continue
            for _, row in fight.iterrows():
                fighter_id = str(row["fighter_id"])
                obs, quality = observation(row, pools)
                if obs is not None and quality > 0.0:
                    baseline = population_baseline(weighted_sum, quality_sum)
                    expected = expected_intrinsic(ratings[fighter_id], baseline)
                    delta = (
                        k_factor(updates[fighter_id])
                        * quality
                        * (float(obs) - expected)
                    )
                    date_deltas[fighter_id] += delta
                    date_updates[fighter_id] += 1
                    date_weighted_sum += quality * float(obs)
                    date_quality_sum += quality
                date_fights[fighter_id] += 1

        for fighter_id, delta in date_deltas.items():
            ratings[fighter_id] = clamp(
                ratings[fighter_id] + delta,
                MIN_RATING,
                MAX_RATING,
            )
            updates[fighter_id] += date_updates[fighter_id]
        for fighter_id, count in date_fights.items():
            fights[fighter_id] += count

        weighted_sum += date_weighted_sum
        quality_sum += date_quality_sum
        append_date_to_pools(date_rows, pools)

    out = pd.DataFrame(snapshots)
    if out.empty:
        raise RuntimeError("distance-pressure FSR replay produced no snapshots")
    if out.duplicated(["fight_id", "fighter_id"]).any():
        raise RuntimeError("distance-pressure FSR snapshots violate fighter-fight grain")
    return out


def main():
    if not RFS_PATH.exists():
        raise RuntimeError(f"RFS history not found: {RFS_PATH}")
    snapshots = build_prefight_snapshots(pd.read_parquet(RFS_PATH), progress=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "distance_striking_pressure_v1_prefight_snapshots.parquet"
    snapshots.to_parquet(path, index=False)
    print(f"Wrote {len(snapshots):,} distance-pressure FSR pre-fight rows to {path}")


if __name__ == "__main__":
    main()
