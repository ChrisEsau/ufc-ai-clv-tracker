"""Shadow Elo-style FSR family for clinch striking.

Adds three candidate persistent ratings:
- clinch_striking_pressure
- clinch_striking_precision
- clinch_striking_defense

Mirrors the locked ground-striking family. Pressure is a population-centered
fighter tendency; precision and defense are opponent-adjusted paired abilities.
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
OUTPUT_DIR = Path("data/simulation/rfs_mc_v2_shared_state/fsr_25_shadow")
BASE_RATING = 50.0
MIN_RATING = 10.0
MAX_RATING = 90.0
RATING_SCALE = 12.0
BASE_K = 7.0
LOGIT_EPSILON = 1e-4

SKILLS = (
    "clinch_striking_pressure",
    "clinch_striking_precision",
    "clinch_striking_defense",
)

C = {
    "clinch_attempts_per_round": "rfs_phase_base_fight_clinch_attempts_per_round",
    "clinch_attempt_share": "rfs_phase_base_fight_clinch_attempt_share",
    "clinch_attempts": "rfs_phase_interact_fight_clinch_attempts",
    "clinch_accuracy": "rfs_phase_interact_fight_clinch_accuracy",
    "opp_clinch_attempts": "rfs_phase_interact_fight_opp_clinch_attempts",
    "clinch_accuracy_allowed": "rfs_phase_interact_fight_clinch_accuracy_allowed",
}
POOL_KEYS = (
    "clinch_attempts_per_round",
    "clinch_attempt_share",
    "clinch_accuracy",
    "clinch_accuracy_allowed",
)


def finite(value):
    if value is None or pd.isna(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if isfinite(out) else None


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def sigmoid(v):
    return 1.0 / (1.0 + exp(-v))


def logit(p):
    p = clamp(float(p), LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    return log(p / (1.0 - p))


def k_factor(n):
    return BASE_K / sqrt(1.0 + float(n) / 6.0)


def q_exp(u):
    return 1.0 - exp(-max(0.0, float(u)))


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
    return (
        sum(weight * float(value) for weight, value in available) / total
        if total > 0
        else None
    )


def population_baseline(weighted_sum, quality_sum, skill):
    quality = float(quality_sum[skill])
    return (
        0.50
        if quality <= 0
        else clamp(float(weighted_sum[skill]) / quality, 0.0, 1.0)
    )


def expected_intrinsic(rating, baseline):
    return sigmoid(logit(baseline) + (float(rating) - BASE_RATING) / RATING_SCALE)


def expected_matchup(offense, defense, baseline):
    return sigmoid(
        logit(baseline) + (float(offense) - float(defense)) / RATING_SCALE
    )


def observation_bundle(row, pools):
    attempts = row_value(row, "clinch_attempts") or 0.0
    opp_attempts = row_value(row, "opp_clinch_attempts") or 0.0
    pressure = weighted_available(
        (
            (
                0.60,
                percentile(
                    pools["clinch_attempts_per_round"],
                    row_value(row, "clinch_attempts_per_round"),
                ),
            ),
            (
                0.40,
                percentile(
                    pools["clinch_attempt_share"],
                    row_value(row, "clinch_attempt_share"),
                ),
            ),
        )
    )
    pressure_quality = q_exp(attempts / 10.0) if attempts > 0 else 0.0
    precision = percentile(
        pools["clinch_accuracy"], row_value(row, "clinch_accuracy")
    )
    allowed = percentile(
        pools["clinch_accuracy_allowed"],
        row_value(row, "clinch_accuracy_allowed"),
    )
    defense = None if allowed is None else 1.0 - allowed
    defense_quality = q_exp(opp_attempts / 10.0) if opp_attempts > 0 else 0.0
    return {
        "clinch_striking_pressure": (
            pressure if pressure_quality > 0 else None,
            pressure_quality,
        ),
        "clinch_striking_precision": (
            precision if pressure_quality > 0 else None,
            pressure_quality,
        ),
        "clinch_striking_defense": (
            defense if defense_quality > 0 else None,
            defense_quality,
        ),
    }


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
            f"RFS history missing required clinch-striking FSR columns: {missing}"
        )


def build_prefight_snapshots(
    rfs: pd.DataFrame,
    *,
    progress_every_dates: int | None = None,
) -> pd.DataFrame:
    """Replay leakage-safe clinch ratings and return pre-fight snapshots.

    ``progress_every_dates`` is optional so unit tests/library callers remain
    quiet. Long-running command-line builders can pass a positive integer to
    emit an immediate-flush heartbeat every N event dates.
    """
    validate_columns(rfs)
    if progress_every_dates is not None and progress_every_dates <= 0:
        raise ValueError("progress_every_dates must be positive when provided")

    df = rfs.copy()
    date_col = "date" if "date" in df.columns else "event_date"
    df["date"] = pd.to_datetime(df[date_col], errors="raise")
    df["fight_id"] = df["fight_id"].astype(str)
    df["fighter_id"] = df["fighter_id"].astype(str)
    df = df.sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)

    ratings = defaultdict(lambda: {skill: BASE_RATING for skill in SKILLS})
    updates = defaultdict(lambda: {skill: 0 for skill in SKILLS})
    fights = defaultdict(int)
    pools = {key: [] for key in POOL_KEYS}
    weighted_sum = defaultdict(float)
    quality_sum = defaultdict(float)
    snapshots = []

    total_dates = int(df["date"].nunique())
    date_groups = df.groupby("date", sort=True)

    for date_index, (fight_date, date_rows) in enumerate(date_groups, start=1):
        if progress_every_dates is not None and (
            date_index == 1
            or date_index % progress_every_dates == 0
            or date_index == total_dates
        ):
            print(
                "[clinch FSR] "
                f"event dates {date_index:,}/{total_dates:,} "
                f"through {pd.Timestamp(fight_date).date()} | "
                f"snapshots {len(snapshots):,}",
                flush=True,
            )

        for fight_id, fight in date_rows.groupby("fight_id", sort=False):
            if len(fight) != 2:
                continue
            for row in fight.itertuples(index=False):
                fighter_id = str(row.fighter_id)
                _ = ratings[fighter_id]
                snapshot = {
                    "fight_id": str(fight_id),
                    "date": pd.Timestamp(fight_date),
                    "fighter_id": fighter_id,
                    "fighter_name": str(row.fighter_name),
                    "prior_ufc_fights": int(fights[fighter_id]),
                }
                snapshot.update(
                    {skill: float(ratings[fighter_id][skill]) for skill in SKILLS}
                )
                snapshot.update(
                    {
                        f"{skill}_updates": int(updates[fighter_id][skill])
                        for skill in SKILLS
                    }
                )
                snapshots.append(snapshot)

        deltas = defaultdict(lambda: {skill: 0.0 for skill in SKILLS})
        date_updates = defaultdict(lambda: {skill: 0 for skill in SKILLS})
        date_fights = defaultdict(int)
        date_weighted = defaultdict(float)
        date_quality = defaultdict(float)

        for _, fight in date_rows.groupby("fight_id", sort=False):
            if len(fight) != 2:
                continue
            rows = [row for _, row in fight.iterrows()]
            for index, row in enumerate(rows):
                opponent = rows[1 - index]
                fighter_id = str(row["fighter_id"])
                opponent_id = str(opponent["fighter_id"])
                _ = ratings[fighter_id]
                _ = ratings[opponent_id]
                bundle = observation_bundle(row, pools)
                for skill in SKILLS:
                    observation, quality = bundle[skill]
                    if observation is None or quality <= 0:
                        continue
                    baseline = population_baseline(
                        weighted_sum, quality_sum, skill
                    )
                    if skill == "clinch_striking_pressure":
                        expected = expected_intrinsic(
                            ratings[fighter_id][skill], baseline
                        )
                    elif skill == "clinch_striking_precision":
                        expected = expected_matchup(
                            ratings[fighter_id][skill],
                            ratings[opponent_id]["clinch_striking_defense"],
                            baseline,
                        )
                    else:
                        expected = expected_matchup(
                            ratings[fighter_id][skill],
                            ratings[opponent_id]["clinch_striking_precision"],
                            baseline,
                        )
                    delta = (
                        k_factor(updates[fighter_id][skill])
                        * quality
                        * (float(observation) - expected)
                    )
                    deltas[fighter_id][skill] += delta
                    date_updates[fighter_id][skill] += 1
                    date_weighted[skill] += quality * float(observation)
                    date_quality[skill] += quality
                date_fights[fighter_id] += 1

        for fighter_id, skill_deltas in deltas.items():
            for skill, delta in skill_deltas.items():
                ratings[fighter_id][skill] = clamp(
                    ratings[fighter_id][skill] + delta,
                    MIN_RATING,
                    MAX_RATING,
                )
                updates[fighter_id][skill] += date_updates[fighter_id][skill]
        for fighter_id, count in date_fights.items():
            fights[fighter_id] += count
        for skill in SKILLS:
            weighted_sum[skill] += date_weighted[skill]
            quality_sum[skill] += date_quality[skill]
        append_date_to_pools(date_rows, pools)

    out = pd.DataFrame(snapshots)
    if out.empty:
        raise RuntimeError("clinch-striking FSR replay produced no snapshots")
    if out.duplicated(["fight_id", "fighter_id"]).any():
        raise RuntimeError("clinch-striking FSR snapshots violate fighter-fight grain")
    return out


def main():
    if not RFS_PATH.exists():
        raise RuntimeError(f"RFS history not found: {RFS_PATH}")
    print(f"[clinch FSR] loading {RFS_PATH}", flush=True)
    snapshots = build_prefight_snapshots(
        pd.read_parquet(RFS_PATH),
        progress_every_dates=100,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "clinch_striking_fsr_v1_prefight_snapshots.parquet"
    print(f"[clinch FSR] writing {len(snapshots):,} rows", flush=True)
    snapshots.to_parquet(path, index=False)
    print(
        f"Wrote {len(snapshots):,} clinch-striking FSR pre-fight rows to {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
