"""Shadow Elo-style FSR family for ground striking.

Adds three candidate persistent ratings without modifying the locked 13 or
existing five dynamic-response ratings:

- ground_striking_pressure
- ground_striking_precision
- ground_striking_defense

All ratings use the established FSR update form:

    R_new = R_old + K * Q * (O - E)

Pressure is a population-centered fighter tendency. Precision and defense are
opponent-adjusted paired abilities. Population pools and baselines use only
prior-date evidence; same-date updates are simultaneous.

Shadow/research only.
"""

from __future__ import annotations

from bisect import bisect_right, insort
from collections import defaultdict
from math import exp, isfinite, log, sqrt
from pathlib import Path

import pandas as pd


RFS_PATH = Path("data/features/round_fighter_state_history.parquet")
OUTPUT_DIR = Path("data/simulation/rfs_mc_v2_shared_state/fsr_21_shadow")

BASE_RATING = 50.0
MIN_RATING = 10.0
MAX_RATING = 90.0
RATING_SCALE = 12.0
BASE_K = 7.0
LOGIT_EPSILON = 1e-4

SKILLS = (
    "ground_striking_pressure",
    "ground_striking_precision",
    "ground_striking_defense",
)

C = {
    "ground_attempts_per_round": "rfs_phase_base_fight_ground_attempts_per_round",
    "ground_attempt_share": "rfs_phase_base_fight_ground_attempt_share",
    "ground_attempts": "rfs_phase_interact_fight_ground_attempts",
    "ground_accuracy": "rfs_phase_interact_fight_ground_accuracy",
    "opp_ground_attempts": "rfs_phase_interact_fight_opp_ground_attempts",
    "ground_accuracy_allowed": "rfs_phase_interact_fight_ground_accuracy_allowed",
}

POOL_KEYS = (
    "ground_attempts_per_round",
    "ground_attempt_share",
    "ground_accuracy",
    "ground_accuracy_allowed",
)


def finite(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def logit(probability: float) -> float:
    selected = clamp(float(probability), LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    return log(selected / (1.0 - selected))


def k_factor(update_count: int) -> float:
    return BASE_K / sqrt(1.0 + float(update_count) / 6.0)


def q_exp(units: float) -> float:
    return 1.0 - exp(-max(0.0, float(units)))


def percentile(pool: list[float], value: float | None) -> float | None:
    if value is None:
        return None
    if not pool:
        return 0.5
    return bisect_right(pool, float(value)) / len(pool)


def row_value(row: pd.Series, key: str) -> float | None:
    return finite(row.get(C[key]))


def weighted_available(parts: tuple[tuple[float, float | None], ...]) -> float | None:
    available = [(weight, value) for weight, value in parts if value is not None]
    if not available:
        return None
    total = sum(weight for weight, _ in available)
    if total <= 0.0:
        return None
    return sum(weight * float(value) for weight, value in available) / total


def population_baseline(
    weighted_observation_sum: dict[str, float],
    quality_sum: dict[str, float],
    skill: str,
) -> float:
    total_quality = float(quality_sum[skill])
    if total_quality <= 0.0:
        return 0.50
    return clamp(float(weighted_observation_sum[skill]) / total_quality, 0.0, 1.0)


def expected_intrinsic(rating: float, baseline: float) -> float:
    return sigmoid(logit(baseline) + (float(rating) - BASE_RATING) / RATING_SCALE)


def expected_matchup(
    offense_rating: float,
    defense_rating: float,
    baseline: float,
) -> float:
    return sigmoid(
        logit(baseline)
        + (float(offense_rating) - float(defense_rating)) / RATING_SCALE
    )


def observation_bundle(
    row: pd.Series,
    pools: dict[str, list[float]],
) -> dict[str, tuple[float | None, float]]:
    ground_attempts = row_value(row, "ground_attempts") or 0.0
    opp_ground_attempts = row_value(row, "opp_ground_attempts") or 0.0

    pressure = weighted_available((
        (
            0.60,
            percentile(
                pools["ground_attempts_per_round"],
                row_value(row, "ground_attempts_per_round"),
            ),
        ),
        (
            0.40,
            percentile(
                pools["ground_attempt_share"],
                row_value(row, "ground_attempt_share"),
            ),
        ),
    ))

    # Pressure should not update from a fight with no observed ground offense.
    pressure_quality = q_exp(ground_attempts / 10.0) if ground_attempts > 0.0 else 0.0

    precision = percentile(
        pools["ground_accuracy"],
        row_value(row, "ground_accuracy"),
    )
    precision_quality = q_exp(ground_attempts / 10.0) if ground_attempts > 0.0 else 0.0

    allowed_pct = percentile(
        pools["ground_accuracy_allowed"],
        row_value(row, "ground_accuracy_allowed"),
    )
    defense = None if allowed_pct is None else 1.0 - allowed_pct
    defense_quality = (
        q_exp(opp_ground_attempts / 10.0)
        if opp_ground_attempts > 0.0
        else 0.0
    )

    return {
        "ground_striking_pressure": (
            pressure if pressure_quality > 0.0 else None,
            pressure_quality,
        ),
        "ground_striking_precision": (
            precision if precision_quality > 0.0 else None,
            precision_quality,
        ),
        "ground_striking_defense": (
            defense if defense_quality > 0.0 else None,
            defense_quality,
        ),
    }


def append_date_to_pools(date_rows: pd.DataFrame, pools: dict[str, list[float]]) -> None:
    for _, row in date_rows.iterrows():
        for pool_key, source_key in (
            ("ground_attempts_per_round", "ground_attempts_per_round"),
            ("ground_attempt_share", "ground_attempt_share"),
            ("ground_accuracy", "ground_accuracy"),
            ("ground_accuracy_allowed", "ground_accuracy_allowed"),
        ):
            value = row_value(row, source_key)
            if value is not None:
                insort(pools[pool_key], value)


def validate_columns(df: pd.DataFrame) -> None:
    required = {"fight_id", "fighter_id", "fighter_name", *C.values()}
    if "date" not in df.columns and "event_date" not in df.columns:
        required.add("date")
    missing = sorted(column for column in required if column not in df.columns)
    if missing:
        raise ValueError(
            f"RFS history missing required ground-striking FSR columns: {missing}"
        )


def build_prefight_snapshots(rfs: pd.DataFrame) -> pd.DataFrame:
    validate_columns(rfs)

    df = rfs.copy()
    date_col = "date" if "date" in df.columns else "event_date"
    df["date"] = pd.to_datetime(df[date_col], errors="raise")
    df["fight_id"] = df["fight_id"].astype(str)
    df["fighter_id"] = df["fighter_id"].astype(str)
    df = df.sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)

    ratings: dict[str, dict[str, float]] = defaultdict(
        lambda: {skill: BASE_RATING for skill in SKILLS}
    )
    update_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {skill: 0 for skill in SKILLS}
    )
    fight_counts: dict[str, int] = defaultdict(int)
    pools = {key: [] for key in POOL_KEYS}
    weighted_observation_sum: dict[str, float] = defaultdict(float)
    quality_sum: dict[str, float] = defaultdict(float)
    snapshots: list[dict[str, object]] = []

    for fight_date, date_rows in df.groupby("date", sort=True):
        for fight_id, fight in date_rows.groupby("fight_id", sort=False):
            if len(fight) != 2:
                continue
            for row in fight.itertuples(index=False):
                fighter_id = str(row.fighter_id)
                _ = ratings[fighter_id]
                snapshot: dict[str, object] = {
                    "fight_id": str(fight_id),
                    "date": pd.Timestamp(fight_date),
                    "fighter_id": fighter_id,
                    "fighter_name": str(row.fighter_name),
                    "prior_ufc_fights": int(fight_counts[fighter_id]),
                }
                snapshot.update(
                    {skill: float(ratings[fighter_id][skill]) for skill in SKILLS}
                )
                snapshot.update(
                    {
                        f"{skill}_updates": int(update_counts[fighter_id][skill])
                        for skill in SKILLS
                    }
                )
                snapshots.append(snapshot)

        date_deltas: dict[str, dict[str, float]] = defaultdict(
            lambda: {skill: 0.0 for skill in SKILLS}
        )
        date_updates: dict[str, dict[str, int]] = defaultdict(
            lambda: {skill: 0 for skill in SKILLS}
        )
        date_fights: dict[str, int] = defaultdict(int)
        date_weighted_sum: dict[str, float] = defaultdict(float)
        date_quality_sum: dict[str, float] = defaultdict(float)

        for _, fight in date_rows.groupby("fight_id", sort=False):
            if len(fight) != 2:
                continue
            rows = [row for _, row in fight.iterrows()]
            for index, row in enumerate(rows):
                opponent_row = rows[1 - index]
                fighter_id = str(row["fighter_id"])
                opponent_id = str(opponent_row["fighter_id"])
                _ = ratings[fighter_id]
                _ = ratings[opponent_id]

                bundle = observation_bundle(row, pools)
                for skill in SKILLS:
                    observation, quality = bundle[skill]
                    if observation is None or quality <= 0.0:
                        continue

                    baseline = population_baseline(
                        weighted_observation_sum,
                        quality_sum,
                        skill,
                    )
                    if skill == "ground_striking_pressure":
                        expected = expected_intrinsic(
                            ratings[fighter_id][skill],
                            baseline,
                        )
                    elif skill == "ground_striking_precision":
                        expected = expected_matchup(
                            ratings[fighter_id][skill],
                            ratings[opponent_id]["ground_striking_defense"],
                            baseline,
                        )
                    else:
                        expected = expected_matchup(
                            ratings[fighter_id][skill],
                            ratings[opponent_id]["ground_striking_precision"],
                            baseline,
                        )

                    delta = (
                        k_factor(update_counts[fighter_id][skill])
                        * quality
                        * (float(observation) - expected)
                    )
                    date_deltas[fighter_id][skill] += delta
                    date_updates[fighter_id][skill] += 1
                    date_weighted_sum[skill] += quality * float(observation)
                    date_quality_sum[skill] += quality

                date_fights[fighter_id] += 1

        for fighter_id, skill_deltas in date_deltas.items():
            for skill, delta in skill_deltas.items():
                ratings[fighter_id][skill] = clamp(
                    ratings[fighter_id][skill] + delta,
                    MIN_RATING,
                    MAX_RATING,
                )
                update_counts[fighter_id][skill] += date_updates[fighter_id][skill]

        for fighter_id, count in date_fights.items():
            fight_counts[fighter_id] += count
        for skill in SKILLS:
            weighted_observation_sum[skill] += date_weighted_sum[skill]
            quality_sum[skill] += date_quality_sum[skill]
        append_date_to_pools(date_rows, pools)

    out = pd.DataFrame(snapshots)
    if out.empty:
        raise RuntimeError("ground-striking FSR replay produced no snapshots")
    if out.duplicated(["fight_id", "fighter_id"]).any():
        raise RuntimeError("ground-striking FSR snapshots violate fighter-fight grain")
    return out


def main() -> None:
    if not RFS_PATH.exists():
        raise RuntimeError(f"RFS history not found: {RFS_PATH}")

    rfs = pd.read_parquet(RFS_PATH)
    snapshots = build_prefight_snapshots(rfs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "ground_striking_fsr_v1_prefight_snapshots.parquet"
    snapshots.to_parquet(output_path, index=False)
    print(
        f"Wrote {len(snapshots):,} ground-striking FSR pre-fight rows to {output_path}"
    )


if __name__ == "__main__":
    main()
