"""Shadow Elo-style FSR rating for reversal ability.

Adds one candidate persistent rating:

- reversal_ability

UFCStats records reversals directly as ``rev``. The RFS Phase Interaction layer
carries those counts forward together with opponent control seconds. This
replay treats opponent control as the opportunity exposure: zero reversals
while controlled are negative evidence, while recorded reversals are positive
evidence whose strength also reflects reversal rate per opponent-control time.

The matchup expectation uses the opponent's leakage-safe pre-fight
``control_imposition`` rating from the existing FSR-21 database.

All updates use the established FSR form:

    R_new = R_old + K * Q * (O - E)

with prior-date population baselines and simultaneous same-date updates.
Shadow/research only.
"""

from __future__ import annotations

from bisect import bisect_right, insort
from collections import defaultdict
from math import exp, isfinite, log, sqrt
from pathlib import Path

import pandas as pd


RFS_PATH = Path("data/features/round_fighter_state_history.parquet")
FSR21_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_21_shadow/"
    "fsr_21_prefight_snapshots.parquet"
)
OUTPUT_DIR = Path("data/simulation/rfs_mc_v2_shared_state/fsr_22_shadow")

BASE_RATING = 50.0
MIN_RATING = 10.0
MAX_RATING = 90.0
RATING_SCALE = 12.0
BASE_K = 7.0
LOGIT_EPSILON = 1e-4
SKILL = "reversal_ability"

C = {
    "reversals": "rfs_phase_interact_fight_reversals",
    "opp_control_seconds": "rfs_phase_interact_fight_opp_control_seconds",
}


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


def population_baseline(weighted_sum: float, quality_sum: float) -> float:
    if quality_sum <= 0.0:
        return 0.50
    return clamp(float(weighted_sum) / float(quality_sum), 0.0, 1.0)


def expected_matchup(
    reversal_rating: float,
    opponent_control_imposition: float,
    baseline: float,
) -> float:
    return sigmoid(
        logit(baseline)
        + (float(reversal_rating) - float(opponent_control_imposition))
        / RATING_SCALE
    )


def observation(
    reversals: float,
    opponent_control_seconds: float,
    positive_rate_pool: list[float],
) -> tuple[float | None, float, float | None]:
    """Return reversal observation, quality, and positive reversal rate.

    A fighter must have observed opponent-control exposure to receive an
    update. Zero reversals under control are explicit negative evidence.
    Positive observations blend event occurrence (60%) with the percentile of
    reversals per 15 minutes of opponent control (40%).
    """

    rev = max(0.0, float(reversals))
    opp_ctrl = max(0.0, float(opponent_control_seconds))
    if opp_ctrl <= 0.0:
        return None, 0.0, None

    # Exposure makes a zero more informative; an observed reversal also adds
    # event evidence so a short successful scramble is not discarded.
    quality = q_exp(opp_ctrl / 180.0 + rev)

    if rev <= 0.0:
        return 0.0, quality, None

    rate_per_15min = rev / (opp_ctrl / 900.0)
    rate_pct = percentile(positive_rate_pool, rate_per_15min)
    if rate_pct is None:
        rate_pct = 0.5
    obs = 0.60 + 0.40 * float(rate_pct)
    return clamp(obs, 0.0, 1.0), quality, rate_per_15min


def validate_inputs(rfs: pd.DataFrame, fsr21: pd.DataFrame) -> None:
    rfs_required = {"fight_id", "fighter_id", "fighter_name", *C.values()}
    if "date" not in rfs.columns and "event_date" not in rfs.columns:
        rfs_required.add("date")
    missing_rfs = sorted(c for c in rfs_required if c not in rfs.columns)
    if missing_rfs:
        raise ValueError(f"RFS history missing reversal columns: {missing_rfs}")

    fsr_required = {"fight_id", "fighter_id", "control_imposition"}
    missing_fsr = sorted(c for c in fsr_required if c not in fsr21.columns)
    if missing_fsr:
        raise ValueError(f"FSR-21 missing reversal matchup columns: {missing_fsr}")


def build_prefight_snapshots(rfs: pd.DataFrame, fsr21: pd.DataFrame) -> pd.DataFrame:
    validate_inputs(rfs, fsr21)

    df = rfs.copy()
    date_col = "date" if "date" in df.columns else "event_date"
    df["date"] = pd.to_datetime(df[date_col], errors="raise")
    df["fight_id"] = df["fight_id"].astype(str)
    df["fighter_id"] = df["fighter_id"].astype(str)

    base = fsr21[["fight_id", "fighter_id", "control_imposition"]].copy()
    base["fight_id"] = base["fight_id"].astype(str)
    base["fighter_id"] = base["fighter_id"].astype(str)

    if base.duplicated(["fight_id", "fighter_id"]).any():
        raise ValueError("FSR-21 violates fighter-fight grain")

    df = df.merge(
        base,
        on=["fight_id", "fighter_id"],
        how="left",
        validate="one_to_one",
    )
    if df["control_imposition"].isna().any():
        raise ValueError("RFS/FSR-21 fighter-fight keys do not align")

    df = df.sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)

    ratings: dict[str, float] = defaultdict(lambda: BASE_RATING)
    update_counts: dict[str, int] = defaultdict(int)
    fight_counts: dict[str, int] = defaultdict(int)
    positive_rate_pool: list[float] = []
    weighted_observation_sum = 0.0
    quality_sum = 0.0
    snapshots: list[dict[str, object]] = []

    for fight_date, date_rows in df.groupby("date", sort=True):
        # Snapshot all ratings before any fights on this date update.
        for fight_id, fight in date_rows.groupby("fight_id", sort=False):
            if len(fight) != 2:
                continue
            for row in fight.itertuples(index=False):
                fighter_id = str(row.fighter_id)
                snapshots.append(
                    {
                        "fight_id": str(fight_id),
                        "date": pd.Timestamp(fight_date),
                        "fighter_id": fighter_id,
                        "fighter_name": str(row.fighter_name),
                        "prior_ufc_fights": int(fight_counts[fighter_id]),
                        SKILL: float(ratings[fighter_id]),
                        f"{SKILL}_updates": int(update_counts[fighter_id]),
                    }
                )

        date_deltas: dict[str, float] = defaultdict(float)
        date_updates: dict[str, int] = defaultdict(int)
        date_fights: dict[str, int] = defaultdict(int)
        date_weighted_sum = 0.0
        date_quality_sum = 0.0
        date_positive_rates: list[float] = []

        for _, fight in date_rows.groupby("fight_id", sort=False):
            if len(fight) != 2:
                continue
            rows = [row for _, row in fight.iterrows()]
            for index, row in enumerate(rows):
                opponent_row = rows[1 - index]
                fighter_id = str(row["fighter_id"])
                reversals = finite(row.get(C["reversals"])) or 0.0
                opp_ctrl = finite(row.get(C["opp_control_seconds"])) or 0.0

                obs, quality, positive_rate = observation(
                    reversals,
                    opp_ctrl,
                    positive_rate_pool,
                )
                if obs is not None and quality > 0.0:
                    baseline = population_baseline(
                        weighted_observation_sum,
                        quality_sum,
                    )
                    opponent_control = float(opponent_row["control_imposition"])
                    expected = expected_matchup(
                        ratings[fighter_id],
                        opponent_control,
                        baseline,
                    )
                    delta = (
                        k_factor(update_counts[fighter_id])
                        * quality
                        * (float(obs) - expected)
                    )
                    date_deltas[fighter_id] += delta
                    date_updates[fighter_id] += 1
                    date_weighted_sum += quality * float(obs)
                    date_quality_sum += quality
                    if positive_rate is not None:
                        date_positive_rates.append(float(positive_rate))

                date_fights[fighter_id] += 1

        for fighter_id, delta in date_deltas.items():
            ratings[fighter_id] = clamp(
                ratings[fighter_id] + delta,
                MIN_RATING,
                MAX_RATING,
            )
            update_counts[fighter_id] += date_updates[fighter_id]
        for fighter_id, count in date_fights.items():
            fight_counts[fighter_id] += count

        weighted_observation_sum += date_weighted_sum
        quality_sum += date_quality_sum
        for rate in date_positive_rates:
            insort(positive_rate_pool, rate)

    out = pd.DataFrame(snapshots)
    if out.empty:
        raise RuntimeError("reversal FSR replay produced no snapshots")
    if out.duplicated(["fight_id", "fighter_id"]).any():
        raise RuntimeError("reversal FSR snapshots violate fighter-fight grain")
    return out


def main() -> None:
    if not RFS_PATH.exists():
        raise RuntimeError(f"RFS history not found: {RFS_PATH}")
    if not FSR21_PATH.exists():
        raise RuntimeError(f"FSR-21 database not found: {FSR21_PATH}")

    rfs = pd.read_parquet(RFS_PATH)
    fsr21 = pd.read_parquet(FSR21_PATH)
    snapshots = build_prefight_snapshots(rfs, fsr21)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "reversal_fsr_v1_prefight_snapshots.parquet"
    snapshots.to_parquet(output_path, index=False)
    print(f"Wrote {len(snapshots):,} reversal FSR pre-fight rows to {output_path}")


if __name__ == "__main__":
    main()
