"""Shadow Elo-style FSR extension for five dynamic-response fighter traits.

This module is intentionally isolated from the locked 13-skill FSR replay.
It adds five candidate persistent ratings:

- fatigue_accumulation_resistance
- fatigue_performance_resilience
- recovery_ability
- adversity_resistance
- adversity_recovery

The first four consume existing leakage-safe RFS fight observations. Recovery
ability uses a distinct non-adversity between-round rebound observation built
from authoritative UFCStats round rows so it does not duplicate adversity
recovery.

All ratings use the locked FSR V1.1 update form:

    R_new = R_old + K * Q * (O - E)

with skill-specific prior-date population baselines and simultaneous same-date
updates. Shadow/research only.
"""

from __future__ import annotations

from bisect import bisect_right, insort
from collections import defaultdict
from math import exp, isfinite, log, sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RFS_PATH = Path("data/features/round_fighter_state_history.parquet")
ROUND_PATH = Path("data/fight_details/ufc_round_stats.parquet")
OUTPUT_DIR = Path("data/simulation/rfs_mc_v2_shared_state/fsr_18_shadow")

BASE_RATING = 50.0
MIN_RATING = 10.0
MAX_RATING = 90.0
RATING_SCALE = 12.0
BASE_K = 7.0
LOGIT_EPSILON = 1e-4

SKILLS = (
    "fatigue_accumulation_resistance",
    "fatigue_performance_resilience",
    "recovery_ability",
    "adversity_resistance",
    "adversity_recovery",
)

C = {
    "rounds": "rfs_dynamic_response_fight_rounds_observed",
    "sig_attempts": "rfs_dynamic_response_fight_sig_strike_attempts",
    "td_attempts": "rfs_dynamic_response_fight_td_attempts",
    "control_seconds": "rfs_dynamic_response_fight_control_seconds",
    "sig_attempt_slope": "rfs_dynamic_response_fight_sig_strike_attempt_slope",
    "total_attempt_slope": "rfs_dynamic_response_fight_total_strike_attempt_slope",
    "td_attempt_slope": "rfs_dynamic_response_fight_td_attempt_slope",
    "control_slope": "rfs_dynamic_response_fight_control_seconds_slope",
    "sig_attempt_first_last": "rfs_dynamic_response_fight_sig_strike_attempt_first_last_ratio",
    "total_attempt_first_last": "rfs_dynamic_response_fight_total_strike_attempt_first_last_ratio",
    "late_early_workload": "rfs_dynamic_response_fight_late_early_workload_ratio",
    "late_early_output": "rfs_dynamic_response_fight_late_early_output_ratio",
    "sig_landed_slope": "rfs_dynamic_response_fight_sig_strike_landed_slope",
    "total_landed_slope": "rfs_dynamic_response_fight_total_strike_landed_slope",
    "sig_accuracy_change": "rfs_dynamic_response_fight_sig_strike_accuracy_change",
    "total_accuracy_change": "rfs_dynamic_response_fight_total_strike_accuracy_change",
    "adversity_round_count": "rfs_dynamic_response_fight_adversity_round_count",
    "post_adv_sig_rebound": "rfs_dynamic_response_fight_post_adversity_sig_strike_rebound",
    "post_adv_output_rebound": "rfs_dynamic_response_fight_post_adversity_output_rebound",
    "post_adv_efficiency": "rfs_dynamic_response_fight_post_adversity_efficiency_preservation",
    "same_round_output": "rfs_finish_state_fight_same_round_output_preservation",
    "same_round_efficiency": "rfs_finish_state_fight_same_round_efficiency_preservation",
    "kd_absorbed": "rfs_finish_state_fight_knockdowns_absorbed",
    "head_absorbed": "rfs_finish_state_fight_head_strikes_absorbed",
    "ground_absorbed": "rfs_finish_state_fight_ground_strikes_absorbed",
    "opp_control_seconds": "rfs_finish_state_fight_opponent_control_seconds",
}

POOL_KEYS = (
    "far_late_early_workload",
    "far_sig_first_last",
    "far_total_first_last",
    "far_sig_slope",
    "far_total_slope",
    "far_td_slope",
    "far_control_slope",
    "fpr_late_early_output",
    "fpr_sig_accuracy_change",
    "fpr_total_accuracy_change",
    "fpr_sig_landed_slope",
    "fpr_total_landed_slope",
    "recovery_workload",
    "recovery_output",
    "recovery_efficiency",
    "asr_output",
    "asr_efficiency",
    "asrec_sig_rebound",
    "asrec_output_rebound",
    "asrec_efficiency",
)

ROUND_REQUIRED = (
    "fight_id",
    "fighter_id",
    "round",
    "sig_str_landed",
    "sig_str_attempted",
    "total_str_landed",
    "total_str_attempted",
    "td_landed",
    "td_attempted",
    "ctrl_sec",
    "kd",
    "head_landed",
    "ground_landed",
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


def weighted_available(parts: Iterable[tuple[float, float | None]]) -> float | None:
    available = [(weight, value) for weight, value in parts if value is not None]
    if not available:
        return None
    total_weight = sum(weight for weight, _ in available)
    if total_weight <= 0.0:
        return None
    return sum(weight * float(value) for weight, value in available) / total_weight


def row_value(row: pd.Series, key: str) -> float | None:
    return finite(row.get(C[key]))


def population_baseline(
    weighted_observation_sum: dict[str, float],
    quality_sum: dict[str, float],
    skill: str,
) -> float:
    total_quality = float(quality_sum[skill])
    if total_quality <= 0.0:
        return 0.50
    return clamp(float(weighted_observation_sum[skill]) / total_quality, 0.0, 1.0)


def expected_probability(rating: float, baseline: float) -> float:
    return sigmoid(logit(baseline) + (float(rating) - BASE_RATING) / RATING_SCALE)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _accuracy(row: pd.Series, landed: str, attempted: str) -> float | None:
    return _safe_ratio(float(row[landed]), float(row[attempted]))


def _round_adversity_mask(
    fighter_rows: pd.DataFrame,
    opponent_rows: pd.DataFrame,
) -> pd.Series:
    fighter = fighter_rows.sort_values("round").reset_index(drop=True)
    opponent = opponent_rows.sort_values("round").reset_index(drop=True)
    if fighter["round"].tolist() != opponent["round"].tolist():
        raise ValueError("fighter and opponent round sets do not match")

    knockdown = pd.to_numeric(opponent["kd"], errors="coerce").fillna(0.0) > 0.0
    elevated_parts = []
    for column in ("head_landed", "ground_landed", "ctrl_sec"):
        values = pd.to_numeric(opponent[column], errors="coerce")
        elevated_parts.append(values > values.shift(1).cummax())
    elevated = pd.concat(elevated_parts, axis=1).any(axis=1).fillna(False)
    return (knockdown | elevated).astype(bool)


def _component_recovery(
    previous: float,
    dip: float,
    after: float,
) -> tuple[float | None, float]:
    loss = float(previous) - float(dip)
    if loss <= 0.0:
        return None, 0.0
    restored = clamp((float(after) - float(dip)) / loss, 0.0, 1.0)
    relative_loss = loss / max(abs(float(previous)), 1.0)
    return restored, q_exp(2.0 * relative_loss)


def build_non_adversity_recovery_observations(rounds: pd.DataFrame) -> pd.DataFrame:
    """Build ordinary between-round recovery evidence from non-adversity triplets."""
    missing = [column for column in ROUND_REQUIRED if column not in rounds.columns]
    if missing:
        raise ValueError(f"round stats missing required columns: {missing}")

    df = rounds.copy()
    df["fight_id"] = df["fight_id"].astype(str)
    df["fighter_id"] = df["fighter_id"].astype(str)
    for column in ROUND_REQUIRED[2:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[list(ROUND_REQUIRED[2:])].isna().any().any():
        raise ValueError("round stats contain nonnumeric required values")

    output_rows: list[dict[str, object]] = []
    for fight_id, fight in df.groupby("fight_id", sort=False):
        fighter_ids = fight["fighter_id"].drop_duplicates().tolist()
        if len(fighter_ids) != 2:
            continue
        for fighter_id in fighter_ids:
            fighter = fight.loc[fight["fighter_id"] == fighter_id].sort_values("round").reset_index(drop=True)
            opponent = fight.loc[fight["fighter_id"] != fighter_id].sort_values("round").reset_index(drop=True)
            workload_values: list[tuple[float, float]] = []
            output_values: list[tuple[float, float]] = []
            efficiency_values: list[tuple[float, float]] = []
            opportunities = 0

            if len(fighter) >= 3 and fighter["round"].tolist() == opponent["round"].tolist():
                adversity = _round_adversity_mask(fighter, opponent)
                for index in range(1, len(fighter) - 1):
                    if bool(adversity.iloc[index - 1:index + 2].any()):
                        continue
                    previous = fighter.iloc[index - 1]
                    dip = fighter.iloc[index]
                    after = fighter.iloc[index + 1]
                    had_opportunity = False

                    for container, columns in (
                        (workload_values, ("sig_str_attempted", "total_str_attempted", "td_attempted", "ctrl_sec")),
                        (output_values, ("sig_str_landed", "total_str_landed", "td_landed", "ctrl_sec")),
                    ):
                        parts: list[tuple[float, float]] = []
                        for column in columns:
                            observation, quality = _component_recovery(previous[column], dip[column], after[column])
                            if observation is not None:
                                parts.append((observation, quality))
                        if parts:
                            had_opportunity = True
                            weight_sum = sum(max(q, 1e-9) for _, q in parts)
                            container.append((
                                sum(obs * max(q, 1e-9) for obs, q in parts) / weight_sum,
                                max(q for _, q in parts),
                            ))

                    previous_accuracy = _accuracy(previous, "sig_str_landed", "sig_str_attempted")
                    dip_accuracy = _accuracy(dip, "sig_str_landed", "sig_str_attempted")
                    after_accuracy = _accuracy(after, "sig_str_landed", "sig_str_attempted")
                    if None not in (previous_accuracy, dip_accuracy, after_accuracy):
                        observation, quality = _component_recovery(
                            float(previous_accuracy),
                            float(dip_accuracy),
                            float(after_accuracy),
                        )
                        if observation is not None:
                            had_opportunity = True
                            efficiency_values.append((observation, quality))

                    if had_opportunity:
                        opportunities += 1

            def collapse(values: list[tuple[float, float]]) -> float:
                if not values:
                    return float("nan")
                weight_sum = sum(max(q, 1e-9) for _, q in values)
                return sum(obs * max(q, 1e-9) for obs, q in values) / weight_sum

            qualities = [q for _, q in workload_values + output_values + efficiency_values]
            output_rows.append({
                "fight_id": str(fight_id),
                "fighter_id": str(fighter_id),
                "recovery_workload": collapse(workload_values),
                "recovery_output": collapse(output_values),
                "recovery_efficiency": collapse(efficiency_values),
                "recovery_quality": min(1.0, sum(qualities) / 3.0) if qualities else 0.0,
                "recovery_opportunities": int(opportunities),
            })

    return pd.DataFrame(output_rows)


def _workload_quality(row: pd.Series) -> float:
    rounds = row_value(row, "rounds") or 0.0
    if rounds < 2.0:
        return 0.0
    sig = row_value(row, "sig_attempts") or 0.0
    td = row_value(row, "td_attempts") or 0.0
    control = row_value(row, "control_seconds") or 0.0
    work_units = sig / 60.0 + td / 4.0 + control / 180.0
    duration = 1.0 - exp(-(rounds - 1.0) / 2.0)
    return clamp(duration * q_exp(work_units), 0.0, 1.0)


def observation_bundle(
    row: pd.Series,
    pools: dict[str, list[float]],
) -> dict[str, tuple[float | None, float]]:
    result: dict[str, tuple[float | None, float]] = {}
    work_q = _workload_quality(row)

    far = weighted_available((
        (0.40, percentile(pools["far_late_early_workload"], row_value(row, "late_early_workload"))),
        (0.20, percentile(pools["far_sig_first_last"], row_value(row, "sig_attempt_first_last"))),
        (0.15, percentile(pools["far_total_first_last"], row_value(row, "total_attempt_first_last"))),
        (0.0625, percentile(pools["far_sig_slope"], row_value(row, "sig_attempt_slope"))),
        (0.0625, percentile(pools["far_total_slope"], row_value(row, "total_attempt_slope"))),
        (0.0625, percentile(pools["far_td_slope"], row_value(row, "td_attempt_slope"))),
        (0.0625, percentile(pools["far_control_slope"], row_value(row, "control_slope"))),
    ))
    result["fatigue_accumulation_resistance"] = (far, work_q if far is not None else 0.0)

    fpr = weighted_available((
        (0.40, percentile(pools["fpr_late_early_output"], row_value(row, "late_early_output"))),
        (0.25, percentile(pools["fpr_sig_accuracy_change"], row_value(row, "sig_accuracy_change"))),
        (0.15, percentile(pools["fpr_total_accuracy_change"], row_value(row, "total_accuracy_change"))),
        (0.10, percentile(pools["fpr_sig_landed_slope"], row_value(row, "sig_landed_slope"))),
        (0.10, percentile(pools["fpr_total_landed_slope"], row_value(row, "total_landed_slope"))),
    ))
    result["fatigue_performance_resilience"] = (fpr, work_q if fpr is not None else 0.0)

    recovery = weighted_available((
        (0.30, percentile(pools["recovery_workload"], finite(row.get("recovery_workload")))),
        (0.40, percentile(pools["recovery_output"], finite(row.get("recovery_output")))),
        (0.30, percentile(pools["recovery_efficiency"], finite(row.get("recovery_efficiency")))),
    ))
    recovery_q = finite(row.get("recovery_quality")) or 0.0
    result["recovery_ability"] = (recovery, clamp(recovery_q, 0.0, 1.0) if recovery is not None else 0.0)

    asr = weighted_available((
        (0.60, percentile(pools["asr_output"], row_value(row, "same_round_output"))),
        (0.40, percentile(pools["asr_efficiency"], row_value(row, "same_round_efficiency"))),
    ))
    adversity_count = row_value(row, "adversity_round_count") or 0.0
    kd = row_value(row, "kd_absorbed") or 0.0
    head = row_value(row, "head_absorbed") or 0.0
    ground = row_value(row, "ground_absorbed") or 0.0
    opp_control = row_value(row, "opp_control_seconds") or 0.0
    adversity_units = adversity_count + 1.5 * kd + head / 25.0 + ground / 15.0 + opp_control / 180.0
    asr_q = q_exp(adversity_units) if adversity_count > 0.0 else 0.0
    result["adversity_resistance"] = (asr, asr_q if asr is not None else 0.0)

    asrec = weighted_available((
        (0.30, percentile(pools["asrec_sig_rebound"], row_value(row, "post_adv_sig_rebound"))),
        (0.40, percentile(pools["asrec_output_rebound"], row_value(row, "post_adv_output_rebound"))),
        (0.30, percentile(pools["asrec_efficiency"], row_value(row, "post_adv_efficiency"))),
    ))
    asrec_q = q_exp(adversity_count) if asrec is not None and adversity_count > 0.0 else 0.0
    result["adversity_recovery"] = (asrec, asrec_q)
    return result


def append_date_to_pools(date_rows: pd.DataFrame, pools: dict[str, list[float]]) -> None:
    mapping = {
        "far_late_early_workload": "late_early_workload",
        "far_sig_first_last": "sig_attempt_first_last",
        "far_total_first_last": "total_attempt_first_last",
        "far_sig_slope": "sig_attempt_slope",
        "far_total_slope": "total_attempt_slope",
        "far_td_slope": "td_attempt_slope",
        "far_control_slope": "control_slope",
        "fpr_late_early_output": "late_early_output",
        "fpr_sig_accuracy_change": "sig_accuracy_change",
        "fpr_total_accuracy_change": "total_accuracy_change",
        "fpr_sig_landed_slope": "sig_landed_slope",
        "fpr_total_landed_slope": "total_landed_slope",
        "asr_output": "same_round_output",
        "asr_efficiency": "same_round_efficiency",
        "asrec_sig_rebound": "post_adv_sig_rebound",
        "asrec_output_rebound": "post_adv_output_rebound",
        "asrec_efficiency": "post_adv_efficiency",
    }
    for _, row in date_rows.iterrows():
        for pool_key, row_key in mapping.items():
            value = row_value(row, row_key)
            if value is not None:
                insort(pools[pool_key], value)
        for pool_key, column in (
            ("recovery_workload", "recovery_workload"),
            ("recovery_output", "recovery_output"),
            ("recovery_efficiency", "recovery_efficiency"),
        ):
            value = finite(row.get(column))
            if value is not None:
                insort(pools[pool_key], value)


def validate_columns(df: pd.DataFrame) -> None:
    required = {"fight_id", "fighter_id", "fighter_name", *C.values()}
    if "date" not in df.columns and "event_date" not in df.columns:
        required.add("date")
    missing = sorted(column for column in required if column not in df.columns)
    if missing:
        raise ValueError(f"RFS history missing required dynamic FSR columns: {missing}")


def build_prefight_snapshots(rfs: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    validate_columns(rfs)
    recovery = build_non_adversity_recovery_observations(rounds)

    df = rfs.copy()
    date_col = "date" if "date" in df.columns else "event_date"
    df["date"] = pd.to_datetime(df[date_col], errors="raise")
    df["fight_id"] = df["fight_id"].astype(str)
    df["fighter_id"] = df["fighter_id"].astype(str)
    df = df.merge(recovery, on=["fight_id", "fighter_id"], how="left", validate="one_to_one")
    df = df.sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)

    ratings: dict[str, dict[str, float]] = defaultdict(lambda: {skill: BASE_RATING for skill in SKILLS})
    update_counts: dict[str, dict[str, int]] = defaultdict(lambda: {skill: 0 for skill in SKILLS})
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
                snapshot.update({skill: float(ratings[fighter_id][skill]) for skill in SKILLS})
                snapshot.update({f"{skill}_updates": int(update_counts[fighter_id][skill]) for skill in SKILLS})
                snapshots.append(snapshot)

        date_deltas: dict[str, dict[str, float]] = defaultdict(lambda: {skill: 0.0 for skill in SKILLS})
        date_updates: dict[str, dict[str, int]] = defaultdict(lambda: {skill: 0 for skill in SKILLS})
        date_fights: dict[str, int] = defaultdict(int)
        date_weighted_sum: dict[str, float] = defaultdict(float)
        date_quality_sum: dict[str, float] = defaultdict(float)

        for _, fight in date_rows.groupby("fight_id", sort=False):
            if len(fight) != 2:
                continue
            for _, row in fight.iterrows():
                fighter_id = str(row["fighter_id"])
                _ = ratings[fighter_id]
                bundle = observation_bundle(row, pools)
                for skill in SKILLS:
                    observation, quality = bundle[skill]
                    baseline = population_baseline(weighted_observation_sum, quality_sum, skill)
                    expected = expected_probability(ratings[fighter_id][skill], baseline)
                    if observation is None or quality <= 0.0:
                        continue
                    delta = k_factor(update_counts[fighter_id][skill]) * quality * (float(observation) - expected)
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
        raise RuntimeError("dynamic FSR replay produced no snapshots")
    if out.duplicated(["fight_id", "fighter_id"]).any():
        raise RuntimeError("dynamic FSR snapshots violate fighter-fight grain")
    return out


def main() -> None:
    if not RFS_PATH.exists():
        raise RuntimeError(f"RFS history not found: {RFS_PATH}")
    if not ROUND_PATH.exists():
        raise RuntimeError(f"round stats not found: {ROUND_PATH}")

    rfs = pd.read_parquet(RFS_PATH)
    rounds = pd.read_parquet(ROUND_PATH)
    snapshots = build_prefight_snapshots(rfs, rounds)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "dynamic_fsr_v1_prefight_snapshots.parquet"
    snapshots.to_parquet(output_path, index=False)
    print(f"Wrote {len(snapshots):,} dynamic FSR pre-fight rows to {output_path}")


if __name__ == "__main__":
    main()
