"""Shadow historical FSR replay for the locked equation families.

This script is intentionally isolated from production.  It replays only fights
strictly before a requested target fight and produces a PRE-target fighter card
for the equation families already reviewed in the RFS/FSR ontology work.

Usage
-----
PYTHONPATH=. python scripts/experimental/fsr_locked_families_v1.py <fight_id>

Output
------
data/simulation/rfs_mc_v2_shared_state/
    fsr_<fight_id>_locked_families_v1_target_card.csv
    fsr_<fight_id>_locked_families_v1_rating_history.csv
"""

from __future__ import annotations

import argparse
from bisect import bisect_right, insort
from collections import defaultdict
from math import exp, isfinite, sqrt
from pathlib import Path

import pandas as pd


RFS_PATH = Path("data/features/round_fighter_state_history.parquet")
OUTPUT_DIR = Path("data/simulation/rfs_mc_v2_shared_state")

BASE_RATING = 50.0
MIN_RATING = 10.0
MAX_RATING = 90.0
RATING_SCALE = 12.0
BASE_K = 7.0
DAMAGE_FAILURE_HURDLE = 10.0

SKILLS = (
    "distance_precision",
    "distance_defense",
    "wrestling_entry",
    "wrestling_conversion",
    "td_defense",
    "control_imposition",
    "control_resistance",
    "submission_pressure",
    "submission_conversion",
    "submission_resistance",
    "striking_power",
    "chin_resistance",
    "damage_resistance",
)

C = {
    "distance_accuracy": "rfs_phase_interact_fight_distance_accuracy",
    "distance_accuracy_allowed": "rfs_phase_interact_fight_distance_accuracy_allowed",
    "distance_attempts": "rfs_phase_interact_fight_distance_attempts",
    "opp_distance_attempts": "rfs_phase_interact_fight_opp_distance_attempts",
    "td_pressure_share": "rfs_phase_interact_fight_td_pressure_share",
    "td_attempts": "rfs_phase_interact_fight_td_attempts",
    "opp_td_attempts": "rfs_phase_interact_fight_opp_td_attempts",
    "td_attempts_per_round": "rfs_phase_base_fight_td_attempts_per_round",
    "td_completion_rate": "rfs_phase_base_fight_td_completion_rate",
    "control_seconds_per_td": "rfs_phase_base_fight_control_seconds_per_td_landed",
    "td_defense_rate": "rfs_phase_interact_fight_td_defense_rate",
    "control_share": "rfs_phase_interact_fight_control_share",
    "control_seconds_per_round": "rfs_phase_base_fight_control_seconds_per_round",
    "control_exchange_balance": "rfs_phase_interact_fight_control_exchange_balance",
    "ground_pressure_share": "rfs_phase_interact_fight_ground_pressure_share",
    "control_seconds": "rfs_phase_interact_fight_control_seconds",
    "opp_control_seconds": "rfs_phase_interact_fight_opp_control_seconds",
    "control_seconds_allowed_per_round": "rfs_phase_interact_fight_control_seconds_allowed_per_round",
    "reversal_rate": "rfs_phase_interact_fight_reversal_rate_per_opponent_control_min",
    "ground_landed_allowed_per_control_min": "rfs_phase_interact_fight_ground_landed_allowed_per_control_min",
    "sub_attempts_allowed_per_control_min": "rfs_phase_interact_fight_sub_attempts_allowed_per_control_min",
    "rounds": "rfs_finish_state_fight_rounds_observed",
    "sub_attempts": "rfs_finish_state_fight_submission_attempts",
    "sub_per_ground_opp": "rfs_finish_state_fight_submission_attempts_per_ground_opportunity_proxy",
    "sub_per_control_min": "rfs_finish_state_fight_submission_attempts_per_control_minute",
    "td_landed": "rfs_finish_state_fight_takedowns_landed",
    "ground_attempts": "rfs_finish_state_fight_ground_strike_attempts",
    "finish_control_seconds": "rfs_finish_state_fight_control_seconds",
    "opp_sub_attempts": "rfs_finish_state_fight_opponent_submission_attempts",
    "submission_loss": "rfs_finish_state_fight_submission_loss_indicator",
    "sig_landed": "rfs_finish_state_fight_sig_strikes_landed",
    "kd_scored": "rfs_finish_state_fight_knockdowns_scored",
    "kd_absorbed": "rfs_finish_state_fight_knockdowns_absorbed",
    "ko_loss": "rfs_finish_state_fight_ko_tko_loss_indicator",
    "sig_absorbed": "rfs_finish_state_fight_sig_strikes_absorbed",
}

POOL_KEYS = (
    "distance_accuracy",
    "distance_accuracy_allowed",
    "td_pressure_share",
    "td_attempts_per_round",
    "td_completion_rate",
    "control_seconds_per_td",
    "td_defense_rate",
    "control_share",
    "control_seconds_per_round",
    "control_exchange_balance",
    "ground_pressure_share",
    "control_seconds_allowed_per_round",
    "reversal_rate",
    "ground_landed_allowed_per_control_min",
    "sub_attempts_allowed_per_control_min",
    "sub_per_ground_opp_positive",
    "sub_per_control_min_positive",
    "sub_per_round_positive",
    "kd_rate_positive",
    "damage_stress",
)


def finite(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if isfinite(out) else None


def row_value(row: pd.Series, key: str) -> float | None:
    return finite(row.get(C[key]))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def k_factor(update_count: int) -> float:
    return BASE_K / sqrt(1.0 + float(update_count) / 6.0)


def q_exp(units: float) -> float:
    return 1.0 - exp(-max(0.0, units))


def percentile(pool: list[float], value: float | None) -> float | None:
    if value is None:
        return None
    if not pool:
        return 0.5
    return bisect_right(pool, float(value)) / len(pool)


def weighted_available(parts: tuple[tuple[float, float | None], ...]) -> float | None:
    available = [(w, v) for w, v in parts if v is not None]
    if not available:
        return None
    weight_sum = sum(w for w, _ in available)
    if weight_sum <= 0.0:
        return None
    return sum(w * float(v) for w, v in available) / weight_sum


def expected_probability(
    ratings: dict[str, dict[str, float]],
    fighter_id: str,
    opponent_id: str,
    skill: str,
) -> float:
    own = ratings[fighter_id][skill]

    if skill == "distance_precision":
        defense = ratings[opponent_id]["distance_defense"]
    elif skill == "distance_defense":
        defense = ratings[opponent_id]["distance_precision"]
    elif skill == "wrestling_entry":
        defense = BASE_RATING
    elif skill == "wrestling_conversion":
        defense = ratings[opponent_id]["td_defense"]
    elif skill == "td_defense":
        defense = ratings[opponent_id]["wrestling_conversion"]
    elif skill == "control_imposition":
        defense = ratings[opponent_id]["control_resistance"]
    elif skill == "control_resistance":
        defense = ratings[opponent_id]["control_imposition"]
    elif skill == "submission_pressure":
        defense = ratings[opponent_id]["submission_resistance"]
    elif skill == "submission_conversion":
        defense = ratings[opponent_id]["submission_resistance"]
    elif skill == "submission_resistance":
        defense = ratings[opponent_id]["submission_conversion"]
    elif skill == "striking_power":
        defense = BASE_RATING
    elif skill in {"chin_resistance", "damage_resistance"}:
        defense = ratings[opponent_id]["striking_power"]
    else:
        defense = BASE_RATING

    return sigmoid((own - defense) / RATING_SCALE)


def observation_bundle(
    row: pd.Series,
    opponent_row: pd.Series,
    pools: dict[str, list[float]],
) -> dict[str, tuple[float | None, float]]:
    result: dict[str, tuple[float | None, float]] = {}

    distance_accuracy = row_value(row, "distance_accuracy")
    distance_attempts = row_value(row, "distance_attempts") or 0.0
    result["distance_precision"] = (
        percentile(pools["distance_accuracy"], distance_accuracy),
        q_exp(distance_attempts / 20.0),
    ) if distance_attempts > 0 else (None, 0.0)

    allowed = row_value(row, "distance_accuracy_allowed")
    opp_distance_attempts = row_value(row, "opp_distance_attempts") or 0.0
    allowed_pct = percentile(pools["distance_accuracy_allowed"], allowed)
    result["distance_defense"] = (
        None if allowed_pct is None else 1.0 - allowed_pct,
        q_exp(opp_distance_attempts / 20.0),
    ) if opp_distance_attempts > 0 else (None, 0.0)

    td_attempts = row_value(row, "td_attempts") or 0.0

    # LEGACY WRESTLING-ENTRY DEFINITION — retained for audit/recovery only.
    # This mixed initiation, conversion and downstream control into one trait:
    # entry_obs = weighted_available(
    #     (
    #         (0.40, percentile(pools["td_pressure_share"], row_value(row, "td_pressure_share"))),
    #         (0.25, percentile(pools["td_attempts_per_round"], row_value(row, "td_attempts_per_round"))),
    #         (0.20, percentile(pools["td_completion_rate"], row_value(row, "td_completion_rate"))),
    #         (0.15, percentile(pools["control_seconds_per_td"], row_value(row, "control_seconds_per_td"))),
    #     )
    # )
    # result["wrestling_entry"] = (
    #     entry_obs if td_attempts > 0 else None,
    #     q_exp(0.5 * td_attempts) if td_attempts > 0 else 0.0,
    # )

    # Canonical wrestling_entry is now a single-purpose initiation trait:
    # how frequently the fighter attempts takedowns.  Zero-attempt fights are
    # real low-entry observations, and confidence comes from fight exposure
    # rather than from the number of attempts themselves.
    td_attempts_per_round = row_value(row, "td_attempts_per_round")
    entry_rounds = row_value(row, "rounds") or 0.0
    entry_obs = percentile(
        pools["td_attempts_per_round"],
        td_attempts_per_round,
    )
    result["wrestling_entry"] = (
        entry_obs,
        q_exp(entry_rounds / 2.0)
        if entry_obs is not None and entry_rounds > 0.0
        else 0.0,
    )

    conversion_obs = weighted_available(
        (
            (0.75, percentile(pools["td_completion_rate"], row_value(row, "td_completion_rate"))),
            (0.25, percentile(pools["control_seconds_per_td"], row_value(row, "control_seconds_per_td"))),
        )
    )
    result["wrestling_conversion"] = (
        conversion_obs if td_attempts > 0 else None,
        q_exp(td_attempts) if td_attempts > 0 else 0.0,
    )

    opp_td_attempts = row_value(row, "opp_td_attempts") or 0.0
    td_def_pct = percentile(pools["td_defense_rate"], row_value(row, "td_defense_rate"))
    result["td_defense"] = (
        td_def_pct if opp_td_attempts > 0 else None,
        q_exp(0.5 * opp_td_attempts) if opp_td_attempts > 0 else 0.0,
    )

    own_control = row_value(row, "control_seconds") or 0.0
    opp_control = row_value(row, "opp_control_seconds") or 0.0
    control_obs = weighted_available(
        (
            (0.35, percentile(pools["control_share"], row_value(row, "control_share"))),
            (0.25, percentile(pools["control_seconds_per_round"], row_value(row, "control_seconds_per_round"))),
            (0.25, percentile(pools["control_exchange_balance"], row_value(row, "control_exchange_balance"))),
            (0.15, percentile(pools["ground_pressure_share"], row_value(row, "ground_pressure_share"))),
        )
    )
    control_opportunity = own_control + opp_control
    result["control_imposition"] = (
        control_obs if control_opportunity > 0 else None,
        q_exp(control_opportunity / 60.0) if control_opportunity > 0 else 0.0,
    )

    p_control = percentile(
        pools["control_seconds_allowed_per_round"],
        row_value(row, "control_seconds_allowed_per_round"),
    )
    p_ground = percentile(
        pools["ground_landed_allowed_per_control_min"],
        row_value(row, "ground_landed_allowed_per_control_min"),
    )
    p_sub = percentile(
        pools["sub_attempts_allowed_per_control_min"],
        row_value(row, "sub_attempts_allowed_per_control_min"),
    )
    resistance_obs = weighted_available(
        (
            (0.45, None if p_control is None else 1.0 - p_control),
            (0.25, percentile(pools["reversal_rate"], row_value(row, "reversal_rate"))),
            (0.20, None if p_ground is None else 1.0 - p_ground),
            (0.10, None if p_sub is None else 1.0 - p_sub),
        )
    )
    result["control_resistance"] = (
        resistance_obs if opp_control > 0 else None,
        q_exp(opp_control / 60.0) if opp_control > 0 else 0.0,
    )

    sub_attempts = row_value(row, "sub_attempts") or 0.0
    rounds = row_value(row, "rounds") or 0.0
    td_landed = row_value(row, "td_landed") or 0.0
    ground_attempts = row_value(row, "ground_attempts") or 0.0
    finish_control = row_value(row, "finish_control_seconds") or 0.0
    meaningful_sub_opportunity = (
        td_landed > 0
        or ground_attempts > 0
        or finish_control > 0
        or sub_attempts > 0
    )

    if not meaningful_sub_opportunity:
        pressure_obs = None
        pressure_q = 0.0
    elif sub_attempts <= 0:
        pressure_obs = 0.0
        opportunity_units = (
            0.25 * (td_landed + ground_attempts)
            + 0.50 * (finish_control / 60.0)
        )
        pressure_q = q_exp(opportunity_units)
    else:
        sub_per_round = sub_attempts / rounds if rounds > 0 else None
        pressure_obs = weighted_available(
            (
                (0.45, percentile(pools["sub_per_ground_opp_positive"], row_value(row, "sub_per_ground_opp"))),
                (0.35, percentile(pools["sub_per_control_min_positive"], row_value(row, "sub_per_control_min"))),
                (0.20, percentile(pools["sub_per_round_positive"], sub_per_round)),
            )
        )
        opportunity_units = (
            0.25 * (td_landed + ground_attempts)
            + 0.50 * (finish_control / 60.0)
            + 1.0
        )
        pressure_q = q_exp(opportunity_units)

    result["submission_pressure"] = (pressure_obs, pressure_q)

    opponent_sub_loss = row_value(opponent_row, "submission_loss") or 0.0
    submission_win = 1.0 if opponent_sub_loss >= 0.5 else 0.0
    attack_count = max(sub_attempts, submission_win)
    if attack_count <= 0:
        result["submission_conversion"] = (None, 0.0)
    else:
        result["submission_conversion"] = (
            1.0 if submission_win > 0 else 0.0,
            q_exp(min(attack_count, 3.0)),
        )

    opp_sub_attempts = row_value(row, "opp_sub_attempts") or 0.0
    submission_loss = row_value(row, "submission_loss") or 0.0
    if submission_loss >= 0.5:
        result["submission_resistance"] = (0.0, 1.0)
    elif opp_sub_attempts > 0:
        result["submission_resistance"] = (
            1.0,
            q_exp(min(opp_sub_attempts, 3.0)),
        )
    else:
        result["submission_resistance"] = (None, 0.0)

    sig_landed = row_value(row, "sig_landed") or 0.0
    kd_scored = row_value(row, "kd_scored") or 0.0
    if sig_landed <= 0:
        result["striking_power"] = (None, 0.0)
    elif kd_scored <= 0:
        result["striking_power"] = (
            0.0,
            q_exp(sig_landed / 25.0),
        )
    else:
        kd_rate = kd_scored / sig_landed
        result["striking_power"] = (
            percentile(pools["kd_rate_positive"], kd_rate),
            q_exp(max(sig_landed / 25.0, 1.0)),
        )

    kd_absorbed = row_value(row, "kd_absorbed") or 0.0
    ko_loss = row_value(row, "ko_loss") or 0.0
    if kd_absorbed <= 0:
        result["chin_resistance"] = (None, 0.0)
    elif ko_loss >= 0.5:
        result["chin_resistance"] = (0.0, 1.0)
    else:
        result["chin_resistance"] = (
            1.0,
            q_exp(0.5 * min(kd_absorbed, 4.0)),
        )

    sig_absorbed = row_value(row, "sig_absorbed") or 0.0
    stress = sig_absorbed / rounds if rounds > 0 else None
    if kd_absorbed > 0 or sig_absorbed <= 0 or stress is None:
        result["damage_resistance"] = (None, 0.0)
    elif ko_loss >= 0.5 and sig_absorbed <= DAMAGE_FAILURE_HURDLE:
        result["damage_resistance"] = (None, 0.0)
    elif ko_loss >= 0.5:
        result["damage_resistance"] = (
            percentile(pools["damage_stress"], stress),
            1.0,
        )
    else:
        stress_pct = percentile(pools["damage_stress"], stress)
        result["damage_resistance"] = (
            1.0,
            0.0 if stress_pct is None else (
                stress_pct ** 2
                * (1.0 - exp(-rounds))
            ),
        )

    return result


def append_date_to_pools(
    rows: pd.DataFrame,
    pools: dict[str, list[float]],
) -> None:
    for _, row in rows.iterrows():
        direct_map = {
            "distance_accuracy": row_value(row, "distance_accuracy"),
            "distance_accuracy_allowed": row_value(row, "distance_accuracy_allowed"),
            "td_pressure_share": row_value(row, "td_pressure_share"),
            "td_attempts_per_round": row_value(row, "td_attempts_per_round"),
            "td_completion_rate": row_value(row, "td_completion_rate"),
            "control_seconds_per_td": row_value(row, "control_seconds_per_td"),
            "td_defense_rate": row_value(row, "td_defense_rate"),
            "control_share": row_value(row, "control_share"),
            "control_seconds_per_round": row_value(row, "control_seconds_per_round"),
            "control_exchange_balance": row_value(row, "control_exchange_balance"),
            "ground_pressure_share": row_value(row, "ground_pressure_share"),
            "control_seconds_allowed_per_round": row_value(row, "control_seconds_allowed_per_round"),
            "reversal_rate": row_value(row, "reversal_rate"),
            "ground_landed_allowed_per_control_min": row_value(row, "ground_landed_allowed_per_control_min"),
            "sub_attempts_allowed_per_control_min": row_value(row, "sub_attempts_allowed_per_control_min"),
        }

        for key, value in direct_map.items():
            if value is not None:
                insort(pools[key], value)

        sub_attempts = row_value(row, "sub_attempts") or 0.0
        rounds = row_value(row, "rounds") or 0.0
        if sub_attempts > 0:
            for key, value in (
                ("sub_per_ground_opp_positive", row_value(row, "sub_per_ground_opp")),
                ("sub_per_control_min_positive", row_value(row, "sub_per_control_min")),
                ("sub_per_round_positive", sub_attempts / rounds if rounds > 0 else None),
            ):
                if value is not None and value > 0:
                    insort(pools[key], value)

        sig_landed = row_value(row, "sig_landed") or 0.0
        kd_scored = row_value(row, "kd_scored") or 0.0
        if sig_landed > 0 and kd_scored > 0:
            insort(pools["kd_rate_positive"], kd_scored / sig_landed)

        kd_abs = row_value(row, "kd_absorbed") or 0.0
        sig_abs = row_value(row, "sig_absorbed") or 0.0
        rounds_obs = row_value(row, "rounds") or 0.0
        if kd_abs == 0 and sig_abs > 0 and rounds_obs > 0:
            insort(pools["damage_stress"], sig_abs / rounds_obs)


def validate_columns(df: pd.DataFrame) -> None:
    required = {
        "fight_id",
        "fighter_id",
        "fighter_name",
        *C.values(),
    }
    date_candidates = {"date", "event_date"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "RFS history is missing required locked-family columns: "
            + ", ".join(missing)
        )
    if not date_candidates.intersection(df.columns):
        raise RuntimeError("RFS history has neither date nor event_date")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fight_id", help="Target historical UFCStats fight ID.")
    args = parser.parse_args()
    target_fight_id = str(args.fight_id)

    if not RFS_PATH.exists():
        raise RuntimeError(f"RFS history not found: {RFS_PATH}")

    df = pd.read_parquet(RFS_PATH)
    validate_columns(df)

    date_col = "date" if "date" in df.columns else "event_date"
    df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    df["fight_id"] = df["fight_id"].astype(str)
    df["fighter_id"] = df["fighter_id"].astype(str)

    target = df.loc[df["fight_id"] == target_fight_id].copy()
    if len(target) != 2:
        raise RuntimeError(
            f"Target fight must have exactly two fighter rows; found {len(target)}"
        )

    target_date = pd.Timestamp(target[date_col].iloc[0])
    target_fighters = {
        str(row.fighter_id): str(row.fighter_name)
        for row in target[["fighter_id", "fighter_name"]].itertuples(index=False)
    }

    history = df.loc[df[date_col] < target_date].copy()
    history = history.sort_values([date_col, "fight_id", "fighter_id"]).reset_index(drop=True)

    ratings: dict[str, dict[str, float]] = defaultdict(
        lambda: {skill: BASE_RATING for skill in SKILLS}
    )
    update_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {skill: 0 for skill in SKILLS}
    )
    fight_counts: dict[str, int] = defaultdict(int)
    pools = {key: [] for key in POOL_KEYS}
    history_rows: list[dict[str, object]] = []

    for fight_date, date_rows in history.groupby(date_col, sort=True):
        date_deltas: dict[str, dict[str, float]] = defaultdict(
            lambda: {skill: 0.0 for skill in SKILLS}
        )
        date_updates: dict[str, dict[str, int]] = defaultdict(
            lambda: {skill: 0 for skill in SKILLS}
        )
        date_fights: dict[str, int] = defaultdict(int)
        date_history_rows: list[dict[str, object]] = []

        for fight_id, fight in date_rows.groupby("fight_id", sort=False):
            if len(fight) != 2:
                continue

            first = fight.iloc[0]
            second = fight.iloc[1]
            pairs = ((first, second), (second, first))

            for row, opponent_row in pairs:
                fighter_id = str(row["fighter_id"])
                opponent_id = str(opponent_row["fighter_id"])
                name = str(row["fighter_name"])
                opponent_name = str(opponent_row["fighter_name"])

                _ = ratings[fighter_id]
                _ = ratings[opponent_id]

                bundle = observation_bundle(row, opponent_row, pools)

                history_record: dict[str, object] = {
                    "fight_id": str(fight_id),
                    "date": pd.Timestamp(fight_date),
                    "fighter_id": fighter_id,
                    "fighter_name": name,
                    "opponent_id": opponent_id,
                    "opponent_name": opponent_name,
                }

                for skill in SKILLS:
                    obs, quality = bundle[skill]
                    pre = ratings[fighter_id][skill]
                    expected = expected_probability(
                        ratings,
                        fighter_id,
                        opponent_id,
                        skill,
                    )

                    if obs is None or quality <= 0.0:
                        delta = 0.0
                    else:
                        delta = (
                            k_factor(update_counts[fighter_id][skill])
                            * quality
                            * (float(obs) - expected)
                        )
                        date_updates[fighter_id][skill] += 1

                    date_deltas[fighter_id][skill] += delta

                    history_record[f"{skill}_pre"] = pre
                    history_record[f"{skill}_O"] = obs
                    history_record[f"{skill}_Q"] = quality
                    history_record[f"{skill}_E"] = expected
                    history_record[f"{skill}_delta"] = delta

                date_fights[fighter_id] += 1
                date_history_rows.append(history_record)

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

        for record in date_history_rows:
            fighter_id = str(record["fighter_id"])
            for skill in SKILLS:
                record[f"{skill}_post"] = ratings[fighter_id][skill]
            record["fight_count_post"] = fight_counts[fighter_id]
            history_rows.append(record)

        append_date_to_pools(date_rows, pools)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    target_rows = []
    for fighter_id, fighter_name in target_fighters.items():
        card = ratings[fighter_id]
        row = {
            "fight_id": target_fight_id,
            "target_date": target_date,
            "fighter_id": fighter_id,
            "fighter_name": fighter_name,
            "prior_ufc_fights": fight_counts[fighter_id],
        }
        row.update(card)
        for skill in SKILLS:
            row[f"{skill}_updates"] = update_counts[fighter_id][skill]
        target_rows.append(row)

    target_path = OUTPUT_DIR / (
        f"fsr_{target_fight_id}_locked_families_v1_target_card.csv"
    )
    history_path = OUTPUT_DIR / (
        f"fsr_{target_fight_id}_locked_families_v1_rating_history.csv"
    )

    pd.DataFrame(target_rows).to_csv(target_path, index=False)
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    print()
    print("=" * 110)
    print("LOCKED FSR FAMILY PRE-TARGET CARD")
    print("=" * 110)
    display_cols = [
        "fighter_name",
        "prior_ufc_fights",
        *SKILLS,
    ]
    print(
        pd.DataFrame(target_rows)[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )
    print()
    print("Saved:", target_path)
    print("Saved:", history_path)


if __name__ == "__main__":
    main()
