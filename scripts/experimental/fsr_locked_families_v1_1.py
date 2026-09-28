"""Shadow FSR V1.1 replay with leakage-safe skill-specific expectations.

V1.1 inherits the current V1 observation (O) and evidence-quality (Q)
functions and changes expected performance (E): equal 50-rated fighters are
expected to produce the leakage-safe Q-weighted population observation for that
skill, rather than an unconditional 0.50 observation.

For skill s:

    B_s = sum(Q_i * O_i) / sum(Q_i)

using only observations from dates strictly before the fight being updated.
Then:

    E_s = sigmoid(logit(B_s) + (R_s - D_s) / RATING_SCALE)

This keeps 50 as the unknown/population-prior rating while allowing hurdle and
binary observations to have natural population baselines away from 0.50.

For wrestling_entry specifically, D_s is the population-prior rating rather
than opponent td_defense.  Entry measures takedown initiation frequency;
opponent defense belongs to takedown conversion, not the decision to initiate.

Shadow/research only.  No production contracts are changed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from math import log
from pathlib import Path

import pandas as pd

from scripts.experimental import fsr_locked_families_v1 as v1


OUTPUT_DIR = v1.OUTPUT_DIR
MIN_CENTER_UPDATES = 3
LOGIT_EPSILON = 1e-4


def logit(probability: float) -> float:
    """Return a numerically safe logit for one population baseline."""

    selected = v1.clamp(
        float(probability),
        LOGIT_EPSILON,
        1.0 - LOGIT_EPSILON,
    )
    return log(selected / (1.0 - selected))


def population_baseline(
    weighted_observation_sum: dict[str, float],
    quality_sum: dict[str, float],
    skill: str,
) -> float:
    """Return the prior-date Q-weighted observation mean for one skill."""

    total_quality = float(quality_sum[skill])
    if total_quality <= 0.0:
        return 0.50

    return v1.clamp(
        float(weighted_observation_sum[skill]) / total_quality,
        0.0,
        1.0,
    )


def defense_rating(
    ratings: dict[str, dict[str, float]],
    opponent_id: str,
    skill: str,
) -> float:
    """Return the paired opponent rating used by the current ontology."""

    if skill == "distance_precision":
        return ratings[opponent_id]["distance_defense"]
    if skill == "distance_defense":
        return ratings[opponent_id]["distance_precision"]
    if skill == "wrestling_entry":
        return v1.BASE_RATING
    if skill == "wrestling_conversion":
        return ratings[opponent_id]["td_defense"]
    if skill == "td_defense":
        return ratings[opponent_id]["wrestling_conversion"]
    if skill == "control_imposition":
        return ratings[opponent_id]["control_resistance"]
    if skill == "control_resistance":
        return ratings[opponent_id]["control_imposition"]
    if skill == "submission_pressure":
        return ratings[opponent_id]["submission_resistance"]
    if skill == "submission_conversion":
        return ratings[opponent_id]["submission_resistance"]
    if skill == "submission_resistance":
        return ratings[opponent_id]["submission_conversion"]
    if skill == "striking_power":
        return v1.BASE_RATING
    if skill in {"chin_resistance", "damage_resistance"}:
        return ratings[opponent_id]["striking_power"]

    return v1.BASE_RATING


def expected_probability(
    ratings: dict[str, dict[str, float]],
    fighter_id: str,
    opponent_id: str,
    skill: str,
    baseline: float,
) -> float:
    """Return skill expectation centered on its prior population observation."""

    own = ratings[fighter_id][skill]
    defense = defense_rating(
        ratings,
        opponent_id,
        skill,
    )

    return v1.sigmoid(
        logit(baseline)
        + (own - defense) / v1.RATING_SCALE
    )


def build_center_diagnostics(
    ratings: dict[str, dict[str, float]],
    update_counts: dict[str, dict[str, int]],
    weighted_observation_sum: dict[str, float],
    quality_sum: dict[str, float],
) -> pd.DataFrame:
    """Summarize target-date population rating centers by skill."""

    rows: list[dict[str, float | int | str]] = []

    for skill in v1.SKILLS:
        values = [
            float(card[skill])
            for fighter_id, card in ratings.items()
            if update_counts[fighter_id][skill] >= MIN_CENTER_UPDATES
        ]

        series = pd.Series(values, dtype=float)

        rows.append(
            {
                "skill": skill,
                "population_baseline": population_baseline(
                    weighted_observation_sum,
                    quality_sum,
                    skill,
                ),
                "baseline_evidence_q": float(quality_sum[skill]),
                "fighters_with_3plus_updates": len(values),
                "rating_mean": (
                    float(series.mean())
                    if len(series) > 0
                    else float("nan")
                ),
                "rating_median": (
                    float(series.median())
                    if len(series) > 0
                    else float("nan")
                ),
                "rating_mean_minus_50": (
                    float(series.mean()) - v1.BASE_RATING
                    if len(series) > 0
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fight_id",
        help="Target historical UFCStats fight ID.",
    )
    args = parser.parse_args()
    target_fight_id = str(args.fight_id)

    if not v1.RFS_PATH.exists():
        raise RuntimeError(
            f"RFS history not found: {v1.RFS_PATH}"
        )

    df = pd.read_parquet(v1.RFS_PATH)
    v1.validate_columns(df)

    date_col = (
        "date"
        if "date" in df.columns
        else "event_date"
    )
    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="raise",
    )
    df["fight_id"] = df["fight_id"].astype(str)
    df["fighter_id"] = df["fighter_id"].astype(str)

    target = df.loc[
        df["fight_id"] == target_fight_id
    ].copy()

    if len(target) != 2:
        raise RuntimeError(
            "Target fight must have exactly two fighter rows; "
            f"found {len(target)}"
        )

    target_date = pd.Timestamp(
        target[date_col].iloc[0]
    )
    target_fighters = {
        str(row.fighter_id): str(row.fighter_name)
        for row in target[
            ["fighter_id", "fighter_name"]
        ].itertuples(index=False)
    }

    history = df.loc[
        df[date_col] < target_date
    ].copy()
    history = history.sort_values(
        [date_col, "fight_id", "fighter_id"]
    ).reset_index(drop=True)

    ratings: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            skill: v1.BASE_RATING
            for skill in v1.SKILLS
        }
    )
    update_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            skill: 0
            for skill in v1.SKILLS
        }
    )
    fight_counts: dict[str, int] = defaultdict(int)
    pools = {
        key: []
        for key in v1.POOL_KEYS
    }

    # Leakage-safe expected-observation baselines.
    weighted_observation_sum: dict[str, float] = defaultdict(float)
    quality_sum: dict[str, float] = defaultdict(float)

    history_rows: list[dict[str, object]] = []

    for fight_date, date_rows in history.groupby(
        date_col,
        sort=True,
    ):
        date_deltas: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                skill: 0.0
                for skill in v1.SKILLS
            }
        )
        date_updates: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                skill: 0
                for skill in v1.SKILLS
            }
        )
        date_fights: dict[str, int] = defaultdict(int)
        date_history_rows: list[dict[str, object]] = []

        # Observations from the current date are accumulated separately and
        # become eligible baselines only after every fight on the date has
        # been updated. This prevents same-date leakage.
        date_weighted_observation_sum: dict[str, float] = defaultdict(float)
        date_quality_sum: dict[str, float] = defaultdict(float)

        for fight_id, fight in date_rows.groupby(
            "fight_id",
            sort=False,
        ):
            if len(fight) != 2:
                continue

            first = fight.iloc[0]
            second = fight.iloc[1]
            pairs = (
                (first, second),
                (second, first),
            )

            for row, opponent_row in pairs:
                fighter_id = str(row["fighter_id"])
                opponent_id = str(opponent_row["fighter_id"])
                name = str(row["fighter_name"])
                opponent_name = str(
                    opponent_row["fighter_name"]
                )

                _ = ratings[fighter_id]
                _ = ratings[opponent_id]

                bundle = v1.observation_bundle(
                    row,
                    opponent_row,
                    pools,
                )

                history_record: dict[str, object] = {
                    "fight_id": str(fight_id),
                    "date": pd.Timestamp(fight_date),
                    "fighter_id": fighter_id,
                    "fighter_name": name,
                    "opponent_id": opponent_id,
                    "opponent_name": opponent_name,
                }

                for skill in v1.SKILLS:
                    obs, quality = bundle[skill]
                    pre = ratings[fighter_id][skill]
                    baseline = population_baseline(
                        weighted_observation_sum,
                        quality_sum,
                        skill,
                    )
                    expected = expected_probability(
                        ratings,
                        fighter_id,
                        opponent_id,
                        skill,
                        baseline,
                    )

                    if obs is None or quality <= 0.0:
                        delta = 0.0
                    else:
                        delta = (
                            v1.k_factor(
                                update_counts[fighter_id][skill]
                            )
                            * quality
                            * (float(obs) - expected)
                        )
                        date_updates[fighter_id][skill] += 1

                        date_weighted_observation_sum[skill] += (
                            quality * float(obs)
                        )
                        date_quality_sum[skill] += quality

                    date_deltas[fighter_id][skill] += delta

                    history_record[f"{skill}_pre"] = pre
                    history_record[f"{skill}_O"] = obs
                    history_record[f"{skill}_Q"] = quality
                    history_record[f"{skill}_B"] = baseline
                    history_record[f"{skill}_E"] = expected
                    history_record[f"{skill}_delta"] = delta

                date_fights[fighter_id] += 1
                date_history_rows.append(history_record)

        # Apply all same-date fighter updates simultaneously.
        for fighter_id, skill_deltas in date_deltas.items():
            for skill, delta in skill_deltas.items():
                ratings[fighter_id][skill] = v1.clamp(
                    ratings[fighter_id][skill] + delta,
                    v1.MIN_RATING,
                    v1.MAX_RATING,
                )
                update_counts[fighter_id][skill] += (
                    date_updates[fighter_id][skill]
                )

        for fighter_id, count in date_fights.items():
            fight_counts[fighter_id] += count

        for record in date_history_rows:
            fighter_id = str(record["fighter_id"])
            for skill in v1.SKILLS:
                record[f"{skill}_post"] = (
                    ratings[fighter_id][skill]
                )
            record["fight_count_post"] = (
                fight_counts[fighter_id]
            )
            history_rows.append(record)

        # Current-date observations become prior information only now.
        for skill in v1.SKILLS:
            weighted_observation_sum[skill] += (
                date_weighted_observation_sum[skill]
            )
            quality_sum[skill] += date_quality_sum[skill]

        # Preserve V1's leakage-safe percentile pools.
        v1.append_date_to_pools(
            date_rows,
            pools,
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_rows: list[dict[str, object]] = []

    for fighter_id, fighter_name in target_fighters.items():
        card = ratings[fighter_id]
        row: dict[str, object] = {
            "fight_id": target_fight_id,
            "target_date": target_date,
            "fighter_id": fighter_id,
            "fighter_name": fighter_name,
            "prior_ufc_fights": fight_counts[fighter_id],
        }
        row.update(card)

        for skill in v1.SKILLS:
            row[f"{skill}_updates"] = (
                update_counts[fighter_id][skill]
            )
            row[f"{skill}_population_baseline"] = (
                population_baseline(
                    weighted_observation_sum,
                    quality_sum,
                    skill,
                )
            )
            row[f"{skill}_baseline_evidence_q"] = (
                quality_sum[skill]
            )

        target_rows.append(row)

    center_df = build_center_diagnostics(
        ratings,
        update_counts,
        weighted_observation_sum,
        quality_sum,
    )

    target_path = OUTPUT_DIR / (
        f"fsr_{target_fight_id}"
        "_locked_families_v1_1_target_card.csv"
    )
    history_path = OUTPUT_DIR / (
        f"fsr_{target_fight_id}"
        "_locked_families_v1_1_rating_history.csv"
    )
    center_path = OUTPUT_DIR / (
        f"fsr_{target_fight_id}"
        "_locked_families_v1_1_population_center.csv"
    )

    pd.DataFrame(target_rows).to_csv(
        target_path,
        index=False,
    )
    pd.DataFrame(history_rows).to_csv(
        history_path,
        index=False,
    )
    center_df.to_csv(
        center_path,
        index=False,
    )

    print()
    print("=" * 118)
    print("LOCKED FSR V1.1 — POPULATION-CENTERED PRE-TARGET CARD")
    print("=" * 118)

    display_cols = [
        "fighter_name",
        "prior_ufc_fights",
        *v1.SKILLS,
    ]
    print(
        pd.DataFrame(target_rows)[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print()
    print("=" * 118)
    print("LEAKAGE-SAFE SKILL BASELINES AND ESTABLISHED-FIGHTER CENTERS")
    print("=" * 118)
    print(
        center_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("Saved:", target_path)
    print("Saved:", history_path)
    print("Saved:", center_path)


if __name__ == "__main__":
    main()
