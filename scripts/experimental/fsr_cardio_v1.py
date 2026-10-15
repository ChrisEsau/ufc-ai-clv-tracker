"""Shadow FSR cardio V1 bridge from leakage-safe RFS Dynamic Response state.

Purpose
-------
Create three persistent-looking fighter-card traits for MC V2 without changing
production RFS artifacts or simulator contracts:

- fatigue_accumulation_resistance
- fatigue_performance_resilience
- recovery_ability

This first checkpoint deliberately reuses the existing leakage-safe RFS Dynamic
Response evidence, population-percentile calibration, and reliability shrinkage
already implemented by ``rfs_parameter_resolver``.  It is a bridge experiment,
not the final opponent-surprise FSR updater.

For a target historical fight, only RFS state and population rows from fights
strictly before the target date are used.

Outputs
-------
``data/simulation/rfs_mc_v2_shared_state/``
``fsr_<fight_id>_cardio_v1_target_card.csv``

Shadow/research only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline.simulation.rfs_mc_v2_shared_state.historical_matchup_loader import (
    load_historical_matchup,
)
from pipeline.simulation.rfs_mc_v2_shared_state.rfs_parameter_resolver import (
    resolve_fighter_parameters,
)


HISTORY_PATH = Path(
    "data/features/round_fighter_state_history.parquet"
)
MASTER_PATH = Path(
    "data/master/ufc_master.parquet"
)
OUTPUT_DIR = Path(
    "data/simulation/rfs_mc_v2_shared_state"
)

MIN_PRIOR_FIGHTS = 0

CARDIO_TARGETS = (
    "dynamic.fatigue_accumulation_resistance",
    "dynamic.fatigue_performance_resilience",
    "dynamic.recovery_ability",
)


def clamp(
    value: float,
    low: float,
    high: float,
) -> float:
    """Clamp one numeric value."""

    return max(
        low,
        min(high, float(value)),
    )


def unit_to_card_rating(value: float) -> float:
    """Map normalized [0, 1] simulator trait to a 10-90 FSR card scale.

    0.00 -> 10
    0.50 -> 50
    1.00 -> 90

    Keeping this mapping linear makes the card display transparent and exactly
    reversible during the current bridge experiment.
    """

    selected = clamp(
        value,
        0.0,
        1.0,
    )

    return 10.0 + 80.0 * selected


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load authoritative RFS history and fight master tables."""

    if not HISTORY_PATH.exists():
        raise RuntimeError(
            f"RFS history not found: {HISTORY_PATH}"
        )

    if not MASTER_PATH.exists():
        raise RuntimeError(
            f"Master fight table not found: {MASTER_PATH}"
        )

    history = pd.read_parquet(HISTORY_PATH)
    master = pd.read_parquet(MASTER_PATH)

    history["date"] = pd.to_datetime(
        history["date"],
        errors="coerce",
    )
    master["date"] = pd.to_datetime(
        master["date"],
        errors="coerce",
    )

    return history, master


def build_target_card(
    fight_id: str,
) -> pd.DataFrame:
    """Build leakage-safe cardio card for both target-fight fighters."""

    history, master = load_inputs()

    matchup = load_historical_matchup(
        history,
        master,
        fight_id,
        min_prior_fights=MIN_PRIOR_FIGHTS,
    )

    population = history.loc[
        history["date"] < matchup.date
    ].copy()

    population = population.loc[
        pd.to_numeric(
            population["rfs_traj_prior_fight_count"],
            errors="coerce",
        ) > 0
    ].copy()

    if population.empty:
        raise RuntimeError(
            f"{fight_id}: leakage-safe population is empty"
        )

    rows: list[dict[str, object]] = []

    for fighter in (
        matchup.red,
        matchup.blue,
    ):
        resolved = resolve_fighter_parameters(
            profile=fighter.features,
            prior_fight_count=fighter.prior_fight_count,
            population_history=population,
        )

        row: dict[str, object] = {
            "fight_id": fight_id,
            "date": matchup.date,
            "fighter_id": fighter.fighter_id,
            "fighter_name": fighter.fighter_name,
            "prior_ufc_fights": fighter.prior_fight_count,
        }

        for target in CARDIO_TARGETS:
            estimate = resolved.estimates[target]
            suffix = target.removeprefix("dynamic.")
            engine_value = float(
                estimate.shrunk_estimate
            )

            row[f"{suffix}_engine"] = engine_value
            row[f"{suffix}_rating"] = unit_to_card_rating(
                engine_value
            )
            row[f"{suffix}_reliability"] = float(
                estimate.reliability
            )
            row[f"{suffix}_used_fallback"] = bool(
                estimate.used_fallback
            )

        rows.append(row)

    return pd.DataFrame(rows)


def print_card(card: pd.DataFrame) -> None:
    """Print a compact human-readable cardio card."""

    print()
    print("=" * 100)
    print("FSR CARDIO V1 — PRE-FIGHT TARGET CARD")
    print("=" * 100)

    for row in card.itertuples(index=False):
        print()
        print(
            f"{row.fighter_name} "
            f"({row.prior_ufc_fights} prior UFC fights)"
        )
        print("-" * 70)
        print(
            "fatigue_accumulation_resistance "
            f"{row.fatigue_accumulation_resistance_rating:6.2f} "
            f"(engine {row.fatigue_accumulation_resistance_engine:.3f})"
        )
        print(
            "fatigue_performance_resilience  "
            f"{row.fatigue_performance_resilience_rating:6.2f} "
            f"(engine {row.fatigue_performance_resilience_engine:.3f})"
        )
        print(
            "recovery_ability                "
            f"{row.recovery_ability_rating:6.2f} "
            f"(engine {row.recovery_ability_engine:.3f})"
        )


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fight_id",
        help="Historical UFC fight ID",
    )
    args = parser.parse_args()

    card = build_target_card(
        str(args.fight_id)
    )

    print_card(card)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / (
            f"fsr_{args.fight_id}"
            "_cardio_v1_target_card.csv"
        )
    )

    card.to_csv(
        output_path,
        index=False,
    )

    print()
    print("Saved FSR cardio V1 target card:")
    print(output_path)


if __name__ == "__main__":
    main()
