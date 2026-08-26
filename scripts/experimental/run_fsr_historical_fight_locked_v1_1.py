"""Run MC V2 with population-centered locked FSR V1.1 ratings.

This runner preserves the exact locked V1 FSR -> MC mapping and MC calibration.
Only the historical rating builder changes: V1.1 uses leakage-safe,
skill-specific Q-weighted population expectations.

Shadow/research only.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.experimental import fsr_locked_families_v1 as equations_v1
from scripts.experimental import run_fsr_historical_fight_v1 as base
from scripts.experimental import run_fsr_historical_fight_locked_v1 as locked_v1


LOCKED_BUILDER = Path(
    "scripts/experimental/fsr_locked_families_v1_1.py"
)
OUTPUT_DIR = base.OUTPUT_DIR
LOCKED_SKILLS = list(equations_v1.SKILLS)


def run_rating_builders(fight_id: str) -> None:
    """Build V1.1 locked ratings plus the unchanged cardio bridge."""

    commands = [
        [
            sys.executable,
            str(LOCKED_BUILDER),
            fight_id,
        ],
        [
            sys.executable,
            str(base.CARDIO_SCRIPT),
            fight_id,
        ],
    ]

    for command in commands:
        subprocess.run(
            command,
            check=True,
        )


def load_locked_card(
    fight_id: str,
    fighter_id: str,
) -> tuple[dict[str, float], int]:
    """Read one population-centered PRE-target locked-family card."""

    path = OUTPUT_DIR / (
        f"fsr_{fight_id}"
        "_locked_families_v1_1_target_card.csv"
    )

    df = pd.read_csv(path)
    df["fighter_id"] = df["fighter_id"].astype(str)

    rows = df.loc[
        df["fighter_id"] == str(fighter_id)
    ].copy()

    if rows.empty:
        locked = {
            skill: 50.0
            for skill in LOCKED_SKILLS
        }
        fight_count = 0
    else:
        if len(rows) != 1:
            raise RuntimeError(
                "Expected exactly one V1.1 locked-family target-card row "
                f"for fighter {fighter_id}; found {len(rows)}"
            )

        row = rows.iloc[0]
        locked = {
            skill: float(row[skill])
            for skill in LOCKED_SKILLS
        }
        fight_count = int(row["prior_ufc_fights"])

    # Compatibility aliases expected by the existing generic runner.
    card = {
        "distance_volume": 50.0,
        "distance_accuracy": locked["distance_precision"],
        "distance_defense": locked["distance_defense"],
        "td_initiative": locked["wrestling_entry"],
        "td_completion": locked["wrestling_conversion"],
        "td_defense": locked["td_defense"],
        "control_imposition": locked["control_imposition"],
        "control_resistance": locked["control_resistance"],
        "submission_pressure": locked["submission_pressure"],
        "submission_conversion": locked["submission_conversion"],
        "submission_resistance": locked["submission_resistance"],
        "finishing_power": locked["striking_power"],
        "chin_resistance": locked["chin_resistance"],
        "damage_absorption": locked["damage_resistance"],
    }
    card.update(locked)

    return card, fight_count


def build_full_card(
    fight_id: str,
    fighter_id: str,
) -> tuple[dict[str, float], int]:
    """Combine V1.1 locked families with unchanged cardio traits."""

    card, fight_count = load_locked_card(
        fight_id,
        fighter_id,
    )
    card.update(
        base.load_cardio_card(
            fight_id,
            fighter_id,
        )
    )
    return card, fight_count


def install_overrides() -> None:
    """Patch only the generic experimental runner extension points."""

    base.run_rating_builders = run_rating_builders
    base.build_full_card = build_full_card

    # Preserve the already-tested V1 adapter exactly so this is a clean A/B
    # test of the expectation normalization only.
    base.build_transition = locked_v1.build_transition
    base.build_phase = locked_v1.build_phase
    base.build_dynamic = locked_v1.build_dynamic
    base.print_card = locked_v1.print_card
    base.print_matchup_parameters = (
        locked_v1.print_matchup_parameters
    )


if __name__ == "__main__":
    install_overrides()
    base.main()
