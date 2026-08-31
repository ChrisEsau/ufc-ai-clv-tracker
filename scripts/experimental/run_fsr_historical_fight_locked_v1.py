"""Run MC V2 with the locked/reviewed FSR equation families.

This is a shadow-only validation shim around
``run_fsr_historical_fight_v1.py``. It reuses that runner's Monte Carlo
calibration, seeds, detailed diagnostics, and result reporting, while replacing
only the rating builder and FSR -> MC adapter functions.

Usage
-----
PYTHONPATH=. python \
    scripts/experimental/run_fsr_historical_fight_locked_v1.py \
    <fight_id>

The original runner remains untouched for direct A/B comparison.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.experimental import run_fsr_historical_fight_v1 as base


LOCKED_BUILDER = Path(
    "scripts/experimental/fsr_locked_families_v1.py"
)

OUTPUT_DIR = base.OUTPUT_DIR

LOCKED_SKILLS = [
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
]

_original_print_matchup_parameters = (
    base.print_matchup_parameters
)


def run_rating_builders(
    fight_id: str,
) -> None:
    """Build the reviewed FSR card plus the unchanged cardio bridge."""

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
    """Read one PRE-target locked-family card."""

    path = (
        OUTPUT_DIR
        / (
            f"fsr_{fight_id}"
            "_locked_families_v1_target_card.csv"
        )
    )

    df = pd.read_csv(path)
    df["fighter_id"] = (
        df["fighter_id"]
        .astype(str)
    )

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
                "Expected exactly one locked-family target-card row "
                f"for fighter {fighter_id}; found {len(rows)}"
            )

        row = rows.iloc[0]
        locked = {
            skill: float(row[skill])
            for skill in LOCKED_SKILLS
        }
        fight_count = int(
            row["prior_ufc_fights"]
        )

    # Compatibility aliases expected by the existing diagnostic runner.
    # Distance volume/pace was rejected as a persistent FSR skill, so it is
    # held neutral. The physical distance-attempt rate comes directly from
    # the leakage-safe population baseline in build_phase().
    card = {
        "distance_volume": 50.0,
        "distance_accuracy": locked[
            "distance_precision"
        ],
        "distance_defense": locked[
            "distance_defense"
        ],
        "td_initiative": locked[
            "wrestling_entry"
        ],
        "td_completion": locked[
            "wrestling_conversion"
        ],
        "td_defense": locked[
            "td_defense"
        ],
        "control_imposition": locked[
            "control_imposition"
        ],
        "control_resistance": locked[
            "control_resistance"
        ],
        "submission_pressure": locked[
            "submission_pressure"
        ],
        "submission_conversion": locked[
            "submission_conversion"
        ],
        "submission_resistance": locked[
            "submission_resistance"
        ],
        "finishing_power": locked[
            "striking_power"
        ],
        "chin_resistance": locked[
            "chin_resistance"
        ],
        "damage_absorption": locked[
            "damage_resistance"
        ],
    }

    card.update(locked)

    return card, fight_count


def build_full_card(
    fight_id: str,
    fighter_id: str,
) -> tuple[dict[str, float], int]:
    """Combine the locked families with the unchanged cardio bridge."""

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


def build_transition(
    fighter: dict[str, float],
) -> base.FighterTransitionParameters:
    """Map the reviewed wrestling/control ontology into MC transitions."""

    td_entry = base.normalized_skill(
        fighter["wrestling_entry"]
    )
    td_conversion = base.normalized_skill(
        fighter["wrestling_conversion"]
    )
    td_defense = base.normalized_skill(
        fighter["td_defense"]
    )
    control = base.normalized_skill(
        fighter["control_imposition"]
    )
    control_resistance = base.normalized_skill(
        fighter["control_resistance"]
    )

    distance_retention = 0.50

    phase_imposition = (
        td_entry
        + control
    ) / 2.0

    phase_resistance = (
        td_defense
        + control_resistance
    ) / 2.0

    return base.FighterTransitionParameters(
        distance_retention=distance_retention,

        clinch_entry_tendency=(
            0.50 * td_entry
            + 0.50 * control
        ),

        clinch_entry_resistance=(
            control_resistance
        ),

        takedown_entry_tendency=td_entry,
        takedown_completion_ability=(
            td_conversion
        ),
        takedown_resistance=td_defense,

        takedown_persistence=(
            0.50 * td_entry
            + 0.50 * control
        ),

        failed_takedown_persistence=(
            td_entry
        ),

        clinch_retention=control,
        clinch_escape_ability=(
            control_resistance
        ),

        ground_retention=control,
        ground_escape_ability=(
            control_resistance
        ),

        reversal_ability=0.50,

        phase_imposition=phase_imposition,
        phase_resistance=phase_resistance,
    )


def build_phase(
    fighter: dict[str, float],
    opponent: dict[str, float],
    baselines: dict[str, float],
) -> base.FighterPhaseParameters:
    """Build matchup activity from the reviewed FSR equations."""

    # Pace/volume remains a physical RFS/population tendency rather than a
    # persistent FSR skill.
    distance_attempt_rate = max(
        0.0,
        baselines[
            "distance_attempts_per_round"
        ] / 10.0,
    )

    distance_accuracy = (
        base.matchup_probability(
            baseline=(
                baselines[
                    "distance_accuracy"
                ]
            ),
            offense_rating=(
                fighter[
                    "distance_precision"
                ]
            ),
            defense_rating=(
                opponent[
                    "distance_defense"
                ]
            ),
        )
    )

    knockdown_probability = (
        base.matchup_probability(
            baseline=0.012,
            offense_rating=(
                fighter[
                    "striking_power"
                ]
            ),
            defense_rating=50.0,
        )
    )

    control_strength = (
        base.matchup_probability(
            baseline=0.50,
            offense_rating=(
                fighter[
                    "control_imposition"
                ]
            ),
            defense_rating=(
                opponent[
                    "control_resistance"
                ]
            ),
        )
    )

    clinch_control_seconds = (
        3.0
        + 6.0
        * control_strength
    )

    ground_control_seconds = (
        4.0
        + 8.0
        * control_strength
    )

    # Pressure controls attack generation.
    submission_attempt_rate = (
        base.matchup_rate(
            baseline=(
                baselines[
                    "sub_attempts_per_round"
                ]
                / 10.0
            ),
            offense_rating=(
                fighter[
                    "submission_pressure"
                ]
            ),
            defense_rating=(
                opponent[
                    "submission_resistance"
                ]
            ),
        )
    )

    # Conversion and resistance interact only after a legitimate attack.
    # MC V2 consumes this through the defender's submission-defense
    # probability, keeping attack pressure separate from finish conversion.
    submission_defense = (
        base.matchup_probability(
            baseline=0.50,
            offense_rating=(
                fighter[
                    "submission_resistance"
                ]
            ),
            defense_rating=(
                opponent[
                    "submission_conversion"
                ]
            ),
        )
    )

    return base.FighterPhaseParameters(
        distance=base.DistanceRateParameters(
            sig_strike_attempt_rate=(
                distance_attempt_rate
            ),
            sig_strike_accuracy=(
                distance_accuracy
            ),
            knockdown_probability_per_landed=(
                knockdown_probability
            ),
        ),

        clinch=base.ClinchRateParameters(
            clinch_strike_attempt_rate=0.30,
            clinch_strike_accuracy=0.45,
            control_seconds_mean=(
                clinch_control_seconds
            ),
            damaging_clinch_probability=0.10,
        ),

        ground_owner=base.GroundOwnerRateParameters(
            ground_strike_attempt_rate=0.35,
            ground_strike_accuracy=0.50,
            control_seconds_mean=(
                ground_control_seconds
            ),
            submission_attempt_rate=(
                submission_attempt_rate
            ),
            position_advancement_probability=0.20,
        ),

        ground_defender=base.GroundDefenderRateParameters(
            escape_attempt_rate=(
                base.normalized_skill(
                    fighter[
                        "control_resistance"
                    ]
                )
            ),
            reversal_attempt_rate=0.10,
            scramble_attempt_rate=0.25,
            submission_defense=(
                submission_defense
            ),
        ),
    )


def build_dynamic(
    fighter: dict[str, float],
) -> base.FighterDynamicParameters:
    """Map reviewed Damage Resistance while leaving cardio bridge unchanged."""

    return base.FighterDynamicParameters(
        fatigue_accumulation_resistance=(
            fighter[
                "fatigue_accumulation_resistance_engine"
            ]
        ),
        fatigue_performance_resilience=(
            fighter[
                "fatigue_performance_resilience_engine"
            ]
        ),
        recovery_ability=(
            fighter[
                "recovery_ability_engine"
            ]
        ),
        damage_resistance=(
            base.normalized_skill(
                fighter[
                    "damage_resistance"
                ]
            )
        ),
        acute_stress_resistance=0.50,
        acute_stress_recovery=0.50,
    )


def print_card(
    name: str,
    fight_count: int,
    card: dict[str, float],
) -> None:
    """Display the reviewed FSR dimensions entering MC V2."""

    print()
    print(
        f"{name} "
        f"({fight_count} prior UFC fights)"
    )
    print("-" * 70)

    for key in (
        LOCKED_SKILLS
        + [
            "fatigue_accumulation_resistance_rating",
            "fatigue_performance_resilience_rating",
            "recovery_ability_rating",
        ]
    ):
        print(
            f"{key:<32} "
            f"{card[key]:8.2f}"
        )


def print_matchup_parameters(
    name: str,
    card: dict[str, float],
    transition: base.FighterTransitionParameters,
    phase: base.FighterPhaseParameters,
    dynamic: base.FighterDynamicParameters,
) -> None:
    """Retain existing display and add submission-specific parameters."""

    _original_print_matchup_parameters(
        name,
        card,
        transition,
        phase,
        dynamic,
    )

    print(
        "submission pressure rate:",
        round(
            phase.ground_owner
            .submission_attempt_rate,
            5,
        ),
    )

    print(
        "submission defense      :",
        round(
            phase.ground_defender
            .submission_defense,
            4,
        ),
    )

    print(
        "submission conversion FSR:",
        round(
            card[
                "submission_conversion"
            ],
            2,
        ),
    )

    print(
        "submission resistance FSR:",
        round(
            card[
                "submission_resistance"
            ],
            2,
        ),
    )


def install_locked_overrides() -> None:
    """Patch only the experimental runner module's extension points."""

    base.run_rating_builders = (
        run_rating_builders
    )
    base.build_full_card = (
        build_full_card
    )
    base.build_transition = (
        build_transition
    )
    base.build_phase = (
        build_phase
    )
    base.build_dynamic = (
        build_dynamic
    )
    base.print_card = (
        print_card
    )
    base.print_matchup_parameters = (
        print_matchup_parameters
    )


if __name__ == "__main__":
    install_locked_overrides()
    base.main()
