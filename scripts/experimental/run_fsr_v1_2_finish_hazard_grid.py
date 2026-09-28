"""Same-fight finish-hazard scale audit for corrected FSR V1.2 activity.

Shadow/research only.

This script does NOT change the Monte Carlo engine and does not fit a final
calibration to one matchup.  It keeps the population-centered FSR V1.1 cards,
corrected V1.2 phase activity rates, transition mapping, dynamic-state model,
and scoring fixed while varying only three existing finish-calibration values:

- landed-strike KO/TKO hazard;
- incremental knockdown KO/TKO hazard;
- base submission probability per simulated submission attempt.

The purpose is to locate the sensible finish-hazard scale after correcting the
previously under-generated phase activity.  A broader population audit is still
required before any candidate is frozen.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from scripts.experimental import run_fsr_historical_fight_v1 as base
from scripts.experimental import run_fsr_historical_fight_locked_v1_2 as v1_2
from scripts.experimental import run_fsr_locked_family_sensitivity as sensitivity


DEFAULT_SIMULATIONS = 500
DEFAULT_SEED = 2026080700

LANDED_KO_HAZARDS = (0.0025, 0.0040, 0.0060)
KNOCKDOWN_BONUS_HAZARDS = (0.040, 0.080)
SUBMISSION_HAZARDS = (0.30, 0.50, 0.65)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fight_id")
    parser.add_argument(
        "--simulations",
        type=int,
        default=DEFAULT_SIMULATIONS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    args = parser.parse_args()

    # Install the already-reviewed V1.2 shadow adapter.  This changes only the
    # experimental runner module; the MC engine remains untouched.
    v1_2.install_overrides()

    # Reuse the compact population runner, but point its adapter hooks at the
    # currently installed V1.2 mappings rather than the older V1 mapping.
    sensitivity.locked.build_transition = base.build_transition
    sensitivity.locked.build_phase = base.build_phase
    sensitivity.locked.build_dynamic = base.build_dynamic

    (
        rounds,
        target_date,
        red_info,
        blue_info,
        scheduled_rounds,
    ) = base.load_target_fight(str(args.fight_id))

    base.run_rating_builders(str(args.fight_id))

    red_card, _ = base.build_full_card(
        str(args.fight_id),
        red_info["fighter_id"],
    )
    blue_card, _ = base.build_full_card(
        str(args.fight_id),
        blue_info["fighter_id"],
    )

    baselines = base.population_baselines(
        rounds,
        target_date,
    )

    original_finish_calibration = base.finish_calibration

    rows: list[dict[str, float]] = []

    for landed_hazard in LANDED_KO_HAZARDS:
        for knockdown_hazard in KNOCKDOWN_BONUS_HAZARDS:
            for submission_hazard in SUBMISSION_HAZARDS:
                selected_candidate = base.Candidate(
                    landed_ko_hazard=landed_hazard,
                    knockdown_bonus_hazard=knockdown_hazard,
                )

                selected_finish = original_finish_calibration(
                    selected_candidate
                )
                selected_finish = replace(
                    selected_finish,
                    submission=replace(
                        selected_finish.submission,
                        base_probability_per_attempt=submission_hazard,
                    ),
                )

                # sensitivity.run_population() obtains its finish calibration
                # through base.finish_calibration().  Override that one shadow
                # extension point for this candidate only.
                base.finish_calibration = (
                    lambda _candidate, calibration=selected_finish: calibration
                )

                metrics = sensitivity.run_population(
                    red_card=red_card,
                    blue_card=blue_card,
                    baselines=baselines,
                    scheduled_rounds=scheduled_rounds,
                    simulations=args.simulations,
                    seed_start=args.seed,
                )

                total_ko = (
                    metrics["red_ko_pct"]
                    + metrics["blue_ko_pct"]
                )
                total_sub = (
                    metrics["red_submission_pct"]
                    + metrics["blue_submission_pct"]
                )
                scheduled_distance = metrics["decision_pct"]

                rows.append(
                    {
                        "landed_ko_hazard": landed_hazard,
                        "knockdown_bonus_hazard": knockdown_hazard,
                        "submission_hazard": submission_hazard,
                        "red_win_pct": metrics["red_win_pct"],
                        "blue_win_pct": metrics["blue_win_pct"],
                        "total_ko_pct": total_ko,
                        "total_sub_pct": total_sub,
                        "scheduled_distance_pct": scheduled_distance,
                        "red_distance_attempts": metrics[
                            "red_distance_attempted"
                        ],
                        "blue_distance_attempts": metrics[
                            "blue_distance_attempted"
                        ],
                        "red_submission_attempts": metrics[
                            "red_submission_attempts"
                        ],
                        "blue_submission_attempts": metrics[
                            "blue_submission_attempts"
                        ],
                        "red_knockdowns": metrics["red_knockdowns"],
                        "blue_knockdowns": metrics["blue_knockdowns"],
                    }
                )

    base.finish_calibration = original_finish_calibration

    result = pd.DataFrame(rows).sort_values(
        [
            "scheduled_distance_pct",
            "total_ko_pct",
            "total_sub_pct",
        ],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    output_path = (
        Path(base.OUTPUT_DIR)
        / f"fsr_{args.fight_id}_v1_2_finish_hazard_grid.csv"
    )
    result.to_csv(output_path, index=False)

    print()
    print("=" * 132)
    print("FSR V1.2 CORRECTED-ACTIVITY FINISH HAZARD GRID")
    print("=" * 132)
    print(
        f"Fight: {red_info['fighter_name']} vs "
        f"{blue_info['fighter_name']}"
    )
    print(f"Simulations per candidate: {args.simulations}")
    print(f"Seed start: {args.seed}")
    print()
    print(
        "This is a scale-location audit, NOT a one-fight calibration target."
    )
    print()

    display = result[
        [
            "landed_ko_hazard",
            "knockdown_bonus_hazard",
            "submission_hazard",
            "total_ko_pct",
            "total_sub_pct",
            "scheduled_distance_pct",
            "red_win_pct",
            "blue_win_pct",
            "red_distance_attempts",
            "blue_distance_attempts",
            "red_submission_attempts",
            "blue_submission_attempts",
        ]
    ]

    print(
        display.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print()
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
