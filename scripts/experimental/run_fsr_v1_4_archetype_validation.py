"""Cross-archetype full-fight validation for the provisional FSR/MC V1.4 stack.

Shadow/research only.

This diagnostic rolls the currently selected V1.4 mechanics across the six
historical archetype fights available in the local UFCStats round dataset.
It does not change production behavior or any locked FSR observation/update
contract.

Selected checkpoint under test
------------------------------
- locked FSR V1.1 population-centered fighter cards;
- V1.2 phase-conditioned activity conversion, with neutral exposure recomputed
  under the current V1.4 transition environment;
- RFS Phase Baseline style controls phase-transition tendency;
- two-stage, multi-attempt takedown sequences;
- takedown sequence-initiation scale = 1.75;
- V1.3 KO/TKO hazards remain 0.0025 landed / 0.080 knockdown bonus;
- submission attempt generation remains 1.00x;
- provisional submission base probability per attempt = 0.12;
- cardio, dynamic-state, judging, and final-result logic remain unchanged.

The output is intended to answer a broad question: after the wrestling and
submission fixes, does the simulator produce distinct, plausible fight shapes
without breaking striker control matchups?
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from scripts.experimental import run_fsr_historical_fight_v1 as base
from scripts.experimental import run_fsr_historical_fight_locked_v1_2 as v1_2
from scripts.experimental import run_fsr_historical_fight_locked_v1_3 as v1_3
from scripts.experimental import run_fsr_v1_3_archetype_validation as validation
from scripts.experimental import run_fsr_v1_4_transition_style_grid as style
from scripts.experimental import run_fsr_v1_4_two_fight_full_validation as v1_4


DEFAULT_SIMULATIONS = 500
DEFAULT_SEED = 2026080960
SUBMISSION_HAZARD = 0.12

RFS_HISTORY_PATH = Path("data/features/round_fighter_state_history.parquet")
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_v1_4_archetype_validation_summary.csv"
)

# These are the same six locally available archetypes used in the earlier
# V1.3 validation. Islam Makhachev vs Dustin Poirier remains absent from the
# exact normalized local pair and is intentionally not substituted.
TARGET_FIGHTS = (
    (
        "power strikers",
        "7208e40818401e88",
    ),
    (
        "wrestler vs striker",
        "3146e5a47a922976",
    ),
    (
        "submission / grappling",
        "40e8bf8ce508c436",
    ),
    (
        "high power / chin",
        "bca5d01f8775f852",
    ),
    (
        "high-volume striker vs striker",
        "a4817b7e46028b4a",
    ),
    (
        "wrestler vs wrestler",
        "31b3ae9352d9389b",
    ),
)


def v1_4_finish_calibration(base_candidate):
    """Preserve V1.3 KO hazards and lower only submission conversion."""

    calibration = v1_3.finish_calibration(base_candidate)
    return replace(
        calibration,
        submission=replace(
            calibration.submission,
            base_probability_per_attempt=SUBMISSION_HAZARD,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
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

    if args.simulations <= 0:
        raise ValueError("--simulations must be positive")

    # Install the already validated V1.1/V1.2/V1.3 adapter stack first.
    v1_3.install_overrides()

    # V1.2 activity rates are per active phase segment. Because V1.4 changed
    # phase occupancy, use V1.4 neutral exposure as the denominator reference.
    v1_2.neutral_phase_exposure = v1_4.neutral_phase_exposure_v1_4

    # Use RFS style -> transition tendency while preserving FSR ability inputs.
    validation.build_inputs = v1_4.build_inputs_v1_4

    # Inject the selected V1.4 shared transition calibration into every path.
    base.run_finish_enabled_dynamic_path = v1_4.run_v1_4_path

    # Preserve V1.3 KO calibration and change only submission conversion.
    base.finish_calibration = v1_4_finish_calibration

    rounds = pd.read_parquet(base.ROUND_PATH)
    rounds["fight_id"] = rounds["fight_id"].astype(str)
    rounds["event_date"] = pd.to_datetime(
        rounds["event_date"],
        errors="raise",
    )

    history = pd.read_parquet(
        RFS_HISTORY_PATH,
        columns=[
            "fight_id",
            "fighter_id",
            *style.STYLE_COLUMNS.values(),
        ],
    )
    history["fight_id"] = history["fight_id"].astype(str)
    history["fighter_id"] = history["fighter_id"].astype(str)

    exposure = v1_4.neutral_phase_exposure_v1_4()

    print()
    print("=" * 170)
    print("FSR / MC V1.4 FULL ARCHETYPE VALIDATION")
    print("=" * 170)
    print(
        f"Simulations/fight: {args.simulations} | seed start: {args.seed} | "
        f"TD scale: {v1_4.ATTEMPT_SCALE:.2f} | "
        f"submission hazard: {SUBMISSION_HAZARD:.2f}"
    )
    print(
        "Neutral segments/round: "
        f"distance={exposure['reference_distance_segments_per_round']:.3f}, "
        f"clinch={exposure['reference_clinch_segments_per_round']:.3f}, "
        f"ground={exposure['reference_ground_segments_per_round']:.3f}, "
        "ground-owner/fighter="
        f"{exposure['reference_ground_owner_segments_per_fighter_round']:.3f}"
    )

    rows: list[dict[str, object]] = []

    for index, (archetype, fight_id) in enumerate(TARGET_FIGHTS):
        (
            _all_rounds,
            target_date,
            red_info,
            blue_info,
            scheduled_rounds,
        ) = base.load_target_fight(fight_id)

        base.run_rating_builders(fight_id)

        red_card, red_prior_fights = base.build_full_card(
            fight_id,
            red_info["fighter_id"],
        )
        blue_card, blue_prior_fights = base.build_full_card(
            fight_id,
            blue_info["fighter_id"],
        )

        red_card = style.attach_style(
            red_card,
            history,
            fight_id=fight_id,
            fighter_id=red_info["fighter_id"],
        )
        blue_card = style.attach_style(
            blue_card,
            history,
            fight_id=fight_id,
            fighter_id=blue_info["fighter_id"],
        )

        baselines = base.population_baselines(rounds, target_date)

        # The path wrapper accumulates chain-wrestling sequence/attempt metrics.
        v1_4.reset_td_audit()

        metrics = validation.run_compact_population(
            red_card=red_card,
            blue_card=blue_card,
            baselines=baselines,
            scheduled_rounds=scheduled_rounds,
            simulations=args.simulations,
            seed_start=args.seed + index * 10000,
        )
        td = v1_4.td_population_metrics(args.simulations)

        row: dict[str, object] = {
            "archetype": archetype,
            "fight_id": fight_id,
            "event_date": str(target_date.date()),
            "red_name": red_info["fighter_name"],
            "blue_name": blue_info["fighter_name"],
            "scheduled_rounds": scheduled_rounds,
            "red_prior_fights": red_prior_fights,
            "blue_prior_fights": blue_prior_fights,
            "red_style_td_attempts_per_round": red_card.get(
                "style_td_attempts_per_round",
                float("nan"),
            ),
            "blue_style_td_attempts_per_round": blue_card.get(
                "style_td_attempts_per_round",
                float("nan"),
            ),
            **metrics,
            **td,
        }
        rows.append(row)

        print()
        print("-" * 170)
        print(
            f"{archetype.upper()}: "
            f"{red_info['fighter_name']} vs {blue_info['fighter_name']} "
            f"({target_date.date()}, {fight_id})"
        )
        print(
            "Outcome: "
            f"RED {metrics['red_win_pct']:.1f}% | "
            f"BLUE {metrics['blue_win_pct']:.1f}% | "
            f"DRAW {metrics['draw_pct']:.1f}%"
        )
        print(
            "Finish: "
            f"KO {metrics['ko_pct']:.1f}% | "
            f"SUB {metrics['submission_pct']:.1f}% | "
            f"DIST {metrics['scheduled_distance_pct']:.1f}% | "
            f"R1 {metrics['round1_finish_pct']:.1f}%"
        )
        print(
            "Phase: "
            f"distance {metrics['distance_phase_pct']:.1f}% | "
            f"clinch {metrics['clinch_phase_pct']:.1f}% | "
            f"ground {metrics['ground_phase_pct']:.1f}%"
        )
        print(
            f"RED TD: style {row['red_style_td_attempts_per_round']:.3f}/rnd | "
            f"sim {td['red_td_attempts_per_reached_round']:.3f} att/rnd | "
            f"{td['red_mean_attempts_per_sequence']:.2f} att/seq | "
            f"{100.0 * td['red_td_completion_rate']:.1f}% completion"
        )
        print(
            f"BLUE TD: style {row['blue_style_td_attempts_per_round']:.3f}/rnd | "
            f"sim {td['blue_td_attempts_per_reached_round']:.3f} att/rnd | "
            f"{td['blue_mean_attempts_per_sequence']:.2f} att/seq | "
            f"{100.0 * td['blue_td_completion_rate']:.1f}% completion"
        )
        print(
            "Activity: "
            f"control sec/rnd RED {metrics['red_control_seconds_per_reached_round']:.1f} "
            f"BLUE {metrics['blue_control_seconds_per_reached_round']:.1f} | "
            f"sub att/fight RED {metrics['red_submission_attempts_per_fight']:.2f} "
            f"BLUE {metrics['blue_submission_attempts_per_fight']:.2f} | "
            f"KD/fight RED {metrics['red_knockdowns_per_fight']:.2f} "
            f"BLUE {metrics['blue_knockdowns_per_fight']:.2f}"
        )
        print(
            "Fatigue: "
            f"R3 RED {metrics['red_r3_fatigue']:.3f} "
            f"BLUE {metrics['blue_r3_fatigue']:.3f} | "
            f"R5 RED {metrics['red_r5_fatigue']:.3f} "
            f"BLUE {metrics['blue_r5_fatigue']:.3f}"
        )

    result = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print()
    print("=" * 170)
    print("CROSS-FIGHT SUMMARY")
    print("=" * 170)

    summary_columns = [
        "archetype",
        "red_name",
        "blue_name",
        "red_win_pct",
        "blue_win_pct",
        "draw_pct",
        "ko_pct",
        "submission_pct",
        "scheduled_distance_pct",
        "distance_phase_pct",
        "clinch_phase_pct",
        "ground_phase_pct",
        "red_td_attempts_per_reached_round",
        "blue_td_attempts_per_reached_round",
        "red_submission_attempts_per_fight",
        "blue_submission_attempts_per_fight",
    ]

    print(
        result[summary_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
