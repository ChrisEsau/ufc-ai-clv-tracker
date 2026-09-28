"""Full-fight validation for the V1.4 RFS-style / chain-wrestling MC.

Shadow/research only.

This diagnostic intentionally runs only two contrasting historical fights:

1. Sean O'Malley vs Merab Dvalishvili -- wrestler vs striker stress test.
2. Max Holloway vs Calvin Kattar -- high-volume striker control matchup.

Frozen from earlier checkpoints:
- locked FSR observation/update equations and V1.1 population centering;
- V1.2 phase-conditioned activity conversion logic;
- V1.3 finish hazards;
- cardio/dynamic-state mappings;
- judging/final-result logic.

V1.4 changes under test:
- leakage-safe PRE-fight RFS Phase Baseline style controls transition tendency;
- current two-stage, multi-attempt takedown engine;
- takedown sequence-initiation scale = 1.75;
- neutral phase exposure is recomputed under that same V1.4 calibration before
  converting historical whole-round activity into per-active-segment rates.

The script reports full outcome/mechanics metrics plus chain-wrestling sequence,
attempt, completion, and success rates from the actually simulated (finish-
censored) paths. Historical target-fight TD statistics are printed only as a
realized-fight reference, not as a training/calibration target.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import FighterSide
from pipeline.simulation.rfs_mc_v2_shared_state.path_runner import (
    SharedPathCalibration,
    run_shared_state_path,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_contracts import (
    TAKEDOWN_EVENTS,
    TransitionEvent,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
    ClinchTransitionCalibration,
    DistanceTransitionCalibration,
    GroundTransitionCalibration,
)

from scripts.experimental import run_fsr_historical_fight_v1 as base
from scripts.experimental import run_fsr_historical_fight_locked_v1_2 as v1_2
from scripts.experimental import run_fsr_historical_fight_locked_v1_3 as v1_3
from scripts.experimental import run_fsr_v1_3_archetype_validation as validation
from scripts.experimental import run_fsr_v1_4_transition_style_grid as style


DEFAULT_SIMULATIONS = 500
DEFAULT_SEED = 2026080930
ATTEMPT_SCALE = 1.75
REFERENCE_PATHS = 5000
REFERENCE_ROUNDS = 3
REFERENCE_SEED_START = 2026080940

RFS_HISTORY_PATH = Path("data/features/round_fighter_state_history.parquet")

TARGET_FIGHTS = (
    (
        "wrestler vs striker",
        "3146e5a47a922976",
    ),
    (
        "high-volume striker vs striker",
        "a4817b7e46028b4a",
    ),
)


def v1_4_calibration() -> SharedPathCalibration:
    """Return the provisional x1.75 transition calibration."""

    return SharedPathCalibration(
        distance=DistanceTransitionCalibration(
            stay_base_weight=6.0,
            clinch_entry_base_weight=1.0,
            takedown_base_weight=0.75 * ATTEMPT_SCALE,
            matchup_effect_strength=1.0,
        ),
        clinch=ClinchTransitionCalibration(
            stay_base_weight=4.5,
            break_base_weight=2.5,
            ownership_change_base_weight=1.0,
            owner_takedown_base_weight=1.5 * ATTEMPT_SCALE,
            defender_takedown_base_weight=0.5 * ATTEMPT_SCALE,
            matchup_effect_strength=1.0,
        ),
        ground=GroundTransitionCalibration(),
    )


V1_4_CALIBRATION = v1_4_calibration()


@lru_cache(maxsize=1)
def neutral_phase_exposure_v1_4() -> dict[str, float]:
    """Recompute neutral physical exposure under the selected V1.4 MC.

    V1.2 converts historical whole-round distance/submission rates into rates
    per active phase segment. Those denominators must move when transition
    calibration changes, otherwise the full-fight test would mix V1.4 phase
    occupancy with stale V1.2 activity scaling.
    """

    neutral_card = {
        "wrestling_entry": 50.0,
        "wrestling_conversion": 50.0,
        "td_defense": 50.0,
        "control_imposition": 50.0,
        "control_resistance": 50.0,
    }
    neutral_transition = style.build_style_transition(neutral_card)

    distance_segments = 0
    clinch_segments = 0
    ground_segments = 0
    ground_owner_segments = 0

    for index in range(REFERENCE_PATHS):
        path = run_shared_state_path(
            neutral_transition,
            neutral_transition,
            scheduled_rounds=REFERENCE_ROUNDS,
            seed=REFERENCE_SEED_START + index,
            calibration=V1_4_CALIBRATION,
        )

        for segment in path.segments:
            state = segment.state
            if state.phase is FightPhase.DISTANCE:
                distance_segments += 1
            elif state.phase is FightPhase.CLINCH:
                clinch_segments += 1
            elif state.phase is FightPhase.GROUND:
                ground_segments += 1
                if state.phase_owner not in {
                    FighterSide.RED,
                    FighterSide.BLUE,
                }:
                    raise RuntimeError(
                        "Neutral ground reference segment has no owner."
                    )
                ground_owner_segments += 1

    total_rounds = float(REFERENCE_PATHS * REFERENCE_ROUNDS)
    distance_per_round = distance_segments / total_rounds
    clinch_per_round = clinch_segments / total_rounds
    ground_per_round = ground_segments / total_rounds
    ground_owner_per_fighter_round = (
        ground_owner_segments / (2.0 * total_rounds)
    )

    total = distance_per_round + clinch_per_round + ground_per_round
    if abs(total - 10.0) > 1e-9:
        raise RuntimeError(
            "V1.4 neutral exposure does not sum to 10 segments/round: "
            f"{total}"
        )
    if distance_per_round <= 0.0:
        raise RuntimeError("V1.4 neutral reference has no distance exposure.")
    if ground_owner_per_fighter_round <= 0.0:
        raise RuntimeError("V1.4 neutral reference has no ground-owner exposure.")

    return {
        "reference_distance_segments_per_round": distance_per_round,
        "reference_clinch_segments_per_round": clinch_per_round,
        "reference_ground_segments_per_round": ground_per_round,
        "reference_ground_owner_segments_per_fighter_round": (
            ground_owner_per_fighter_round
        ),
    }


def build_inputs_v1_4(
    red_card: dict[str, float],
    blue_card: dict[str, float],
    baselines: dict[str, float],
):
    """Use RFS style for tendency while preserving frozen FSR ability inputs."""

    return (
        style.build_style_transition(red_card),
        style.build_style_transition(blue_card),
        base.build_phase(red_card, blue_card, baselines),
        base.build_phase(blue_card, red_card, baselines),
        base.build_dynamic(red_card),
        base.build_dynamic(blue_card),
    )


# Counters populated by the path-runner wrapper during one matchup population.
_td_audit: dict[str, float] = defaultdict(float)
_original_path_runner = base.run_finish_enabled_dynamic_path


def reset_td_audit() -> None:
    """Reset finish-censored chain-wrestling counters for one matchup."""

    _td_audit.clear()


def run_v1_4_path(*args, **kwargs):
    """Inject V1.4 shared calibration and audit sampled wrestling chains."""

    kwargs["shared_path_calibration"] = V1_4_CALIBRATION
    path = _original_path_runner(*args, **kwargs)

    reached_rounds = {
        int(segment.state.round_number)
        for segment in path.segments
    }
    _td_audit["reached_rounds"] += float(len(reached_rounds))

    for segment in path.segments:
        transition = segment.transition
        if transition is None or transition.event not in TAKEDOWN_EVENTS:
            continue

        if transition.actor is FighterSide.RED:
            side = "red"
        elif transition.actor is FighterSide.BLUE:
            side = "blue"
        else:
            raise RuntimeError("Takedown transition requires an actor.")

        _td_audit[f"{side}_sequences"] += 1.0
        _td_audit[f"{side}_attempts"] += float(transition.attempt_count)
        if transition.event is TransitionEvent.TAKEDOWN:
            _td_audit[f"{side}_successes"] += 1.0

    return path


def td_population_metrics(simulations: int) -> dict[str, float]:
    """Return finish-censored wrestling metrics accumulated by the wrapper."""

    reached_rounds = _td_audit["reached_rounds"]
    if reached_rounds <= 0.0:
        raise RuntimeError("No simulated rounds were reached.")

    result: dict[str, float] = {}
    for side in ("red", "blue"):
        sequences = _td_audit[f"{side}_sequences"]
        attempts = _td_audit[f"{side}_attempts"]
        successes = _td_audit[f"{side}_successes"]

        result[f"{side}_td_sequences_per_reached_round"] = (
            sequences / reached_rounds
        )
        result[f"{side}_td_attempts_per_reached_round"] = (
            attempts / reached_rounds
        )
        result[f"{side}_mean_attempts_per_sequence"] = (
            attempts / sequences if sequences > 0.0 else 0.0
        )
        result[f"{side}_td_completion_rate"] = (
            successes / attempts if attempts > 0.0 else 0.0
        )
        result[f"{side}_td_successes_per_fight"] = (
            successes / float(simulations)
        )

    return result


def actual_td_metrics(
    fight_rounds: pd.DataFrame,
    fighter_id: str,
) -> dict[str, float]:
    """Return realized target-fight TD statistics for reference only."""

    rows = fight_rounds.loc[
        fight_rounds["fighter_id"].astype(str) == str(fighter_id)
    ]
    attempts = float(pd.to_numeric(rows["td_attempted"]).fillna(0.0).sum())
    landed = float(pd.to_numeric(rows["td_landed"]).fillna(0.0).sum())
    observed_rounds = float(rows["round"].nunique())

    return {
        "attempts": attempts,
        "landed": landed,
        "attempts_per_round": (
            attempts / observed_rounds if observed_rounds > 0.0 else 0.0
        ),
        "completion": landed / attempts if attempts > 0.0 else 0.0,
    }


def print_matchup(
    *,
    archetype: str,
    fight_id: str,
    red_info,
    blue_info,
    red_actual: dict[str, float],
    blue_actual: dict[str, float],
    metrics: dict[str, float],
    td_metrics: dict[str, float],
) -> None:
    """Print one compact full-fight validation block."""

    red_name = str(red_info["fighter_name"])
    blue_name = str(blue_info["fighter_name"])

    print()
    print("=" * 120)
    print(f"{archetype.upper()}: {red_name} vs {blue_name} | {fight_id}")
    print("=" * 120)
    print(
        "Outcome: "
        f"RED {metrics['red_win_pct']:.1f}% | "
        f"BLUE {metrics['blue_win_pct']:.1f}% | "
        f"DRAW {metrics['draw_pct']:.1f}%"
    )
    print(
        "Finish mix: "
        f"KO {metrics['ko_pct']:.1f}% | "
        f"SUB {metrics['submission_pct']:.1f}% | "
        f"DISTANCE {metrics['scheduled_distance_pct']:.1f}% | "
        f"R1 FINISH {metrics['round1_finish_pct']:.1f}%"
    )
    print(
        "Phase mix: "
        f"distance {metrics['distance_phase_pct']:.1f}% | "
        f"clinch {metrics['clinch_phase_pct']:.1f}% | "
        f"ground {metrics['ground_phase_pct']:.1f}%"
    )

    for side, name, actual in (
        ("red", red_name, red_actual),
        ("blue", blue_name, blue_actual),
    ):
        print(
            f"{name}: actual TD {actual['attempts']:.0f} att / "
            f"{actual['landed']:.0f} landed "
            f"({actual['attempts_per_round']:.2f}/round, "
            f"{100.0 * actual['completion']:.1f}%); "
            f"sim {td_metrics[f'{side}_td_attempts_per_reached_round']:.2f} "
            f"att/reached-round, "
            f"{td_metrics[f'{side}_td_sequences_per_reached_round']:.2f} "
            f"sequences/reached-round, "
            f"{td_metrics[f'{side}_mean_attempts_per_sequence']:.2f} "
            f"att/sequence, "
            f"{100.0 * td_metrics[f'{side}_td_completion_rate']:.1f}% completion, "
            f"{td_metrics[f'{side}_td_successes_per_fight']:.2f} TD/fight"
        )

    print(
        "Control sec/reached-round: "
        f"{red_name} {metrics['red_control_seconds_per_reached_round']:.1f} | "
        f"{blue_name} {metrics['blue_control_seconds_per_reached_round']:.1f}"
    )
    print(
        "Sub attempts/fight: "
        f"{red_name} {metrics['red_submission_attempts_per_fight']:.2f} | "
        f"{blue_name} {metrics['blue_submission_attempts_per_fight']:.2f}"
    )
    print(
        "Knockdowns/fight: "
        f"{red_name} {metrics['red_knockdowns_per_fight']:.2f} | "
        f"{blue_name} {metrics['blue_knockdowns_per_fight']:.2f}"
    )
    print(
        "R3 fatigue: "
        f"{red_name} {metrics['red_r3_fatigue']:.3f} | "
        f"{blue_name} {metrics['blue_r3_fatigue']:.3f}"
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

    # Install frozen V1.1/V1.2/V1.3 adapters first.
    v1_3.install_overrides()

    # Repoint the V1.2 activity denominator to the current V1.4 transition MC.
    v1_2.neutral_phase_exposure = neutral_phase_exposure_v1_4

    # Replace only the transition-input builder and shared path calibration in
    # the diagnostic validator. All other simulator/model layers stay frozen.
    validation.build_inputs = build_inputs_v1_4
    base.run_finish_enabled_dynamic_path = run_v1_4_path

    rounds = pd.read_parquet(base.ROUND_PATH)
    rounds["fight_id"] = rounds["fight_id"].astype(str)

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

    exposure = neutral_phase_exposure_v1_4()
    print()
    print("=" * 120)
    print("FSR / MC V1.4 TWO-FIGHT FULL VALIDATION")
    print("=" * 120)
    print(
        f"Simulations/fight: {args.simulations} | seed start: {args.seed} | "
        f"TD attempt scale: {ATTEMPT_SCALE:.2f}"
    )
    print(
        "Neutral V1.4 segments/round: "
        f"distance={exposure['reference_distance_segments_per_round']:.3f}, "
        f"clinch={exposure['reference_clinch_segments_per_round']:.3f}, "
        f"ground={exposure['reference_ground_segments_per_round']:.3f}, "
        "ground-owner/fighter="
        f"{exposure['reference_ground_owner_segments_per_fighter_round']:.3f}"
    )

    for index, (archetype, fight_id) in enumerate(TARGET_FIGHTS):
        (
            fight_rounds,
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

        reset_td_audit()
        metrics = validation.run_compact_population(
            red_card=red_card,
            blue_card=blue_card,
            baselines=baselines,
            scheduled_rounds=scheduled_rounds,
            simulations=args.simulations,
            seed_start=args.seed + index * 10000,
        )
        td_metrics = td_population_metrics(args.simulations)

        red_actual = actual_td_metrics(
            fight_rounds,
            red_info["fighter_id"],
        )
        blue_actual = actual_td_metrics(
            fight_rounds,
            blue_info["fighter_id"],
        )

        print_matchup(
            archetype=archetype,
            fight_id=fight_id,
            red_info=red_info,
            blue_info=blue_info,
            red_actual=red_actual,
            blue_actual=blue_actual,
            metrics=metrics,
            td_metrics=td_metrics,
        )
        print(
            "Prior UFC fights entering target: "
            f"{red_info['fighter_name']}={red_prior_fights}, "
            f"{blue_info['fighter_name']}={blue_prior_fights}"
        )


if __name__ == "__main__":
    main()
