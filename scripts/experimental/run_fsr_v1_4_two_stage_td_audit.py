"""Audit V1.4 multi-attempt MC takedown sequences across archetypes.

Shadow/research only. This script does not modify FSR ratings, production
artifacts, finish calibration, judging, cardio, or simulator source.

The MC separates:

    style propensity -> wrestling sequence initiation
    conversion matchup -> per-shot success/failure
    failed-shot persistence -> additional attempts in the same 30-second segment

This audit measures initiated sequences, total attempts, completed takedowns,
failed attempts, completion rate, and mean attempts per sequence separately.
Leakage-safe PRE-fight EWM TD states are scale references, not exact matchup
targets because opponent defense should change realized completion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import pandas as pd

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import FighterSide
from pipeline.simulation.rfs_mc_v2_shared_state.path_runner import (
    SharedPathCalibration,
    run_shared_state_path,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_contracts import (
    TransitionEvent,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
    ClinchTransitionCalibration,
    DistanceTransitionCalibration,
    GroundTransitionCalibration,
)

from scripts.experimental import run_fsr_v1_4_td_contrast_grid as contrast
from scripts.experimental import run_fsr_v1_4_transition_style_grid as style


DEFAULT_SIMULATIONS = 500
DEFAULT_SEED = 2026080920
TD_COMPLETION_COLUMN = "rfs_phase_base_ewm_td_completion_rate"
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_v1_4_two_stage_td_audit.csv"
)


@dataclass(frozen=True)
class Candidate:
    """Sequence-initiation weight calibration with chain logic frozen."""

    name: str
    td_weight_scale: float


CANDIDATES = (
    Candidate("attempt_x0_75", 0.75),
    Candidate("attempt_x1_00", 1.00),
    Candidate("attempt_x1_25", 1.25),
    Candidate("attempt_x1_50", 1.50),
    Candidate("attempt_x1_75", 1.75),
    Candidate("attempt_x2_00", 2.00),
)


def calibration(candidate: Candidate) -> SharedPathCalibration:
    """Scale only wrestling-sequence initiation opportunity weights."""

    scale = candidate.td_weight_scale
    return SharedPathCalibration(
        distance=DistanceTransitionCalibration(
            stay_base_weight=6.0,
            clinch_entry_base_weight=1.0,
            takedown_base_weight=0.75 * scale,
            matchup_effect_strength=1.0,
        ),
        clinch=ClinchTransitionCalibration(
            stay_base_weight=4.5,
            break_base_weight=2.5,
            ownership_change_base_weight=1.0,
            owner_takedown_base_weight=1.5 * scale,
            defender_takedown_base_weight=0.5 * scale,
            matchup_effect_strength=1.0,
        ),
        ground=GroundTransitionCalibration(),
    )


def _finite(value: object) -> float | None:
    try:
        selected = float(value)
    except (TypeError, ValueError):
        return None
    return selected if isfinite(selected) else None


def run_candidate(
    red_transition,
    blue_transition,
    *,
    selected_calibration: SharedPathCalibration,
    scheduled_rounds: int,
    simulations: int,
    seed_start: int,
) -> dict[str, float]:
    """Count phase exposure plus wrestling sequences and every shot inside them."""

    totals = {
        "segments": 0.0,
        "distance": 0.0,
        "clinch": 0.0,
        "ground": 0.0,
        "red_sequences": 0.0,
        "blue_sequences": 0.0,
        "red_attempts": 0.0,
        "blue_attempts": 0.0,
        "red_success": 0.0,
        "blue_success": 0.0,
        "red_failed": 0.0,
        "blue_failed": 0.0,
        "red_attempts_distance": 0.0,
        "blue_attempts_distance": 0.0,
        "red_attempts_clinch": 0.0,
        "blue_attempts_clinch": 0.0,
    }

    for index in range(simulations):
        path = run_shared_state_path(
            red_transition,
            blue_transition,
            scheduled_rounds=scheduled_rounds,
            seed=seed_start + index,
            calibration=selected_calibration,
        )

        for segment in path.segments:
            totals["segments"] += 1.0
            phase = segment.state.phase
            if phase is FightPhase.DISTANCE:
                totals["distance"] += 1.0
            elif phase is FightPhase.CLINCH:
                totals["clinch"] += 1.0
            elif phase is FightPhase.GROUND:
                totals["ground"] += 1.0

            transition = segment.transition
            if transition is None or transition.event not in {
                TransitionEvent.TAKEDOWN,
                TransitionEvent.TAKEDOWN_ATTEMPT_FAILED,
            }:
                continue

            side = (
                "red"
                if transition.actor is FighterSide.RED
                else "blue"
            )
            attempt_count = max(1, int(transition.attempt_count))

            totals[f"{side}_sequences"] += 1.0
            totals[f"{side}_attempts"] += float(attempt_count)

            if phase is FightPhase.DISTANCE:
                totals[f"{side}_attempts_distance"] += float(attempt_count)
            elif phase is FightPhase.CLINCH:
                totals[f"{side}_attempts_clinch"] += float(attempt_count)

            if transition.event is TransitionEvent.TAKEDOWN:
                totals[f"{side}_success"] += 1.0
                totals[f"{side}_failed"] += float(attempt_count - 1)
            else:
                totals[f"{side}_failed"] += float(attempt_count)

    fighter_rounds = float(simulations * scheduled_rounds)
    segments = totals["segments"]
    result = {
        "distance_phase_pct": 100.0 * totals["distance"] / segments,
        "clinch_phase_pct": 100.0 * totals["clinch"] / segments,
        "ground_phase_pct": 100.0 * totals["ground"] / segments,
    }

    for side in ("red", "blue"):
        sequences = totals[f"{side}_sequences"]
        attempts = totals[f"{side}_attempts"]
        successes = totals[f"{side}_success"]
        result[f"{side}_td_sequences_per_round"] = sequences / fighter_rounds
        result[f"{side}_td_attempts_per_round"] = attempts / fighter_rounds
        result[f"{side}_td_success_per_round"] = successes / fighter_rounds
        result[f"{side}_td_failed_per_round"] = (
            totals[f"{side}_failed"] / fighter_rounds
        )
        result[f"{side}_td_completion_rate"] = (
            successes / attempts if attempts > 0.0 else 0.0
        )
        result[f"{side}_mean_attempts_per_sequence"] = (
            attempts / sequences if sequences > 0.0 else 0.0
        )
        result[f"{side}_td_attempts_distance_per_round"] = (
            totals[f"{side}_attempts_distance"] / fighter_rounds
        )
        result[f"{side}_td_attempts_clinch_per_round"] = (
            totals[f"{side}_attempts_clinch"] / fighter_rounds
        )

    return result


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

    rounds = pd.read_parquet(
        style.ROUND_STATS_PATH,
        columns=[
            "fight_id",
            "fighter_id",
            "fighter_name",
            "corner",
            "total_rounds",
        ],
    )
    rounds["fight_id"] = rounds["fight_id"].astype(str)
    rounds["fighter_id"] = rounds["fighter_id"].astype(str)

    history = pd.read_parquet(
        style.RFS_HISTORY_PATH,
        columns=[
            "fight_id",
            "fighter_id",
            *style.STYLE_COLUMNS.values(),
            TD_COMPLETION_COLUMN,
        ],
    )
    history["fight_id"] = history["fight_id"].astype(str)
    history["fighter_id"] = history["fighter_id"].astype(str)

    rows: list[dict[str, object]] = []

    for archetype, fight_id in style.TARGET_FIGHTS:
        target = rounds.loc[rounds["fight_id"] == fight_id].copy()
        if target.empty:
            continue

        fighters = target[
            ["fighter_id", "fighter_name", "corner", "total_rounds"]
        ].drop_duplicates(subset=["fighter_id"])
        red_row = fighters.loc[
            fighters["corner"].astype(str).str.upper() == "RED"
        ].iloc[0]
        blue_row = fighters.loc[
            fighters["corner"].astype(str).str.upper() == "BLUE"
        ].iloc[0]
        scheduled_rounds = int(float(red_row["total_rounds"]))

        target_card_path = style.v1_1.OUTPUT_DIR / (
            f"fsr_{fight_id}_locked_families_v1_1_target_card.csv"
        )
        if not target_card_path.exists():
            style.v1_1.run_rating_builders(fight_id)

        cards: dict[str, dict[str, float]] = {}
        names: dict[str, str] = {}
        historical_completion: dict[str, float] = {}

        for side, fighter_row in (("red", red_row), ("blue", blue_row)):
            fighter_id = str(fighter_row["fighter_id"])
            card, _ = style.v1_1.load_locked_card(
                fight_id,
                fighter_id,
            )
            card = style.attach_style(
                card,
                history,
                fight_id=fight_id,
                fighter_id=fighter_id,
            )
            state = history.loc[
                (history["fight_id"] == fight_id)
                & (history["fighter_id"] == fighter_id)
            ].iloc[0]
            completion = _finite(state[TD_COMPLETION_COLUMN])

            cards[side] = card
            names[side] = str(fighter_row["fighter_name"])
            historical_completion[side] = (
                float("nan")
                if completion is None
                else completion
            )

        # Linear V1.4 style is deliberate. The MC odds transform inverts
        # t/(t+1), while persistence now controls repeat shots separately.
        red_transition = contrast.build_transition(
            cards["red"],
            hill_power=1.0,
        )
        blue_transition = contrast.build_transition(
            cards["blue"],
            hill_power=1.0,
        )

        for candidate in CANDIDATES:
            metrics = run_candidate(
                red_transition,
                blue_transition,
                selected_calibration=calibration(candidate),
                scheduled_rounds=scheduled_rounds,
                simulations=args.simulations,
                seed_start=args.seed,
            )

            row: dict[str, object] = {
                "candidate": candidate.name,
                "archetype": archetype,
                "fight_id": fight_id,
                "red_name": names["red"],
                "blue_name": names["blue"],
                **metrics,
            }
            for side in ("red", "blue"):
                historical_attempts = float(
                    cards[side]["style_td_attempts_per_round"]
                )
                historical_comp = historical_completion[side]
                row[f"{side}_historical_td_attempts_per_round"] = (
                    historical_attempts
                )
                row[f"{side}_historical_td_completion_rate"] = (
                    historical_comp
                )
                row[f"{side}_historical_td_success_reference"] = (
                    historical_attempts * historical_comp
                    if isfinite(historical_comp)
                    else float("nan")
                )
            rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No chain-wrestling TD audit rows produced")

    print()
    print("=" * 210)
    print("V1.4 MULTI-ATTEMPT TAKEDOWN AUDIT")
    print("=" * 210)
    display = [
        "candidate",
        "archetype",
        "red_name",
        "blue_name",
        "distance_phase_pct",
        "clinch_phase_pct",
        "ground_phase_pct",
        "red_historical_td_attempts_per_round",
        "red_td_sequences_per_round",
        "red_td_attempts_per_round",
        "red_mean_attempts_per_sequence",
        "red_historical_td_completion_rate",
        "red_td_completion_rate",
        "red_td_success_per_round",
        "blue_historical_td_attempts_per_round",
        "blue_td_sequences_per_round",
        "blue_td_attempts_per_round",
        "blue_mean_attempts_per_sequence",
        "blue_historical_td_completion_rate",
        "blue_td_completion_rate",
        "blue_td_success_per_round",
    ]
    print(
        result[display].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    long_rows: list[dict[str, object]] = []
    for _, row in result.iterrows():
        for side in ("red", "blue"):
            hist_attempt = float(
                row[f"{side}_historical_td_attempts_per_round"]
            )
            sim_attempt = float(
                row[f"{side}_td_attempts_per_round"]
            )
            long_rows.append(
                {
                    "candidate": row["candidate"],
                    "fighter": row[f"{side}_name"],
                    "historical_attempts": hist_attempt,
                    "simulated_attempts": sim_attempt,
                    "attempt_abs_error": abs(
                        sim_attempt - hist_attempt
                    ),
                    "low_attempt_style": hist_attempt <= 0.30,
                    "high_attempt_style": hist_attempt >= 1.20,
                }
            )
    long = pd.DataFrame(long_rows)

    summary = (
        long.groupby("candidate", sort=False)
        .agg(
            mean_historical_attempts=("historical_attempts", "mean"),
            mean_simulated_attempts=("simulated_attempts", "mean"),
            attempt_mae=("attempt_abs_error", "mean"),
            attempt_max_error=("attempt_abs_error", "max"),
        )
        .reset_index()
    )
    low = (
        long.loc[long["low_attempt_style"]]
        .groupby("candidate", sort=False)
        .agg(
            low_style_attempt_mae=("attempt_abs_error", "mean"),
        )
        .reset_index()
    )
    high = (
        long.loc[long["high_attempt_style"]]
        .groupby("candidate", sort=False)
        .agg(
            high_style_attempt_mae=("attempt_abs_error", "mean"),
        )
        .reset_index()
    )
    summary = summary.merge(low, on="candidate", how="left").merge(
        high,
        on="candidate",
        how="left",
    )

    print()
    print("=" * 210)
    print("ATTEMPT-SCALE SUMMARY")
    print("=" * 210)
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
