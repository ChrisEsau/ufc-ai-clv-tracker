"""Takedown-contrast audit for style-aware FSR/MC V1.4 transitions.

Shadow/research only.  This script changes no simulator source, locked FSR
formula, cardio, finish, judging, or production artifact.

The first V1.4 transition grid showed two facts at once:
1. leakage-safe RFS style inputs correctly distinguish wrestling propensity;
2. simply lowering global takedown weights fixes low-use strikers but suppresses
   high-use wrestlers too strongly.

This audit asks whether a sharper style mapping plus stronger existing matchup
contrast can separate those cases without changing the engine architecture.
It also decomposes successful takedown transitions by source phase and compares
simulated success scale with a leakage-safe historical reference:

    EWM TD attempts/round * EWM TD completion rate

That reference is descriptive, not a matchup target. Opponent TD defense should
move an individual matchup away from it. It is used only to locate the correct
order of magnitude.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
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

from scripts.experimental import run_fsr_v1_4_transition_style_grid as prior


DEFAULT_SIMULATIONS = 500
DEFAULT_SEED = 2026080910
TD_COMPLETION_COLUMN = "rfs_phase_base_ewm_td_completion_rate"
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_v1_4_td_contrast_grid.csv"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    hill_power: float
    distance_strength: float
    clinch_strength: float
    td_weight_scale: float = 1.0


def hill_ratio(value: float, half_saturation: float, power: float) -> float:
    """Bounded Hill curve; power > 1 sharply suppresses low-use styles."""

    value = max(0.0, float(value))
    if value == 0.0:
        return 0.0
    numerator = value ** power
    denominator = numerator + half_saturation ** power
    return prior.clamp01(numerator / denominator)


def build_transition(card: dict[str, float], *, hill_power: float):
    """Use prior V1.4 mapping, changing only the TD-style gate shape."""

    transition = prior.build_style_transition(card)
    if not prior.has_style(card):
        return transition

    td_attempts = max(0.0, float(card["style_td_attempts_per_round"]))
    failed = max(0.0, float(card["style_failed_td_attempts_per_round"]))
    persistence_raw = max(0.0, float(card["style_td_persistence_ratio"]))

    td_style = hill_ratio(td_attempts, 1.0, hill_power)
    persistence_shape = prior.saturating_ratio(persistence_raw, 1.0)
    failed_shape = prior.saturating_ratio(failed, 1.0)

    td_persistence = td_style * (0.50 + 0.50 * persistence_shape)
    failed_td_persistence = td_style * (0.50 + 0.50 * failed_shape)

    return replace(
        transition,
        takedown_entry_tendency=prior.clamp01(td_style),
        takedown_persistence=prior.clamp01(td_persistence),
        failed_takedown_persistence=prior.clamp01(failed_td_persistence),
    )


def calibration(candidate: Candidate) -> SharedPathCalibration:
    """Use existing calibration interfaces; no transition-engine code changes."""

    scale = candidate.td_weight_scale
    return SharedPathCalibration(
        distance=DistanceTransitionCalibration(
            stay_base_weight=6.0,
            clinch_entry_base_weight=1.0,
            takedown_base_weight=0.75 * scale,
            matchup_effect_strength=candidate.distance_strength,
        ),
        clinch=ClinchTransitionCalibration(
            stay_base_weight=4.5,
            break_base_weight=2.5,
            ownership_change_base_weight=1.0,
            owner_takedown_base_weight=1.5 * scale,
            defender_takedown_base_weight=0.5 * scale,
            matchup_effect_strength=candidate.clinch_strength,
        ),
        ground=GroundTransitionCalibration(),
    )


CANDIDATES = (
    Candidate("linear_s1_default", 1.0, 1.0, 1.0, 1.0),
    Candidate("hill2_s2_default", 2.0, 2.0, 2.0, 1.0),
    Candidate("hill2_s2_5_default", 2.0, 2.5, 2.5, 1.0),
    Candidate("hill2_s3_default", 2.0, 3.0, 3.0, 1.0),
    Candidate("hill2_s3_td70", 2.0, 3.0, 3.0, 0.70),
    Candidate("hill2_5_s3_default", 2.5, 3.0, 3.0, 1.0),
    Candidate("hill3_s3_default", 3.0, 3.0, 3.0, 1.0),
)


def run_candidate(
    red_transition,
    blue_transition,
    *,
    selected_calibration: SharedPathCalibration,
    scheduled_rounds: int,
    simulations: int,
    seed_start: int,
) -> dict[str, float]:
    totals = {
        "segments": 0.0,
        "distance": 0.0,
        "clinch": 0.0,
        "ground": 0.0,
        "red_td": 0.0,
        "blue_td": 0.0,
        "red_td_distance": 0.0,
        "blue_td_distance": 0.0,
        "red_td_clinch": 0.0,
        "blue_td_clinch": 0.0,
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
            if transition is None or transition.event is not TransitionEvent.TAKEDOWN:
                continue

            if transition.actor is FighterSide.RED:
                totals["red_td"] += 1.0
                if phase is FightPhase.DISTANCE:
                    totals["red_td_distance"] += 1.0
                elif phase is FightPhase.CLINCH:
                    totals["red_td_clinch"] += 1.0
            elif transition.actor is FighterSide.BLUE:
                totals["blue_td"] += 1.0
                if phase is FightPhase.DISTANCE:
                    totals["blue_td_distance"] += 1.0
                elif phase is FightPhase.CLINCH:
                    totals["blue_td_clinch"] += 1.0

    fighter_rounds = float(simulations * scheduled_rounds)
    segments = totals["segments"]

    result = {
        "distance_phase_pct": 100.0 * totals["distance"] / segments,
        "clinch_phase_pct": 100.0 * totals["clinch"] / segments,
        "ground_phase_pct": 100.0 * totals["ground"] / segments,
    }
    for side in ("red", "blue"):
        result[f"{side}_td_success_per_round"] = totals[f"{side}_td"] / fighter_rounds
        result[f"{side}_td_from_distance_per_round"] = (
            totals[f"{side}_td_distance"] / fighter_rounds
        )
        result[f"{side}_td_from_clinch_per_round"] = (
            totals[f"{side}_td_clinch"] / fighter_rounds
        )
    return result


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.simulations <= 0:
        raise ValueError("--simulations must be positive")

    rounds = pd.read_parquet(
        prior.ROUND_STATS_PATH,
        columns=["fight_id", "fighter_id", "fighter_name", "corner", "total_rounds"],
    )
    rounds["fight_id"] = rounds["fight_id"].astype(str)
    rounds["fighter_id"] = rounds["fighter_id"].astype(str)

    history_columns = [
        "fight_id",
        "fighter_id",
        *prior.STYLE_COLUMNS.values(),
        TD_COMPLETION_COLUMN,
    ]
    history = pd.read_parquet(prior.RFS_HISTORY_PATH, columns=history_columns)
    history["fight_id"] = history["fight_id"].astype(str)
    history["fighter_id"] = history["fighter_id"].astype(str)

    rows: list[dict[str, object]] = []

    for archetype, fight_id in prior.TARGET_FIGHTS:
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

        target_card_path = prior.v1_1.OUTPUT_DIR / (
            f"fsr_{fight_id}_locked_families_v1_1_target_card.csv"
        )
        if not target_card_path.exists():
            prior.v1_1.run_rating_builders(fight_id)

        cards: dict[str, dict[str, float]] = {}
        names: dict[str, str] = {}
        refs: dict[str, float] = {}

        for side, fighter_row in (("red", red_row), ("blue", blue_row)):
            fighter_id = str(fighter_row["fighter_id"])
            card, _ = prior.v1_1.load_locked_card(fight_id, fighter_id)
            card = prior.attach_style(
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
            attempts = float(card["style_td_attempts_per_round"])
            reference = float("nan") if completion is None else attempts * completion

            cards[side] = card
            names[side] = str(fighter_row["fighter_name"])
            refs[side] = reference
            card["style_td_completion_rate"] = (
                float("nan") if completion is None else completion
            )

        for candidate in CANDIDATES:
            red_transition = build_transition(
                cards["red"], hill_power=candidate.hill_power
            )
            blue_transition = build_transition(
                cards["blue"], hill_power=candidate.hill_power
            )
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
                sim = float(metrics[f"{side}_td_success_per_round"])
                ref = refs[side]
                row[f"{side}_td_attempts_per_round"] = cards[side][
                    "style_td_attempts_per_round"
                ]
                row[f"{side}_td_completion_rate"] = cards[side][
                    "style_td_completion_rate"
                ]
                row[f"{side}_historical_td_success_reference"] = ref
                row[f"{side}_sim_minus_reference"] = (
                    float("nan") if not isfinite(ref) else sim - ref
                )
            rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No TD contrast rows produced")

    print()
    print("=" * 170)
    print("V1.4 TAKEDOWN CONTRAST GRID")
    print("=" * 170)
    display = [
        "candidate", "archetype", "red_name", "blue_name",
        "distance_phase_pct", "clinch_phase_pct", "ground_phase_pct",
        "red_td_attempts_per_round", "red_td_completion_rate",
        "red_historical_td_success_reference", "red_td_success_per_round",
        "red_td_from_distance_per_round", "red_td_from_clinch_per_round",
        "blue_td_attempts_per_round", "blue_td_completion_rate",
        "blue_historical_td_success_reference", "blue_td_success_per_round",
        "blue_td_from_distance_per_round", "blue_td_from_clinch_per_round",
    ]
    print(result[display].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    long_rows: list[dict[str, object]] = []
    for _, row in result.iterrows():
        for side in ("red", "blue"):
            ref = row[f"{side}_historical_td_success_reference"]
            sim = row[f"{side}_td_success_per_round"]
            if pd.isna(ref):
                continue
            long_rows.append({
                "candidate": row["candidate"],
                "fighter": row[f"{side}_name"],
                "reference": float(ref),
                "simulated": float(sim),
                "abs_error": abs(float(sim) - float(ref)),
                "low_attempt_style": (
                    float(row[f"{side}_td_attempts_per_round"]) <= 0.30
                ),
            })
    long = pd.DataFrame(long_rows)

    print()
    print("=" * 170)
    print("CANDIDATE SCALE SUMMARY")
    print("=" * 170)
    summary = (
        long.groupby("candidate", sort=False)
        .agg(
            mean_reference=("reference", "mean"),
            mean_simulated=("simulated", "mean"),
            mean_abs_error=("abs_error", "mean"),
            max_abs_error=("abs_error", "max"),
        )
        .reset_index()
    )
    low = long.loc[long["low_attempt_style"]].groupby("candidate", sort=False).agg(
        low_style_mean_reference=("reference", "mean"),
        low_style_mean_simulated=("simulated", "mean"),
        low_style_mean_abs_error=("abs_error", "mean"),
    ).reset_index()
    summary = summary.merge(low, on="candidate", how="left")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print()
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
