"""Batch archetype validation for frozen shadow FSR/MC V1.3.

This script is diagnostic only. It does NOT change the Monte Carlo engine,
locked FSR equations, V1.1 population centering, V1.2 phase-conditioned
activity conversion, cardio bridge, judging, or V1.3 provisional finish
hazards.

It resolves a small curated set of historical fights from fighter names and
event dates, builds each leakage-safe PRE-fight FSR card, runs the same frozen
V1.3 simulator configuration, and prints compact cross-matchup mechanics.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.experimental import run_fsr_historical_fight_v1 as base
from scripts.experimental import run_fsr_historical_fight_locked_v1_3 as v1_3


DEFAULT_SIMULATIONS = 500
DEFAULT_SEED = 2026080700
OUTPUT_PATH = (
    Path("data/simulation/rfs_mc_v2_shared_state")
    / "fsr_v1_3_archetype_validation_summary.csv"
)


@dataclass(frozen=True)
class FightSpec:
    archetype: str
    fighter_a: str
    fighter_b: str
    event_date: str


CURATED_FIGHTS = (
    FightSpec(
        "corrected-activity reference / power strikers",
        "Ilia Topuria",
        "Justin Gaethje",
        "2026-06-14",
    ),
    FightSpec(
        "wrestler vs striker",
        "Merab Dvalishvili",
        "Sean O'Malley",
        "2024-09-14",
    ),
    FightSpec(
        "wrestling-submission vs striker",
        "Islam Makhachev",
        "Dustin Poirier",
        "2024-06-01",
    ),
    FightSpec(
        "submission / grappling",
        "Charles Oliveira",
        "Beneil Dariush",
        "2023-06-10",
    ),
    FightSpec(
        "high power / chin",
        "Alex Pereira",
        "Jiri Prochazka",
        "2024-06-29",
    ),
    FightSpec(
        "high-volume striker vs striker",
        "Max Holloway",
        "Calvin Kattar",
        "2021-01-16",
    ),
    FightSpec(
        "wrestler vs wrestler",
        "Kamaru Usman",
        "Colby Covington",
        "2021-11-06",
    ),
)


def normalize_name(value: str) -> str:
    """Normalize UFCStats names for robust curated-pair matching."""

    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def resolve_fight_id(rounds: pd.DataFrame, spec: FightSpec) -> str:
    """Resolve one exact-date fight from an unordered normalized name pair."""

    target_date = pd.Timestamp(spec.event_date).normalize()
    day = rounds.loc[
        rounds["event_date"].dt.normalize() == target_date
    ].copy()

    if day.empty:
        raise RuntimeError(
            f"No UFCStats rows on {spec.event_date} for "
            f"{spec.fighter_a} vs {spec.fighter_b}"
        )

    wanted = {
        normalize_name(spec.fighter_a),
        normalize_name(spec.fighter_b),
    }
    matches: list[str] = []

    for fight_id, fight_rows in day.groupby("fight_id", sort=False):
        names = {
            normalize_name(name)
            for name in fight_rows["fighter_name"].dropna().unique()
        }
        if names == wanted:
            matches.append(str(fight_id))

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one curated fight match for "
            f"{spec.fighter_a} vs {spec.fighter_b} on {spec.event_date}; "
            f"found {len(matches)}: {matches}"
        )

    return matches[0]


def build_inputs(
    red_card: dict[str, float],
    blue_card: dict[str, float],
    baselines: dict[str, float],
):
    """Translate two frozen V1.3 FSR cards through installed adapter."""

    return (
        base.build_transition(red_card),
        base.build_transition(blue_card),
        base.build_phase(red_card, blue_card, baselines),
        base.build_phase(blue_card, red_card, baselines),
        base.build_dynamic(red_card),
        base.build_dynamic(blue_card),
    )


def run_compact_population(
    *,
    red_card: dict[str, float],
    blue_card: dict[str, float],
    baselines: dict[str, float],
    scheduled_rounds: int,
    simulations: int,
    seed_start: int,
) -> dict[str, float]:
    """Run one matchup and retain cross-archetype calibration diagnostics."""

    (
        red_transition,
        blue_transition,
        red_phase,
        blue_phase,
        red_dynamic,
        blue_dynamic,
    ) = build_inputs(red_card, blue_card, baselines)

    candidate = base.Candidate(
        landed_ko_hazard=base.V1_LANDED_KO_HAZARD,
        knockdown_bonus_hazard=base.V1_KNOCKDOWN_BONUS_HAZARD,
    )
    dynamic_cal = base.state_calibration(candidate)
    phase_cal = base.phase_effect_calibration(candidate)
    transition_cal = base.zero_transition_effect_calibration()
    finish_cal = base.finish_calibration(candidate)

    totals = defaultdict(float)
    finish_rounds = Counter()
    round_reached = Counter()
    round_completed = Counter()
    round_fatigue = {
        r: defaultdict(float)
        for r in range(1, scheduled_rounds + 1)
    }

    for simulation_index in range(simulations):
        seed = seed_start + simulation_index

        path = base.run_finish_enabled_dynamic_path(
            red_transition,
            blue_transition,
            red_phase,
            blue_phase,
            red_dynamic,
            blue_dynamic,
            dynamic_state_calibration=dynamic_cal,
            phase_effect_calibration=phase_cal,
            transition_effect_calibration=transition_cal,
            finish_probability_calibration=finish_cal,
            scheduled_rounds=scheduled_rounds,
            seed=seed,
            red_intrinsic_power_multiplier=(
                base.intrinsic_power_multiplier(red_card["finishing_power"])
            ),
            blue_intrinsic_power_multiplier=(
                base.intrinsic_power_multiplier(blue_card["finishing_power"])
            ),
            red_intrinsic_ko_vulnerability_multiplier=(
                base.intrinsic_ko_vulnerability_multiplier(
                    red_card["chin_resistance"]
                )
            ),
            blue_intrinsic_ko_vulnerability_multiplier=(
                base.intrinsic_ko_vulnerability_multiplier(
                    blue_card["chin_resistance"]
                )
            ),
        )

        segments_by_round = defaultdict(list)

        for segment in path.segments:
            round_number = int(segment.state.round_number)
            segments_by_round[round_number].append(segment)

            phase = segment.state.phase.value
            totals[f"{phase}_segments"] += 1.0

            red_activity = segment.activity.red
            blue_activity = segment.activity.blue

            totals["red_distance_attempted"] += getattr(
                red_activity, "sig_str_attempted", 0
            )
            totals["blue_distance_attempted"] += getattr(
                blue_activity, "sig_str_attempted", 0
            )
            totals["red_distance_landed"] += getattr(
                red_activity, "sig_str_landed", 0
            )
            totals["blue_distance_landed"] += getattr(
                blue_activity, "sig_str_landed", 0
            )
            totals["red_control_seconds"] += getattr(
                red_activity, "control_seconds", 0
            )
            totals["blue_control_seconds"] += getattr(
                blue_activity, "control_seconds", 0
            )
            totals["red_submission_attempts"] += getattr(
                red_activity, "submission_attempts", 0
            )
            totals["blue_submission_attempts"] += getattr(
                blue_activity, "submission_attempts", 0
            )
            totals["red_knockdowns"] += getattr(
                red_activity, "knockdowns", 0
            )
            totals["blue_knockdowns"] += getattr(
                blue_activity, "knockdowns", 0
            )

            transition = segment.transition
            if (
                transition is not None
                and transition.event.value == "takedown"
            ):
                if transition.actor is base.FighterSide.RED:
                    totals["red_takedowns"] += 1
                elif transition.actor is base.FighterSide.BLUE:
                    totals["blue_takedowns"] += 1

        for round_number, segments in segments_by_round.items():
            round_reached[round_number] += 1
            last_segment = segments[-1]

            if int(last_segment.state.segment_number) == 10:
                round_completed[round_number] += 1
                state = last_segment.dynamic_state_after_activity
                round_fatigue[round_number]["red"] += state.red.fatigue
                round_fatigue[round_number]["blue"] += state.blue.fatigue

        final_state = path.segments[-1].dynamic_state_after_segment
        totals["red_final_damage"] += final_state.red.damage
        totals["blue_final_damage"] += final_state.blue.damage

        result = base.resolve_final_fight_result(path)

        if result.winner is base.FighterSide.RED:
            totals["red_wins"] += 1
        elif result.winner is base.FighterSide.BLUE:
            totals["blue_wins"] += 1
        else:
            totals["draws"] += 1

        if result.finish is None:
            totals["decisions"] += 1
        else:
            finish_rounds[int(result.finish.round_number)] += 1
            method = result.finish.method.value.lower()
            if "ko" in method:
                totals["ko_finishes"] += 1
            if "submission" in method:
                totals["submission_finishes"] += 1

    total_segments = (
        totals["distance_segments"]
        + totals["clinch_segments"]
        + totals["ground_segments"]
    )
    total_rounds_reached = float(sum(round_reached.values()))

    if total_segments <= 0 or total_rounds_reached <= 0:
        raise RuntimeError("Simulator produced no usable exposure.")

    def pct(count: float) -> float:
        return 100.0 * float(count) / float(simulations)

    def round_fatigue_mean(round_number: int, side: str) -> float:
        completed = round_completed.get(round_number, 0)
        if completed <= 0:
            return float("nan")
        return round_fatigue[round_number][side] / completed

    decisions = totals["decisions"]
    decision_draw_pct = (
        100.0 * totals["draws"] / decisions
        if decisions > 0
        else float("nan")
    )

    return {
        "red_win_pct": pct(totals["red_wins"]),
        "blue_win_pct": pct(totals["blue_wins"]),
        "draw_pct": pct(totals["draws"]),
        "ko_pct": pct(totals["ko_finishes"]),
        "submission_pct": pct(totals["submission_finishes"]),
        "scheduled_distance_pct": pct(totals["decisions"]),
        "round1_finish_pct": pct(finish_rounds[1]),
        "decision_draw_pct": decision_draw_pct,
        "distance_phase_pct": (
            100.0 * totals["distance_segments"] / total_segments
        ),
        "clinch_phase_pct": (
            100.0 * totals["clinch_segments"] / total_segments
        ),
        "ground_phase_pct": (
            100.0 * totals["ground_segments"] / total_segments
        ),
        "red_distance_attempts_per_reached_round": (
            totals["red_distance_attempted"] / total_rounds_reached
        ),
        "blue_distance_attempts_per_reached_round": (
            totals["blue_distance_attempted"] / total_rounds_reached
        ),
        "red_takedowns_per_fight": totals["red_takedowns"] / simulations,
        "blue_takedowns_per_fight": totals["blue_takedowns"] / simulations,
        "red_control_seconds_per_reached_round": (
            totals["red_control_seconds"] / total_rounds_reached
        ),
        "blue_control_seconds_per_reached_round": (
            totals["blue_control_seconds"] / total_rounds_reached
        ),
        "red_submission_attempts_per_fight": (
            totals["red_submission_attempts"] / simulations
        ),
        "blue_submission_attempts_per_fight": (
            totals["blue_submission_attempts"] / simulations
        ),
        "red_knockdowns_per_fight": totals["red_knockdowns"] / simulations,
        "blue_knockdowns_per_fight": totals["blue_knockdowns"] / simulations,
        "red_r3_fatigue": round_fatigue_mean(3, "red"),
        "blue_r3_fatigue": round_fatigue_mean(3, "blue"),
        "red_r5_fatigue": round_fatigue_mean(5, "red"),
        "blue_r5_fatigue": round_fatigue_mean(5, "blue"),
        "red_final_damage": totals["red_final_damage"] / simulations,
        "blue_final_damage": totals["blue_final_damage"] / simulations,
    }


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

    v1_3.install_overrides()

    rounds = pd.read_parquet(base.ROUND_PATH)
    rounds["event_date"] = pd.to_datetime(rounds["event_date"])
    rounds["fight_id"] = rounds["fight_id"].astype(str)

    rows: list[dict[str, object]] = []

    print()
    print("=" * 150)
    print("FSR / MC V1.3 MULTI-FIGHT ARCHETYPE VALIDATION")
    print("=" * 150)
    print(
        f"Simulations per fight: {args.simulations} | Seed start: {args.seed}"
    )
    print(
        "Frozen: FSR equations, V1.1 centering, V1.2 activity, "
        "V1.3 finish hazards, MC engine."
    )
    print()

    for index, spec in enumerate(CURATED_FIGHTS, start=1):
        fight_id = resolve_fight_id(rounds, spec)

        (
            fight_rounds,
            target_date,
            red_info,
            blue_info,
            scheduled_rounds,
        ) = base.load_target_fight(fight_id)

        print(
            f"[{index}/{len(CURATED_FIGHTS)}] {spec.archetype}: "
            f"{red_info['fighter_name']} vs {blue_info['fighter_name']} "
            f"({target_date.date()}, {fight_id})"
        )

        base.run_rating_builders(fight_id)

        red_card, red_prior_fights = base.build_full_card(
            fight_id, red_info["fighter_id"]
        )
        blue_card, blue_prior_fights = base.build_full_card(
            fight_id, blue_info["fighter_id"]
        )

        if red_prior_fights < 3 or blue_prior_fights < 3:
            print(
                "  WARNING: outside primary established-fighter cohort: "
                f"{red_prior_fights} / {blue_prior_fights} prior fights"
            )

        baselines = base.population_baselines(fight_rounds, target_date)

        metrics = run_compact_population(
            red_card=red_card,
            blue_card=blue_card,
            baselines=baselines,
            scheduled_rounds=scheduled_rounds,
            simulations=args.simulations,
            seed_start=args.seed,
        )

        row: dict[str, object] = {
            "archetype": spec.archetype,
            "fight_id": fight_id,
            "event_date": str(target_date.date()),
            "red_name": red_info["fighter_name"],
            "blue_name": blue_info["fighter_name"],
            "scheduled_rounds": scheduled_rounds,
            "red_prior_ufc_fights": red_prior_fights,
            "blue_prior_ufc_fights": blue_prior_fights,
            "red_distance_precision": red_card["distance_precision"],
            "blue_distance_precision": blue_card["distance_precision"],
            "red_wrestling_entry": red_card["wrestling_entry"],
            "blue_wrestling_entry": blue_card["wrestling_entry"],
            "red_control_imposition": red_card["control_imposition"],
            "blue_control_imposition": blue_card["control_imposition"],
            "red_submission_pressure": red_card["submission_pressure"],
            "blue_submission_pressure": blue_card["submission_pressure"],
            "red_striking_power": red_card["striking_power"],
            "blue_striking_power": blue_card["striking_power"],
            "red_chin_resistance": red_card["chin_resistance"],
            "blue_chin_resistance": blue_card["chin_resistance"],
            "red_cardio_resistance": red_card[
                "fatigue_accumulation_resistance_rating"
            ],
            "blue_cardio_resistance": blue_card[
                "fatigue_accumulation_resistance_rating"
            ],
        }
        row.update(metrics)
        rows.append(row)

    result = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    compact_columns = [
        "archetype",
        "red_name",
        "blue_name",
        "red_win_pct",
        "blue_win_pct",
        "draw_pct",
        "ko_pct",
        "submission_pct",
        "scheduled_distance_pct",
        "round1_finish_pct",
        "distance_phase_pct",
        "clinch_phase_pct",
        "ground_phase_pct",
        "red_r3_fatigue",
        "blue_r3_fatigue",
        "decision_draw_pct",
    ]

    print()
    print("=" * 150)
    print("CROSS-ARCHETYPE SUMMARY")
    print("=" * 150)
    print(
        result[compact_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print()
    print("=" * 150)
    print("ACTIVITY / CONTROL SUMMARY")
    print("=" * 150)

    activity_columns = [
        "archetype",
        "red_name",
        "blue_name",
        "red_distance_attempts_per_reached_round",
        "blue_distance_attempts_per_reached_round",
        "red_takedowns_per_fight",
        "blue_takedowns_per_fight",
        "red_control_seconds_per_reached_round",
        "blue_control_seconds_per_reached_round",
        "red_submission_attempts_per_fight",
        "blue_submission_attempts_per_fight",
        "red_knockdowns_per_fight",
        "blue_knockdowns_per_fight",
    ]

    print(
        result[activity_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print()
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
