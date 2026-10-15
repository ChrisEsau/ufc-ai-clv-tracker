"""Audit RFS style-aware transition candidates for FSR/MC V1.4.

Shadow/research only.

This script does not modify the simulator engine, locked FSR equations, V1.1
population centering, V1.2 activity conversion, V1.3 finish hazards, cardio,
judging, or production artifacts.

Purpose
-------
The V1.3 multi-fight archetype audit showed that transition behavior is too
compressed across fighter styles.  The existing leakage-safe Phase Baseline
RFS state contains strong PRE-fight tendency separation, so this script tests
whether those style signals can gate phase transitions while FSR ratings remain
responsible for skill/ability.

The grid is intentionally transition-only:
- it runs the static shared-state path;
- there is no activity, fatigue, damage, finish, or judging censoring;
- all rounds complete;
- identical seed blocks are used for every candidate.

The first candidate is the untouched V1.3 transition mapping and default engine
calibration.  Later candidates use the same proposed style mapping with a small
range of static transition-calibration bundles passed through the engine's
existing SharedPathCalibration interface.  No engine source is changed.
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
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)

from scripts.experimental import run_fsr_historical_fight_locked_v1 as locked_v1
from scripts.experimental import run_fsr_historical_fight_locked_v1_1 as v1_1


DEFAULT_SIMULATIONS = 500
DEFAULT_SEED = 2026080900

ROUND_STATS_PATH = Path("data/fight_details/ufc_round_stats.parquet")
RFS_HISTORY_PATH = Path("data/features/round_fighter_state_history.parquet")
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_v1_4_transition_style_grid.csv"
)

TARGET_FIGHTS = (
    ("power strikers", "7208e40818401e88"),
    ("wrestler vs striker", "3146e5a47a922976"),
    ("submission / grappling", "40e8bf8ce508c436"),
    ("high power / chin", "bca5d01f8775f852"),
    ("high-volume striker vs striker", "a4817b7e46028b4a"),
    ("wrestler vs wrestler", "31b3ae9352d9389b"),
)

STYLE_COLUMNS = {
    "distance_share": "rfs_phase_base_ewm_distance_attempt_share",
    "clinch_share": "rfs_phase_base_ewm_clinch_attempt_share",
    "ground_share": "rfs_phase_base_ewm_ground_attempt_share",
    "td_attempts_per_round": "rfs_phase_base_ewm_td_attempts_per_round",
    "failed_td_attempts_per_round": (
        "rfs_phase_base_ewm_failed_td_attempts_per_round"
    ),
    "td_persistence_ratio": "rfs_phase_base_ewm_td_persistence_ratio",
    "control_seconds_per_round": (
        "rfs_phase_base_ewm_control_seconds_per_round"
    ),
    "non_distance_clinch_share": (
        "rfs_phase_base_ewm_non_distance_clinch_share"
    ),
    "non_distance_ground_share": (
        "rfs_phase_base_ewm_non_distance_ground_share"
    ),
}


@dataclass(frozen=True)
class Candidate:
    name: str
    use_style_mapping: bool
    calibration: SharedPathCalibration


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def finite_or_none(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(parsed):
        return None
    return parsed


def saturating_ratio(value: float, half_saturation: float) -> float:
    """Map nonnegative tendency evidence to [0, 1) monotonically."""

    if value <= 0.0:
        return 0.0
    return clamp01(value / (value + half_saturation))


def attach_style(
    card: dict[str, float],
    history: pd.DataFrame,
    *,
    fight_id: str,
    fighter_id: str,
) -> dict[str, float]:
    """Attach one fighter's leakage-safe PRE-fight Phase Baseline EWM state."""

    rows = history.loc[
        (history["fight_id"] == str(fight_id))
        & (history["fighter_id"] == str(fighter_id))
    ]

    if len(rows) != 1:
        raise RuntimeError(
            "Expected exactly one RFS history row for "
            f"fight={fight_id}, fighter={fighter_id}; found {len(rows)}"
        )

    row = rows.iloc[0]
    out = dict(card)

    for short_name, column in STYLE_COLUMNS.items():
        value = finite_or_none(row[column])
        out[f"style_{short_name}"] = (
            float("nan") if value is None else value
        )

    return out


def has_style(card: dict[str, float]) -> bool:
    required = (
        "style_distance_share",
        "style_clinch_share",
        "style_td_attempts_per_round",
        "style_failed_td_attempts_per_round",
        "style_td_persistence_ratio",
        "style_control_seconds_per_round",
        "style_non_distance_clinch_share",
    )
    return all(
        finite_or_none(card.get(key)) is not None
        for key in required
    )


def build_style_transition(
    fighter: dict[str, float],
) -> FighterTransitionParameters:
    """Candidate V1.4 separation of style/tendency from FSR skill.

    Style controls how often the fighter tries to move the fight:
    - distance attempt share -> distance retention tendency;
    - TD attempts/round -> takedown entry tendency;
    - TD persistence + failed-TD frequency -> repeat-attempt tendencies;
    - non-distance/clinch evidence + control exposure -> clinch entry tendency.

    FSR remains responsible for ability:
    - Wrestling Conversion -> takedown completion ability;
    - TD Defense -> takedown resistance;
    - Control Imposition/Resistance -> phase retention/escape/resistance.

    If style evidence is unavailable, preserve the old locked V1 mapping.
    """

    if not has_style(fighter):
        return locked_v1.build_transition(fighter)

    td_entry_skill = locked_v1.base.normalized_skill(
        fighter["wrestling_entry"]
    )
    td_conversion = locked_v1.base.normalized_skill(
        fighter["wrestling_conversion"]
    )
    td_defense = locked_v1.base.normalized_skill(
        fighter["td_defense"]
    )
    control = locked_v1.base.normalized_skill(
        fighter["control_imposition"]
    )
    control_resistance = locked_v1.base.normalized_skill(
        fighter["control_resistance"]
    )

    distance_share = clamp01(fighter["style_distance_share"])
    clinch_share = clamp01(fighter["style_clinch_share"])
    non_distance_clinch = clamp01(
        fighter["style_non_distance_clinch_share"]
    )

    td_attempts = max(0.0, fighter["style_td_attempts_per_round"])
    failed_td_attempts = max(
        0.0,
        fighter["style_failed_td_attempts_per_round"],
    )
    td_persistence_raw = max(
        0.0,
        fighter["style_td_persistence_ratio"],
    )
    control_seconds = max(
        0.0,
        fighter["style_control_seconds_per_round"],
    )

    # A half-saturation of 1 TD attempt/round keeps low-use strikers near zero
    # while allowing persistent wrestlers to approach one smoothly.
    td_style = saturating_ratio(td_attempts, 1.0)

    persistence_shape = saturating_ratio(td_persistence_raw, 1.0)
    failed_shape = saturating_ratio(failed_td_attempts, 1.0)

    # Persistence cannot become high merely because a noisy ratio is high when
    # the fighter almost never attempts takedowns.
    td_persistence = td_style * (0.50 + 0.50 * persistence_shape)
    failed_td_persistence = td_style * (0.50 + 0.50 * failed_shape)

    # UFCStats does not expose exact clinch-entry counts. Use two independent
    # tendency clues without pretending either is exact phase time:
    # 1) the fighter's non-distance striking allocation toward clinch;
    # 2) accumulated control exposure, which captures non-striking clinch/cage
    #    work missed by strike-share evidence.
    non_distance_share = max(0.0, 1.0 - distance_share)
    allocated_clinch_signal = non_distance_share * non_distance_clinch
    clinch_signal = (
        0.50 * clinch_share
        + 0.50 * allocated_clinch_signal
    )
    clinch_style = saturating_ratio(clinch_signal, 0.08)
    control_style = saturating_ratio(control_seconds, 60.0)
    clinch_entry = clamp01(
        0.70 * clinch_style
        + 0.30 * control_style
    )

    # Broad ability terms remain FSR-based.  Entry style is deliberately not
    # folded into these fields so preference and skill remain separable.
    phase_imposition = (
        td_entry_skill
        + control
    ) / 2.0
    phase_resistance = (
        td_defense
        + control_resistance
    ) / 2.0

    return FighterTransitionParameters(
        distance_retention=distance_share,
        clinch_entry_tendency=clinch_entry,
        clinch_entry_resistance=control_resistance,
        takedown_entry_tendency=td_style,
        takedown_completion_ability=td_conversion,
        takedown_resistance=td_defense,
        takedown_persistence=clamp01(td_persistence),
        failed_takedown_persistence=clamp01(failed_td_persistence),
        clinch_retention=control,
        clinch_escape_ability=control_resistance,
        ground_retention=control,
        ground_escape_ability=control_resistance,
        reversal_ability=0.50,
        phase_imposition=clamp01(phase_imposition),
        phase_resistance=clamp01(phase_resistance),
    )


def path_calibration(
    *,
    distance_stay: float = 6.0,
    distance_clinch: float = 1.0,
    distance_td: float = 0.75,
    distance_strength: float = 1.0,
    clinch_stay: float = 4.5,
    clinch_break: float = 2.5,
    clinch_change: float = 1.0,
    clinch_owner_td: float = 1.5,
    clinch_defender_td: float = 0.5,
    clinch_strength: float = 1.0,
) -> SharedPathCalibration:
    return SharedPathCalibration(
        distance=DistanceTransitionCalibration(
            stay_base_weight=distance_stay,
            clinch_entry_base_weight=distance_clinch,
            takedown_base_weight=distance_td,
            matchup_effect_strength=distance_strength,
        ),
        clinch=ClinchTransitionCalibration(
            stay_base_weight=clinch_stay,
            break_base_weight=clinch_break,
            ownership_change_base_weight=clinch_change,
            owner_takedown_base_weight=clinch_owner_td,
            defender_takedown_base_weight=clinch_defender_td,
            matchup_effect_strength=clinch_strength,
        ),
        ground=GroundTransitionCalibration(),
    )


CANDIDATES = (
    Candidate(
        "v1_3_old_mapping",
        False,
        SharedPathCalibration(),
    ),
    Candidate(
        "style_default_cal",
        True,
        SharedPathCalibration(),
    ),
    Candidate(
        "style_td_half",
        True,
        path_calibration(
            distance_td=0.35,
            distance_strength=1.25,
            clinch_owner_td=0.70,
            clinch_defender_td=0.20,
            clinch_strength=1.25,
        ),
    ),
    Candidate(
        "style_td_low_strong",
        True,
        path_calibration(
            distance_td=0.20,
            distance_strength=2.0,
            clinch_owner_td=0.40,
            clinch_defender_td=0.10,
            clinch_strength=2.0,
        ),
    ),
    Candidate(
        "style_balanced",
        True,
        path_calibration(
            distance_stay=7.0,
            distance_clinch=0.75,
            distance_td=0.20,
            distance_strength=2.0,
            clinch_stay=4.5,
            clinch_break=3.0,
            clinch_change=0.80,
            clinch_owner_td=0.40,
            clinch_defender_td=0.10,
            clinch_strength=2.0,
        ),
    ),
    Candidate(
        "style_conservative",
        True,
        path_calibration(
            distance_stay=8.0,
            distance_clinch=0.55,
            distance_td=0.12,
            distance_strength=2.5,
            clinch_stay=4.0,
            clinch_break=3.5,
            clinch_change=0.60,
            clinch_owner_td=0.25,
            clinch_defender_td=0.05,
            clinch_strength=2.5,
        ),
    ),
)


def run_static_candidate(
    *,
    red_transition: FighterTransitionParameters,
    blue_transition: FighterTransitionParameters,
    calibration: SharedPathCalibration,
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
        "red_clinch_entry": 0.0,
        "blue_clinch_entry": 0.0,
    }

    for index in range(simulations):
        path = run_shared_state_path(
            red_transition,
            blue_transition,
            scheduled_rounds=scheduled_rounds,
            seed=seed_start + index,
            calibration=calibration,
        )

        for segment in path.segments:
            totals["segments"] += 1.0
            if segment.state.phase is FightPhase.DISTANCE:
                totals["distance"] += 1.0
            elif segment.state.phase is FightPhase.CLINCH:
                totals["clinch"] += 1.0
            elif segment.state.phase is FightPhase.GROUND:
                totals["ground"] += 1.0

            transition = segment.transition
            if transition is None:
                continue

            if transition.event is TransitionEvent.TAKEDOWN:
                if transition.actor is FighterSide.RED:
                    totals["red_td"] += 1.0
                elif transition.actor is FighterSide.BLUE:
                    totals["blue_td"] += 1.0

            if transition.event is TransitionEvent.CLINCH_ENTRY:
                if transition.actor is FighterSide.RED:
                    totals["red_clinch_entry"] += 1.0
                elif transition.actor is FighterSide.BLUE:
                    totals["blue_clinch_entry"] += 1.0

    fighter_rounds = float(simulations * scheduled_rounds)
    segments = totals["segments"]

    return {
        "distance_phase_pct": 100.0 * totals["distance"] / segments,
        "clinch_phase_pct": 100.0 * totals["clinch"] / segments,
        "ground_phase_pct": 100.0 * totals["ground"] / segments,
        "red_td_success_per_round": totals["red_td"] / fighter_rounds,
        "blue_td_success_per_round": totals["blue_td"] / fighter_rounds,
        "red_clinch_entries_per_round": (
            totals["red_clinch_entry"] / fighter_rounds
        ),
        "blue_clinch_entries_per_round": (
            totals["blue_clinch_entry"] / fighter_rounds
        ),
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

    if not ROUND_STATS_PATH.exists():
        raise FileNotFoundError(ROUND_STATS_PATH)
    if not RFS_HISTORY_PATH.exists():
        raise FileNotFoundError(RFS_HISTORY_PATH)

    rounds = pd.read_parquet(
        ROUND_STATS_PATH,
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

    history_columns = [
        "fight_id",
        "fighter_id",
        *STYLE_COLUMNS.values(),
    ]
    history = pd.read_parquet(
        RFS_HISTORY_PATH,
        columns=history_columns,
    )
    history["fight_id"] = history["fight_id"].astype(str)
    history["fighter_id"] = history["fighter_id"].astype(str)

    rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []

    for archetype, fight_id in TARGET_FIGHTS:
        target = rounds.loc[rounds["fight_id"] == fight_id].copy()
        if target.empty:
            print(f"SKIP {fight_id}: fight not found")
            continue

        fighters = (
            target[
                [
                    "fighter_id",
                    "fighter_name",
                    "corner",
                    "total_rounds",
                ]
            ]
            .drop_duplicates(subset=["fighter_id"])
            .copy()
        )

        if len(fighters) != 2:
            raise RuntimeError(
                f"Expected two fighters for {fight_id}; found {len(fighters)}"
            )

        red_row = fighters.loc[
            fighters["corner"].astype(str).str.upper() == "RED"
        ].iloc[0]
        blue_row = fighters.loc[
            fighters["corner"].astype(str).str.upper() == "BLUE"
        ].iloc[0]

        scheduled_rounds = int(float(red_row["total_rounds"]))

        # The V1.1 target cards were generated by the preceding archetype
        # validation. Rebuild only if a user runs this grid independently.
        target_card_path = v1_1.OUTPUT_DIR / (
            f"fsr_{fight_id}_locked_families_v1_1_target_card.csv"
        )
        if not target_card_path.exists():
            v1_1.run_rating_builders(fight_id)

        red_card, _ = v1_1.load_locked_card(
            fight_id,
            str(red_row["fighter_id"]),
        )
        blue_card, _ = v1_1.load_locked_card(
            fight_id,
            str(blue_row["fighter_id"]),
        )

        red_card = attach_style(
            red_card,
            history,
            fight_id=fight_id,
            fighter_id=str(red_row["fighter_id"]),
        )
        blue_card = attach_style(
            blue_card,
            history,
            fight_id=fight_id,
            fighter_id=str(blue_row["fighter_id"]),
        )

        # Print the proposed style-mapping parameters once per fighter.
        for fighter_name, card in (
            (str(red_row["fighter_name"]), red_card),
            (str(blue_row["fighter_name"]), blue_card),
        ):
            transition = build_style_transition(card)
            mapping_rows.append(
                {
                    "archetype": archetype,
                    "fighter_name": fighter_name,
                    "style_td_att_per_round": card[
                        "style_td_attempts_per_round"
                    ],
                    "style_distance_share": card["style_distance_share"],
                    "mapped_distance_retention": transition.distance_retention,
                    "mapped_clinch_entry": transition.clinch_entry_tendency,
                    "mapped_td_entry": transition.takedown_entry_tendency,
                    "mapped_td_persistence": transition.takedown_persistence,
                    "mapped_failed_td_persistence": (
                        transition.failed_takedown_persistence
                    ),
                    "fsr_td_conversion": transition.takedown_completion_ability,
                    "fsr_td_resistance": transition.takedown_resistance,
                }
            )

        for candidate in CANDIDATES:
            if candidate.use_style_mapping:
                red_transition = build_style_transition(red_card)
                blue_transition = build_style_transition(blue_card)
            else:
                red_transition = locked_v1.build_transition(red_card)
                blue_transition = locked_v1.build_transition(blue_card)

            metrics = run_static_candidate(
                red_transition=red_transition,
                blue_transition=blue_transition,
                calibration=candidate.calibration,
                scheduled_rounds=scheduled_rounds,
                simulations=args.simulations,
                seed_start=args.seed,
            )

            rows.append(
                {
                    "candidate": candidate.name,
                    "archetype": archetype,
                    "fight_id": fight_id,
                    "red_name": str(red_row["fighter_name"]),
                    "blue_name": str(blue_row["fighter_name"]),
                    "red_style_td_attempts_per_round": red_card[
                        "style_td_attempts_per_round"
                    ],
                    "blue_style_td_attempts_per_round": blue_card[
                        "style_td_attempts_per_round"
                    ],
                    **metrics,
                }
            )

    result = pd.DataFrame(rows)
    mapping = pd.DataFrame(mapping_rows).drop_duplicates()

    if result.empty:
        raise RuntimeError("No transition-grid rows were produced.")

    print()
    print("=" * 150)
    print("V1.4 PROPOSED RFS STYLE -> TRANSITION PARAMETER MAPPING")
    print("=" * 150)
    print(
        mapping.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print()
    print("=" * 150)
    print("V1.4 TRANSITION-ONLY CANDIDATE GRID")
    print("=" * 150)
    display_columns = [
        "candidate",
        "archetype",
        "red_name",
        "blue_name",
        "distance_phase_pct",
        "clinch_phase_pct",
        "ground_phase_pct",
        "red_style_td_attempts_per_round",
        "red_td_success_per_round",
        "blue_style_td_attempts_per_round",
        "blue_td_success_per_round",
    ]
    print(
        result[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print()
    print("=" * 150)
    print("LOW-TD STYLE OVERSHOOT CHECK")
    print("=" * 150)
    low_style_rows: list[dict[str, object]] = []

    for _, row in result.iterrows():
        for side in ("red", "blue"):
            style = float(row[f"{side}_style_td_attempts_per_round"])
            simulated = float(row[f"{side}_td_success_per_round"])
            if style <= 0.30:
                low_style_rows.append(
                    {
                        "candidate": row["candidate"],
                        "archetype": row["archetype"],
                        "fighter": row[f"{side}_name"],
                        "style_td_attempts_per_round": style,
                        "sim_td_success_per_round": simulated,
                        "success_minus_attempt_tendency": simulated - style,
                    }
                )

    low_style = pd.DataFrame(low_style_rows)
    if not low_style.empty:
        print(
            low_style.to_string(
                index=False,
                float_format=lambda value: f"{value:.3f}",
            )
        )

    print()
    print("=" * 150)
    print("CANDIDATE AGGREGATES")
    print("=" * 150)
    aggregate = (
        result.groupby("candidate", sort=False)
        .agg(
            mean_distance_phase_pct=("distance_phase_pct", "mean"),
            mean_clinch_phase_pct=("clinch_phase_pct", "mean"),
            mean_ground_phase_pct=("ground_phase_pct", "mean"),
            mean_red_td_success_per_round=("red_td_success_per_round", "mean"),
            mean_blue_td_success_per_round=("blue_td_success_per_round", "mean"),
        )
        .reset_index()
    )
    print(
        aggregate.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print()
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
