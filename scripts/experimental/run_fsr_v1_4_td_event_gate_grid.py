"""Audit propensity-gated successful-takedown transitions for FSR/MC V1.4.

Shadow/research only.

The V1.4 TD contrast grid showed that Hill-gating entry/persistence improves
style separation, but low-use strikers still receive too many successful
TAKEDOWN transitions.  The reason is structural: the transition engine's
TAKEDOWN event already represents a *completed* takedown, while its score adds
ungated completion/phase-imposition terms alongside entry tendency.

This audit leaves the simulator engine unchanged and tests an adapter-level
interpretation that matches those semantics:

    effective completion contribution = TD propensity gate * FSR conversion

Optionally, broad offensive phase-imposition is also style-gated so neutral
wrestling/control ability cannot create offensive phase changes by itself.
Raw FSR ratings are never changed; only effective transition parameters passed
to the existing engine are modified.

The historical reference remains descriptive only:

    PRE-fight EWM TD attempts/round * PRE-fight EWM TD completion rate

Opponent TD defense should move a target matchup away from that reference.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path

import pandas as pd

from pipeline.simulation.rfs_mc_v2_shared_state.path_runner import (
    SharedPathCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
    ClinchTransitionCalibration,
    DistanceTransitionCalibration,
    GroundTransitionCalibration,
)

from scripts.experimental import run_fsr_historical_fight_v1 as base
from scripts.experimental import run_fsr_v1_4_td_contrast_grid as contrast


DEFAULT_SIMULATIONS = 500
DEFAULT_SEED = 2026080920
TD_COMPLETION_COLUMN = "rfs_phase_base_ewm_td_completion_rate"
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_v1_4_td_event_gate_grid.csv"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    hill_power: float
    matchup_strength: float
    gate_completion: bool
    gate_phase_imposition: bool


def calibration(candidate: Candidate) -> SharedPathCalibration:
    """Preserve base weights; vary only existing matchup contrast strength."""

    strength = candidate.matchup_strength
    return SharedPathCalibration(
        distance=DistanceTransitionCalibration(
            stay_base_weight=6.0,
            clinch_entry_base_weight=1.0,
            takedown_base_weight=0.75,
            matchup_effect_strength=strength,
        ),
        clinch=ClinchTransitionCalibration(
            stay_base_weight=4.5,
            break_base_weight=2.5,
            ownership_change_base_weight=1.0,
            owner_takedown_base_weight=1.5,
            defender_takedown_base_weight=0.5,
            matchup_effect_strength=strength,
        ),
        ground=GroundTransitionCalibration(),
    )


def build_transition(card: dict[str, float], candidate: Candidate):
    """Build one style-aware effective transition profile.

    Entry and persistence are Hill-gated by the prior contrast audit.  This
    audit optionally gates completion and broad offensive phase imposition by
    the same attempt propensity.  Defensive/retention abilities remain raw FSR
    abilities because once a phase exists, preference should not erase skill.
    """

    transition = contrast.build_transition(
        card,
        hill_power=candidate.hill_power,
    )

    if not contrast.prior.has_style(card):
        return transition

    td_attempts = max(0.0, float(card["style_td_attempts_per_round"]))
    td_style = contrast.hill_ratio(
        td_attempts,
        1.0,
        candidate.hill_power,
    )

    updates: dict[str, float] = {}

    if candidate.gate_completion:
        updates["takedown_completion_ability"] = contrast.prior.clamp01(
            td_style * transition.takedown_completion_ability
        )

    if candidate.gate_phase_imposition:
        td_entry_skill = base.normalized_skill(card["wrestling_entry"])
        control_skill = base.normalized_skill(card["control_imposition"])

        # Offensive phase imposition requires both ability and a demonstrated
        # tendency to use the relevant route.  Clinch style remains separate
        # from TD style so a clinch-heavy non-wrestler can still force clinches.
        td_imposition = td_style * td_entry_skill
        clinch_imposition = (
            transition.clinch_entry_tendency * control_skill
        )
        updates["phase_imposition"] = contrast.prior.clamp01(
            0.50 * td_imposition
            + 0.50 * clinch_imposition
        )

    return replace(transition, **updates)


CANDIDATES = (
    Candidate("hill3_s3_current", 3.0, 3.0, False, False),
    Candidate("hill3_s3_gate_completion", 3.0, 3.0, True, False),
    Candidate("hill3_s4_gate_completion", 3.0, 4.0, True, False),
    Candidate("hill3_s3_gate_both", 3.0, 3.0, True, True),
    Candidate("hill3_s4_gate_both", 3.0, 4.0, True, True),
    Candidate("hill3_s5_gate_both", 3.0, 5.0, True, True),
    Candidate("hill4_s4_gate_both", 4.0, 4.0, True, True),
)


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

    prior = contrast.prior

    rounds = pd.read_parquet(
        prior.ROUND_STATS_PATH,
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
        *prior.STYLE_COLUMNS.values(),
        TD_COMPLETION_COLUMN,
    ]
    history = pd.read_parquet(
        prior.RFS_HISTORY_PATH,
        columns=history_columns,
    )
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
            reference = (
                float("nan")
                if completion is None
                else attempts * completion
            )

            cards[side] = card
            names[side] = str(fighter_row["fighter_name"])
            refs[side] = reference

        for candidate in CANDIDATES:
            red_transition = build_transition(cards["red"], candidate)
            blue_transition = build_transition(cards["blue"], candidate)

            metrics = contrast.run_candidate(
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
                attempts = float(cards[side]["style_td_attempts_per_round"])
                ref = refs[side]
                sim = float(metrics[f"{side}_td_success_per_round"])
                transition = (
                    red_transition if side == "red" else blue_transition
                )
                row[f"{side}_td_attempts_per_round"] = attempts
                row[f"{side}_historical_td_success_reference"] = ref
                row[f"{side}_effective_td_entry"] = (
                    transition.takedown_entry_tendency
                )
                row[f"{side}_effective_td_completion"] = (
                    transition.takedown_completion_ability
                )
                row[f"{side}_effective_phase_imposition"] = (
                    transition.phase_imposition
                )
                row[f"{side}_sim_minus_reference"] = (
                    float("nan") if not isfinite(ref) else sim - ref
                )

            rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No TD event-gate rows produced")

    print()
    print("=" * 180)
    print("V1.4 SUCCESSFUL-TAKEDOWN EVENT-GATE GRID")
    print("=" * 180)
    display = [
        "candidate",
        "archetype",
        "red_name",
        "blue_name",
        "distance_phase_pct",
        "clinch_phase_pct",
        "ground_phase_pct",
        "red_historical_td_success_reference",
        "red_td_success_per_round",
        "red_td_from_distance_per_round",
        "red_td_from_clinch_per_round",
        "blue_historical_td_success_reference",
        "blue_td_success_per_round",
        "blue_td_from_distance_per_round",
        "blue_td_from_clinch_per_round",
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
            ref = row[f"{side}_historical_td_success_reference"]
            if pd.isna(ref):
                continue
            sim = float(row[f"{side}_td_success_per_round"])
            attempts = float(row[f"{side}_td_attempts_per_round"])
            long_rows.append(
                {
                    "candidate": row["candidate"],
                    "fighter": row[f"{side}_name"],
                    "reference": float(ref),
                    "simulated": sim,
                    "abs_error": abs(sim - float(ref)),
                    "low_attempt_style": attempts <= 0.30,
                    "high_attempt_style": attempts >= 1.0,
                }
            )

    long = pd.DataFrame(long_rows)

    print()
    print("=" * 180)
    print("CANDIDATE SCALE SUMMARY")
    print("=" * 180)
    overall = (
        long.groupby("candidate", sort=False)
        .agg(
            mean_reference=("reference", "mean"),
            mean_simulated=("simulated", "mean"),
            mean_abs_error=("abs_error", "mean"),
            max_abs_error=("abs_error", "max"),
        )
        .reset_index()
    )
    low = (
        long.loc[long["low_attempt_style"]]
        .groupby("candidate", sort=False)
        .agg(
            low_ref=("reference", "mean"),
            low_sim=("simulated", "mean"),
            low_mae=("abs_error", "mean"),
        )
        .reset_index()
    )
    high = (
        long.loc[long["high_attempt_style"]]
        .groupby("candidate", sort=False)
        .agg(
            high_ref=("reference", "mean"),
            high_sim=("simulated", "mean"),
            high_mae=("abs_error", "mean"),
        )
        .reset_index()
    )
    summary = overall.merge(low, on="candidate", how="left").merge(
        high,
        on="candidate",
        how="left",
    )
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print()
    print("=" * 180)
    print("ANCHOR FIGHTERS")
    print("=" * 180)
    anchor_names = {
        "Sean O'Malley",
        "Merab Dvalishvili",
        "Alex Pereira",
        "Max Holloway",
        "Kamaru Usman",
        "Colby Covington",
    }
    anchors = long.loc[long["fighter"].isin(anchor_names)].copy()
    print(
        anchors.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print()
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
