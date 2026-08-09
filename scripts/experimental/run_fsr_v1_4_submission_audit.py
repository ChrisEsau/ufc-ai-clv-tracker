"""Focused submission audit for the V1.4 RFS/FSR Monte Carlo.

Shadow/research only.

Purpose
-------
The V1.4 transition and chain-wrestling work materially increased realistic
matchup-specific ground exposure.  The previously frozen V1.3 submission
calibration was fitted under the older transition environment, so this script
separates two possible causes of excessive submission finishes:

1. submission-attempt generation while the fighter owns the ground phase;
2. conversion of each simulated attempt into a submission finish.

The audit deliberately changes no simulator source and no locked FSR equation.
It runs two contrasting grappling-relevant historical matchups with identical
seed blocks across candidates:

- Sean O'Malley vs Merab Dvalishvili
- Charles Oliveira vs Beneil Dariush

Frozen unless explicitly varied below:
- locked FSR V1.1 ratings and population centering;
- V1.2 phase-conditioned activity conversion;
- V1.3 KO/TKO hazards;
- V1.4 RFS style transitions;
- V1.4 x1.75 takedown initiation scale and chain wrestling;
- cardio, dynamic state, judging, and final-result logic.

Historical target-fight submission attempts are descriptive references only;
they are not exact matchup calibration targets.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import FighterSide

from scripts.experimental import run_fsr_historical_fight_v1 as base
from scripts.experimental import run_fsr_historical_fight_locked_v1_2 as v1_2
from scripts.experimental import run_fsr_historical_fight_locked_v1_3 as v1_3
from scripts.experimental import run_fsr_v1_3_archetype_validation as validation
from scripts.experimental import run_fsr_v1_4_transition_style_grid as style
from scripts.experimental import run_fsr_v1_4_two_fight_full_validation as v1_4


DEFAULT_SIMULATIONS = 250
DEFAULT_SEED = 2026080950
RFS_HISTORY_PATH = Path("data/features/round_fighter_state_history.parquet")
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_v1_4_submission_audit.csv"
)

TARGET_FIGHTS = (
    (
        "wrestler vs striker",
        "3146e5a47a922976",
    ),
    (
        "submission / grappling",
        "40e8bf8ce508c436",
    ),
)


@dataclass(frozen=True)
class Candidate:
    """One isolated submission-generation/conversion candidate."""

    name: str
    attempt_rate_scale: float
    submission_hazard: float

    def __post_init__(self) -> None:
        if self.attempt_rate_scale <= 0.0:
            raise ValueError("attempt_rate_scale must be positive")
        if not 0.0 < self.submission_hazard < 1.0:
            raise ValueError("submission_hazard must be between zero and one")


# Start with hazard-only changes at the current attempt rate, then reduce
# attempt generation separately.  This lets us identify which layer is doing
# the damage rather than tuning both simultaneously from the beginning.
CANDIDATES = (
    Candidate("attempt1_00_hazard0_36_current", 1.00, 0.36),
    Candidate("attempt1_00_hazard0_24", 1.00, 0.24),
    Candidate("attempt1_00_hazard0_18", 1.00, 0.18),
    Candidate("attempt1_00_hazard0_12", 1.00, 0.12),
    Candidate("attempt0_75_hazard0_24", 0.75, 0.24),
    Candidate("attempt0_75_hazard0_18", 0.75, 0.18),
    Candidate("attempt0_50_hazard0_24", 0.50, 0.24),
    Candidate("attempt0_50_hazard0_18", 0.50, 0.18),
)


_CURRENT_CANDIDATE: Candidate | None = None
_SUB_AUDIT: dict[str, float] = defaultdict(float)

# Capture the real path runner before installing our diagnostic wrapper.
_ORIGINAL_PATH_RUNNER = base.run_finish_enabled_dynamic_path


def current_candidate() -> Candidate:
    """Return the candidate selected by the outer audit loop."""

    if _CURRENT_CANDIDATE is None:
        raise RuntimeError("submission audit candidate is not selected")
    return _CURRENT_CANDIDATE


def build_inputs_candidate(
    red_card: dict[str, float],
    blue_card: dict[str, float],
    baselines: dict[str, float],
):
    """Build V1.4 inputs while scaling only submission-attempt generation."""

    candidate = current_candidate()

    red_phase = base.build_phase(red_card, blue_card, baselines)
    blue_phase = base.build_phase(blue_card, red_card, baselines)

    red_phase = replace(
        red_phase,
        ground_owner=replace(
            red_phase.ground_owner,
            submission_attempt_rate=(
                red_phase.ground_owner.submission_attempt_rate
                * candidate.attempt_rate_scale
            ),
        ),
    )
    blue_phase = replace(
        blue_phase,
        ground_owner=replace(
            blue_phase.ground_owner,
            submission_attempt_rate=(
                blue_phase.ground_owner.submission_attempt_rate
                * candidate.attempt_rate_scale
            ),
        ),
    )

    return (
        style.build_style_transition(red_card),
        style.build_style_transition(blue_card),
        red_phase,
        blue_phase,
        base.build_dynamic(red_card),
        base.build_dynamic(blue_card),
    )


def candidate_finish_calibration(base_candidate):
    """Preserve V1.3 KO calibration and vary only submission conversion."""

    candidate = current_candidate()
    calibration = v1_3.finish_calibration(base_candidate)

    return replace(
        calibration,
        submission=replace(
            calibration.submission,
            base_probability_per_attempt=candidate.submission_hazard,
        ),
    )


def reset_submission_audit() -> None:
    """Reset per-population counters before one candidate/fight run."""

    _SUB_AUDIT.clear()


def run_audited_v1_4_path(*args, **kwargs):
    """Inject V1.4 transitions and record submission exposure/conversion."""

    kwargs["shared_path_calibration"] = v1_4.V1_4_CALIBRATION
    path = _ORIGINAL_PATH_RUNNER(*args, **kwargs)

    reached_rounds = {
        int(segment.state.round_number)
        for segment in path.segments
    }
    _SUB_AUDIT["reached_rounds"] += float(len(reached_rounds))

    for segment in path.segments:
        if segment.state.phase is not FightPhase.GROUND:
            continue

        owner = segment.state.phase_owner
        if owner is FighterSide.RED:
            owner_side = "red"
        elif owner is FighterSide.BLUE:
            owner_side = "blue"
        else:
            raise RuntimeError("ground segment has no authoritative owner")

        _SUB_AUDIT[f"{owner_side}_ground_owner_segments"] += 1.0

        red_attempts = float(
            getattr(segment.activity.red, "submission_attempts", 0)
        )
        blue_attempts = float(
            getattr(segment.activity.blue, "submission_attempts", 0)
        )
        _SUB_AUDIT["red_submission_attempts"] += red_attempts
        _SUB_AUDIT["blue_submission_attempts"] += blue_attempts

    result = base.resolve_final_fight_result(path)
    if result.finish is not None:
        method = result.finish.method.value.lower()
        if "submission" in method:
            if result.winner is FighterSide.RED:
                _SUB_AUDIT["red_submission_finishes"] += 1.0
            elif result.winner is FighterSide.BLUE:
                _SUB_AUDIT["blue_submission_finishes"] += 1.0
            else:
                raise RuntimeError("submission finish has no winner")

    return path


def submission_population_metrics(simulations: int) -> dict[str, float]:
    """Return submission exposure and conversion diagnostics."""

    reached_rounds = _SUB_AUDIT["reached_rounds"]
    if reached_rounds <= 0.0:
        raise RuntimeError("no simulated rounds were reached")

    result: dict[str, float] = {}

    for side in ("red", "blue"):
        owner_segments = _SUB_AUDIT[f"{side}_ground_owner_segments"]
        attempts = _SUB_AUDIT[f"{side}_submission_attempts"]
        finishes = _SUB_AUDIT[f"{side}_submission_finishes"]

        result[f"{side}_ground_owner_segments_per_reached_round"] = (
            owner_segments / reached_rounds
        )
        result[f"{side}_submission_attempts_per_reached_round"] = (
            attempts / reached_rounds
        )
        result[f"{side}_submission_attempts_per_fight"] = (
            attempts / float(simulations)
        )
        result[f"{side}_submission_attempts_per_ground_owner_segment"] = (
            attempts / owner_segments if owner_segments > 0.0 else 0.0
        )
        result[f"{side}_submission_finish_pct"] = (
            100.0 * finishes / float(simulations)
        )
        result[f"{side}_submission_finishes_per_generated_attempt"] = (
            finishes / attempts if attempts > 0.0 else 0.0
        )

    return result


def actual_submission_metrics(
    fight_rows: pd.DataFrame,
    fighter_id: str,
) -> dict[str, float]:
    """Return realized target-fight submission-attempt counts for reference."""

    rows = fight_rows.loc[
        fight_rows["fighter_id"].astype(str) == str(fighter_id)
    ]
    attempts = float(
        pd.to_numeric(rows["sub_att"], errors="coerce")
        .fillna(0.0)
        .sum()
    )
    observed_rounds = float(rows["round"].nunique())

    return {
        "attempts": attempts,
        "attempts_per_round": (
            attempts / observed_rounds if observed_rounds > 0.0 else 0.0
        ),
    }


def main() -> None:
    global _CURRENT_CANDIDATE

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

    # Install the already-validated V1.1/V1.2/V1.3 shadow adapters first.
    v1_3.install_overrides()

    # The V1.2 whole-round -> active-segment activity conversion must use the
    # current V1.4 transition environment, not the obsolete V1.2 phase mix.
    v1_2.neutral_phase_exposure = v1_4.neutral_phase_exposure_v1_4

    # Override only the diagnostic seams used by run_compact_population().
    validation.build_inputs = build_inputs_candidate
    base.finish_calibration = candidate_finish_calibration
    base.run_finish_enabled_dynamic_path = run_audited_v1_4_path

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

    prepared: list[dict[str, object]] = []

    print()
    print("=" * 170)
    print("FSR / MC V1.4 SUBMISSION AUDIT")
    print("=" * 170)
    print(
        f"Simulations per candidate/fight: {args.simulations} | "
        f"seed start: {args.seed}"
    )
    print(
        "Frozen: V1.1 FSR, V1.4 style/chain wrestling x1.75, "
        "V1.3 KO hazards, cardio, judging."
    )

    for fight_index, (archetype, fight_id) in enumerate(TARGET_FIGHTS):
        (
            all_rounds,
            target_date,
            red_info,
            blue_info,
            scheduled_rounds,
        ) = base.load_target_fight(fight_id)

        # load_target_fight() returns the full historical dataset; restrict the
        # realized-fight reference frame explicitly to avoid career totals.
        fight_rows = all_rounds.loc[
            all_rounds["fight_id"].astype(str) == str(fight_id)
        ].copy()

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

        prepared.append(
            {
                "fight_index": fight_index,
                "archetype": archetype,
                "fight_id": fight_id,
                "red_info": red_info,
                "blue_info": blue_info,
                "scheduled_rounds": scheduled_rounds,
                "red_prior_fights": red_prior_fights,
                "blue_prior_fights": blue_prior_fights,
                "red_card": red_card,
                "blue_card": blue_card,
                "baselines": baselines,
                "red_actual": actual_submission_metrics(
                    fight_rows,
                    red_info["fighter_id"],
                ),
                "blue_actual": actual_submission_metrics(
                    fight_rows,
                    blue_info["fighter_id"],
                ),
            }
        )

    rows: list[dict[str, object]] = []

    for item in prepared:
        fight_index = int(item["fight_index"])
        archetype = str(item["archetype"])
        fight_id = str(item["fight_id"])
        red_info = item["red_info"]
        blue_info = item["blue_info"]
        red_actual = item["red_actual"]
        blue_actual = item["blue_actual"]

        print()
        print("=" * 170)
        print(
            f"{archetype.upper()}: "
            f"{red_info['fighter_name']} vs {blue_info['fighter_name']} | "
            f"{fight_id}"
        )
        print("=" * 170)
        print(
            "Actual target-fight SUB attempts: "
            f"{red_info['fighter_name']}={red_actual['attempts']:.0f} "
            f"({red_actual['attempts_per_round']:.2f}/round), "
            f"{blue_info['fighter_name']}={blue_actual['attempts']:.0f} "
            f"({blue_actual['attempts_per_round']:.2f}/round)"
        )

        for candidate in CANDIDATES:
            _CURRENT_CANDIDATE = candidate
            reset_submission_audit()

            metrics = validation.run_compact_population(
                red_card=item["red_card"],
                blue_card=item["blue_card"],
                baselines=item["baselines"],
                scheduled_rounds=int(item["scheduled_rounds"]),
                simulations=args.simulations,
                seed_start=args.seed + fight_index * 10000,
            )
            sub = submission_population_metrics(args.simulations)

            row: dict[str, object] = {
                "candidate": candidate.name,
                "attempt_rate_scale": candidate.attempt_rate_scale,
                "submission_hazard": candidate.submission_hazard,
                "archetype": archetype,
                "fight_id": fight_id,
                "red_name": red_info["fighter_name"],
                "blue_name": blue_info["fighter_name"],
                "red_actual_submission_attempts": red_actual["attempts"],
                "blue_actual_submission_attempts": blue_actual["attempts"],
                "red_prior_fights": item["red_prior_fights"],
                "blue_prior_fights": item["blue_prior_fights"],
                **metrics,
                **sub,
            }
            rows.append(row)

            print(
                f"{candidate.name:30s} | "
                f"SUB finish={metrics['submission_pct']:5.1f}% | "
                f"distance={metrics['scheduled_distance_pct']:5.1f}% | "
                f"ground={metrics['ground_phase_pct']:5.1f}% | "
                f"RED sub att/fight={sub['red_submission_attempts_per_fight']:.2f} "
                f"finish={sub['red_submission_finish_pct']:.1f}% | "
                f"BLUE sub att/fight={sub['blue_submission_attempts_per_fight']:.2f} "
                f"finish={sub['blue_submission_finish_pct']:.1f}%"
            )

    _CURRENT_CANDIDATE = None

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("submission audit produced no rows")

    print()
    print("=" * 170)
    print("CROSS-FIGHT CANDIDATE SUMMARY")
    print("=" * 170)

    summary = (
        result.groupby(
            ["candidate", "attempt_rate_scale", "submission_hazard"],
            sort=False,
        )
        .agg(
            mean_submission_finish_pct=("submission_pct", "mean"),
            mean_distance_pct=("scheduled_distance_pct", "mean"),
            mean_ground_phase_pct=("ground_phase_pct", "mean"),
            mean_red_sub_attempts=("red_submission_attempts_per_fight", "mean"),
            mean_blue_sub_attempts=("blue_submission_attempts_per_fight", "mean"),
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
