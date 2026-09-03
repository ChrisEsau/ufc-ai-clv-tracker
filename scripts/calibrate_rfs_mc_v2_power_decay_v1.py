"""V0 empirical finish calibration search for RFS Monte Carlo V2.

Purpose
-------
Fit the static finish layer before introducing dynamic-state amplification.

This stage calibrates only:

- base landed-strike KO/TKO hazard
- incremental knockdown KO/TKO hazard
- base submission probability per attempt

Dynamic fatigue/damage/stress amplifiers remain zero.

This is deliberately a coarse search. The goal is to locate the correct
finish-hazard scale, not declare production calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.audit_rfs_mc_v2_finish_paths import (
    zero_phase_effect_calibration,
    zero_state_calibration,
    zero_transition_effect_calibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_calibration import (
    ActivityWorkloadCalibration,
    AdversityCalibration,
    DynamicStateCalibration,
    PhaseWorkloadCalibration,
    RecoveryCalibration,
    ResistanceScalingCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_effect_calibration import (
    DynamicEffectCalibration,
    StatePenaltyWeights,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_calibration import (
    FinishProbabilityCalibration,
    KnockoutFinishCalibration,
    SubmissionFinishCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.historical_matchup_loader import (
    HistoricalMatchupLoadError,
    load_historical_matchup,
)
from pipeline.simulation.rfs_mc_v2_shared_state.monte_carlo_runner import (
    run_matchup_monte_carlo,
)
from pipeline.simulation.rfs_mc_v2_shared_state.rfs_parameter_resolver import (
    resolve_fighter_parameters,
)


HISTORY_PATH = (
    REPO_ROOT
    / "data/simulation/rfs_mc_v2_shared_state/"
    / "historical_fighter_state.parquet"
)

MASTER_PATH = (
    REPO_ROOT
    / "data/master/ufc_master.parquet"
)


# Primary cohort contract.
MIN_PRIOR_FIGHTS = 3

# Keep this first search modest.
SAMPLE_FIGHTS = 300
SIMULATIONS_PER_FIGHT = 20

# ---------------------------------------------------------------------------
# Frozen RFS MC V2 V1 shadow calibration
#
# Validated on 300 historical UFC matchups x 100 Monte Carlo paths.
# These are simulator-wide calibration values. Fighter-specific parameters
# continue to come from leakage-safe pre-fight RFS profiles.
# ---------------------------------------------------------------------------

V1_LANDED_KO_HAZARD = 0.013
V1_KNOCKDOWN_BONUS_HAZARD = 0.080
V1_SUBMISSION_HAZARD = 0.65

V1_FATIGUE_PER_SEGMENT = 0.05
V1_POWER_FATIGUE_PENALTY = 0.80

# Historical calibration objectives from the completed audit.
TARGET_KO_RATE = 0.3390
TARGET_SUB_RATE = 0.1666
TARGET_DECISION_RATE = 0.4944

TARGET_FINISH_ROUND_SHARE = {
    1: 0.4626,
    2: 0.3267,
    3: 0.1680,
    4: 0.0235,
    5: 0.0192,
}


@dataclass(frozen=True)
class Candidate:
    """One static KO calibration candidate with fixed power decay."""

    landed_ko_hazard: float
    knockdown_bonus_hazard: float


@dataclass(frozen=True)
class CandidateResult:
    """Aggregated evaluation result for one candidate."""

    candidate: Candidate

    simulated_fights: int

    ko_count: int
    submission_count: int
    decision_count: int

    finish_round_counts: tuple[int, int, int, int, int]

    method_error: float
    round_error: float
    total_error: float


def candidate_grid() -> tuple[Candidate, ...]:
    """Return KO hazard candidates for corrected-bridge recalibration."""

    landed_hazards = (
        0.0025,
        0.0040,
        0.0060,
        0.0090,
        0.0130,
        0.0180,
    )

    return tuple(
        Candidate(
            landed_ko_hazard=hazard,
            knockdown_bonus_hazard=V1_KNOCKDOWN_BONUS_HAZARD,
        )
        for hazard in landed_hazards
    )



def finish_calibration(
    candidate: Candidate,
) -> FinishProbabilityCalibration:
    """Return the fixed V0 method-balanced finish baseline."""

    landed_ko_hazard = candidate.landed_ko_hazard
    knockdown_bonus_hazard = candidate.knockdown_bonus_hazard
    submission_hazard = V1_SUBMISSION_HAZARD

    return FinishProbabilityCalibration(
        knockout=KnockoutFinishCalibration(
            distance_landed_probability=landed_ko_hazard,
            distance_knockdown_probability=knockdown_bonus_hazard,
            clinch_landed_probability=(
                landed_ko_hazard * 1.15
            ),
            damaging_clinch_probability=(
                knockdown_bonus_hazard * 0.40
            ),
            ground_landed_probability=(
                landed_ko_hazard * 1.35
            ),

            # V1 still does not amplify finish hazard from defender state.
            defender_fatigue_amplifier=0.0,
            defender_damage_amplifier=0.0,
            defender_acute_stress_amplifier=0.0,

            maximum_segment_probability=0.50,
        ),
        submission=SubmissionFinishCalibration(
            base_probability_per_attempt=submission_hazard,
            position_quality_amplifier=0.35,
            minimum_submission_defense_effect_multiplier=0.10,

            defender_fatigue_amplifier=0.0,
            defender_damage_amplifier=0.0,
            defender_acute_stress_amplifier=0.0,

            maximum_probability_per_attempt=0.75,
            maximum_segment_probability=0.75,
        ),
    )


def zero_weights() -> StatePenaltyWeights:
    """Return a capability family with no dynamic penalty."""

    return StatePenaltyWeights(
        fatigue=0.0,
        damage=0.0,
        acute_stress=0.0,
    )


def state_calibration(
    candidate: Candidate,
) -> DynamicStateCalibration:
    """Accumulate fatigue while leaving damage and stress disabled."""

    fatigue = V1_FATIGUE_PER_SEGMENT

    return DynamicStateCalibration(
        phase_workload=PhaseWorkloadCalibration(
            distance=fatigue,
            clinch_owner=fatigue,
            clinch_defender=fatigue,
            ground_owner=fatigue,
            ground_defender=fatigue,
        ),

        # Do not make fatigue depend on pace yet.
        activity_workload=ActivityWorkloadCalibration(
            strike_attempt=0.0,
            control_second=0.0,
            submission_attempt=0.0,
            position_advancement=0.0,
            escape_attempt=0.0,
            reversal_attempt=0.0,
            scramble_attempt=0.0,
        ),

        # V1 isolates fatigue from adversity.
        adversity=AdversityCalibration(
            distance_landed_damage=0.0,
            clinch_landed_damage=0.0,
            damaging_clinch_bonus_damage=0.0,
            ground_landed_damage=0.0,
            knockdown_damage=0.0,

            distance_landed_stress=0.0,
            clinch_landed_stress=0.0,
            damaging_clinch_bonus_stress=0.0,
            ground_landed_stress=0.0,
            knockdown_stress=0.0,

            control_second_received_stress=0.0,
            submission_attempt_received_stress=0.0,
            position_advancement_received_stress=0.0,
        ),

        resistance_scaling=ResistanceScalingCalibration(
            minimum_fatigue_accumulation_multiplier=0.25,
            minimum_damage_accumulation_multiplier=0.20,
            minimum_acute_stress_accumulation_multiplier=0.15,
        ),

        # Small between-round recovery prevents fatigue from becoming
        # a simple round-number lookup and preserves fighter recovery traits.
        recovery=RecoveryCalibration(
            low_workload_threshold=0.0,
            segment_fatigue_recovery=0.0,
            round_break_fatigue_recovery=0.025,
            segment_acute_stress_recovery=0.0,
            round_break_acute_stress_recovery=0.0,
        ),
    )


def phase_effect_calibration(
    candidate: Candidate,
) -> DynamicEffectCalibration:
    """Apply accumulated fatigue only to finishing power."""

    return DynamicEffectCalibration(
        minimum_fatigue_effect_multiplier=0.20,
        minimum_effective_capability_multiplier=0.10,

        output=zero_weights(),
        accuracy=zero_weights(),

        power=StatePenaltyWeights(
            fatigue=V1_POWER_FATIGUE_PENALTY,
            damage=0.0,
            acute_stress=0.0,
        ),

        control=zero_weights(),
        grappling=zero_weights(),
        defense=zero_weights(),
    )



def eligible_fight_ids(
    history: pd.DataFrame,
) -> list[str]:
    """Return fights where both fighters satisfy the primary cohort."""

    work = history.copy()

    work["_prior"] = pd.to_numeric(
        work["rfs_traj_prior_fight_count"],
        errors="coerce",
    )

    eligible = work.loc[
        work["_prior"] >= MIN_PRIOR_FIGHTS
    ]

    eligible = (
        eligible.groupby("fight_id")
        .filter(
            lambda group: (
                len(group) == 2
                and group["fighter_id"].nunique() == 2
            )
        )
    )

    fights = (
        eligible[
            [
                "fight_id",
                "date",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "date",
                "fight_id",
            ]
        )
        .reset_index(drop=True)
    )

    if len(fights) <= SAMPLE_FIGHTS:
        return (
            fights["fight_id"]
            .astype(str)
            .tolist()
        )

    # Deterministic evenly-spaced sample across history rather than
    # selecting only recent or early fights.
    indices = [
        round(
            index
            * (len(fights) - 1)
            / (SAMPLE_FIGHTS - 1)
        )
        for index in range(SAMPLE_FIGHTS)
    ]

    return (
        fights.iloc[indices][
            "fight_id"
        ]
        .astype(str)
        .tolist()
    )


def build_matchup_inputs(
    *,
    history: pd.DataFrame,
    master: pd.DataFrame,
    fight_id: str,
):
    """Build leakage-safe fighter parameters for one historical fight."""

    matchup = load_historical_matchup(
        history,
        master,
        fight_id,
        min_prior_fights=MIN_PRIOR_FIGHTS,
    )

    population = history.loc[
        pd.to_datetime(
            history["date"]
        )
        < matchup.date
    ].copy()

    population = population.loc[
        pd.to_numeric(
            population[
                "rfs_traj_prior_fight_count"
            ],
            errors="coerce",
        ) > 0
    ]

    if population.empty:
        raise RuntimeError(
            f"{fight_id}: leakage-safe population is empty"
        )

    red = resolve_fighter_parameters(
        profile=matchup.red.features,
        prior_fight_count=(
            matchup.red.prior_fight_count
        ),
        population_history=population,
    )

    blue = resolve_fighter_parameters(
        profile=matchup.blue.features,
        prior_fight_count=(
            matchup.blue.prior_fight_count
        ),
        population_history=population,
    )

    return (
        matchup,
        red,
        blue,
    )


def method_error(
    *,
    ko_rate: float,
    sub_rate: float,
    decision_rate: float,
) -> float:
    """Squared error against historical method proportions."""

    return (
        (ko_rate - TARGET_KO_RATE) ** 2
        + (sub_rate - TARGET_SUB_RATE) ** 2
        + (
            decision_rate
            - TARGET_DECISION_RATE
        ) ** 2
    )


def finish_round_error(
    finish_round_counts: tuple[
        int,
        int,
        int,
        int,
        int,
    ],
) -> float:
    """Squared error for finish-round conditional distribution."""

    total_finishes = sum(
        finish_round_counts
    )

    if total_finishes == 0:
        return 1.0

    error = 0.0

    for round_number in range(
        1,
        6,
    ):
        observed = (
            finish_round_counts[
                round_number - 1
            ]
            / total_finishes
        )

        target = TARGET_FINISH_ROUND_SHARE[
            round_number
        ]

        error += (
            observed
            - target
        ) ** 2

    return error


def evaluate_candidate(
    *,
    candidate: Candidate,
    prepared_matchups: list[tuple],
) -> CandidateResult:
    """Evaluate one candidate on the prepared historical sample."""

    calibration = finish_calibration(candidate)

    dynamic_state_calibration = (
        state_calibration(candidate)
    )

    dynamic_phase_effect_calibration = (
        phase_effect_calibration(candidate)
    )

    ko_count = 0
    submission_count = 0
    decision_count = 0

    finish_round_counts = [
        0,
        0,
        0,
        0,
        0,
    ]

    total_simulations = 0

    for matchup_index, (
        matchup,
        red,
        blue,
    ) in enumerate(prepared_matchups):

        # Scheduled distance is authoritative fight metadata.
        # Do not infer 3 vs 5 rounds when it is unavailable.
        if matchup.scheduled_rounds not in {3, 5}:
            continue

        seed_start = (
            1_000_000
            + matchup_index
            * 10_000
        )

        summary = run_matchup_monte_carlo(
            red.transition,
            blue.transition,
            red.phase,
            blue.phase,
            red.dynamic,
            blue.dynamic,
            dynamic_state_calibration=(
            dynamic_state_calibration
        ),
        phase_effect_calibration=(
            dynamic_phase_effect_calibration
        ),
            transition_effect_calibration=(
                zero_transition_effect_calibration()
            ),
            finish_probability_calibration=(
                calibration
            ),
            simulation_count=(
                SIMULATIONS_PER_FIGHT
            ),
            seed_start=seed_start,
            scheduled_rounds=int(
                matchup.scheduled_rounds
            ),
        )

        ko_count += summary.ko_tko_count

        submission_count += (
            summary.submission_count
        )

        decision_count += (
            summary.scheduled_distance_count
        )

        for round_index, count in enumerate(
            summary.finish_round_counts
        ):
            finish_round_counts[
                round_index
            ] += count

        total_simulations += (
            summary.simulation_count
        )

    ko_rate = (
        ko_count
        / total_simulations
    )

    sub_rate = (
        submission_count
        / total_simulations
    )

    decision_rate = (
        decision_count
        / total_simulations
    )

    selected_method_error = method_error(
        ko_rate=ko_rate,
        sub_rate=sub_rate,
        decision_rate=decision_rate,
    )

    selected_round_error = (
        finish_round_error(
            tuple(
                finish_round_counts
            )
        )
    )

    # Method mix is the main V0 objective.
    # Finish timing gets a smaller secondary weight.
    total_error = (
        selected_method_error
        + 0.25
        * selected_round_error
    )

    return CandidateResult(
        candidate=candidate,
        simulated_fights=total_simulations,
        ko_count=ko_count,
        submission_count=submission_count,
        decision_count=decision_count,
        finish_round_counts=tuple(
            finish_round_counts
        ),
        method_error=selected_method_error,
        round_error=selected_round_error,
        total_error=total_error,
    )


def main() -> None:
    history = pd.read_parquet(
        HISTORY_PATH
    )

    master = pd.read_parquet(
        MASTER_PATH
    )

    history["fight_id"] = (
        history["fight_id"].astype(str)
    )

    master["fight_id"] = (
        master["fight_id"].astype(str)
    )

    fight_ids = eligible_fight_ids(
        history
    )

    print("=" * 78)
    print("RFS MONTE CARLO V2 — FINISH CALIBRATION V1")
    print("=" * 78)

    print(
        "Historical fights sampled:",
        len(fight_ids),
    )

    print(
        "Simulations per matchup:",
        SIMULATIONS_PER_FIGHT,
    )

    print(
        "Candidates:",
        len(candidate_grid()),
    )

    print()
    print(
        "Preparing leakage-safe fighter parameters..."
    )

    prepared_matchups = []

    for index, fight_id in enumerate(
        fight_ids,
        start=1,
    ):
        try:
            prepared_matchups.append(
                build_matchup_inputs(
                    history=history,
                    master=master,
                    fight_id=fight_id,
                )
            )
        except HistoricalMatchupLoadError as exc:
            print(
                "SKIP",
                fight_id,
                exc,
            )

        if (
            index % 10 == 0
            or index == len(fight_ids)
        ):
            print(
                f"  prepared {index}/{len(fight_ids)}"
            )

    if not prepared_matchups:
        raise RuntimeError(
            "No historical matchups prepared."
        )

    print()
    print(
        "Prepared matchups:",
        len(prepared_matchups),
    )

    print()
    print(
        "Searching finish calibration..."
    )

    results = []

    for index, candidate in enumerate(
        candidate_grid(),
        start=1,
    ):
        result = evaluate_candidate(
            candidate=candidate,
            prepared_matchups=(
                prepared_matchups
            ),
        )

        results.append(
            result
        )

        total = (
            result.simulated_fights
        )

        print(
            f"{index:02d}/"
            f"{len(candidate_grid()):02d} "
            f"landed={candidate.landed_ko_hazard:.4f} "
            f"kd={candidate.knockdown_bonus_hazard:.3f} | "
            f"KO={result.ko_count / total:6.2%} "
            f"SUB={result.submission_count / total:6.2%} "
            f"DEC={result.decision_count / total:6.2%} "
            f"score={result.total_error:.6f}"
        )

    results.sort(
        key=lambda item: item.total_error
    )

    print()
    print("=" * 78)
    print("TOP FINISH CALIBRATION CANDIDATES")
    print("=" * 78)

    for rank, result in enumerate(
        results[:5],
        start=1,
    ):
        candidate = result.candidate
        total = result.simulated_fights
        finishes = (
            result.ko_count
            + result.submission_count
        )

        print()
        print(
            f"#{rank} "
            f"score={result.total_error:.6f}"
        )

        print(
            "  landed KO hazard :",
            candidate.landed_ko_hazard,
        )

        print(
            "  knockdown hazard :",
            candidate.knockdown_bonus_hazard,
        )

        print(
            "  fixed fatigue    : 0.05"
        )

        print(
            "  fixed power      : 0.80"
        )

        print(
            "  KO/TKO :",
            f"{result.ko_count / total:.2%}",
            "(target 33.90%)",
        )

        print(
            "  SUB    :",
            f"{result.submission_count / total:.2%}",
            "(target 16.66%)",
        )

        print(
            "  DEC    :",
            f"{result.decision_count / total:.2%}",
            "(target 49.44%)",
        )

        if finishes:
            print(
                "  Finish rounds:",
                ", ".join(
                    (
                        f"R{round_number}="
                        f"{count / finishes:.1%}"
                    )
                    for round_number, count
                    in enumerate(
                        result.finish_round_counts,
                        start=1,
                    )
                ),
            )

    print()
    print("=" * 78)
    print("FINISH CALIBRATION V1 VALIDATION COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
