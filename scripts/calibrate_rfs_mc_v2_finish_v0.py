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
SAMPLE_FIGHTS = 60
SIMULATIONS_PER_FIGHT = 30

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
    """One static finish calibration candidate."""

    landed_ko_hazard: float
    knockdown_bonus_hazard: float
    submission_hazard: float


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
    """Return a deliberately coarse first-stage search grid."""

    landed_hazards = (
        0.0010,
        0.00115,
        0.00130,
    )

    knockdown_hazards = (
        0.060,
        0.0675,
        0.075,
    )

    submission_hazards = (
        0.60,
        0.65,
        0.70,
    )

    return tuple(
        Candidate(
            landed_ko_hazard=landed,
            knockdown_bonus_hazard=knockdown,
            submission_hazard=submission,
        )
        for landed in landed_hazards
        for knockdown in knockdown_hazards
        for submission in submission_hazards
    )


def finish_calibration(
    candidate: Candidate,
) -> FinishProbabilityCalibration:
    """Convert one search candidate into the engine calibration contract."""

    return FinishProbabilityCalibration(
        knockout=KnockoutFinishCalibration(
            # Use one base landed hazard with modest phase-specific scaling.
            distance_landed_probability=(
                candidate.landed_ko_hazard
            ),
            distance_knockdown_probability=(
                candidate.knockdown_bonus_hazard
            ),
            clinch_landed_probability=(
                candidate.landed_ko_hazard
                * 1.15
            ),
            damaging_clinch_probability=(
                min(
                    1.0,
                    candidate.knockdown_bonus_hazard
                    * 0.40,
                )
            ),
            ground_landed_probability=(
                candidate.landed_ko_hazard
                * 1.35
            ),

            # Dynamic amplification is intentionally OFF in V0.
            defender_fatigue_amplifier=0.0,
            defender_damage_amplifier=0.0,
            defender_acute_stress_amplifier=0.0,

            maximum_segment_probability=0.50,
        ),
        submission=SubmissionFinishCalibration(
            base_probability_per_attempt=(
                candidate.submission_hazard
            ),
            position_quality_amplifier=0.35,
            minimum_submission_defense_effect_multiplier=0.10,

            # Dynamic amplification is intentionally OFF in V0.
            defender_fatigue_amplifier=0.0,
            defender_damage_amplifier=0.0,
            defender_acute_stress_amplifier=0.0,

            maximum_probability_per_attempt=0.75,
            maximum_segment_probability=0.75,
        ),
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

    calibration = finish_calibration(
        candidate
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
                zero_state_calibration()
            ),
            phase_effect_calibration=(
                zero_phase_effect_calibration()
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
    print("RFS MONTE CARLO V2 — FINISH CALIBRATION V0")
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
            f"kd={candidate.knockdown_bonus_hazard:.3f} "
            f"sub={candidate.submission_hazard:.3f} | "
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
            "  submission hazard:",
            candidate.submission_hazard,
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
    print("FINISH CALIBRATION V0 SEARCH COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
