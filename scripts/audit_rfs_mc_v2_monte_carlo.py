"""Population audit for the RFS Monte Carlo V2 matchup runner.

The audit validates:

- authoritative count arithmetic
- probability-family arithmetic
- deterministic population replay
- exact equivalence between full and partitioned seed ranges
- symmetric decision corner balance
- symmetric finish corner balance
- KO/TKO method aggregation
- unanimous draws under zero scoring and zero judge variation
- three-round and five-round population handling
- Wilson interval integrity
- reduced Monte Carlo uncertainty at larger population sizes
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from pipeline.simulation.rfs_mc_v2_shared_state.judge_scorecard_generator import (
    JudgeVariabilityCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.monte_carlo_runner import (
    run_matchup_monte_carlo,
)
from pipeline.simulation.rfs_mc_v2_shared_state.monte_carlo_summary import (
    MatchupMonteCarloSummary,
    ProbabilityEstimate,
)
from pipeline.simulation.rfs_mc_v2_shared_state.round_scoring_engine import (
    RoundScoringCalibration,
)
from scripts.audit_rfs_mc_v2_finish_paths import (
    distance_only_transition_parameters,
    dynamic_parameters,
    knockout_finish_calibration,
    phase_parameters,
    zero_finish_calibration,
    zero_phase_effect_calibration,
    zero_state_calibration,
    zero_transition_effect_calibration,
)


@dataclass(frozen=True)
class AuditCheck:
    """One population-audit result."""

    name: str
    passed: bool
    details: str


def zero_variability() -> JudgeVariabilityCalibration:
    """Return judge calibration with no stochastic adjustment."""

    return JudgeVariabilityCalibration(
        fight_bias_stddev=0.0,
        round_noise_stddev=0.0,
        maximum_absolute_adjustment=0.0,
    )


def zero_scoring() -> RoundScoringCalibration:
    """Return scoring calibration that makes all rounds even."""

    return RoundScoringCalibration(
        persistent_damage_weight=0.0,
        acute_stress_weight=0.0,
        knockdown_weight=0.0,
        damaging_clinch_weight=0.0,
        distance_landed_weight=0.0,
        clinch_landed_weight=0.0,
        ground_landed_weight=0.0,
        submission_attempt_weight=0.0,
        position_advancement_weight=0.0,
        reversal_weight=0.0,
        control_second_weight=0.0,
        escape_weight=0.0,
        scramble_weight=0.0,
        primary_close_threshold=0.05,
        secondary_scale=0.0,
        even_round_threshold=0.05,
        ten_eight_threshold=1.0,
        ten_seven_threshold=2.0,
    )


def run_population(
    *,
    simulation_count: int,
    seed_start: int,
    scheduled_rounds: int = 3,
    finish_calibration=None,
    scoring_calibration: RoundScoringCalibration | None = None,
) -> MatchupMonteCarloSummary:
    """Run one symmetric matchup population."""

    transition = distance_only_transition_parameters()
    phase = phase_parameters()
    dynamic = dynamic_parameters()

    selected_finish_calibration = (
        finish_calibration
        if finish_calibration is not None
        else zero_finish_calibration()
    )

    selected_scoring_calibration = (
        scoring_calibration
        if scoring_calibration is not None
        else RoundScoringCalibration()
    )

    return run_matchup_monte_carlo(
        transition,
        transition,
        phase,
        phase,
        dynamic,
        dynamic,
        dynamic_state_calibration=zero_state_calibration(),
        phase_effect_calibration=zero_phase_effect_calibration(),
        transition_effect_calibration=(
            zero_transition_effect_calibration()
        ),
        finish_probability_calibration=(
            selected_finish_calibration
        ),
        simulation_count=simulation_count,
        seed_start=seed_start,
        scheduled_rounds=scheduled_rounds,
        scoring_calibration=selected_scoring_calibration,
        variability_calibration=zero_variability(),
    )


def merge_contiguous_summaries(
    summaries: tuple[MatchupMonteCarloSummary, ...],
) -> MatchupMonteCarloSummary:
    """Merge summaries whose simulation seed ranges are contiguous."""

    if not summaries:
        raise ValueError(
            "at least one summary is required"
        )

    scheduled_rounds = summaries[0].scheduled_rounds
    expected_seed_start = summaries[0].seed_start

    for summary in summaries:
        if summary.scheduled_rounds != scheduled_rounds:
            raise ValueError(
                "all summaries must use the same scheduled rounds"
            )

        if summary.seed_start != expected_seed_start:
            raise ValueError(
                "summary seed ranges must be contiguous"
            )

        expected_seed_start += summary.simulation_count

    def total(field_name: str) -> int:
        """Sum one integer summary field."""

        return sum(
            getattr(summary, field_name)
            for summary in summaries
        )

    finish_round_counts = tuple(
        sum(
            summary.finish_round_counts[round_index]
            for summary in summaries
        )
        for round_index in range(scheduled_rounds)
    )

    return MatchupMonteCarloSummary(
        simulation_count=total("simulation_count"),
        seed_start=summaries[0].seed_start,
        scheduled_rounds=scheduled_rounds,
        red_win_count=total("red_win_count"),
        blue_win_count=total("blue_win_count"),
        draw_count=total("draw_count"),
        finish_count=total("finish_count"),
        scheduled_distance_count=total(
            "scheduled_distance_count"
        ),
        red_ko_tko_count=total("red_ko_tko_count"),
        blue_ko_tko_count=total("blue_ko_tko_count"),
        red_submission_count=total(
            "red_submission_count"
        ),
        blue_submission_count=total(
            "blue_submission_count"
        ),
        red_decision_count=total("red_decision_count"),
        blue_decision_count=total("blue_decision_count"),
        unanimous_decision_count=total(
            "unanimous_decision_count"
        ),
        split_decision_count=total(
            "split_decision_count"
        ),
        majority_decision_count=total(
            "majority_decision_count"
        ),
        unanimous_draw_count=total(
            "unanimous_draw_count"
        ),
        split_draw_count=total("split_draw_count"),
        majority_draw_count=total(
            "majority_draw_count"
        ),
        finish_round_counts=finish_round_counts,
        total_finish_elapsed_seconds_in_fight=total(
            "total_finish_elapsed_seconds_in_fight"
        ),
    )


def arithmetic_violations(
    summary: MatchupMonteCarloSummary,
) -> list[str]:
    """Return violations of authoritative population arithmetic."""

    violations: list[str] = []

    if (
        summary.red_win_count
        + summary.blue_win_count
        + summary.draw_count
        != summary.simulation_count
    ):
        violations.append(
            "winner counts do not total simulations"
        )

    if (
        summary.finish_count
        + summary.scheduled_distance_count
        != summary.simulation_count
    ):
        violations.append(
            "terminal branches do not total simulations"
        )

    if (
        summary.ko_tko_count
        + summary.submission_count
        != summary.finish_count
    ):
        violations.append(
            "finish methods do not total finishes"
        )

    if (
        summary.red_ko_tko_count
        + summary.red_submission_count
        + summary.red_decision_count
        != summary.red_win_count
    ):
        violations.append(
            "red methods do not total red wins"
        )

    if (
        summary.blue_ko_tko_count
        + summary.blue_submission_count
        + summary.blue_decision_count
        != summary.blue_win_count
    ):
        violations.append(
            "blue methods do not total blue wins"
        )

    winning_decision_types = (
        summary.unanimous_decision_count
        + summary.split_decision_count
        + summary.majority_decision_count
    )
    draw_decision_types = (
        summary.unanimous_draw_count
        + summary.split_draw_count
        + summary.majority_draw_count
    )

    if winning_decision_types != (
        summary.red_decision_count
        + summary.blue_decision_count
    ):
        violations.append(
            "winning decision types do not total decision wins"
        )

    if draw_decision_types != summary.draw_count:
        violations.append(
            "draw decision types do not total draws"
        )

    if (
        winning_decision_types
        + draw_decision_types
        != summary.scheduled_distance_count
    ):
        violations.append(
            "decision types do not total distance results"
        )

    if (
        sum(summary.finish_round_counts)
        != summary.finish_count
    ):
        violations.append(
            "finish rounds do not total finishes"
        )

    return violations


def probability_violations(
    summary: MatchupMonteCarloSummary,
) -> list[str]:
    """Return probability and confidence-interval violations."""

    violations: list[str] = []

    estimates: tuple[
        tuple[str, ProbabilityEstimate],
        ...,
    ] = (
        (
            "red win",
            summary.red_win_probability,
        ),
        (
            "blue win",
            summary.blue_win_probability,
        ),
        (
            "draw",
            summary.draw_probability,
        ),
        (
            "finish",
            summary.finish_probability,
        ),
        (
            "scheduled distance",
            summary.scheduled_distance_probability,
        ),
        (
            "KO/TKO",
            summary.ko_tko_probability,
        ),
        (
            "submission",
            summary.submission_probability,
        ),
        (
            "red KO/TKO",
            summary.red_ko_tko_probability,
        ),
        (
            "blue KO/TKO",
            summary.blue_ko_tko_probability,
        ),
        (
            "red submission",
            summary.red_submission_probability,
        ),
        (
            "blue submission",
            summary.blue_submission_probability,
        ),
        (
            "red decision",
            summary.red_decision_probability,
        ),
        (
            "blue decision",
            summary.blue_decision_probability,
        ),
    )

    for name, estimate in estimates:
        if not (
            0.0
            <= estimate.lower_bound
            <= estimate.probability
            <= estimate.upper_bound
            <= 1.0
        ):
            violations.append(
                f"{name} estimate has invalid Wilson bounds"
            )

    winner_probability_total = (
        summary.red_win_probability.probability
        + summary.blue_win_probability.probability
        + summary.draw_probability.probability
    )

    if not math.isclose(
        winner_probability_total,
        1.0,
        abs_tol=1e-12,
    ):
        violations.append(
            "winner probabilities do not total one"
        )

    branch_probability_total = (
        summary.finish_probability.probability
        + summary.scheduled_distance_probability.probability
    )

    if not math.isclose(
        branch_probability_total,
        1.0,
        abs_tol=1e-12,
    ):
        violations.append(
            "terminal branch probabilities do not total one"
        )

    finish_method_probability_total = (
        summary.ko_tko_probability.probability
        + summary.submission_probability.probability
    )

    if not math.isclose(
        finish_method_probability_total,
        summary.finish_probability.probability,
        abs_tol=1e-12,
    ):
        violations.append(
            "finish-method probabilities do not total finish probability"
        )

    for round_number in range(
        1,
        summary.scheduled_rounds + 1,
    ):
        estimate = summary.finish_in_round_probability(
            round_number
        )

        if not (
            0.0
            <= estimate.lower_bound
            <= estimate.probability
            <= estimate.upper_bound
            <= 1.0
        ):
            violations.append(
                f"round {round_number} finish estimate is invalid"
            )

    return violations


def format_percentage(value: float) -> str:
    """Format one probability as a percentage."""

    return f"{value:.2%}"


def population_description(
    summary: MatchupMonteCarloSummary,
) -> str:
    """Return a compact summary for audit output."""

    return (
        f"n={summary.simulation_count}, "
        f"red={format_percentage(summary.red_win_probability.probability)}, "
        f"blue={format_percentage(summary.blue_win_probability.probability)}, "
        f"draw={format_percentage(summary.draw_probability.probability)}, "
        f"finish={format_percentage(summary.finish_probability.probability)}, "
        f"distance={format_percentage(summary.scheduled_distance_probability.probability)}"
    )


def add_check(
    checks: list[AuditCheck],
    *,
    name: str,
    passed: bool,
    details: str,
) -> None:
    """Append one audit check."""

    checks.append(
        AuditCheck(
            name=name,
            passed=passed,
            details=details,
        )
    )


def main() -> None:
    """Run the Monte Carlo population audit."""

    parser = argparse.ArgumentParser(
        description=(
            "Audit RFS Monte Carlo V2 matchup populations"
        )
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=1_000,
        help=(
            "Primary simulations per symmetric population "
            "(minimum 500)"
        ),
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Starting seed for the first population",
    )

    args = parser.parse_args()

    if args.paths < 500:
        parser.error(
            "--paths must be at least 500"
        )

    if args.seed_start < 0:
        parser.error(
            "--seed-start cannot be negative"
        )

    primary_paths = args.paths
    small_paths = max(
        100,
        primary_paths // 10,
    )
    draw_paths = min(
        250,
        primary_paths,
    )
    five_round_paths = min(
        250,
        primary_paths,
    )
    replay_paths = 100
    partition_chunk_paths = max(
        50,
        min(
            100,
            primary_paths // 10,
        ),
    )
    partition_total_paths = (
        partition_chunk_paths * 3
    )

    base_seed = args.seed_start

    symmetric_decisions = run_population(
        simulation_count=primary_paths,
        seed_start=base_seed,
    )

    small_symmetric_decisions = run_population(
        simulation_count=small_paths,
        seed_start=base_seed + 1_000_000,
    )

    high_ko_calibration = knockout_finish_calibration(
        landed_probability=0.15,
        knockdown_probability=0.60,
    )

    symmetric_high_ko = run_population(
        simulation_count=primary_paths,
        seed_start=base_seed + 2_000_000,
        finish_calibration=high_ko_calibration,
    )

    unanimous_draws = run_population(
        simulation_count=draw_paths,
        seed_start=base_seed + 3_000_000,
        scoring_calibration=zero_scoring(),
    )

    five_round_decisions = run_population(
        simulation_count=five_round_paths,
        seed_start=base_seed + 4_000_000,
        scheduled_rounds=5,
    )

    replay_first = run_population(
        simulation_count=replay_paths,
        seed_start=base_seed + 5_000_000,
        finish_calibration=high_ko_calibration,
    )
    replay_second = run_population(
        simulation_count=replay_paths,
        seed_start=base_seed + 5_000_000,
        finish_calibration=high_ko_calibration,
    )

    partition_seed = base_seed + 6_000_000

    full_partition_population = run_population(
        simulation_count=partition_total_paths,
        seed_start=partition_seed,
        finish_calibration=high_ko_calibration,
    )

    partition_summaries = tuple(
        run_population(
            simulation_count=partition_chunk_paths,
            seed_start=(
                partition_seed
                + partition_index
                * partition_chunk_paths
            ),
            finish_calibration=high_ko_calibration,
        )
        for partition_index in range(3)
    )

    merged_partition_population = (
        merge_contiguous_summaries(
            partition_summaries
        )
    )

    audited_summaries = (
        symmetric_decisions,
        small_symmetric_decisions,
        symmetric_high_ko,
        unanimous_draws,
        five_round_decisions,
        replay_first,
        replay_second,
        full_partition_population,
        merged_partition_population,
    )

    arithmetic_failures = sum(
        len(arithmetic_violations(summary))
        for summary in audited_summaries
    )
    probability_failures = sum(
        len(probability_violations(summary))
        for summary in audited_summaries
    )

    decision_corner_difference = abs(
        symmetric_decisions.red_win_probability.probability
        - symmetric_decisions.blue_win_probability.probability
    )
    finish_corner_difference = abs(
        symmetric_high_ko.red_win_probability.probability
        - symmetric_high_ko.blue_win_probability.probability
    )

    checks: list[AuditCheck] = []

    add_check(
        checks,
        name="all population summaries preserve authoritative count arithmetic",
        passed=arithmetic_failures == 0,
        details=f"violations={arithmetic_failures}",
    )

    add_check(
        checks,
        name="all probability families and Wilson intervals are valid",
        passed=probability_failures == 0,
        details=f"violations={probability_failures}",
    )

    add_check(
        checks,
        name="population replay is deterministic",
        passed=replay_first == replay_second,
        details=(
            f"paths={replay_paths}, "
            f"exact equality={replay_first == replay_second}"
        ),
    )

    add_check(
        checks,
        name="partitioned contiguous seed ranges equal one full population",
        passed=(
            merged_partition_population
            == full_partition_population
        ),
        details=(
            f"full paths={partition_total_paths}, "
            f"partitions=3 x {partition_chunk_paths}"
        ),
    )

    add_check(
        checks,
        name="symmetric decision population has no material corner bias",
        passed=decision_corner_difference <= 0.10,
        details=(
            f"red={format_percentage(symmetric_decisions.red_win_probability.probability)}, "
            f"blue={format_percentage(symmetric_decisions.blue_win_probability.probability)}, "
            f"difference={format_percentage(decision_corner_difference)}"
        ),
    )

    add_check(
        checks,
        name="high-KO population resolves overwhelmingly through finishes",
        passed=(
            symmetric_high_ko.finish_probability.probability
            >= 0.98
        ),
        details=(
            f"finish rate={format_percentage(symmetric_high_ko.finish_probability.probability)}"
        ),
    )

    add_check(
        checks,
        name="high-KO population contains only KO/TKO finishes",
        passed=(
            symmetric_high_ko.ko_tko_count
            == symmetric_high_ko.finish_count
            and symmetric_high_ko.submission_count == 0
        ),
        details=(
            f"KO/TKO={symmetric_high_ko.ko_tko_count}, "
            f"submission={symmetric_high_ko.submission_count}"
        ),
    )

    add_check(
        checks,
        name="symmetric finish population has no material corner bias",
        passed=finish_corner_difference <= 0.10,
        details=(
            f"red={format_percentage(symmetric_high_ko.red_win_probability.probability)}, "
            f"blue={format_percentage(symmetric_high_ko.blue_win_probability.probability)}, "
            f"difference={format_percentage(finish_corner_difference)}"
        ),
    )

    add_check(
        checks,
        name="zero scoring and zero judge variation produce unanimous draws",
        passed=(
            unanimous_draws.draw_count
            == unanimous_draws.simulation_count
            and unanimous_draws.unanimous_draw_count
            == unanimous_draws.simulation_count
            and unanimous_draws.red_win_count == 0
            and unanimous_draws.blue_win_count == 0
        ),
        details=(
            f"draws={unanimous_draws.draw_count}, "
            f"unanimous draws={unanimous_draws.unanimous_draw_count}"
        ),
    )

    add_check(
        checks,
        name="five-round zero-finish populations reach scheduled distance",
        passed=(
            five_round_decisions.scheduled_rounds == 5
            and five_round_decisions.finish_count == 0
            and five_round_decisions.scheduled_distance_count
            == five_round_decisions.simulation_count
            and five_round_decisions.finish_round_counts
            == (0, 0, 0, 0, 0)
        ),
        details=(
            f"distance={five_round_decisions.scheduled_distance_count}, "
            f"paths={five_round_decisions.simulation_count}, "
            f"round buckets={five_round_decisions.finish_round_counts}"
        ),
    )

    add_check(
        checks,
        name="larger Monte Carlo populations produce narrower uncertainty",
        passed=(
            symmetric_decisions.red_win_probability.interval_width
            < small_symmetric_decisions.red_win_probability.interval_width
        ),
        details=(
            f"n={small_paths} width="
            f"{small_symmetric_decisions.red_win_probability.interval_width:.6f}, "
            f"n={primary_paths} width="
            f"{symmetric_decisions.red_win_probability.interval_width:.6f}"
        ),
    )

    mean_finish_time = (
        symmetric_high_ko.mean_finish_elapsed_seconds_in_fight
    )

    add_check(
        checks,
        name="aggregated finish timing remains within legal fight bounds",
        passed=(
            mean_finish_time is not None
            and 1.0
            <= mean_finish_time
            <= symmetric_high_ko.scheduled_rounds * 300.0
        ),
        details=(
            f"mean finish seconds={mean_finish_time}"
        ),
    )

    print("=" * 80)
    print("RFS MONTE CARLO V2 MATCHUP POPULATION AUDIT")
    print("=" * 80)
    print(f"Primary paths:              {primary_paths:,}")
    print(f"Small comparison paths:     {small_paths:,}")
    print(f"Replay paths:               {replay_paths:,}")
    print(f"Draw paths:                 {draw_paths:,}")
    print(f"Five-round paths:           {five_round_paths:,}")
    print(
        "Partition paths:            "
        f"{partition_total_paths:,} "
        f"(3 x {partition_chunk_paths:,})"
    )
    print(f"Seed start:                 {base_seed:,}")
    print()

    print("POPULATION SUMMARIES")
    print("-" * 80)
    print(
        "Symmetric decisions: "
        + population_description(
            symmetric_decisions
        )
    )
    print(
        "Small symmetric:     "
        + population_description(
            small_symmetric_decisions
        )
    )
    print(
        "Symmetric high-KO:   "
        + population_description(
            symmetric_high_ko
        )
    )
    print(
        "Zero-scoring draws:  "
        + population_description(
            unanimous_draws
        )
    )
    print(
        "Five-round results:  "
        + population_description(
            five_round_decisions
        )
    )
    print(
        "High-KO finish rounds: "
        f"{symmetric_high_ko.finish_round_counts}"
    )
    print()

    for check in checks:
        status = (
            "PASS"
            if check.passed
            else "FAIL"
        )

        print(f"[{status}] {check.name}")
        print(f"       {check.details}")

    failed_checks = [
        check
        for check in checks
        if not check.passed
    ]

    print()
    print("=" * 80)

    if failed_checks:
        print(
            f"AUDIT FAIL: {len(failed_checks)} check(s) failed"
        )
        print("=" * 80)
        raise SystemExit(1)

    print("AUDIT PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
