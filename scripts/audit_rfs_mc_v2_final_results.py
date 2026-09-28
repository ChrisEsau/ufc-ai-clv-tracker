"""Population audit for unified RFS Monte Carlo V2 fight results.

This audit validates:

- zero-finish paths always resolve through scheduled distance
- high-KO calibration resolves through the finish branch
- finish and scheduled-distance payloads remain mutually exclusive
- scorecards and decision classifications remain legal
- symmetric fighters do not show material red/blue winner bias
- high-KO finish winners remain approximately side-symmetric
- dominant fighter parameters win in either corner
- mirrored dominant matchups produce similar strong-side win rates
- zero scoring and zero judge variability produce unanimous draws
- three- and five-round paths resolve correctly
- repeated seeds reproduce identical final results
- no structural violations occur across the audited populations

These checks establish structural and directional validity. They do not
establish historically calibrated UFC outcome probabilities.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass

from pipeline.simulation.rfs_mc_v1.contracts import (
    FightPhase,
)
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    SEGMENTS_PER_ROUND,
    FighterSide,
)
from pipeline.simulation.rfs_mc_v2_shared_state.decision_contracts import (
    DecisionType,
    resolve_decision,
)
from pipeline.simulation.rfs_mc_v2_shared_state.final_fight_result import (
    FightResultBranch,
    FinalFightResult,
    resolve_final_fight_result,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_calibration import (
    FinishProbabilityCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_contracts import (
    FinishMethod,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_path_runner import (
    run_finish_enabled_dynamic_path,
)
from pipeline.simulation.rfs_mc_v2_shared_state.judge_scorecard_generator import (
    JudgeVariabilityCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    FighterPhaseParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.round_scoring_engine import (
    RoundScoringCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
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
    """One final-result population audit check."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class FinalResultPopulationSummary:
    """Aggregated final results for one simulation scenario."""

    path_count: int

    finish_count: int
    scheduled_distance_count: int

    red_win_count: int
    blue_win_count: int
    draw_count: int

    ko_tko_count: int
    submission_count: int

    decision_counts: Counter[str]
    finish_round_counts: Counter[int]

    structural_violations: int

    @property
    def finish_rate(self) -> float:
        """Return the population finish rate."""

        return self.finish_count / self.path_count

    @property
    def scheduled_distance_rate(self) -> float:
        """Return the scheduled-distance rate."""

        return (
            self.scheduled_distance_count
            / self.path_count
        )

    @property
    def red_win_rate(self) -> float:
        """Return red's official win rate."""

        return self.red_win_count / self.path_count

    @property
    def blue_win_rate(self) -> float:
        """Return blue's official win rate."""

        return self.blue_win_count / self.path_count

    @property
    def draw_rate(self) -> float:
        """Return the official draw rate."""

        return self.draw_count / self.path_count


def zero_variability_calibration(
) -> JudgeVariabilityCalibration:
    """Return judge calibration with no random adjustment."""

    return JudgeVariabilityCalibration(
        fight_bias_stddev=0.0,
        round_noise_stddev=0.0,
        maximum_absolute_adjustment=0.0,
    )


def zero_scoring_calibration(
) -> RoundScoringCalibration:
    """Return scoring calibration producing even rounds."""

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


def strong_striker_parameters() -> FighterPhaseParameters:
    """Return a deliberately dominant distance-striking profile."""

    return phase_parameters(
        distance_attempt_rate=18.0,
        distance_accuracy=0.75,
        distance_knockdown_probability=0.0,
    )


def weak_striker_parameters() -> FighterPhaseParameters:
    """Return a deliberately weak distance-striking profile."""

    return phase_parameters(
        distance_attempt_rate=3.0,
        distance_accuracy=0.25,
        distance_knockdown_probability=0.0,
    )


def run_final_result(
    *,
    seed: int,
    finish_calibration: FinishProbabilityCalibration,
    scheduled_rounds: int = 3,
    red_phase: FighterPhaseParameters | None = None,
    blue_phase: FighterPhaseParameters | None = None,
    red_transition: FighterTransitionParameters | None = None,
    blue_transition: FighterTransitionParameters | None = None,
    scoring_calibration: RoundScoringCalibration | None = None,
    variability_calibration: (
        JudgeVariabilityCalibration | None
    ) = None,
) -> FinalFightResult:
    """Run and fully resolve one controlled simulation."""

    selected_red_transition = (
        red_transition
        if red_transition is not None
        else distance_only_transition_parameters()
    )
    selected_blue_transition = (
        blue_transition
        if blue_transition is not None
        else distance_only_transition_parameters()
    )

    selected_red_phase = (
        red_phase
        if red_phase is not None
        else phase_parameters()
    )
    selected_blue_phase = (
        blue_phase
        if blue_phase is not None
        else phase_parameters()
    )

    dynamic = dynamic_parameters()

    path = run_finish_enabled_dynamic_path(
        selected_red_transition,
        selected_blue_transition,
        selected_red_phase,
        selected_blue_phase,
        dynamic,
        dynamic,
        dynamic_state_calibration=zero_state_calibration(),
        phase_effect_calibration=zero_phase_effect_calibration(),
        transition_effect_calibration=(
            zero_transition_effect_calibration()
        ),
        finish_probability_calibration=finish_calibration,
        scheduled_rounds=scheduled_rounds,
        seed=seed,
    )

    return resolve_final_fight_result(
        path,
        scoring_calibration=scoring_calibration,
        variability_calibration=variability_calibration,
    )


def count_scorecard_violations(
    result: FinalFightResult,
) -> int:
    """Count structural violations in decision scorecards."""

    if result.scorecards is None:
        return 0

    violations = 0
    scorecards = result.scorecards

    if len(scorecards) != 3:
        violations += 1

    if {
        scorecard.judge_number
        for scorecard in scorecards
    } != {
        1,
        2,
        3,
    }:
        violations += 1

    for scorecard in scorecards:
        if (
            scorecard.scheduled_rounds
            != result.scheduled_rounds
        ):
            violations += 1

        if len(scorecard.rounds) != result.scheduled_rounds:
            violations += 1

        for expected_round, score in enumerate(
            scorecard.rounds,
            start=1,
        ):
            if score.round_number != expected_round:
                violations += 1

            if score.red_points not in {
                7,
                8,
                9,
                10,
            }:
                violations += 1

            if score.blue_points not in {
                7,
                8,
                9,
                10,
            }:
                violations += 1

            if max(
                score.red_points,
                score.blue_points,
            ) != 10:
                violations += 1

    try:
        reconstructed = resolve_decision(
            scorecards
        )
    except (
        TypeError,
        ValueError,
    ):
        violations += 1
    else:
        if result.scheduled_distance is None:
            violations += 1
        elif (
            reconstructed
            != result.scheduled_distance.decision
        ):
            violations += 1

    return violations


def count_final_result_violations(
    result: FinalFightResult,
) -> int:
    """Count terminal-branch and payload consistency violations."""

    violations = 0

    if (
        result.path.seed
        != result.seed
    ):
        violations += 1

    if (
        result.path.scheduled_rounds
        != result.scheduled_rounds
    ):
        violations += 1

    if result.branch is FightResultBranch.FINISH:
        if not result.is_finish:
            violations += 1

        if result.is_scheduled_distance:
            violations += 1

        if result.finish is None:
            violations += 1
            return violations

        if result.scheduled_distance is not None:
            violations += 1

        if result.path.finish != result.finish:
            violations += 1

        if result.winner is not result.finish.winner:
            violations += 1

        if result.winner is None:
            violations += 1

        if result.is_draw:
            violations += 1

        if result.finish_method is not result.finish.method:
            violations += 1

        if result.official_method is not result.finish.method:
            violations += 1

        if result.decision_type is not None:
            violations += 1

        if result.scorecards is not None:
            violations += 1

        if not (
            1
            <= result.finish.elapsed_seconds_in_segment
            <= 30
        ):
            violations += 1

        if not (
            1
            <= result.finish.elapsed_seconds_in_round
            <= 300
        ):
            violations += 1

        expected_path_length = (
            (
                result.finish.round_number - 1
            )
            * SEGMENTS_PER_ROUND
            + result.finish.segment_number
        )

        if len(result.path.segments) != expected_path_length:
            violations += 1

        if (
            result.path.segments[-1].finish
            != result.finish
        ):
            violations += 1

        if result.path.segments[-1].transition is not None:
            violations += 1

        if (
            result.path.segments[-1]
            .round_break_recovery_applied
        ):
            violations += 1

        if result.finish.method is FinishMethod.SUBMISSION:
            if result.finish.state.phase is not FightPhase.GROUND:
                violations += 1

            if (
                result.finish.state.phase_owner
                is not result.finish.winner
            ):
                violations += 1

    elif (
        result.branch
        is FightResultBranch.SCHEDULED_DISTANCE
    ):
        if result.is_finish:
            violations += 1

        if not result.is_scheduled_distance:
            violations += 1

        if result.finish is not None:
            violations += 1

        if result.path.finish is not None:
            violations += 1

        if not result.path.reached_scheduled_distance:
            violations += 1

        expected_path_length = (
            result.scheduled_rounds
            * SEGMENTS_PER_ROUND
        )

        if len(result.path.segments) != expected_path_length:
            violations += 1

        if result.scheduled_distance is None:
            violations += 1
            return violations

        if result.scheduled_distance.path != result.path:
            violations += 1

        if (
            result.winner
            is not result.scheduled_distance.winner
        ):
            violations += 1

        if (
            result.decision_type
            is not result.scheduled_distance.decision_type
        ):
            violations += 1

        if (
            result.official_method
            is not result.scheduled_distance.decision_type
        ):
            violations += 1

        if result.finish_method is not None:
            violations += 1

        if result.finish_round is not None:
            violations += 1

        if result.finish_segment is not None:
            violations += 1

        if result.elapsed_seconds_in_round is not None:
            violations += 1

        if result.is_draw is not (
            result.winner is None
        ):
            violations += 1

        violations += count_scorecard_violations(
            result
        )

    else:
        violations += 1

    return violations


def summarize_population(
    *,
    path_count: int,
    seed_start: int,
    finish_calibration: FinishProbabilityCalibration,
    scheduled_rounds: int = 3,
    red_phase: FighterPhaseParameters | None = None,
    blue_phase: FighterPhaseParameters | None = None,
    scoring_calibration: RoundScoringCalibration | None = None,
    variability_calibration: (
        JudgeVariabilityCalibration | None
    ) = None,
) -> FinalResultPopulationSummary:
    """Run and summarize one final-result population."""

    finish_count = 0
    scheduled_distance_count = 0

    red_win_count = 0
    blue_win_count = 0
    draw_count = 0

    ko_tko_count = 0
    submission_count = 0

    decision_counts: Counter[str] = Counter()
    finish_round_counts: Counter[int] = Counter()

    structural_violations = 0

    for path_index in range(path_count):
        result = run_final_result(
            seed=seed_start + path_index,
            finish_calibration=finish_calibration,
            scheduled_rounds=scheduled_rounds,
            red_phase=red_phase,
            blue_phase=blue_phase,
            scoring_calibration=scoring_calibration,
            variability_calibration=(
                variability_calibration
            ),
        )

        structural_violations += (
            count_final_result_violations(
                result
            )
        )

        if result.winner is FighterSide.RED:
            red_win_count += 1
        elif result.winner is FighterSide.BLUE:
            blue_win_count += 1
        else:
            draw_count += 1

        if result.branch is FightResultBranch.FINISH:
            finish_count += 1

            if result.finish_method is FinishMethod.KO_TKO:
                ko_tko_count += 1
            elif (
                result.finish_method
                is FinishMethod.SUBMISSION
            ):
                submission_count += 1

            if result.finish_round is not None:
                finish_round_counts[
                    result.finish_round
                ] += 1

        else:
            scheduled_distance_count += 1

            if result.decision_type is not None:
                decision_counts[
                    result.decision_type.value
                ] += 1

    return FinalResultPopulationSummary(
        path_count=path_count,
        finish_count=finish_count,
        scheduled_distance_count=(
            scheduled_distance_count
        ),
        red_win_count=red_win_count,
        blue_win_count=blue_win_count,
        draw_count=draw_count,
        ko_tko_count=ko_tko_count,
        submission_count=submission_count,
        decision_counts=decision_counts,
        finish_round_counts=finish_round_counts,
        structural_violations=structural_violations,
    )


def count_replay_violations(
    *,
    path_count: int,
    seed_start: int,
    finish_calibration: FinishProbabilityCalibration,
    scheduled_rounds: int = 3,
) -> int:
    """Count differences between repeated seeded final results."""

    violations = 0

    for path_index in range(path_count):
        seed = seed_start + path_index

        first = run_final_result(
            seed=seed,
            finish_calibration=finish_calibration,
            scheduled_rounds=scheduled_rounds,
        )
        second = run_final_result(
            seed=seed,
            finish_calibration=finish_calibration,
            scheduled_rounds=scheduled_rounds,
        )

        if first != second:
            violations += 1

    return violations


def format_summary(
    name: str,
    summary: FinalResultPopulationSummary,
) -> str:
    """Return one concise population summary line."""

    return (
        f"{name}: "
        f"finish={summary.finish_rate:.2%}, "
        f"distance={summary.scheduled_distance_rate:.2%}, "
        f"red={summary.red_win_rate:.2%}, "
        f"blue={summary.blue_win_rate:.2%}, "
        f"draw={summary.draw_rate:.2%}"
    )


def run_audit(
    *,
    path_count: int,
    seed_start: int,
) -> int:
    """Run every unified final-result population scenario."""

    if path_count <= 0:
        raise ValueError(
            "path_count must be positive"
        )

    replay_path_count = min(
        path_count,
        100,
    )
    directional_path_count = min(
        path_count,
        500,
    )
    draw_path_count = min(
        path_count,
        250,
    )
    five_round_path_count = min(
        path_count,
        250,
    )

    zero_finish = zero_finish_calibration()

    high_ko = knockout_finish_calibration(
        landed_probability=0.15,
        knockdown_probability=0.60,
    )

    symmetric_decisions = summarize_population(
        path_count=path_count,
        seed_start=seed_start,
        finish_calibration=zero_finish,
    )

    symmetric_finishes = summarize_population(
        path_count=path_count,
        seed_start=seed_start + 100_000,
        finish_calibration=high_ko,
    )

    unanimous_draws = summarize_population(
        path_count=draw_path_count,
        seed_start=seed_start + 200_000,
        finish_calibration=zero_finish,
        scoring_calibration=zero_scoring_calibration(),
        variability_calibration=(
            zero_variability_calibration()
        ),
    )

    strong_red = summarize_population(
        path_count=directional_path_count,
        seed_start=seed_start + 300_000,
        finish_calibration=zero_finish,
        red_phase=strong_striker_parameters(),
        blue_phase=weak_striker_parameters(),
        variability_calibration=(
            zero_variability_calibration()
        ),
    )

    strong_blue = summarize_population(
        path_count=directional_path_count,
        seed_start=seed_start + 300_000,
        finish_calibration=zero_finish,
        red_phase=weak_striker_parameters(),
        blue_phase=strong_striker_parameters(),
        variability_calibration=(
            zero_variability_calibration()
        ),
    )

    five_round_decisions = summarize_population(
        path_count=five_round_path_count,
        seed_start=seed_start + 400_000,
        finish_calibration=zero_finish,
        scheduled_rounds=5,
    )

    decision_replay_violations = (
        count_replay_violations(
            path_count=replay_path_count,
            seed_start=seed_start + 500_000,
            finish_calibration=zero_finish,
        )
    )

    finish_replay_violations = (
        count_replay_violations(
            path_count=replay_path_count,
            seed_start=seed_start + 600_000,
            finish_calibration=high_ko,
        )
    )

    symmetric_decision_bias = abs(
        symmetric_decisions.red_win_rate
        - symmetric_decisions.blue_win_rate
    )

    symmetric_finish_bias = abs(
        symmetric_finishes.red_win_rate
        - symmetric_finishes.blue_win_rate
    )

    strong_red_rate = strong_red.red_win_rate
    strong_blue_rate = strong_blue.blue_win_rate
    mirrored_rate_difference = abs(
        strong_red_rate
        - strong_blue_rate
    )

    checks = [
        AuditCheck(
            name=(
                "zero-finish calibration resolves only "
                "scheduled-distance results"
            ),
            passed=(
                symmetric_decisions.finish_count == 0
                and symmetric_decisions
                .scheduled_distance_count
                == path_count
            ),
            detail=(
                f"finish={symmetric_decisions.finish_count}, "
                f"distance="
                f"{symmetric_decisions.scheduled_distance_count}"
            ),
        ),
        AuditCheck(
            name=(
                "scheduled-distance population has zero "
                "structural violations"
            ),
            passed=(
                symmetric_decisions
                .structural_violations
                == 0
            ),
            detail=(
                f"violations="
                f"{symmetric_decisions.structural_violations}"
            ),
        ),
        AuditCheck(
            name=(
                "symmetric decision population has no "
                "material corner bias"
            ),
            passed=symmetric_decision_bias <= 0.10,
            detail=(
                f"red={symmetric_decisions.red_win_rate:.2%}, "
                f"blue={symmetric_decisions.blue_win_rate:.2%}, "
                f"absolute difference="
                f"{symmetric_decision_bias:.2%}"
            ),
        ),
        AuditCheck(
            name=(
                "high-KO calibration resolves overwhelmingly "
                "through finish branch"
            ),
            passed=(
                symmetric_finishes.finish_rate
                >= 0.98
            ),
            detail=(
                f"finish rate="
                f"{symmetric_finishes.finish_rate:.2%}"
            ),
        ),
        AuditCheck(
            name=(
                "high-KO calibration produces only "
                "KO/TKO finishes"
            ),
            passed=(
                symmetric_finishes.finish_count > 0
                and symmetric_finishes.ko_tko_count
                == symmetric_finishes.finish_count
                and symmetric_finishes.submission_count
                == 0
            ),
            detail=(
                f"KO/TKO="
                f"{symmetric_finishes.ko_tko_count}, "
                f"submission="
                f"{symmetric_finishes.submission_count}"
            ),
        ),
        AuditCheck(
            name=(
                "finish population has zero structural "
                "violations"
            ),
            passed=(
                symmetric_finishes
                .structural_violations
                == 0
            ),
            detail=(
                f"violations="
                f"{symmetric_finishes.structural_violations}"
            ),
        ),
        AuditCheck(
            name=(
                "symmetric finish population has no "
                "material corner bias"
            ),
            passed=symmetric_finish_bias <= 0.10,
            detail=(
                f"red={symmetric_finishes.red_win_rate:.2%}, "
                f"blue={symmetric_finishes.blue_win_rate:.2%}, "
                f"absolute difference="
                f"{symmetric_finish_bias:.2%}"
            ),
        ),
        AuditCheck(
            name=(
                "zero scoring and zero variability produce "
                "unanimous draws"
            ),
            passed=(
                unanimous_draws.draw_count
                == draw_path_count
                and unanimous_draws.decision_counts[
                    DecisionType.UNANIMOUS_DRAW.value
                ]
                == draw_path_count
            ),
            detail=(
                f"draws={unanimous_draws.draw_count}, "
                f"unanimous draws="
                f"{unanimous_draws.decision_counts[DecisionType.UNANIMOUS_DRAW.value]}"
            ),
        ),
        AuditCheck(
            name=(
                "dominant red striking parameters win "
                "directionally"
            ),
            passed=strong_red_rate >= 0.95,
            detail=(
                f"red win rate="
                f"{strong_red_rate:.2%}"
            ),
        ),
        AuditCheck(
            name=(
                "dominant blue striking parameters win "
                "directionally"
            ),
            passed=strong_blue_rate >= 0.95,
            detail=(
                f"blue win rate="
                f"{strong_blue_rate:.2%}"
            ),
        ),
        AuditCheck(
            name=(
                "mirrored dominant matchups have comparable "
                "strong-side win rates"
            ),
            passed=mirrored_rate_difference <= 0.05,
            detail=(
                f"strong red={strong_red_rate:.2%}, "
                f"strong blue={strong_blue_rate:.2%}, "
                f"difference="
                f"{mirrored_rate_difference:.2%}"
            ),
        ),
        AuditCheck(
            name=(
                "five-round paths resolve only through "
                "scheduled distance"
            ),
            passed=(
                five_round_decisions.finish_count == 0
                and five_round_decisions
                .scheduled_distance_count
                == five_round_path_count
            ),
            detail=(
                f"distance="
                f"{five_round_decisions.scheduled_distance_count}, "
                f"expected={five_round_path_count}"
            ),
        ),
        AuditCheck(
            name=(
                "five-round final results have zero "
                "structural violations"
            ),
            passed=(
                five_round_decisions
                .structural_violations
                == 0
            ),
            detail=(
                f"violations="
                f"{five_round_decisions.structural_violations}"
            ),
        ),
        AuditCheck(
            name=(
                "scheduled-distance final results replay "
                "deterministically"
            ),
            passed=decision_replay_violations == 0,
            detail=(
                f"replayed={replay_path_count}, "
                f"violations="
                f"{decision_replay_violations}"
            ),
        ),
        AuditCheck(
            name=(
                "finish final results replay "
                "deterministically"
            ),
            passed=finish_replay_violations == 0,
            detail=(
                f"replayed={replay_path_count}, "
                f"violations="
                f"{finish_replay_violations}"
            ),
        ),
    ]

    print("=" * 80)
    print("RFS MONTE CARLO V2 FINAL RESULT POPULATION AUDIT")
    print("=" * 80)
    print(f"Primary paths per scenario: {path_count:,}")
    print(f"Replay paths per branch:    {replay_path_count:,}")
    print(f"Directional paths:          {directional_path_count:,}")
    print(f"Five-round paths:           {five_round_path_count:,}")
    print(f"Seed start:                 {seed_start:,}")
    print()

    print("POPULATION SUMMARIES")
    print("-" * 80)
    print(
        format_summary(
            "Symmetric decisions",
            symmetric_decisions,
        )
    )
    print(
        "Decision types: "
        f"{dict(sorted(symmetric_decisions.decision_counts.items()))}"
    )
    print(
        format_summary(
            "Symmetric high-KO",
            symmetric_finishes,
        )
    )
    print(
        "Finish rounds: "
        f"{dict(sorted(symmetric_finishes.finish_round_counts.items()))}"
    )
    print(
        format_summary(
            "Strong red",
            strong_red,
        )
    )
    print(
        format_summary(
            "Strong blue",
            strong_blue,
        )
    )
    print(
        format_summary(
            "Five-round decisions",
            five_round_decisions,
        )
    )
    print()

    all_passed = True

    for check in checks:
        status = (
            "PASS"
            if check.passed
            else "FAIL"
        )
        all_passed = (
            all_passed
            and check.passed
        )

        print(f"[{status}] {check.name}")
        print(f"       {check.detail}")

    print()
    print("=" * 80)
    print(
        "AUDIT PASS"
        if all_passed
        else "AUDIT FAIL"
    )
    print("=" * 80)

    return (
        0
        if all_passed
        else 1
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Audit unified RFS Monte Carlo V2 "
            "final-result populations."
        )
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=1_000,
        help=(
            "Number of paths in each primary "
            "population scenario."
        ),
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First deterministic simulation seed.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the command-line population audit."""

    args = parse_args()

    return run_audit(
        path_count=args.paths,
        seed_start=args.seed_start,
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
