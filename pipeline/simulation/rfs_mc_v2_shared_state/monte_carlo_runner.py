"""Matchup Monte Carlo population runner for RFS Monte Carlo V2.

Each simulation:

1. generates one finish-enabled dynamic path
2. resolves the authoritative final fight result
3. aggregates winner, method, decision, round, and timing outcomes

Simulation seeds are sequential:

    seed_start
    seed_start + 1
    ...
    seed_start + simulation_count - 1

The runner returns only the population summary. Individual path retention,
parallel execution, serialization, and production artifact generation belong
to later integration layers.
"""

from __future__ import annotations

from pipeline.simulation.rfs_mc_v2_shared_state.decision_contracts import (
    DecisionType,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_calibration import (
    DynamicStateCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_effect_calibration import (
    DynamicEffectCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_parameters import (
    FighterDynamicParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_transition_effect_calibration import (
    DynamicTransitionEffectCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.final_fight_result import (
    FightResultBranch,
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
from pipeline.simulation.rfs_mc_v2_shared_state.monte_carlo_summary import (
    MatchupMonteCarloSummary,
)
from pipeline.simulation.rfs_mc_v2_shared_state.path_runner import (
    SharedPathCalibration,
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
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
)


def _validate_runner_inputs(
    *,
    simulation_count: int,
    seed_start: int,
    scheduled_rounds: int,
    dynamic_state_calibration: DynamicStateCalibration,
    phase_effect_calibration: DynamicEffectCalibration,
    transition_effect_calibration: (
        DynamicTransitionEffectCalibration
    ),
    finish_probability_calibration: FinishProbabilityCalibration,
    scoring_calibration: RoundScoringCalibration,
    variability_calibration: JudgeVariabilityCalibration,
    shared_path_calibration: SharedPathCalibration | None,
) -> None:
    """Validate population controls and calibration contracts."""

    if type(simulation_count) is not int:
        raise TypeError(
            "simulation_count must be an integer"
        )

    if simulation_count <= 0:
        raise ValueError(
            "simulation_count must be positive"
        )

    if type(seed_start) is not int:
        raise TypeError(
            "seed_start must be an integer"
        )

    if seed_start < 0:
        raise ValueError(
            "seed_start cannot be negative"
        )

    if type(scheduled_rounds) is not int:
        raise TypeError(
            "scheduled_rounds must be an integer"
        )

    if scheduled_rounds not in {
        3,
        5,
    }:
        raise ValueError(
            "scheduled_rounds must be 3 or 5"
        )

    if not isinstance(
        dynamic_state_calibration,
        DynamicStateCalibration,
    ):
        raise TypeError(
            "dynamic_state_calibration must be "
            "DynamicStateCalibration"
        )

    if not isinstance(
        phase_effect_calibration,
        DynamicEffectCalibration,
    ):
        raise TypeError(
            "phase_effect_calibration must be "
            "DynamicEffectCalibration"
        )

    if not isinstance(
        transition_effect_calibration,
        DynamicTransitionEffectCalibration,
    ):
        raise TypeError(
            "transition_effect_calibration must be "
            "DynamicTransitionEffectCalibration"
        )

    if not isinstance(
        finish_probability_calibration,
        FinishProbabilityCalibration,
    ):
        raise TypeError(
            "finish_probability_calibration must be "
            "FinishProbabilityCalibration"
        )

    if not isinstance(
        scoring_calibration,
        RoundScoringCalibration,
    ):
        raise TypeError(
            "scoring_calibration must be "
            "RoundScoringCalibration"
        )

    if not isinstance(
        variability_calibration,
        JudgeVariabilityCalibration,
    ):
        raise TypeError(
            "variability_calibration must be "
            "JudgeVariabilityCalibration"
        )

    if (
        shared_path_calibration is not None
        and not isinstance(
            shared_path_calibration,
            SharedPathCalibration,
        )
    ):
        raise TypeError(
            "shared_path_calibration must be "
            "SharedPathCalibration or None"
        )


def run_matchup_monte_carlo(
    red_transition_baseline: FighterTransitionParameters,
    blue_transition_baseline: FighterTransitionParameters,
    red_phase_baseline: FighterPhaseParameters,
    blue_phase_baseline: FighterPhaseParameters,
    red_dynamic_parameters: FighterDynamicParameters,
    blue_dynamic_parameters: FighterDynamicParameters,
    *,
    dynamic_state_calibration: DynamicStateCalibration,
    phase_effect_calibration: DynamicEffectCalibration,
    transition_effect_calibration: (
        DynamicTransitionEffectCalibration
    ),
    finish_probability_calibration: FinishProbabilityCalibration,
    simulation_count: int,
    seed_start: int,
    scheduled_rounds: int,
    red_intrinsic_power_multiplier: float = 1.0,
    blue_intrinsic_power_multiplier: float = 1.0,
    red_intrinsic_ko_vulnerability_multiplier: float = 1.0,
    blue_intrinsic_ko_vulnerability_multiplier: float = 1.0,
    scoring_calibration: RoundScoringCalibration | None = None,
    variability_calibration: JudgeVariabilityCalibration | None = None,
    shared_path_calibration: SharedPathCalibration | None = None,
) -> MatchupMonteCarloSummary:
    """Run and summarize one matchup simulation population."""

    selected_scoring = (
        scoring_calibration
        if scoring_calibration is not None
        else RoundScoringCalibration()
    )

    selected_variability = (
        variability_calibration
        if variability_calibration is not None
        else JudgeVariabilityCalibration()
    )

    _validate_runner_inputs(
        simulation_count=simulation_count,
        seed_start=seed_start,
        scheduled_rounds=scheduled_rounds,
        dynamic_state_calibration=(
            dynamic_state_calibration
        ),
        phase_effect_calibration=(
            phase_effect_calibration
        ),
        transition_effect_calibration=(
            transition_effect_calibration
        ),
        finish_probability_calibration=(
            finish_probability_calibration
        ),
        scoring_calibration=selected_scoring,
        variability_calibration=selected_variability,
        shared_path_calibration=(
            shared_path_calibration
        ),
    )

    red_win_count = 0
    blue_win_count = 0
    draw_count = 0

    finish_count = 0
    scheduled_distance_count = 0

    red_ko_tko_count = 0
    blue_ko_tko_count = 0
    red_submission_count = 0
    blue_submission_count = 0

    red_decision_count = 0
    blue_decision_count = 0

    unanimous_decision_count = 0
    split_decision_count = 0
    majority_decision_count = 0

    unanimous_draw_count = 0
    split_draw_count = 0
    majority_draw_count = 0

    finish_round_counts = [
        0
        for _ in range(scheduled_rounds)
    ]
    total_finish_elapsed_seconds_in_fight = 0

    for simulation_index in range(
        simulation_count
    ):
        seed = (
            seed_start
            + simulation_index
        )

        path = run_finish_enabled_dynamic_path(
            red_transition_baseline,
            blue_transition_baseline,
            red_phase_baseline,
            blue_phase_baseline,
            red_dynamic_parameters,
            blue_dynamic_parameters,
            dynamic_state_calibration=(
                dynamic_state_calibration
            ),
            phase_effect_calibration=(
                phase_effect_calibration
            ),
            transition_effect_calibration=(
                transition_effect_calibration
            ),
            finish_probability_calibration=(
                finish_probability_calibration
            ),
            scheduled_rounds=scheduled_rounds,
            seed=seed,
            red_intrinsic_power_multiplier=(
                red_intrinsic_power_multiplier
            ),
            blue_intrinsic_power_multiplier=(
                blue_intrinsic_power_multiplier
            ),
            red_intrinsic_ko_vulnerability_multiplier=(
                red_intrinsic_ko_vulnerability_multiplier
            ),
            blue_intrinsic_ko_vulnerability_multiplier=(
                blue_intrinsic_ko_vulnerability_multiplier
            ),
            shared_path_calibration=(
                shared_path_calibration
            ),
        )

        result = resolve_final_fight_result(
            path,
            scoring_calibration=selected_scoring,
            variability_calibration=selected_variability,
        )

        if result.winner is FighterSide.RED:
            red_win_count += 1
        elif result.winner is FighterSide.BLUE:
            blue_win_count += 1
        else:
            draw_count += 1

        if result.branch is FightResultBranch.FINISH:
            finish_count += 1

            if result.finish is None:
                raise RuntimeError(
                    "finish branch has no finish payload"
                )

            winner = result.finish.winner
            method = result.finish.method

            if method is FinishMethod.KO_TKO:
                if winner is FighterSide.RED:
                    red_ko_tko_count += 1
                else:
                    blue_ko_tko_count += 1

            elif method is FinishMethod.SUBMISSION:
                if winner is FighterSide.RED:
                    red_submission_count += 1
                else:
                    blue_submission_count += 1

            else:
                raise RuntimeError(
                    "unsupported finish method"
                )

            finish_round = result.finish.round_number

            if not 1 <= finish_round <= scheduled_rounds:
                raise RuntimeError(
                    "finish round is outside scheduled rounds"
                )

            finish_round_counts[
                finish_round - 1
            ] += 1

            total_finish_elapsed_seconds_in_fight += (
                (
                    finish_round - 1
                )
                * 300
                + result.finish.elapsed_seconds_in_round
            )

            continue

        if (
            result.branch
            is not FightResultBranch.SCHEDULED_DISTANCE
        ):
            raise RuntimeError(
                "unsupported final result branch"
            )

        scheduled_distance_count += 1

        if result.decision_type is None:
            raise RuntimeError(
                "scheduled-distance branch has no "
                "decision type"
            )

        if result.winner is FighterSide.RED:
            red_decision_count += 1
        elif result.winner is FighterSide.BLUE:
            blue_decision_count += 1

        if (
            result.decision_type
            is DecisionType.UNANIMOUS_DECISION
        ):
            unanimous_decision_count += 1

        elif (
            result.decision_type
            is DecisionType.SPLIT_DECISION
        ):
            split_decision_count += 1

        elif (
            result.decision_type
            is DecisionType.MAJORITY_DECISION
        ):
            majority_decision_count += 1

        elif (
            result.decision_type
            is DecisionType.UNANIMOUS_DRAW
        ):
            unanimous_draw_count += 1

        elif (
            result.decision_type
            is DecisionType.SPLIT_DRAW
        ):
            split_draw_count += 1

        elif (
            result.decision_type
            is DecisionType.MAJORITY_DRAW
        ):
            majority_draw_count += 1

        else:
            raise RuntimeError(
                "unsupported decision type"
            )

    return MatchupMonteCarloSummary(
        simulation_count=simulation_count,
        seed_start=seed_start,
        scheduled_rounds=scheduled_rounds,
        red_win_count=red_win_count,
        blue_win_count=blue_win_count,
        draw_count=draw_count,
        finish_count=finish_count,
        scheduled_distance_count=(
            scheduled_distance_count
        ),
        red_ko_tko_count=red_ko_tko_count,
        blue_ko_tko_count=blue_ko_tko_count,
        red_submission_count=red_submission_count,
        blue_submission_count=blue_submission_count,
        red_decision_count=red_decision_count,
        blue_decision_count=blue_decision_count,
        unanimous_decision_count=(
            unanimous_decision_count
        ),
        split_decision_count=split_decision_count,
        majority_decision_count=(
            majority_decision_count
        ),
        unanimous_draw_count=unanimous_draw_count,
        split_draw_count=split_draw_count,
        majority_draw_count=majority_draw_count,
        finish_round_counts=tuple(
            finish_round_counts
        ),
        total_finish_elapsed_seconds_in_fight=(
            total_finish_elapsed_seconds_in_fight
        ),
    )
