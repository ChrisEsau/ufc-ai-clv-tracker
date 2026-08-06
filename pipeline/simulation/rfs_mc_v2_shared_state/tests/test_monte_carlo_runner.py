"""Tests for the V2 matchup Monte Carlo population runner."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import pipeline.simulation.rfs_mc_v2_shared_state.monte_carlo_runner as runner_module
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
)
from pipeline.simulation.rfs_mc_v2_shared_state.decision_contracts import (
    DecisionType,
)
from pipeline.simulation.rfs_mc_v2_shared_state.final_fight_result import (
    FightResultBranch,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_contracts import (
    FinishMethod,
)
from pipeline.simulation.rfs_mc_v2_shared_state.judge_scorecard_generator import (
    JudgeVariabilityCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.monte_carlo_runner import (
    run_matchup_monte_carlo,
)
from pipeline.simulation.rfs_mc_v2_shared_state.monte_carlo_summary import (
    MatchupMonteCarloSummary,
)
from pipeline.simulation.rfs_mc_v2_shared_state.path_runner import (
    SharedPathCalibration,
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


def zero_variability() -> JudgeVariabilityCalibration:
    """Return judge calibration with no random variation."""

    return JudgeVariabilityCalibration(
        fight_bias_stddev=0.0,
        round_noise_stddev=0.0,
        maximum_absolute_adjustment=0.0,
    )


def zero_scoring() -> RoundScoringCalibration:
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


def positional_inputs() -> tuple[object, ...]:
    """Return valid fighter parameter inputs."""

    transition = distance_only_transition_parameters()
    phase = phase_parameters()
    dynamic = dynamic_parameters()

    return (
        transition,
        transition,
        phase,
        phase,
        dynamic,
        dynamic,
    )


def runner_kwargs(
    *,
    simulation_count: int = 5,
    seed_start: int = 100,
    scheduled_rounds: int = 3,
) -> dict[str, object]:
    """Return valid keyword arguments for the runner."""

    return {
        "dynamic_state_calibration": zero_state_calibration(),
        "phase_effect_calibration": (
            zero_phase_effect_calibration()
        ),
        "transition_effect_calibration": (
            zero_transition_effect_calibration()
        ),
        "finish_probability_calibration": (
            zero_finish_calibration()
        ),
        "simulation_count": simulation_count,
        "seed_start": seed_start,
        "scheduled_rounds": scheduled_rounds,
        "scoring_calibration": RoundScoringCalibration(),
        "variability_calibration": zero_variability(),
    }


def run_actual(
    *,
    simulation_count: int = 5,
    seed_start: int = 100,
    scheduled_rounds: int = 3,
    finish_calibration=None,
    scoring_calibration=None,
    variability_calibration=None,
) -> MatchupMonteCarloSummary:
    """Run one small real simulation population."""

    kwargs = runner_kwargs(
        simulation_count=simulation_count,
        seed_start=seed_start,
        scheduled_rounds=scheduled_rounds,
    )

    if finish_calibration is not None:
        kwargs["finish_probability_calibration"] = (
            finish_calibration
        )

    if scoring_calibration is not None:
        kwargs["scoring_calibration"] = scoring_calibration

    if variability_calibration is not None:
        kwargs["variability_calibration"] = (
            variability_calibration
        )

    return run_matchup_monte_carlo(
        *positional_inputs(),
        **kwargs,
    )


def fake_finish_result(
    *,
    winner: FighterSide,
    method: FinishMethod,
    round_number: int,
    elapsed_seconds_in_round: int,
) -> SimpleNamespace:
    """Build the result fields consumed by the aggregator."""

    return SimpleNamespace(
        winner=winner,
        branch=FightResultBranch.FINISH,
        finish=SimpleNamespace(
            winner=winner,
            method=method,
            round_number=round_number,
            elapsed_seconds_in_round=(
                elapsed_seconds_in_round
            ),
        ),
        decision_type=None,
    )


def fake_decision_result(
    *,
    winner: FighterSide | None,
    decision_type: DecisionType,
) -> SimpleNamespace:
    """Build a scheduled-distance result for aggregation tests."""

    return SimpleNamespace(
        winner=winner,
        branch=FightResultBranch.SCHEDULED_DISTANCE,
        finish=None,
        decision_type=decision_type,
    )


def install_fake_population(
    monkeypatch: pytest.MonkeyPatch,
    results: tuple[SimpleNamespace, ...],
    *,
    seed_start: int,
) -> tuple[list[int], list[dict[str, object]]]:
    """Replace path generation and final resolution with fixed results."""

    captured_seeds: list[int] = []
    captured_path_kwargs: list[dict[str, object]] = []

    def fake_path_runner(
        *args: object,
        **kwargs: object,
    ) -> SimpleNamespace:
        del args

        seed = kwargs["seed"]
        captured_seeds.append(seed)
        captured_path_kwargs.append(dict(kwargs))

        return SimpleNamespace(
            seed=seed,
        )

    def fake_resolver(
        path: SimpleNamespace,
        **kwargs: object,
    ) -> SimpleNamespace:
        del kwargs

        return results[
            path.seed - seed_start
        ]

    monkeypatch.setattr(
        runner_module,
        "run_finish_enabled_dynamic_path",
        fake_path_runner,
    )
    monkeypatch.setattr(
        runner_module,
        "resolve_final_fight_result",
        fake_resolver,
    )

    return (
        captured_seeds,
        captured_path_kwargs,
    )


def exact_fake_results() -> tuple[SimpleNamespace, ...]:
    """Return ten outcomes covering every aggregation family."""

    return (
        fake_finish_result(
            winner=FighterSide.RED,
            method=FinishMethod.KO_TKO,
            round_number=1,
            elapsed_seconds_in_round=100,
        ),
        fake_finish_result(
            winner=FighterSide.BLUE,
            method=FinishMethod.KO_TKO,
            round_number=2,
            elapsed_seconds_in_round=50,
        ),
        fake_finish_result(
            winner=FighterSide.RED,
            method=FinishMethod.SUBMISSION,
            round_number=3,
            elapsed_seconds_in_round=30,
        ),
        fake_finish_result(
            winner=FighterSide.BLUE,
            method=FinishMethod.SUBMISSION,
            round_number=1,
            elapsed_seconds_in_round=200,
        ),
        fake_decision_result(
            winner=FighterSide.RED,
            decision_type=DecisionType.UNANIMOUS_DECISION,
        ),
        fake_decision_result(
            winner=FighterSide.BLUE,
            decision_type=DecisionType.SPLIT_DECISION,
        ),
        fake_decision_result(
            winner=FighterSide.RED,
            decision_type=DecisionType.MAJORITY_DECISION,
        ),
        fake_decision_result(
            winner=FighterSide.BLUE,
            decision_type=DecisionType.UNANIMOUS_DECISION,
        ),
        fake_decision_result(
            winner=None,
            decision_type=DecisionType.SPLIT_DRAW,
        ),
        fake_decision_result(
            winner=None,
            decision_type=DecisionType.MAJORITY_DRAW,
        ),
    )


def test_exact_population_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_start = 500
    results = exact_fake_results()

    install_fake_population(
        monkeypatch,
        results,
        seed_start=seed_start,
    )

    selected = run_matchup_monte_carlo(
        *positional_inputs(),
        **runner_kwargs(
            simulation_count=10,
            seed_start=seed_start,
        ),
    )

    assert selected.simulation_count == 10
    assert selected.seed_start == 500

    assert selected.red_win_count == 4
    assert selected.blue_win_count == 4
    assert selected.draw_count == 2

    assert selected.finish_count == 4
    assert selected.scheduled_distance_count == 6

    assert selected.red_ko_tko_count == 1
    assert selected.blue_ko_tko_count == 1
    assert selected.red_submission_count == 1
    assert selected.blue_submission_count == 1

    assert selected.red_decision_count == 2
    assert selected.blue_decision_count == 2

    assert selected.unanimous_decision_count == 2
    assert selected.split_decision_count == 1
    assert selected.majority_decision_count == 1

    assert selected.unanimous_draw_count == 0
    assert selected.split_draw_count == 1
    assert selected.majority_draw_count == 1

    assert selected.finish_round_counts == (
        2,
        1,
        1,
    )

    # Fight times:
    # R1 1:40 = 100
    # R2 0:50 = 350 total
    # R3 0:30 = 630 total
    # R1 3:20 = 200
    assert (
        selected.total_finish_elapsed_seconds_in_fight
        == 1_280
    )
    assert (
        selected.mean_finish_elapsed_seconds_in_fight
        == pytest.approx(320.0)
    )


def test_runner_uses_sequential_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_start = 700
    results = tuple(
        fake_decision_result(
            winner=FighterSide.RED,
            decision_type=DecisionType.UNANIMOUS_DECISION,
        )
        for _ in range(4)
    )

    captured_seeds, _ = install_fake_population(
        monkeypatch,
        results,
        seed_start=seed_start,
    )

    run_matchup_monte_carlo(
        *positional_inputs(),
        **runner_kwargs(
            simulation_count=4,
            seed_start=seed_start,
        ),
    )

    assert captured_seeds == [
        700,
        701,
        702,
        703,
    ]


def test_shared_path_calibration_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_start = 10
    selected_shared_calibration = SharedPathCalibration()

    _, captured_kwargs = install_fake_population(
        monkeypatch,
        (
            fake_decision_result(
                winner=FighterSide.RED,
                decision_type=(
                    DecisionType.UNANIMOUS_DECISION
                ),
            ),
        ),
        seed_start=seed_start,
    )

    kwargs = runner_kwargs(
        simulation_count=1,
        seed_start=seed_start,
    )
    kwargs["shared_path_calibration"] = (
        selected_shared_calibration
    )

    run_matchup_monte_carlo(
        *positional_inputs(),
        **kwargs,
    )

    assert (
        captured_kwargs[0]["shared_path_calibration"]
        is selected_shared_calibration
    )


def test_scoring_and_variability_are_forwarded_to_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_start = 20
    scoring = zero_scoring()
    variability = zero_variability()
    captured: list[dict[str, object]] = []

    def fake_path_runner(
        *args: object,
        **kwargs: object,
    ) -> SimpleNamespace:
        del args

        return SimpleNamespace(
            seed=kwargs["seed"],
        )

    def fake_resolver(
        path: SimpleNamespace,
        **kwargs: object,
    ) -> SimpleNamespace:
        del path
        captured.append(dict(kwargs))

        return fake_decision_result(
            winner=None,
            decision_type=DecisionType.UNANIMOUS_DRAW,
        )

    monkeypatch.setattr(
        runner_module,
        "run_finish_enabled_dynamic_path",
        fake_path_runner,
    )
    monkeypatch.setattr(
        runner_module,
        "resolve_final_fight_result",
        fake_resolver,
    )

    kwargs = runner_kwargs(
        simulation_count=2,
        seed_start=seed_start,
    )
    kwargs["scoring_calibration"] = scoring
    kwargs["variability_calibration"] = variability

    run_matchup_monte_carlo(
        *positional_inputs(),
        **kwargs,
    )

    assert len(captured) == 2

    for call in captured:
        assert call["scoring_calibration"] is scoring
        assert call["variability_calibration"] is variability


def test_default_optional_calibrations_are_supported() -> None:
    kwargs = runner_kwargs(
        simulation_count=1,
    )
    kwargs.pop("scoring_calibration")
    kwargs.pop("variability_calibration")

    selected = run_matchup_monte_carlo(
        *positional_inputs(),
        **kwargs,
    )

    assert isinstance(
        selected,
        MatchupMonteCarloSummary,
    )
    assert selected.simulation_count == 1


@pytest.mark.parametrize(
    ("scheduled_rounds", "expected_round_counts"),
    [
        (
            3,
            3,
        ),
        (
            5,
            5,
        ),
    ],
)
def test_real_zero_finish_population_reaches_distance(
    scheduled_rounds: int,
    expected_round_counts: int,
) -> None:
    selected = run_actual(
        simulation_count=5,
        scheduled_rounds=scheduled_rounds,
    )

    assert selected.finish_count == 0
    assert selected.scheduled_distance_count == 5
    assert len(
        selected.finish_round_counts
    ) == expected_round_counts
    assert sum(
        selected.finish_round_counts
    ) == 0
    assert (
        selected.total_finish_elapsed_seconds_in_fight
        == 0
    )


def test_real_zero_scoring_population_produces_unanimous_draws() -> None:
    selected = run_actual(
        simulation_count=10,
        scoring_calibration=zero_scoring(),
        variability_calibration=zero_variability(),
    )

    assert selected.red_win_count == 0
    assert selected.blue_win_count == 0
    assert selected.draw_count == 10
    assert selected.unanimous_draw_count == 10
    assert selected.split_draw_count == 0
    assert selected.majority_draw_count == 0


def test_real_high_ko_population_aggregates_only_ko_finishes() -> None:
    selected = run_actual(
        simulation_count=50,
        seed_start=10_000,
        finish_calibration=knockout_finish_calibration(
            landed_probability=0.15,
            knockdown_probability=0.60,
        ),
    )

    assert selected.finish_count >= 49
    assert selected.ko_tko_count == selected.finish_count
    assert selected.submission_count == 0
    assert (
        sum(selected.finish_round_counts)
        == selected.finish_count
    )


def test_same_population_replays_identically() -> None:
    first = run_actual(
        simulation_count=10,
        seed_start=919,
    )
    second = run_actual(
        simulation_count=10,
        seed_start=919,
    )

    assert first == second


def test_changing_seed_start_changes_population_seed_identity() -> None:
    first = run_actual(
        simulation_count=3,
        seed_start=1,
    )
    second = run_actual(
        simulation_count=3,
        seed_start=2,
    )

    assert first.seed_start == 1
    assert second.seed_start == 2
    assert first != second


def test_runner_does_not_mutate_fighter_baselines() -> None:
    inputs = positional_inputs()
    originals = positional_inputs()

    run_matchup_monte_carlo(
        *inputs,
        **runner_kwargs(
            simulation_count=3,
        ),
    )

    assert inputs == originals


@pytest.mark.parametrize(
    "invalid_value",
    [
        1.0,
        True,
        "1",
    ],
)
def test_simulation_count_requires_exact_integer(
    invalid_value: object,
) -> None:
    kwargs = runner_kwargs()
    kwargs["simulation_count"] = invalid_value

    with pytest.raises(
        TypeError,
        match="simulation_count must be an integer",
    ):
        run_matchup_monte_carlo(
            *positional_inputs(),
            **kwargs,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        0,
        -1,
    ],
)
def test_simulation_count_must_be_positive(
    invalid_value: int,
) -> None:
    kwargs = runner_kwargs()
    kwargs["simulation_count"] = invalid_value

    with pytest.raises(
        ValueError,
        match="simulation_count must be positive",
    ):
        run_matchup_monte_carlo(
            *positional_inputs(),
            **kwargs,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        1.0,
        True,
        "1",
    ],
)
def test_seed_start_requires_exact_integer(
    invalid_value: object,
) -> None:
    kwargs = runner_kwargs()
    kwargs["seed_start"] = invalid_value

    with pytest.raises(
        TypeError,
        match="seed_start must be an integer",
    ):
        run_matchup_monte_carlo(
            *positional_inputs(),
            **kwargs,
        )


def test_seed_start_cannot_be_negative() -> None:
    kwargs = runner_kwargs()
    kwargs["seed_start"] = -1

    with pytest.raises(
        ValueError,
        match="seed_start cannot be negative",
    ):
        run_matchup_monte_carlo(
            *positional_inputs(),
            **kwargs,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        3.0,
        True,
        "3",
    ],
)
def test_scheduled_rounds_requires_exact_integer(
    invalid_value: object,
) -> None:
    kwargs = runner_kwargs()
    kwargs["scheduled_rounds"] = invalid_value

    with pytest.raises(
        TypeError,
        match="scheduled_rounds must be an integer",
    ):
        run_matchup_monte_carlo(
            *positional_inputs(),
            **kwargs,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        2,
        4,
    ],
)
def test_scheduled_rounds_supports_only_three_or_five(
    invalid_value: int,
) -> None:
    kwargs = runner_kwargs()
    kwargs["scheduled_rounds"] = invalid_value

    with pytest.raises(
        ValueError,
        match="scheduled_rounds must be 3 or 5",
    ):
        run_matchup_monte_carlo(
            *positional_inputs(),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("field_name", "expected_message"),
    [
        (
            "dynamic_state_calibration",
            "dynamic_state_calibration must be "
            "DynamicStateCalibration",
        ),
        (
            "phase_effect_calibration",
            "phase_effect_calibration must be "
            "DynamicEffectCalibration",
        ),
        (
            "transition_effect_calibration",
            "transition_effect_calibration must be "
            "DynamicTransitionEffectCalibration",
        ),
        (
            "finish_probability_calibration",
            "finish_probability_calibration must be "
            "FinishProbabilityCalibration",
        ),
        (
            "scoring_calibration",
            "scoring_calibration must be "
            "RoundScoringCalibration",
        ),
        (
            "variability_calibration",
            "variability_calibration must be "
            "JudgeVariabilityCalibration",
        ),
    ],
)
def test_runner_requires_calibration_contracts(
    field_name: str,
    expected_message: str,
) -> None:
    kwargs = runner_kwargs()
    kwargs[field_name] = "invalid"

    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        run_matchup_monte_carlo(
            *positional_inputs(),
            **kwargs,
        )


def test_shared_path_calibration_requires_contract_or_none() -> None:
    kwargs = runner_kwargs()
    kwargs["shared_path_calibration"] = "invalid"

    with pytest.raises(
        TypeError,
        match=(
            "shared_path_calibration must be "
            "SharedPathCalibration or None"
        ),
    ):
        run_matchup_monte_carlo(
            *positional_inputs(),
            **kwargs,
        )


def test_unsupported_finish_method_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_start = 40

    install_fake_population(
        monkeypatch,
        (
            fake_finish_result(
                winner=FighterSide.RED,
                method=FinishMethod.KO_TKO,
                round_number=1,
                elapsed_seconds_in_round=10,
            ),
        ),
        seed_start=seed_start,
    )

    original_method = FinishMethod.KO_TKO
    result = fake_finish_result(
        winner=FighterSide.RED,
        method=original_method,
        round_number=1,
        elapsed_seconds_in_round=10,
    )
    result.finish.method = "invalid"

    install_fake_population(
        monkeypatch,
        (
            result,
        ),
        seed_start=seed_start,
    )

    with pytest.raises(
        RuntimeError,
        match="unsupported finish method",
    ):
        run_matchup_monte_carlo(
            *positional_inputs(),
            **runner_kwargs(
                simulation_count=1,
                seed_start=seed_start,
            ),
        )


def test_unsupported_decision_type_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_start = 50
    result = fake_decision_result(
        winner=None,
        decision_type=DecisionType.UNANIMOUS_DRAW,
    )
    result.decision_type = "invalid"

    install_fake_population(
        monkeypatch,
        (
            result,
        ),
        seed_start=seed_start,
    )

    with pytest.raises(
        RuntimeError,
        match="unsupported decision type",
    ):
        run_matchup_monte_carlo(
            *positional_inputs(),
            **runner_kwargs(
                simulation_count=1,
                seed_start=seed_start,
            ),
        )
