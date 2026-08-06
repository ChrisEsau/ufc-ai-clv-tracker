"""Tests for V2 matchup Monte Carlo population summaries."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from pipeline.simulation.rfs_mc_v2_shared_state.monte_carlo_summary import (
    MatchupMonteCarloSummary,
    ProbabilityEstimate,
)


COUNT_FIELDS = (
    "red_win_count",
    "blue_win_count",
    "draw_count",
    "finish_count",
    "scheduled_distance_count",
    "red_ko_tko_count",
    "blue_ko_tko_count",
    "red_submission_count",
    "blue_submission_count",
    "red_decision_count",
    "blue_decision_count",
    "unanimous_decision_count",
    "split_decision_count",
    "majority_decision_count",
    "unanimous_draw_count",
    "split_draw_count",
    "majority_draw_count",
    "total_finish_elapsed_seconds_in_fight",
)


def summary() -> MatchupMonteCarloSummary:
    """Build one internally consistent three-round summary."""

    return MatchupMonteCarloSummary(
        simulation_count=100,
        seed_start=500,
        scheduled_rounds=3,
        red_win_count=45,
        blue_win_count=45,
        draw_count=10,
        finish_count=40,
        scheduled_distance_count=60,
        red_ko_tko_count=12,
        blue_ko_tko_count=10,
        red_submission_count=8,
        blue_submission_count=10,
        red_decision_count=25,
        blue_decision_count=25,
        unanimous_decision_count=30,
        split_decision_count=12,
        majority_decision_count=8,
        unanimous_draw_count=4,
        split_draw_count=2,
        majority_draw_count=4,
        finish_round_counts=(
            15,
            15,
            10,
        ),
        total_finish_elapsed_seconds_in_fight=10_000,
    )


def five_round_summary() -> MatchupMonteCarloSummary:
    """Build one internally consistent five-round summary."""

    return MatchupMonteCarloSummary(
        simulation_count=20,
        seed_start=0,
        scheduled_rounds=5,
        red_win_count=9,
        blue_win_count=10,
        draw_count=1,
        finish_count=5,
        scheduled_distance_count=15,
        red_ko_tko_count=2,
        blue_ko_tko_count=1,
        red_submission_count=1,
        blue_submission_count=1,
        red_decision_count=6,
        blue_decision_count=8,
        unanimous_decision_count=8,
        split_decision_count=4,
        majority_decision_count=2,
        unanimous_draw_count=0,
        split_draw_count=1,
        majority_draw_count=0,
        finish_round_counts=(
            1,
            1,
            1,
            1,
            1,
        ),
        total_finish_elapsed_seconds_in_fight=2_500,
    )


def no_finish_summary() -> MatchupMonteCarloSummary:
    """Build a valid population containing only decisions."""

    return MatchupMonteCarloSummary(
        simulation_count=10,
        seed_start=0,
        scheduled_rounds=3,
        red_win_count=4,
        blue_win_count=5,
        draw_count=1,
        finish_count=0,
        scheduled_distance_count=10,
        red_ko_tko_count=0,
        blue_ko_tko_count=0,
        red_submission_count=0,
        blue_submission_count=0,
        red_decision_count=4,
        blue_decision_count=5,
        unanimous_decision_count=9,
        split_decision_count=0,
        majority_decision_count=0,
        unanimous_draw_count=1,
        split_draw_count=0,
        majority_draw_count=0,
        finish_round_counts=(
            0,
            0,
            0,
        ),
        total_finish_elapsed_seconds_in_fight=0,
    )


def all_finish_summary() -> MatchupMonteCarloSummary:
    """Build a valid population containing only finishes."""

    return MatchupMonteCarloSummary(
        simulation_count=10,
        seed_start=0,
        scheduled_rounds=3,
        red_win_count=6,
        blue_win_count=4,
        draw_count=0,
        finish_count=10,
        scheduled_distance_count=0,
        red_ko_tko_count=4,
        blue_ko_tko_count=2,
        red_submission_count=2,
        blue_submission_count=2,
        red_decision_count=0,
        blue_decision_count=0,
        unanimous_decision_count=0,
        split_decision_count=0,
        majority_decision_count=0,
        unanimous_draw_count=0,
        split_draw_count=0,
        majority_draw_count=0,
        finish_round_counts=(
            4,
            3,
            3,
        ),
        total_finish_elapsed_seconds_in_fight=3_000,
    )


@pytest.mark.parametrize(
    "invalid_value",
    [
        1.0,
        True,
        "1",
    ],
)
def test_probability_count_requires_exact_integer(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="count must be an integer",
    ):
        ProbabilityEstimate(
            count=invalid_value,
            total=10,
        )


def test_probability_count_cannot_be_negative() -> None:
    with pytest.raises(
        ValueError,
        match="count cannot be negative",
    ):
        ProbabilityEstimate(
            count=-1,
            total=10,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        10.0,
        True,
        "10",
    ],
)
def test_probability_total_requires_exact_integer(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="total must be an integer",
    ):
        ProbabilityEstimate(
            count=1,
            total=invalid_value,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        0,
        -1,
    ],
)
def test_probability_total_must_be_positive(
    invalid_value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="total must be positive",
    ):
        ProbabilityEstimate(
            count=0,
            total=invalid_value,
        )


def test_probability_count_cannot_exceed_total() -> None:
    with pytest.raises(
        ValueError,
        match="count cannot exceed total",
    ):
        ProbabilityEstimate(
            count=11,
            total=10,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        True,
        "0.95",
        None,
    ],
)
def test_confidence_level_requires_numeric_value(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="confidence_level must be numeric",
    ):
        ProbabilityEstimate(
            count=5,
            total=10,
            confidence_level=invalid_value,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_confidence_level_must_be_finite(
    invalid_value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="confidence_level must be finite",
    ):
        ProbabilityEstimate(
            count=5,
            total=10,
            confidence_level=invalid_value,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
    ],
)
def test_confidence_level_must_be_between_zero_and_one(
    invalid_value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "confidence_level must be between "
            "zero and one"
        ),
    ):
        ProbabilityEstimate(
            count=5,
            total=10,
            confidence_level=invalid_value,
        )


def test_probability_estimate_arithmetic() -> None:
    selected = ProbabilityEstimate(
        count=25,
        total=100,
    )

    assert selected.probability == pytest.approx(0.25)
    assert selected.standard_error == pytest.approx(
        0.0433012702
    )
    assert selected.z_score == pytest.approx(
        1.9599639845
    )


def test_wilson_interval_for_even_population() -> None:
    selected = ProbabilityEstimate(
        count=50,
        total=100,
    )

    assert selected.lower_bound == pytest.approx(
        0.4038315304
    )
    assert selected.upper_bound == pytest.approx(
        0.5961684696
    )
    assert selected.interval_width == pytest.approx(
        selected.upper_bound
        - selected.lower_bound
    )


@pytest.mark.parametrize(
    ("count", "expected_probability"),
    [
        (
            0,
            0.0,
        ),
        (
            100,
            1.0,
        ),
    ],
)
def test_wilson_interval_respects_probability_bounds(
    count: int,
    expected_probability: float,
) -> None:
    selected = ProbabilityEstimate(
        count=count,
        total=100,
    )

    assert selected.probability == expected_probability
    assert 0.0 <= selected.lower_bound <= 1.0
    assert 0.0 <= selected.upper_bound <= 1.0
    assert selected.lower_bound <= selected.upper_bound


def test_zero_success_wilson_lower_bound_is_exact_zero() -> None:
    selected = ProbabilityEstimate(
        count=0,
        total=100,
    )

    assert selected.lower_bound == 0.0
    assert (
        selected.lower_bound
        <= selected.probability
        <= selected.upper_bound
    )


def test_all_success_wilson_upper_bound_is_exact_one() -> None:
    selected = ProbabilityEstimate(
        count=100,
        total=100,
    )

    assert selected.upper_bound == 1.0
    assert (
        selected.lower_bound
        <= selected.probability
        <= selected.upper_bound
    )


def test_larger_population_produces_narrower_interval() -> None:
    small = ProbabilityEstimate(
        count=50,
        total=100,
    )
    large = ProbabilityEstimate(
        count=500,
        total=1_000,
    )

    assert large.interval_width < small.interval_width


def test_higher_confidence_produces_wider_interval() -> None:
    lower_confidence = ProbabilityEstimate(
        count=50,
        total=100,
        confidence_level=0.90,
    )
    higher_confidence = ProbabilityEstimate(
        count=50,
        total=100,
        confidence_level=0.99,
    )

    assert (
        higher_confidence.interval_width
        > lower_confidence.interval_width
    )


def test_probability_estimate_is_immutable() -> None:
    selected = ProbabilityEstimate(
        count=5,
        total=10,
    )

    with pytest.raises(FrozenInstanceError):
        selected.count = 6


def test_valid_three_round_summary() -> None:
    selected = summary()

    assert selected.simulation_count == 100
    assert selected.seed_start == 500
    assert selected.scheduled_rounds == 3
    assert len(selected.finish_round_counts) == 3


def test_valid_five_round_summary() -> None:
    selected = five_round_summary()

    assert selected.scheduled_rounds == 5
    assert len(selected.finish_round_counts) == 5
    assert sum(selected.finish_round_counts) == 5


@pytest.mark.parametrize(
    "invalid_value",
    [
        100.0,
        True,
        "100",
    ],
)
def test_simulation_count_requires_exact_integer(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="simulation_count must be an integer",
    ):
        replace(
            summary(),
            simulation_count=invalid_value,
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
    with pytest.raises(
        ValueError,
        match="simulation_count must be positive",
    ):
        replace(
            summary(),
            simulation_count=invalid_value,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        0.0,
        True,
        "0",
    ],
)
def test_seed_start_requires_exact_integer(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="seed_start must be an integer",
    ):
        replace(
            summary(),
            seed_start=invalid_value,
        )


def test_seed_start_cannot_be_negative() -> None:
    with pytest.raises(
        ValueError,
        match="seed_start cannot be negative",
    ):
        replace(
            summary(),
            seed_start=-1,
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
    with pytest.raises(
        TypeError,
        match="scheduled_rounds must be an integer",
    ):
        replace(
            summary(),
            scheduled_rounds=invalid_value,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        2,
        4,
    ],
)
def test_summary_supports_only_three_or_five_rounds(
    invalid_value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="scheduled_rounds must be 3 or 5",
    ):
        replace(
            summary(),
            scheduled_rounds=invalid_value,
        )


@pytest.mark.parametrize(
    "field_name",
    COUNT_FIELDS,
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        1.0,
        True,
        "1",
    ],
)
def test_summary_counts_require_exact_integers(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match=f"{field_name} must be an integer",
    ):
        replace(
            summary(),
            **{
                field_name: invalid_value,
            },
        )


@pytest.mark.parametrize(
    "field_name",
    COUNT_FIELDS,
)
def test_summary_counts_cannot_be_negative(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} cannot be negative",
    ):
        replace(
            summary(),
            **{
                field_name: -1,
            },
        )


def test_finish_round_counts_must_be_tuple() -> None:
    with pytest.raises(
        TypeError,
        match="finish_round_counts must be a tuple",
    ):
        replace(
            summary(),
            finish_round_counts=[
                15,
                15,
                10,
            ],
        )


def test_finish_round_counts_must_match_scheduled_rounds() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "finish_round_counts must contain one "
            "count per scheduled round"
        ),
    ):
        replace(
            summary(),
            finish_round_counts=(
                20,
                20,
            ),
        )


def test_finish_round_counts_require_exact_integers() -> None:
    with pytest.raises(
        TypeError,
        match=(
            r"finish_round_counts\[2\] "
            "must be an integer"
        ),
    ):
        replace(
            summary(),
            finish_round_counts=(
                15,
                15.0,
                10,
            ),
        )


def test_finish_round_counts_cannot_be_negative() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"finish_round_counts\[2\] "
            "cannot be negative"
        ),
    ):
        replace(
            summary(),
            finish_round_counts=(
                15,
                -1,
                26,
            ),
        )


def test_wins_and_draws_must_total_simulation_count() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "red wins, blue wins, and draws must "
            "total simulation_count"
        ),
    ):
        replace(
            summary(),
            draw_count=9,
        )


def test_terminal_branches_must_total_simulation_count() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "finishes and scheduled-distance results "
            "must total simulation_count"
        ),
    ):
        replace(
            summary(),
            scheduled_distance_count=59,
        )


def test_finish_methods_must_total_finish_count() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "KO/TKO and submission counts must "
            "total finish_count"
        ),
    ):
        replace(
            summary(),
            blue_submission_count=9,
        )


def test_red_methods_must_total_red_wins() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "red method counts must total "
            "red_win_count"
        ),
    ):
        replace(
            summary(),
            red_decision_count=24,
            unanimous_decision_count=29,
        )


def test_blue_methods_must_total_blue_wins() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "blue method counts must total "
            "blue_win_count"
        ),
    ):
        replace(
            summary(),
            blue_decision_count=24,
            unanimous_decision_count=29,
        )


def test_winning_decision_types_must_total_decision_wins() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "winning decision types must total "
            "fighter decision wins"
        ),
    ):
        replace(
            summary(),
            unanimous_decision_count=29,
        )


def test_draw_types_must_total_draw_count() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "draw decision types must total "
            "draw_count"
        ),
    ):
        replace(
            summary(),
            unanimous_draw_count=3,
        )


def test_finish_round_counts_must_total_finish_count() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "finish_round_counts must total "
            "finish_count"
        ),
    ):
        replace(
            summary(),
            finish_round_counts=(
                15,
                15,
                9,
            ),
        )


def test_no_finish_population_requires_zero_elapsed_total() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "finish elapsed-time total must be zero "
            "when there are no finishes"
        ),
    ):
        replace(
            no_finish_summary(),
            total_finish_elapsed_seconds_in_fight=1,
        )


@pytest.mark.parametrize(
    "invalid_total",
    [
        9,
        9_001,
    ],
)
def test_finish_elapsed_total_must_be_within_legal_bounds(
    invalid_total: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "finish elapsed-time total is outside "
            "legal fight-time bounds"
        ),
    ):
        replace(
            all_finish_summary(),
            total_finish_elapsed_seconds_in_fight=invalid_total,
        )


def test_summary_probability_builder() -> None:
    selected = summary()

    estimate = selected.probability(
        25,
        confidence_level=0.90,
    )

    assert estimate.count == 25
    assert estimate.total == 100
    assert estimate.confidence_level == 0.90
    assert estimate.probability == pytest.approx(0.25)


def test_primary_outcome_probabilities() -> None:
    selected = summary()

    assert (
        selected.red_win_probability.probability
        == pytest.approx(0.45)
    )
    assert (
        selected.blue_win_probability.probability
        == pytest.approx(0.45)
    )
    assert (
        selected.draw_probability.probability
        == pytest.approx(0.10)
    )
    assert (
        selected.finish_probability.probability
        == pytest.approx(0.40)
    )
    assert (
        selected.scheduled_distance_probability.probability
        == pytest.approx(0.60)
    )


def test_finish_method_counts_and_probabilities() -> None:
    selected = summary()

    assert selected.ko_tko_count == 22
    assert selected.submission_count == 18
    assert (
        selected.ko_tko_probability.probability
        == pytest.approx(0.22)
    )
    assert (
        selected.submission_probability.probability
        == pytest.approx(0.18)
    )


def test_fighter_method_probabilities() -> None:
    selected = summary()

    assert (
        selected.red_ko_tko_probability.probability
        == pytest.approx(0.12)
    )
    assert (
        selected.blue_ko_tko_probability.probability
        == pytest.approx(0.10)
    )
    assert (
        selected.red_submission_probability.probability
        == pytest.approx(0.08)
    )
    assert (
        selected.blue_submission_probability.probability
        == pytest.approx(0.10)
    )
    assert (
        selected.red_decision_probability.probability
        == pytest.approx(0.25)
    )
    assert (
        selected.blue_decision_probability.probability
        == pytest.approx(0.25)
    )


@pytest.mark.parametrize(
    ("round_number", "expected_probability"),
    [
        (
            1,
            0.15,
        ),
        (
            2,
            0.15,
        ),
        (
            3,
            0.10,
        ),
    ],
)
def test_finish_round_probabilities(
    round_number: int,
    expected_probability: float,
) -> None:
    selected = summary()

    estimate = selected.finish_in_round_probability(
        round_number
    )

    assert estimate.count == selected.finish_round_counts[
        round_number - 1
    ]
    assert estimate.total == selected.simulation_count
    assert estimate.probability == pytest.approx(
        expected_probability
    )


@pytest.mark.parametrize(
    "invalid_value",
    [
        1.0,
        True,
        "1",
    ],
)
def test_finish_round_probability_requires_exact_integer(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="round_number must be an integer",
    ):
        summary().finish_in_round_probability(
            invalid_value
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        0,
        4,
    ],
)
def test_finish_round_probability_requires_scheduled_round(
    invalid_value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "round_number must be within "
            "scheduled_rounds"
        ),
    ):
        summary().finish_in_round_probability(
            invalid_value
        )


def test_mean_finish_time() -> None:
    selected = summary()

    assert (
        selected.mean_finish_elapsed_seconds_in_fight
        == pytest.approx(250.0)
    )


def test_no_finish_population_has_no_mean_finish_time() -> None:
    assert (
        no_finish_summary()
        .mean_finish_elapsed_seconds_in_fight
        is None
    )


def test_summary_is_immutable() -> None:
    selected = summary()

    with pytest.raises(FrozenInstanceError):
        selected.red_win_count = 46
