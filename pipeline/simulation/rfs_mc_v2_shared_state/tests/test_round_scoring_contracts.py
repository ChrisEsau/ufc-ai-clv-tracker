"""Tests for V2 round-scoring contracts."""

from dataclasses import FrozenInstanceError

import pytest

from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
)
from pipeline.simulation.rfs_mc_v2_shared_state.round_scoring_contracts import (
    JudgeRoundScore,
    JudgeScorecard,
    VALID_ROUND_SCORE_PAIRS,
)


VALID_SCORE_CASES = [
    (
        10,
        10,
        None,
        None,
        0,
        True,
    ),
    (
        10,
        9,
        FighterSide.RED,
        FighterSide.BLUE,
        1,
        False,
    ),
    (
        9,
        10,
        FighterSide.BLUE,
        FighterSide.RED,
        1,
        False,
    ),
    (
        10,
        8,
        FighterSide.RED,
        FighterSide.BLUE,
        2,
        False,
    ),
    (
        8,
        10,
        FighterSide.BLUE,
        FighterSide.RED,
        2,
        False,
    ),
    (
        10,
        7,
        FighterSide.RED,
        FighterSide.BLUE,
        3,
        False,
    ),
    (
        7,
        10,
        FighterSide.BLUE,
        FighterSide.RED,
        3,
        False,
    ),
]


def round_score(
    *,
    round_number: int = 1,
    red_points: int = 10,
    blue_points: int = 9,
) -> JudgeRoundScore:
    """Build one valid judge round score."""

    return JudgeRoundScore(
        round_number=round_number,
        red_points=red_points,
        blue_points=blue_points,
    )


def red_three_round_scorecard() -> JudgeScorecard:
    """Build a valid three-round red-winning scorecard."""

    return JudgeScorecard(
        judge_number=1,
        scheduled_rounds=3,
        rounds=(
            round_score(
                round_number=1,
                red_points=10,
                blue_points=9,
            ),
            round_score(
                round_number=2,
                red_points=9,
                blue_points=10,
            ),
            round_score(
                round_number=3,
                red_points=10,
                blue_points=9,
            ),
        ),
    )


def blue_five_round_scorecard() -> JudgeScorecard:
    """Build a valid five-round blue-winning scorecard."""

    return JudgeScorecard(
        judge_number=2,
        scheduled_rounds=5,
        rounds=(
            round_score(
                round_number=1,
                red_points=9,
                blue_points=10,
            ),
            round_score(
                round_number=2,
                red_points=10,
                blue_points=9,
            ),
            round_score(
                round_number=3,
                red_points=9,
                blue_points=10,
            ),
            round_score(
                round_number=4,
                red_points=10,
                blue_points=8,
            ),
            round_score(
                round_number=5,
                red_points=8,
                blue_points=10,
            ),
        ),
    )


def draw_three_round_scorecard() -> JudgeScorecard:
    """Build a valid tied three-round scorecard."""

    return JudgeScorecard(
        judge_number=3,
        scheduled_rounds=3,
        rounds=(
            round_score(
                round_number=1,
                red_points=10,
                blue_points=9,
            ),
            round_score(
                round_number=2,
                red_points=9,
                blue_points=10,
            ),
            round_score(
                round_number=3,
                red_points=10,
                blue_points=10,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("red_points", "blue_points"),
    sorted(VALID_ROUND_SCORE_PAIRS),
)
def test_all_supported_round_scores_are_allowed(
    red_points: int,
    blue_points: int,
) -> None:
    selected = round_score(
        red_points=red_points,
        blue_points=blue_points,
    )

    assert selected.red_points == red_points
    assert selected.blue_points == blue_points


@pytest.mark.parametrize(
    "field_name",
    [
        "round_number",
        "red_points",
        "blue_points",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        1.0,
        True,
        "1",
    ],
)
def test_round_score_fields_require_exact_integers(
    field_name: str,
    invalid_value: object,
) -> None:
    values = {
        "round_number": 1,
        "red_points": 10,
        "blue_points": 9,
    }
    values[field_name] = invalid_value

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be an integer",
    ):
        JudgeRoundScore(
            **values
        )


@pytest.mark.parametrize(
    "invalid_round",
    [
        0,
        6,
    ],
)
def test_round_number_must_be_between_one_and_five(
    invalid_round: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="round_number must be between 1 and 5",
    ):
        round_score(
            round_number=invalid_round,
        )


@pytest.mark.parametrize(
    ("red_points", "blue_points"),
    [
        (9, 9),
        (8, 8),
        (10, 6),
        (9, 8),
    ],
)
def test_unsupported_no_foul_scores_are_rejected(
    red_points: int,
    blue_points: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "unsupported no-foul round score: "
            f"{red_points}-{blue_points}"
        ),
    ):
        round_score(
            red_points=red_points,
            blue_points=blue_points,
        )


@pytest.mark.parametrize(
    (
        "red_points",
        "blue_points",
        "expected_winner",
        "expected_loser",
        "expected_margin",
        "expected_even",
    ),
    VALID_SCORE_CASES,
)
def test_round_score_properties(
    red_points: int,
    blue_points: int,
    expected_winner: FighterSide | None,
    expected_loser: FighterSide | None,
    expected_margin: int,
    expected_even: bool,
) -> None:
    selected = round_score(
        red_points=red_points,
        blue_points=blue_points,
    )

    assert selected.winner is expected_winner
    assert selected.loser is expected_loser
    assert selected.point_margin == expected_margin
    assert selected.is_even is expected_even


def test_round_score_is_immutable() -> None:
    selected = round_score()

    with pytest.raises(FrozenInstanceError):
        selected.red_points = 9


def test_valid_three_round_scorecard() -> None:
    selected = red_three_round_scorecard()

    assert selected.judge_number == 1
    assert selected.scheduled_rounds == 3
    assert len(selected.rounds) == 3


def test_valid_five_round_scorecard() -> None:
    selected = blue_five_round_scorecard()

    assert selected.judge_number == 2
    assert selected.scheduled_rounds == 5
    assert len(selected.rounds) == 5


@pytest.mark.parametrize(
    "invalid_value",
    [
        1.0,
        True,
        "1",
    ],
)
def test_judge_number_requires_exact_integer(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="judge_number must be an integer",
    ):
        JudgeScorecard(
            judge_number=invalid_value,
            scheduled_rounds=3,
            rounds=red_three_round_scorecard().rounds,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        0,
        4,
    ],
)
def test_judge_number_must_be_between_one_and_three(
    invalid_value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="judge_number must be between 1 and 3",
    ):
        JudgeScorecard(
            judge_number=invalid_value,
            scheduled_rounds=3,
            rounds=red_three_round_scorecard().rounds,
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
        JudgeScorecard(
            judge_number=1,
            scheduled_rounds=invalid_value,
            rounds=red_three_round_scorecard().rounds,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        1,
        4,
    ],
)
def test_scorecard_supports_only_three_or_five_rounds(
    invalid_value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="scheduled_rounds must be 3 or 5",
    ):
        JudgeScorecard(
            judge_number=1,
            scheduled_rounds=invalid_value,
            rounds=red_three_round_scorecard().rounds,
        )


def test_scorecard_rounds_must_be_tuple() -> None:
    with pytest.raises(
        TypeError,
        match="rounds must be a tuple",
    ):
        JudgeScorecard(
            judge_number=1,
            scheduled_rounds=3,
            rounds=list(
                red_three_round_scorecard().rounds
            ),
        )


@pytest.mark.parametrize(
    "rounds",
    [
        (
            round_score(
                round_number=1,
            ),
            round_score(
                round_number=2,
            ),
        ),
        (
            round_score(
                round_number=1,
            ),
            round_score(
                round_number=2,
            ),
            round_score(
                round_number=3,
            ),
            round_score(
                round_number=4,
            ),
        ),
    ],
)
def test_scorecard_requires_exact_scheduled_round_count(
    rounds: tuple[JudgeRoundScore, ...],
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "scorecard must contain exactly one score "
            "for every scheduled round"
        ),
    ):
        JudgeScorecard(
            judge_number=1,
            scheduled_rounds=3,
            rounds=rounds,
        )


def test_scorecard_rounds_require_round_score_contracts() -> None:
    with pytest.raises(
        TypeError,
        match="rounds must contain JudgeRoundScore values",
    ):
        JudgeScorecard(
            judge_number=1,
            scheduled_rounds=3,
            rounds=(
                round_score(
                    round_number=1,
                ),
                "invalid",
                round_score(
                    round_number=3,
                ),
            ),
        )


@pytest.mark.parametrize(
    "rounds",
    [
        (
            round_score(
                round_number=1,
            ),
            round_score(
                round_number=3,
            ),
            round_score(
                round_number=2,
            ),
        ),
        (
            round_score(
                round_number=2,
            ),
            round_score(
                round_number=1,
            ),
            round_score(
                round_number=3,
            ),
        ),
    ],
)
def test_scorecard_rounds_must_be_sequential(
    rounds: tuple[JudgeRoundScore, ...],
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "scorecard rounds must be sequential "
            "starting at round one"
        ),
    ):
        JudgeScorecard(
            judge_number=1,
            scheduled_rounds=3,
            rounds=rounds,
        )


@pytest.mark.parametrize(
    (
        "selected",
        "expected_red_total",
        "expected_blue_total",
        "expected_winner",
        "expected_draw",
    ),
    [
        (
            red_three_round_scorecard(),
            29,
            28,
            FighterSide.RED,
            False,
        ),
        (
            blue_five_round_scorecard(),
            46,
            47,
            FighterSide.BLUE,
            False,
        ),
        (
            draw_three_round_scorecard(),
            29,
            29,
            None,
            True,
        ),
    ],
)
def test_scorecard_total_and_result_properties(
    selected: JudgeScorecard,
    expected_red_total: int,
    expected_blue_total: int,
    expected_winner: FighterSide | None,
    expected_draw: bool,
) -> None:
    assert selected.red_total == expected_red_total
    assert selected.blue_total == expected_blue_total
    assert selected.winner is expected_winner
    assert selected.is_draw is expected_draw


@pytest.mark.parametrize(
    ("side", "expected_wins"),
    [
        (
            FighterSide.RED,
            2,
        ),
        (
            FighterSide.BLUE,
            1,
        ),
    ],
)
def test_round_wins_counts_awarded_rounds(
    side: FighterSide,
    expected_wins: int,
) -> None:
    selected = red_three_round_scorecard()

    assert selected.round_wins(side) == expected_wins


def test_round_wins_requires_fighter_side() -> None:
    selected = red_three_round_scorecard()

    with pytest.raises(
        TypeError,
        match="side must be FighterSide",
    ):
        selected.round_wins(
            "red"
        )


def test_scorecard_is_immutable() -> None:
    selected = red_three_round_scorecard()

    with pytest.raises(FrozenInstanceError):
        selected.judge_number = 2
