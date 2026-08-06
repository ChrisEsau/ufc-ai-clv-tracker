"""Tests for V2 three-judge decision contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
)
from pipeline.simulation.rfs_mc_v2_shared_state.decision_contracts import (
    DecisionResult,
    DecisionType,
    _classify_votes,
    resolve_decision,
)
from pipeline.simulation.rfs_mc_v2_shared_state.round_scoring_contracts import (
    JudgeRoundScore,
    JudgeScorecard,
)


def scorecard(
    *,
    judge_number: int,
    winner: FighterSide | None,
    scheduled_rounds: int = 3,
) -> JudgeScorecard:
    """Build a complete scorecard with the requested result."""

    if winner is FighterSide.RED:
        red_points = 10
        blue_points = 9
    elif winner is FighterSide.BLUE:
        red_points = 9
        blue_points = 10
    else:
        red_points = 10
        blue_points = 10

    return JudgeScorecard(
        judge_number=judge_number,
        scheduled_rounds=scheduled_rounds,
        rounds=tuple(
            JudgeRoundScore(
                round_number=round_number,
                red_points=red_points,
                blue_points=blue_points,
            )
            for round_number in range(
                1,
                scheduled_rounds + 1,
            )
        ),
    )


def scorecards_from_votes(
    votes: tuple[FighterSide | None, ...],
    *,
    scheduled_rounds: int = 3,
) -> tuple[JudgeScorecard, ...]:
    """Build three numbered scorecards from judge votes."""

    return tuple(
        scorecard(
            judge_number=judge_number,
            winner=winner,
            scheduled_rounds=scheduled_rounds,
        )
        for judge_number, winner in enumerate(
            votes,
            start=1,
        )
    )


def red_unanimous_scorecards(
    *,
    scheduled_rounds: int = 3,
) -> tuple[JudgeScorecard, ...]:
    """Build three scorecards awarding the fight to red."""

    return scorecards_from_votes(
        (
            FighterSide.RED,
            FighterSide.RED,
            FighterSide.RED,
        ),
        scheduled_rounds=scheduled_rounds,
    )


def red_unanimous_result() -> DecisionResult:
    """Build a valid red unanimous-decision result."""

    return DecisionResult(
        scheduled_rounds=3,
        scorecards=red_unanimous_scorecards(),
        winner=FighterSide.RED,
        decision_type=DecisionType.UNANIMOUS_DECISION,
    )


@pytest.mark.parametrize(
    (
        "votes",
        "expected_winner",
        "expected_type",
        "expected_red_votes",
        "expected_blue_votes",
        "expected_draw_votes",
    ),
    [
        (
            (
                FighterSide.RED,
                FighterSide.RED,
                FighterSide.RED,
            ),
            FighterSide.RED,
            DecisionType.UNANIMOUS_DECISION,
            3,
            0,
            0,
        ),
        (
            (
                FighterSide.BLUE,
                FighterSide.BLUE,
                FighterSide.BLUE,
            ),
            FighterSide.BLUE,
            DecisionType.UNANIMOUS_DECISION,
            0,
            3,
            0,
        ),
        (
            (
                FighterSide.RED,
                FighterSide.RED,
                FighterSide.BLUE,
            ),
            FighterSide.RED,
            DecisionType.SPLIT_DECISION,
            2,
            1,
            0,
        ),
        (
            (
                FighterSide.BLUE,
                FighterSide.BLUE,
                FighterSide.RED,
            ),
            FighterSide.BLUE,
            DecisionType.SPLIT_DECISION,
            1,
            2,
            0,
        ),
        (
            (
                FighterSide.RED,
                FighterSide.RED,
                None,
            ),
            FighterSide.RED,
            DecisionType.MAJORITY_DECISION,
            2,
            0,
            1,
        ),
        (
            (
                FighterSide.BLUE,
                FighterSide.BLUE,
                None,
            ),
            FighterSide.BLUE,
            DecisionType.MAJORITY_DECISION,
            0,
            2,
            1,
        ),
        (
            (
                None,
                None,
                None,
            ),
            None,
            DecisionType.UNANIMOUS_DRAW,
            0,
            0,
            3,
        ),
        (
            (
                FighterSide.RED,
                None,
                None,
            ),
            None,
            DecisionType.MAJORITY_DRAW,
            1,
            0,
            2,
        ),
        (
            (
                FighterSide.BLUE,
                None,
                None,
            ),
            None,
            DecisionType.MAJORITY_DRAW,
            0,
            1,
            2,
        ),
        (
            (
                FighterSide.RED,
                FighterSide.BLUE,
                None,
            ),
            None,
            DecisionType.SPLIT_DRAW,
            1,
            1,
            1,
        ),
    ],
)
def test_resolve_all_legal_three_judge_vote_combinations(
    votes: tuple[FighterSide | None, ...],
    expected_winner: FighterSide | None,
    expected_type: DecisionType,
    expected_red_votes: int,
    expected_blue_votes: int,
    expected_draw_votes: int,
) -> None:
    selected = resolve_decision(
        scorecards_from_votes(votes)
    )

    assert selected.winner is expected_winner
    assert selected.decision_type is expected_type
    assert selected.red_votes == expected_red_votes
    assert selected.blue_votes == expected_blue_votes
    assert selected.draw_votes == expected_draw_votes
    assert selected.is_draw is (
        expected_winner is None
    )


@pytest.mark.parametrize(
    "scheduled_rounds",
    [
        3,
        5,
    ],
)
def test_resolver_supports_three_and_five_round_fights(
    scheduled_rounds: int,
) -> None:
    selected = resolve_decision(
        red_unanimous_scorecards(
            scheduled_rounds=scheduled_rounds,
        )
    )

    assert selected.scheduled_rounds == scheduled_rounds
    assert all(
        card.scheduled_rounds == scheduled_rounds
        for card in selected.scorecards
    )


def test_scorecard_order_does_not_change_vote_resolution() -> None:
    cards = scorecards_from_votes(
        (
            FighterSide.RED,
            FighterSide.RED,
            FighterSide.BLUE,
        )
    )

    selected = resolve_decision(
        (
            cards[2],
            cards[0],
            cards[1],
        )
    )

    assert selected.winner is FighterSide.RED
    assert selected.decision_type is DecisionType.SPLIT_DECISION
    assert selected.red_votes == 2
    assert selected.blue_votes == 1


def test_decision_result_preserves_scorecards() -> None:
    cards = red_unanimous_scorecards()
    selected = resolve_decision(cards)

    assert selected.scorecards is cards


def test_decision_result_is_immutable() -> None:
    selected = red_unanimous_result()

    with pytest.raises(FrozenInstanceError):
        selected.winner = FighterSide.BLUE


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
        DecisionResult(
            scheduled_rounds=invalid_value,
            scorecards=red_unanimous_scorecards(),
            winner=FighterSide.RED,
            decision_type=DecisionType.UNANIMOUS_DECISION,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        1,
        4,
    ],
)
def test_decision_supports_only_three_or_five_rounds(
    invalid_value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="scheduled_rounds must be 3 or 5",
    ):
        DecisionResult(
            scheduled_rounds=invalid_value,
            scorecards=red_unanimous_scorecards(),
            winner=FighterSide.RED,
            decision_type=DecisionType.UNANIMOUS_DECISION,
        )


def test_scorecards_must_be_tuple() -> None:
    with pytest.raises(
        TypeError,
        match="scorecards must be a tuple",
    ):
        DecisionResult(
            scheduled_rounds=3,
            scorecards=list(
                red_unanimous_scorecards()
            ),
            winner=FighterSide.RED,
            decision_type=DecisionType.UNANIMOUS_DECISION,
        )


@pytest.mark.parametrize(
    "cards",
    [
        red_unanimous_scorecards()[:2],
        red_unanimous_scorecards()
        + (
            scorecard(
                judge_number=1,
                winner=FighterSide.RED,
            ),
        ),
    ],
)
def test_decision_result_requires_exactly_three_scorecards(
    cards: tuple[JudgeScorecard, ...],
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "decision result must contain exactly "
            "three judge scorecards"
        ),
    ):
        DecisionResult(
            scheduled_rounds=3,
            scorecards=cards,
            winner=FighterSide.RED,
            decision_type=DecisionType.UNANIMOUS_DECISION,
        )


def test_decision_result_requires_scorecard_contracts() -> None:
    cards = red_unanimous_scorecards()

    with pytest.raises(
        TypeError,
        match=(
            "scorecards must contain "
            "JudgeScorecard values"
        ),
    ):
        DecisionResult(
            scheduled_rounds=3,
            scorecards=(
                cards[0],
                "invalid",
                cards[2],
            ),
            winner=FighterSide.RED,
            decision_type=DecisionType.UNANIMOUS_DECISION,
        )


def test_all_scorecards_must_match_scheduled_rounds() -> None:
    three_round_cards = red_unanimous_scorecards()
    five_round_card = scorecard(
        judge_number=3,
        winner=FighterSide.RED,
        scheduled_rounds=5,
    )

    with pytest.raises(
        ValueError,
        match=(
            "all scorecards must match "
            "scheduled_rounds"
        ),
    ):
        DecisionResult(
            scheduled_rounds=3,
            scorecards=(
                three_round_cards[0],
                three_round_cards[1],
                five_round_card,
            ),
            winner=FighterSide.RED,
            decision_type=DecisionType.UNANIMOUS_DECISION,
        )


def test_judge_numbers_must_appear_exactly_once() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "decision scorecards must contain "
            "judge numbers 1, 2, and 3 exactly once"
        ),
    ):
        DecisionResult(
            scheduled_rounds=3,
            scorecards=(
                scorecard(
                    judge_number=1,
                    winner=FighterSide.RED,
                ),
                scorecard(
                    judge_number=1,
                    winner=FighterSide.RED,
                ),
                scorecard(
                    judge_number=3,
                    winner=FighterSide.RED,
                ),
            ),
            winner=FighterSide.RED,
            decision_type=DecisionType.UNANIMOUS_DECISION,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        "red",
        1,
    ],
)
def test_winner_requires_fighter_side_or_none(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="winner must be FighterSide or None",
    ):
        DecisionResult(
            scheduled_rounds=3,
            scorecards=red_unanimous_scorecards(),
            winner=invalid_value,
            decision_type=DecisionType.UNANIMOUS_DECISION,
        )


def test_decision_type_requires_enum() -> None:
    with pytest.raises(
        TypeError,
        match="decision_type must be DecisionType",
    ):
        DecisionResult(
            scheduled_rounds=3,
            scorecards=red_unanimous_scorecards(),
            winner=FighterSide.RED,
            decision_type="unanimous_decision",
        )


@pytest.mark.parametrize(
    "invalid_winner",
    [
        FighterSide.BLUE,
        None,
    ],
)
def test_winner_must_match_scorecards(
    invalid_winner: FighterSide | None,
) -> None:
    with pytest.raises(
        ValueError,
        match="winner does not match judge scorecards",
    ):
        DecisionResult(
            scheduled_rounds=3,
            scorecards=red_unanimous_scorecards(),
            winner=invalid_winner,
            decision_type=DecisionType.UNANIMOUS_DECISION,
        )


@pytest.mark.parametrize(
    "invalid_type",
    [
        DecisionType.SPLIT_DECISION,
        DecisionType.MAJORITY_DECISION,
        DecisionType.UNANIMOUS_DRAW,
    ],
)
def test_decision_type_must_match_scorecards(
    invalid_type: DecisionType,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "decision_type does not match "
            "judge scorecards"
        ),
    ):
        DecisionResult(
            scheduled_rounds=3,
            scorecards=red_unanimous_scorecards(),
            winner=FighterSide.RED,
            decision_type=invalid_type,
        )


def test_resolver_requires_scorecard_tuple() -> None:
    with pytest.raises(
        TypeError,
        match="scorecards must be a tuple",
    ):
        resolve_decision(
            list(
                red_unanimous_scorecards()
            )
        )


@pytest.mark.parametrize(
    "cards",
    [
        red_unanimous_scorecards()[:2],
        red_unanimous_scorecards()
        + red_unanimous_scorecards()[:1],
    ],
)
def test_resolver_requires_exactly_three_scorecards(
    cards: tuple[JudgeScorecard, ...],
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "decision requires exactly three "
            "judge scorecards"
        ),
    ):
        resolve_decision(cards)


def test_resolver_requires_scorecard_contracts() -> None:
    cards = red_unanimous_scorecards()

    with pytest.raises(
        TypeError,
        match=(
            "scorecards must contain "
            "JudgeScorecard values"
        ),
    ):
        resolve_decision(
            (
                cards[0],
                "invalid",
                cards[2],
            )
        )


def test_resolver_rejects_mixed_scheduled_rounds() -> None:
    cards = red_unanimous_scorecards()

    with pytest.raises(
        ValueError,
        match=(
            "all scorecards must match "
            "scheduled_rounds"
        ),
    ):
        resolve_decision(
            (
                cards[0],
                cards[1],
                scorecard(
                    judge_number=3,
                    winner=FighterSide.RED,
                    scheduled_rounds=5,
                ),
            )
        )


def test_resolver_rejects_duplicate_judge_numbers() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "decision scorecards must contain "
            "judge numbers 1, 2, and 3 exactly once"
        ),
    ):
        resolve_decision(
            (
                scorecard(
                    judge_number=1,
                    winner=FighterSide.RED,
                ),
                scorecard(
                    judge_number=1,
                    winner=FighterSide.RED,
                ),
                scorecard(
                    judge_number=3,
                    winner=FighterSide.RED,
                ),
            )
        )


@pytest.mark.parametrize(
    ("red_votes", "blue_votes", "draw_votes"),
    [
        (2, 0, 0),
        (4, 0, 0),
    ],
)
def test_vote_counts_must_total_three(
    red_votes: int,
    blue_votes: int,
    draw_votes: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="judge vote counts must total three",
    ):
        _classify_votes(
            red_votes=red_votes,
            blue_votes=blue_votes,
            draw_votes=draw_votes,
        )


def test_invalid_signed_vote_combination_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="unsupported three-judge vote combination",
    ):
        _classify_votes(
            red_votes=4,
            blue_votes=-1,
            draw_votes=0,
        )
