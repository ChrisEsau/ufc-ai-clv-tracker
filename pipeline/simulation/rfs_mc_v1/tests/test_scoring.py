"""Tests for RFS Monte Carlo V1 round scoring."""

from pipeline.simulation.rfs_mc_v1.contracts import (
    FightPhase,
)
from pipeline.simulation.rfs_mc_v1.scoring import (
    RoundWinner,
    score_decision,
    score_round,
)
from pipeline.simulation.rfs_mc_v1.segment_engine import (
    SegmentActivity,
    SegmentMatchupActivity,
)


def activity(
    *,
    landed: int = 0,
    attempted: int | None = None,
    ground_landed: int = 0,
    knockdowns: int = 0,
    td_landed: int = 0,
    control: int = 0,
    submissions: int = 0,
) -> SegmentActivity:
    if attempted is None:
        attempted = landed

    return SegmentActivity(
        phase=FightPhase.DISTANCE,
        sig_str_attempted=attempted,
        sig_str_landed=landed,
        td_attempted=td_landed,
        td_landed=td_landed,
        control_seconds=control,
        ground_str_attempted=ground_landed,
        ground_str_landed=ground_landed,
        submission_attempts=submissions,
        knockdowns=knockdowns,
    )


def matchup_segment(
    round_number: int,
    segment_number: int,
    *,
    red: SegmentActivity,
    blue: SegmentActivity,
) -> SegmentMatchupActivity:
    return SegmentMatchupActivity(
        round_number=round_number,
        segment_number=segment_number,
        red=red,
        blue=blue,
    )


def test_more_effective_striking_wins_round() -> None:
    segments = [
        matchup_segment(
            1,
            segment_number,
            red=activity(landed=3, attempted=5),
            blue=activity(landed=1, attempted=5),
        )
        for segment_number in range(1, 11)
    ]

    score = score_round(
        round_number=1,
        segments=segments,
    )

    assert score.winner is RoundWinner.RED
    assert score.red_points == 10
    assert score.blue_points in {8, 9}


def test_knockdown_has_meaningful_scoring_value() -> None:
    segments = [
        matchup_segment(
            1,
            segment_number,
            red=activity(
                landed=1,
                knockdowns=int(segment_number == 10),
            ),
            blue=activity(landed=1),
        )
        for segment_number in range(1, 11)
    ]

    score = score_round(
        round_number=1,
        segments=segments,
    )

    assert score.winner is RoundWinner.RED


def test_identical_activity_scores_even_round() -> None:
    segments = [
        matchup_segment(
            1,
            segment_number,
            red=activity(landed=2, attempted=4),
            blue=activity(landed=2, attempted=4),
        )
        for segment_number in range(1, 11)
    ]

    score = score_round(
        round_number=1,
        segments=segments,
    )

    assert score.winner is RoundWinner.EVEN
    assert score.red_points == 10
    assert score.blue_points == 10


def test_score_decision_aggregates_rounds() -> None:
    segments = []

    for round_number in range(1, 4):
        red_landed = 3 if round_number in {1, 2} else 1
        blue_landed = 1 if round_number in {1, 2} else 3

        for segment_number in range(1, 11):
            segments.append(
                matchup_segment(
                    round_number,
                    segment_number,
                    red=activity(landed=red_landed),
                    blue=activity(landed=blue_landed),
                )
            )

    decision = score_decision(
        segments,
        scheduled_rounds=3,
    )

    assert decision.winner == "red"
    assert decision.loser == "blue"
    assert decision.red_rounds_won == 2
    assert decision.blue_rounds_won == 1
    assert len(decision.round_scores) == 3


def test_large_margin_can_produce_ten_eight() -> None:
    segments = [
        matchup_segment(
            1,
            segment_number,
            red=activity(
                landed=5,
                ground_landed=2,
                knockdowns=int(segment_number in {5, 10}),
            ),
            blue=activity(),
        )
        for segment_number in range(1, 11)
    ]

    score = score_round(
        round_number=1,
        segments=segments,
    )

    assert score.red_points == 10
    assert score.blue_points == 8
