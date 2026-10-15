"""Three-judge decision contracts for RFS Monte Carlo V2.

This layer resolves three complete judge scorecards into:

- unanimous decision
- split decision
- majority decision
- unanimous draw
- split draw
- majority draw

Technical decisions, fouls, overturned results, and incomplete-round scoring
remain outside this initial no-foul scheduled-distance model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
)
from pipeline.simulation.rfs_mc_v2_shared_state.round_scoring_contracts import (
    JudgeScorecard,
)


class DecisionType(str, Enum):
    """Supported three-judge scheduled-distance outcomes."""

    UNANIMOUS_DECISION = "unanimous_decision"
    SPLIT_DECISION = "split_decision"
    MAJORITY_DECISION = "majority_decision"

    UNANIMOUS_DRAW = "unanimous_draw"
    SPLIT_DRAW = "split_draw"
    MAJORITY_DRAW = "majority_draw"


def _classify_votes(
    *,
    red_votes: int,
    blue_votes: int,
    draw_votes: int,
) -> tuple[FighterSide | None, DecisionType]:
    """Classify one legal set of three judge votes."""

    if red_votes + blue_votes + draw_votes != 3:
        raise ValueError(
            "judge vote counts must total three"
        )

    if red_votes == 3:
        return (
            FighterSide.RED,
            DecisionType.UNANIMOUS_DECISION,
        )

    if blue_votes == 3:
        return (
            FighterSide.BLUE,
            DecisionType.UNANIMOUS_DECISION,
        )

    if red_votes == 2 and blue_votes == 1:
        return (
            FighterSide.RED,
            DecisionType.SPLIT_DECISION,
        )

    if blue_votes == 2 and red_votes == 1:
        return (
            FighterSide.BLUE,
            DecisionType.SPLIT_DECISION,
        )

    if red_votes == 2 and draw_votes == 1:
        return (
            FighterSide.RED,
            DecisionType.MAJORITY_DECISION,
        )

    if blue_votes == 2 and draw_votes == 1:
        return (
            FighterSide.BLUE,
            DecisionType.MAJORITY_DECISION,
        )

    if draw_votes == 3:
        return (
            None,
            DecisionType.UNANIMOUS_DRAW,
        )

    if draw_votes == 2:
        return (
            None,
            DecisionType.MAJORITY_DRAW,
        )

    if (
        red_votes == 1
        and blue_votes == 1
        and draw_votes == 1
    ):
        return (
            None,
            DecisionType.SPLIT_DRAW,
        )

    raise ValueError(
        "unsupported three-judge vote combination"
    )


@dataclass(frozen=True)
class DecisionResult:
    """Resolved scheduled-distance result from three scorecards."""

    scheduled_rounds: int
    scorecards: tuple[JudgeScorecard, ...]
    winner: FighterSide | None
    decision_type: DecisionType

    def __post_init__(self) -> None:
        """Validate scorecards and outcome consistency."""

        if type(self.scheduled_rounds) is not int:
            raise TypeError(
                "scheduled_rounds must be an integer"
            )

        if self.scheduled_rounds not in {3, 5}:
            raise ValueError(
                "scheduled_rounds must be 3 or 5"
            )

        if not isinstance(
            self.scorecards,
            tuple,
        ):
            raise TypeError(
                "scorecards must be a tuple"
            )

        if len(self.scorecards) != 3:
            raise ValueError(
                "decision result must contain exactly "
                "three judge scorecards"
            )

        for scorecard in self.scorecards:
            if not isinstance(
                scorecard,
                JudgeScorecard,
            ):
                raise TypeError(
                    "scorecards must contain "
                    "JudgeScorecard values"
                )

            if (
                scorecard.scheduled_rounds
                != self.scheduled_rounds
            ):
                raise ValueError(
                    "all scorecards must match "
                    "scheduled_rounds"
                )

        judge_numbers = {
            scorecard.judge_number
            for scorecard in self.scorecards
        }

        if judge_numbers != {1, 2, 3}:
            raise ValueError(
                "decision scorecards must contain "
                "judge numbers 1, 2, and 3 exactly once"
            )

        if (
            self.winner is not None
            and not isinstance(
                self.winner,
                FighterSide,
            )
        ):
            raise TypeError(
                "winner must be FighterSide or None"
            )

        if not isinstance(
            self.decision_type,
            DecisionType,
        ):
            raise TypeError(
                "decision_type must be DecisionType"
            )

        expected_winner, expected_type = _classify_votes(
            red_votes=self.red_votes,
            blue_votes=self.blue_votes,
            draw_votes=self.draw_votes,
        )

        if self.winner is not expected_winner:
            raise ValueError(
                "winner does not match judge scorecards"
            )

        if self.decision_type is not expected_type:
            raise ValueError(
                "decision_type does not match "
                "judge scorecards"
            )

    @property
    def red_votes(self) -> int:
        """Return scorecards awarded to red."""

        return sum(
            scorecard.winner is FighterSide.RED
            for scorecard in self.scorecards
        )

    @property
    def blue_votes(self) -> int:
        """Return scorecards awarded to blue."""

        return sum(
            scorecard.winner is FighterSide.BLUE
            for scorecard in self.scorecards
        )

    @property
    def draw_votes(self) -> int:
        """Return tied scorecards."""

        return sum(
            scorecard.winner is None
            for scorecard in self.scorecards
        )

    @property
    def is_draw(self) -> bool:
        """Return whether the official result is a draw."""

        return self.winner is None


def resolve_decision(
    scorecards: tuple[JudgeScorecard, ...],
) -> DecisionResult:
    """Resolve exactly three complete judge scorecards."""

    if not isinstance(
        scorecards,
        tuple,
    ):
        raise TypeError(
            "scorecards must be a tuple"
        )

    if len(scorecards) != 3:
        raise ValueError(
            "decision requires exactly three "
            "judge scorecards"
        )

    for scorecard in scorecards:
        if not isinstance(
            scorecard,
            JudgeScorecard,
        ):
            raise TypeError(
                "scorecards must contain "
                "JudgeScorecard values"
            )

    red_votes = sum(
        scorecard.winner is FighterSide.RED
        for scorecard in scorecards
    )
    blue_votes = sum(
        scorecard.winner is FighterSide.BLUE
        for scorecard in scorecards
    )
    draw_votes = sum(
        scorecard.winner is None
        for scorecard in scorecards
    )

    winner, decision_type = _classify_votes(
        red_votes=red_votes,
        blue_votes=blue_votes,
        draw_votes=draw_votes,
    )

    return DecisionResult(
        scheduled_rounds=scorecards[0].scheduled_rounds,
        scorecards=scorecards,
        winner=winner,
        decision_type=decision_type,
    )
