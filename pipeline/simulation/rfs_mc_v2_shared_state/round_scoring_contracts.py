"""Round-scoring contracts for RFS Monte Carlo V2.

This initial scoring layer models completed rounds under a no-foul 10-point
structure. Point deductions and technical decisions are intentionally deferred.

Supported round scores:

- 10-10
- 10-9 / 9-10
- 10-8 / 8-10
- 10-7 / 7-10

A complete judge scorecard contains exactly three or five sequential rounds.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
)


VALID_ROUND_SCORE_PAIRS = frozenset(
    {
        (10, 10),
        (10, 9),
        (9, 10),
        (10, 8),
        (8, 10),
        (10, 7),
        (7, 10),
    }
)


@dataclass(frozen=True)
class JudgeRoundScore:
    """One judge's score for one completed round."""

    round_number: int
    red_points: int
    blue_points: int

    def __post_init__(self) -> None:
        """Validate round identity and legal point combinations."""

        for name, value in (
            ("round_number", self.round_number),
            ("red_points", self.red_points),
            ("blue_points", self.blue_points),
        ):
            if type(value) is not int:
                raise TypeError(
                    f"{name} must be an integer"
                )

        if not 1 <= self.round_number <= 5:
            raise ValueError(
                "round_number must be between 1 and 5"
            )

        score_pair = (
            self.red_points,
            self.blue_points,
        )

        if score_pair not in VALID_ROUND_SCORE_PAIRS:
            raise ValueError(
                "unsupported no-foul round score: "
                f"{self.red_points}-{self.blue_points}"
            )

    @property
    def winner(self) -> FighterSide | None:
        """Return the round winner, or None for an even round."""

        if self.red_points > self.blue_points:
            return FighterSide.RED

        if self.blue_points > self.red_points:
            return FighterSide.BLUE

        return None

    @property
    def loser(self) -> FighterSide | None:
        """Return the round loser, or None for an even round."""

        if self.winner is None:
            return None

        return self.winner.opponent

    @property
    def point_margin(self) -> int:
        """Return the absolute score difference."""

        return abs(
            self.red_points
            - self.blue_points
        )

    @property
    def is_even(self) -> bool:
        """Return whether the round was scored evenly."""

        return self.winner is None


@dataclass(frozen=True)
class JudgeScorecard:
    """One complete judge scorecard for a scheduled-distance fight."""

    judge_number: int
    scheduled_rounds: int
    rounds: tuple[JudgeRoundScore, ...]

    def __post_init__(self) -> None:
        """Validate judge identity and complete round sequence."""

        if type(self.judge_number) is not int:
            raise TypeError(
                "judge_number must be an integer"
            )

        if not 1 <= self.judge_number <= 3:
            raise ValueError(
                "judge_number must be between 1 and 3"
            )

        if type(self.scheduled_rounds) is not int:
            raise TypeError(
                "scheduled_rounds must be an integer"
            )

        if self.scheduled_rounds not in {3, 5}:
            raise ValueError(
                "scheduled_rounds must be 3 or 5"
            )

        if not isinstance(
            self.rounds,
            tuple,
        ):
            raise TypeError(
                "rounds must be a tuple"
            )

        if len(self.rounds) != self.scheduled_rounds:
            raise ValueError(
                "scorecard must contain exactly one score "
                "for every scheduled round"
            )

        for index, score in enumerate(
            self.rounds,
            start=1,
        ):
            if not isinstance(
                score,
                JudgeRoundScore,
            ):
                raise TypeError(
                    "rounds must contain JudgeRoundScore values"
                )

            if score.round_number != index:
                raise ValueError(
                    "scorecard rounds must be sequential "
                    "starting at round one"
                )

    @property
    def red_total(self) -> int:
        """Return the red fighter's total points."""

        return sum(
            score.red_points
            for score in self.rounds
        )

    @property
    def blue_total(self) -> int:
        """Return the blue fighter's total points."""

        return sum(
            score.blue_points
            for score in self.rounds
        )

    @property
    def winner(self) -> FighterSide | None:
        """Return this judge's fight winner, or None for a draw."""

        if self.red_total > self.blue_total:
            return FighterSide.RED

        if self.blue_total > self.red_total:
            return FighterSide.BLUE

        return None

    @property
    def is_draw(self) -> bool:
        """Return whether this scorecard is tied."""

        return self.winner is None

    def round_wins(
        self,
        side: FighterSide,
    ) -> int:
        """Return how many rounds this judge awarded to one fighter."""

        if not isinstance(
            side,
            FighterSide,
        ):
            raise TypeError(
                "side must be FighterSide"
            )

        return sum(
            score.winner is side
            for score in self.rounds
        )
