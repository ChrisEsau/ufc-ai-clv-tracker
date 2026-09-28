"""Round scoring for RFS Monte Carlo V1.

Scoring priority follows the simulator architecture:

1. Effective striking and grappling
2. Effective aggression
3. Control as a secondary separator

The initial coefficients are transparent placeholders for later calibration.
They are not intended to reproduce official judging perfectly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from pipeline.simulation.rfs_mc_v1.segment_engine import (
    SegmentActivity,
    SegmentMatchupActivity,
)


class RoundWinner(str, Enum):
    """Possible round-scoring outcomes."""

    RED = "red"
    BLUE = "blue"
    EVEN = "even"


@dataclass(frozen=True)
class ScoringParameters:
    """Weights and margins used by the initial round scorer."""

    sig_strike_landed_weight: float = 1.00
    ground_strike_landed_weight: float = 1.20
    knockdown_weight: float = 6.00

    takedown_landed_weight: float = 1.50
    submission_attempt_weight: float = 2.25

    sig_strike_attempt_weight: float = 0.025
    takedown_attempt_weight: float = 0.10

    control_second_weight: float = 0.015

    # Close rounds should normally have a 10-9 winner. Reserve 10-10
    # scoring for an exact tie in the simulated judging score.
    even_round_margin: float = 0.0
    ten_eight_margin: float = 8.00

    # A 10-8 requires dominance beyond a large arithmetic margin.
    ten_eight_knockdown_advantage: int = 1
    ten_eight_submission_advantage: int = 2
    ten_eight_effective_margin: float = 15.00
    ten_eight_effective_ratio: float = 2.50

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")

        if self.ten_eight_margin <= self.even_round_margin:
            raise ValueError(
                "ten_eight_margin must exceed even_round_margin"
            )

        if self.ten_eight_effective_ratio < 1.0:
            raise ValueError(
                "ten_eight_effective_ratio must be at least 1.0"
            )


DEFAULT_SCORING_PARAMETERS = ScoringParameters()


@dataclass(frozen=True)
class FighterRoundMetrics:
    """Aggregated scoring metrics for one fighter in one round."""

    sig_str_attempted: int
    sig_str_landed: int
    ground_str_landed: int
    knockdowns: int

    td_attempted: int
    td_landed: int
    submission_attempts: int
    control_seconds: int

    effective_score: float
    aggression_score: float
    control_score: float
    total_score: float


@dataclass(frozen=True)
class RoundScore:
    """One simulated judges' round score."""

    round_number: int
    winner: RoundWinner

    red_points: int
    blue_points: int

    red_metrics: FighterRoundMetrics
    blue_metrics: FighterRoundMetrics

    score_margin: float

    def __post_init__(self) -> None:
        if self.round_number <= 0:
            raise ValueError("round_number must be positive")

        if self.red_points not in {8, 9, 10}:
            raise ValueError("red_points must be 8, 9, or 10")

        if self.blue_points not in {8, 9, 10}:
            raise ValueError("blue_points must be 8, 9, or 10")

        if max(self.red_points, self.blue_points) != 10:
            raise ValueError(
                "At least one fighter must receive 10 points"
            )


@dataclass(frozen=True)
class DecisionResult:
    """Aggregate scorecard for a path reaching the scheduled distance."""

    winner: str | None
    loser: str | None

    red_total: int
    blue_total: int

    red_rounds_won: int
    blue_rounds_won: int
    even_rounds: int

    round_scores: tuple[RoundScore, ...]

    def __post_init__(self) -> None:
        if self.winner not in {None, "red", "blue"}:
            raise ValueError("winner must be red, blue, or None")

        if self.loser not in {None, "red", "blue"}:
            raise ValueError("loser must be red, blue, or None")

        if self.winner is None and self.loser is not None:
            raise ValueError("Draw cannot have a loser")

        if self.winner is not None:
            if self.loser is None:
                raise ValueError("Decision winner requires a loser")
            if self.winner == self.loser:
                raise ValueError("winner and loser must differ")


def aggregate_fighter_round_metrics(
    activities: Iterable[SegmentActivity],
    parameters: ScoringParameters = DEFAULT_SCORING_PARAMETERS,
) -> FighterRoundMetrics:
    """Aggregate one fighter's segment activity into round scoring metrics."""

    activities = tuple(activities)

    sig_attempted = sum(
        activity.sig_str_attempted
        for activity in activities
    )
    sig_landed = sum(
        activity.sig_str_landed
        for activity in activities
    )
    ground_landed = sum(
        activity.ground_str_landed
        for activity in activities
    )
    knockdowns = sum(
        activity.knockdowns
        for activity in activities
    )

    td_attempted = sum(
        activity.td_attempted
        for activity in activities
    )
    td_landed = sum(
        activity.td_landed
        for activity in activities
    )
    submission_attempts = sum(
        activity.submission_attempts
        for activity in activities
    )
    control_seconds = sum(
        activity.control_seconds
        for activity in activities
    )

    effective_score = (
        sig_landed * parameters.sig_strike_landed_weight
        + ground_landed
        * parameters.ground_strike_landed_weight
        + knockdowns * parameters.knockdown_weight
        + td_landed * parameters.takedown_landed_weight
        + submission_attempts
        * parameters.submission_attempt_weight
    )

    aggression_score = (
        sig_attempted
        * parameters.sig_strike_attempt_weight
        + td_attempted
        * parameters.takedown_attempt_weight
    )

    control_score = (
        control_seconds * parameters.control_second_weight
    )

    total_score = (
        effective_score
        + aggression_score
        + control_score
    )

    return FighterRoundMetrics(
        sig_str_attempted=sig_attempted,
        sig_str_landed=sig_landed,
        ground_str_landed=ground_landed,
        knockdowns=knockdowns,
        td_attempted=td_attempted,
        td_landed=td_landed,
        submission_attempts=submission_attempts,
        control_seconds=control_seconds,
        effective_score=float(effective_score),
        aggression_score=float(aggression_score),
        control_score=float(control_score),
        total_score=float(total_score),
    )


def _is_ten_eight_round(
    *,
    winner_metrics: FighterRoundMetrics,
    loser_metrics: FighterRoundMetrics,
    score_margin: float,
    parameters: ScoringParameters,
) -> bool:
    """Return whether a winning performance merits a 10-8 round.

    A large weighted-score margin is necessary but not sufficient.
    The winner must also demonstrate meaningful damage, near-finish
    grappling, or overwhelming effective-offense dominance.
    """

    if score_margin < parameters.ten_eight_margin:
        return False

    knockdown_advantage = (
        winner_metrics.knockdowns
        - loser_metrics.knockdowns
    )
    submission_advantage = (
        winner_metrics.submission_attempts
        - loser_metrics.submission_attempts
    )
    effective_margin = (
        winner_metrics.effective_score
        - loser_metrics.effective_score
    )

    if loser_metrics.effective_score > 0:
        effective_ratio = (
            winner_metrics.effective_score
            / loser_metrics.effective_score
        )
    else:
        effective_ratio = (
            float("inf")
            if winner_metrics.effective_score > 0
            else 1.0
        )

    damage_dominance = (
        knockdown_advantage
        >= parameters.ten_eight_knockdown_advantage
    )
    submission_dominance = (
        submission_advantage
        >= parameters.ten_eight_submission_advantage
    )
    overwhelming_effective_offense = (
        effective_margin
        >= parameters.ten_eight_effective_margin
        and effective_ratio
        >= parameters.ten_eight_effective_ratio
    )

    return (
        damage_dominance
        or submission_dominance
        or overwhelming_effective_offense
    )


def score_round(
    *,
    round_number: int,
    segments: Iterable[SegmentMatchupActivity],
    parameters: ScoringParameters = DEFAULT_SCORING_PARAMETERS,
) -> RoundScore:
    """Score one completed simulated round."""

    round_segments = tuple(segments)

    if not round_segments:
        raise ValueError("Cannot score an empty round")

    if any(
        segment.round_number != round_number
        for segment in round_segments
    ):
        raise ValueError(
            "All supplied segments must match round_number"
        )

    red_metrics = aggregate_fighter_round_metrics(
        (segment.red for segment in round_segments),
        parameters,
    )
    blue_metrics = aggregate_fighter_round_metrics(
        (segment.blue for segment in round_segments),
        parameters,
    )

    margin = (
        red_metrics.total_score
        - blue_metrics.total_score
    )
    absolute_margin = abs(margin)

    if absolute_margin <= parameters.even_round_margin:
        winner = RoundWinner.EVEN
        red_points = 10
        blue_points = 10
    elif margin > 0:
        winner = RoundWinner.RED

        is_ten_eight = _is_ten_eight_round(
            winner_metrics=red_metrics,
            loser_metrics=blue_metrics,
            score_margin=absolute_margin,
            parameters=parameters,
        )

        red_points = 10
        blue_points = 8 if is_ten_eight else 9
    else:
        winner = RoundWinner.BLUE

        is_ten_eight = _is_ten_eight_round(
            winner_metrics=blue_metrics,
            loser_metrics=red_metrics,
            score_margin=absolute_margin,
            parameters=parameters,
        )

        blue_points = 10
        red_points = 8 if is_ten_eight else 9

    return RoundScore(
        round_number=round_number,
        winner=winner,
        red_points=red_points,
        blue_points=blue_points,
        red_metrics=red_metrics,
        blue_metrics=blue_metrics,
        score_margin=float(margin),
    )


def score_decision(
    segments: Iterable[SegmentMatchupActivity],
    *,
    scheduled_rounds: int,
    parameters: ScoringParameters = DEFAULT_SCORING_PARAMETERS,
) -> DecisionResult:
    """Score all completed rounds and determine a decision winner."""

    if scheduled_rounds not in {3, 5}:
        raise ValueError("scheduled_rounds must be 3 or 5")

    all_segments = tuple(segments)

    round_scores: list[RoundScore] = []

    for round_number in range(1, scheduled_rounds + 1):
        round_segments = tuple(
            segment
            for segment in all_segments
            if segment.round_number == round_number
        )

        if not round_segments:
            raise ValueError(
                f"Missing activity for round {round_number}"
            )

        round_scores.append(
            score_round(
                round_number=round_number,
                segments=round_segments,
                parameters=parameters,
            )
        )

    red_total = sum(
        round_score.red_points
        for round_score in round_scores
    )
    blue_total = sum(
        round_score.blue_points
        for round_score in round_scores
    )

    red_rounds_won = sum(
        round_score.winner is RoundWinner.RED
        for round_score in round_scores
    )
    blue_rounds_won = sum(
        round_score.winner is RoundWinner.BLUE
        for round_score in round_scores
    )
    even_rounds = sum(
        round_score.winner is RoundWinner.EVEN
        for round_score in round_scores
    )

    # Simplified simulator decision rule:
    # the fighter who wins more rounds wins the fight.
    #
    # Ten-eight scores remain available as round diagnostics, but point
    # totals do not override the number of rounds won.
    if red_rounds_won > blue_rounds_won:
        winner = "red"
        loser = "blue"
    elif blue_rounds_won > red_rounds_won:
        winner = "blue"
        loser = "red"
    else:
        # Exact 10-10 rounds can theoretically leave the round-win count
        # tied. Draws are disabled, so use the latest decisive round.
        decisive_rounds = [
            round_score
            for round_score in round_scores
            if round_score.winner is not RoundWinner.EVEN
        ]

        if decisive_rounds:
            winner = decisive_rounds[-1].winner.value
            loser = "blue" if winner == "red" else "red"
        else:
            # Fully identical scorecards are extremely rare. Use a stable
            # fallback so identical inputs remain reproducible.
            winner = "red"
            loser = "blue"

    return DecisionResult(
        winner=winner,
        loser=loser,
        red_total=red_total,
        blue_total=blue_total,
        red_rounds_won=red_rounds_won,
        blue_rounds_won=blue_rounds_won,
        even_rounds=even_rounds,
        round_scores=tuple(round_scores),
    )
