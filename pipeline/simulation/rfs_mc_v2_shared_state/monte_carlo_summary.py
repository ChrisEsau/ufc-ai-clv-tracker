"""Matchup Monte Carlo population summaries for RFS Monte Carlo V2.

This module stores aggregated outcomes from many fully resolved fight paths.

The summary exposes probabilities for:

- red win, blue win, and draw
- finish versus scheduled distance
- KO/TKO and submission
- each fighter by KO/TKO, submission, or decision
- official decision classifications
- finish round distribution

Probability estimates include Wilson confidence intervals so callers can
distinguish simulation uncertainty from the model's predicted probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist


def _validate_exact_integer(
    name: str,
    value: object,
) -> int:
    """Validate and return one exact integer value."""

    if type(value) is not int:
        raise TypeError(
            f"{name} must be an integer"
        )

    return value


def _validate_nonnegative_count(
    name: str,
    value: object,
) -> int:
    """Validate and return one nonnegative integer count."""

    selected = _validate_exact_integer(
        name,
        value,
    )

    if selected < 0:
        raise ValueError(
            f"{name} cannot be negative"
        )

    return selected


@dataclass(frozen=True)
class ProbabilityEstimate:
    """One binomial probability estimate with a Wilson interval."""

    count: int
    total: int
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        """Validate count, population size, and confidence level."""

        _validate_nonnegative_count(
            "count",
            self.count,
        )
        _validate_exact_integer(
            "total",
            self.total,
        )

        if self.total <= 0:
            raise ValueError(
                "total must be positive"
            )

        if self.count > self.total:
            raise ValueError(
                "count cannot exceed total"
            )

        if type(self.confidence_level) not in {
            int,
            float,
        }:
            raise TypeError(
                "confidence_level must be numeric"
            )

        selected_confidence = float(
            self.confidence_level
        )

        if not math.isfinite(
            selected_confidence
        ):
            raise ValueError(
                "confidence_level must be finite"
            )

        if not 0.0 < selected_confidence < 1.0:
            raise ValueError(
                "confidence_level must be between "
                "zero and one"
            )

    @property
    def probability(self) -> float:
        """Return the observed population proportion."""

        return self.count / self.total

    @property
    def standard_error(self) -> float:
        """Return the unadjusted binomial standard error."""

        probability = self.probability

        return math.sqrt(
            probability
            * (1.0 - probability)
            / self.total
        )

    @property
    def z_score(self) -> float:
        """Return the two-sided normal critical value."""

        return NormalDist().inv_cdf(
            0.5
            + float(self.confidence_level) / 2.0
        )

    @property
    def lower_bound(self) -> float:
        """Return the lower Wilson confidence bound."""

        return self._wilson_interval()[0]

    @property
    def upper_bound(self) -> float:
        """Return the upper Wilson confidence bound."""

        return self._wilson_interval()[1]

    @property
    def interval_width(self) -> float:
        """Return the Wilson confidence interval width."""

        return (
            self.upper_bound
            - self.lower_bound
        )

    def _wilson_interval(
        self,
    ) -> tuple[float, float]:
        """Calculate a two-sided Wilson score interval."""

        probability = self.probability
        z_score = self.z_score
        z_squared = z_score * z_score
        total = float(self.total)

        denominator = (
            1.0
            + z_squared / total
        )

        center = (
            probability
            + z_squared / (2.0 * total)
        ) / denominator

        half_width = (
            z_score
            * math.sqrt(
                (
                    probability
                    * (1.0 - probability)
                    / total
                )
                + (
                    z_squared
                    / (4.0 * total * total)
                )
            )
            / denominator
        )

        lower_bound = max(
            0.0,
            center - half_width,
        )
        upper_bound = min(
            1.0,
            center + half_width,
        )

        # Wilson endpoints are mathematically exact at
        # zero successes and at an all-success population.
        # Explicitly clamp them to avoid floating-point
        # residue excluding the observed proportion.
        if self.count == 0:
            lower_bound = 0.0

        if self.count == self.total:
            upper_bound = 1.0

        return (
            lower_bound,
            upper_bound,
        )


@dataclass(frozen=True)
class MatchupMonteCarloSummary:
    """Aggregated outcomes from one matchup simulation population."""

    simulation_count: int
    seed_start: int
    scheduled_rounds: int

    red_win_count: int
    blue_win_count: int
    draw_count: int

    finish_count: int
    scheduled_distance_count: int

    red_ko_tko_count: int
    blue_ko_tko_count: int
    red_submission_count: int
    blue_submission_count: int

    red_decision_count: int
    blue_decision_count: int

    unanimous_decision_count: int
    split_decision_count: int
    majority_decision_count: int

    unanimous_draw_count: int
    split_draw_count: int
    majority_draw_count: int

    finish_round_counts: tuple[int, ...]
    total_finish_elapsed_seconds_in_fight: int

    def __post_init__(self) -> None:
        """Validate population counts and cross-field invariants."""

        _validate_exact_integer(
            "simulation_count",
            self.simulation_count,
        )

        if self.simulation_count <= 0:
            raise ValueError(
                "simulation_count must be positive"
            )

        _validate_nonnegative_count(
            "seed_start",
            self.seed_start,
        )

        _validate_exact_integer(
            "scheduled_rounds",
            self.scheduled_rounds,
        )

        if self.scheduled_rounds not in {
            3,
            5,
        }:
            raise ValueError(
                "scheduled_rounds must be 3 or 5"
            )

        count_fields = (
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

        for name in count_fields:
            _validate_nonnegative_count(
                name,
                getattr(self, name),
            )

        if not isinstance(
            self.finish_round_counts,
            tuple,
        ):
            raise TypeError(
                "finish_round_counts must be a tuple"
            )

        if (
            len(self.finish_round_counts)
            != self.scheduled_rounds
        ):
            raise ValueError(
                "finish_round_counts must contain one "
                "count per scheduled round"
            )

        for round_index, count in enumerate(
            self.finish_round_counts,
            start=1,
        ):
            _validate_nonnegative_count(
                f"finish_round_counts[{round_index}]",
                count,
            )

        if (
            self.red_win_count
            + self.blue_win_count
            + self.draw_count
            != self.simulation_count
        ):
            raise ValueError(
                "red wins, blue wins, and draws must "
                "total simulation_count"
            )

        if (
            self.finish_count
            + self.scheduled_distance_count
            != self.simulation_count
        ):
            raise ValueError(
                "finishes and scheduled-distance results "
                "must total simulation_count"
            )

        ko_tko_count = (
            self.red_ko_tko_count
            + self.blue_ko_tko_count
        )
        submission_count = (
            self.red_submission_count
            + self.blue_submission_count
        )

        if (
            ko_tko_count
            + submission_count
            != self.finish_count
        ):
            raise ValueError(
                "KO/TKO and submission counts must "
                "total finish_count"
            )

        if (
            self.red_ko_tko_count
            + self.red_submission_count
            + self.red_decision_count
            != self.red_win_count
        ):
            raise ValueError(
                "red method counts must total "
                "red_win_count"
            )

        if (
            self.blue_ko_tko_count
            + self.blue_submission_count
            + self.blue_decision_count
            != self.blue_win_count
        ):
            raise ValueError(
                "blue method counts must total "
                "blue_win_count"
            )

        winning_decision_count = (
            self.unanimous_decision_count
            + self.split_decision_count
            + self.majority_decision_count
        )
        draw_decision_count = (
            self.unanimous_draw_count
            + self.split_draw_count
            + self.majority_draw_count
        )

        if (
            winning_decision_count
            != (
                self.red_decision_count
                + self.blue_decision_count
            )
        ):
            raise ValueError(
                "winning decision types must total "
                "fighter decision wins"
            )

        if draw_decision_count != self.draw_count:
            raise ValueError(
                "draw decision types must total draw_count"
            )

        if (
            sum(self.finish_round_counts)
            != self.finish_count
        ):
            raise ValueError(
                "finish_round_counts must total "
                "finish_count"
            )

        if self.finish_count == 0:
            if (
                self.total_finish_elapsed_seconds_in_fight
                != 0
            ):
                raise ValueError(
                    "finish elapsed-time total must be zero "
                    "when there are no finishes"
                )
        else:
            minimum_total = self.finish_count
            maximum_total = (
                self.finish_count
                * self.scheduled_rounds
                * 300
            )

            if not (
                minimum_total
                <= self.total_finish_elapsed_seconds_in_fight
                <= maximum_total
            ):
                raise ValueError(
                    "finish elapsed-time total is outside "
                    "legal fight-time bounds"
                )

    def probability(
        self,
        count: int,
        *,
        confidence_level: float = 0.95,
    ) -> ProbabilityEstimate:
        """Build a probability estimate from one summary count."""

        return ProbabilityEstimate(
            count=count,
            total=self.simulation_count,
            confidence_level=confidence_level,
        )

    @property
    def red_win_probability(self) -> ProbabilityEstimate:
        """Return red's overall win probability."""

        return self.probability(
            self.red_win_count
        )

    @property
    def blue_win_probability(self) -> ProbabilityEstimate:
        """Return blue's overall win probability."""

        return self.probability(
            self.blue_win_count
        )

    @property
    def draw_probability(self) -> ProbabilityEstimate:
        """Return the official draw probability."""

        return self.probability(
            self.draw_count
        )

    @property
    def finish_probability(self) -> ProbabilityEstimate:
        """Return the probability of any finish."""

        return self.probability(
            self.finish_count
        )

    @property
    def scheduled_distance_probability(
        self,
    ) -> ProbabilityEstimate:
        """Return the probability of reaching scheduled distance."""

        return self.probability(
            self.scheduled_distance_count
        )

    @property
    def ko_tko_count(self) -> int:
        """Return total KO/TKO outcomes."""

        return (
            self.red_ko_tko_count
            + self.blue_ko_tko_count
        )

    @property
    def submission_count(self) -> int:
        """Return total submission outcomes."""

        return (
            self.red_submission_count
            + self.blue_submission_count
        )

    @property
    def ko_tko_probability(self) -> ProbabilityEstimate:
        """Return the probability of any KO/TKO."""

        return self.probability(
            self.ko_tko_count
        )

    @property
    def submission_probability(self) -> ProbabilityEstimate:
        """Return the probability of any submission."""

        return self.probability(
            self.submission_count
        )

    @property
    def red_ko_tko_probability(self) -> ProbabilityEstimate:
        """Return red by KO/TKO probability."""

        return self.probability(
            self.red_ko_tko_count
        )

    @property
    def blue_ko_tko_probability(self) -> ProbabilityEstimate:
        """Return blue by KO/TKO probability."""

        return self.probability(
            self.blue_ko_tko_count
        )

    @property
    def red_submission_probability(
        self,
    ) -> ProbabilityEstimate:
        """Return red by submission probability."""

        return self.probability(
            self.red_submission_count
        )

    @property
    def blue_submission_probability(
        self,
    ) -> ProbabilityEstimate:
        """Return blue by submission probability."""

        return self.probability(
            self.blue_submission_count
        )

    @property
    def red_decision_probability(
        self,
    ) -> ProbabilityEstimate:
        """Return red by decision probability."""

        return self.probability(
            self.red_decision_count
        )

    @property
    def blue_decision_probability(
        self,
    ) -> ProbabilityEstimate:
        """Return blue by decision probability."""

        return self.probability(
            self.blue_decision_count
        )

    def finish_in_round_probability(
        self,
        round_number: int,
    ) -> ProbabilityEstimate:
        """Return unconditional probability of a finish in one round."""

        _validate_exact_integer(
            "round_number",
            round_number,
        )

        if not 1 <= round_number <= self.scheduled_rounds:
            raise ValueError(
                "round_number must be within "
                "scheduled_rounds"
            )

        return self.probability(
            self.finish_round_counts[
                round_number - 1
            ]
        )

    @property
    def mean_finish_elapsed_seconds_in_fight(
        self,
    ) -> float | None:
        """Return mean elapsed fight time among finishes."""

        if self.finish_count == 0:
            return None

        return (
            self.total_finish_elapsed_seconds_in_fight
            / self.finish_count
        )
