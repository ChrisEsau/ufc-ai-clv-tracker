"""Judge-specific scorecard generation for RFS Monte Carlo V2.

The deterministic round-scoring assessment remains the authoritative evidence
interpretation. Judge variability is applied only to its final signed
comparison margin.

Each judge receives:

- one independently seeded fight-level bias, constant across all rounds
- one independently seeded round-specific noise value per round
- a bounded total margin adjustment

Positive adjustments favor red. Negative adjustments favor blue.

The default variability values are modeling baselines and are not yet
empirically calibrated against historical UFC scorecards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pipeline.simulation.rfs_mc_v2_shared_state.round_scoring_contracts import (
    JudgeRoundScore,
    JudgeScorecard,
)
from pipeline.simulation.rfs_mc_v2_shared_state.round_scoring_engine import (
    RoundScoringAssessment,
    RoundScoringCalibration,
    _score_from_margin,
)


@dataclass(frozen=True)
class JudgeVariabilityCalibration:
    """Judge-specific margin variability parameters."""

    fight_bias_stddev: float = 0.20
    round_noise_stddev: float = 0.35
    maximum_absolute_adjustment: float = 1.50

    def __post_init__(self) -> None:
        """Validate finite nonnegative variability parameters."""

        for name in (
            "fight_bias_stddev",
            "round_noise_stddev",
            "maximum_absolute_adjustment",
        ):
            value = getattr(self, name)

            if type(value) not in {
                int,
                float,
            }:
                raise TypeError(
                    f"{name} must be numeric"
                )

            selected = float(value)

            if not math.isfinite(selected):
                raise ValueError(
                    f"{name} must be finite"
                )

            if selected < 0.0:
                raise ValueError(
                    f"{name} cannot be negative"
                )


@dataclass(frozen=True)
class JudgeRoundScoringRecord:
    """Auditable judge adjustment and score for one round."""

    judge_number: int
    round_number: int

    base_comparison_margin: float
    fight_bias: float
    round_noise: float
    applied_adjustment: float
    adjusted_comparison_margin: float

    score: JudgeRoundScore

    def __post_init__(self) -> None:
        """Validate identity, finite margins, and score consistency."""

        for name in (
            "judge_number",
            "round_number",
        ):
            value = getattr(self, name)

            if type(value) is not int:
                raise TypeError(
                    f"{name} must be an integer"
                )

        if not 1 <= self.judge_number <= 3:
            raise ValueError(
                "judge_number must be between 1 and 3"
            )

        if not 1 <= self.round_number <= 5:
            raise ValueError(
                "round_number must be between 1 and 5"
            )

        for name in (
            "base_comparison_margin",
            "fight_bias",
            "round_noise",
            "applied_adjustment",
            "adjusted_comparison_margin",
        ):
            value = getattr(self, name)

            if type(value) not in {
                int,
                float,
            }:
                raise TypeError(
                    f"{name} must be numeric"
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    f"{name} must be finite"
                )

        if not isinstance(
            self.score,
            JudgeRoundScore,
        ):
            raise TypeError(
                "score must be JudgeRoundScore"
            )

        if self.score.round_number != self.round_number:
            raise ValueError(
                "score round_number must match record"
            )

        expected_adjusted_margin = (
            self.base_comparison_margin
            + self.applied_adjustment
        )

        if not math.isclose(
            self.adjusted_comparison_margin,
            expected_adjusted_margin,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "adjusted_comparison_margin must equal "
                "base margin plus applied adjustment"
            )


@dataclass(frozen=True)
class GeneratedJudgeScorecard:
    """One generated judge scorecard and its audit records."""

    judge_number: int
    scheduled_rounds: int
    fight_bias: float
    rounds: tuple[JudgeRoundScoringRecord, ...]
    scorecard: JudgeScorecard

    def __post_init__(self) -> None:
        """Validate complete sequential judge output."""

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

        if type(self.fight_bias) not in {
            int,
            float,
        }:
            raise TypeError(
                "fight_bias must be numeric"
            )

        if not math.isfinite(
            float(self.fight_bias)
        ):
            raise ValueError(
                "fight_bias must be finite"
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
                "generated judge must contain one record "
                "for every scheduled round"
            )

        for expected_round, record in enumerate(
            self.rounds,
            start=1,
        ):
            if not isinstance(
                record,
                JudgeRoundScoringRecord,
            ):
                raise TypeError(
                    "rounds must contain "
                    "JudgeRoundScoringRecord values"
                )

            if record.judge_number != self.judge_number:
                raise ValueError(
                    "round record judge_number must match "
                    "generated judge"
                )

            if record.round_number != expected_round:
                raise ValueError(
                    "judge round records must be sequential"
                )

            if not math.isclose(
                record.fight_bias,
                self.fight_bias,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "all round records must share the "
                    "generated judge fight_bias"
                )

        if not isinstance(
            self.scorecard,
            JudgeScorecard,
        ):
            raise TypeError(
                "scorecard must be JudgeScorecard"
            )

        if self.scorecard.judge_number != self.judge_number:
            raise ValueError(
                "scorecard judge_number must match "
                "generated judge"
            )

        if (
            self.scorecard.scheduled_rounds
            != self.scheduled_rounds
        ):
            raise ValueError(
                "scorecard scheduled_rounds must match "
                "generated judge"
            )

        expected_scores = tuple(
            record.score
            for record in self.rounds
        )

        if self.scorecard.rounds != expected_scores:
            raise ValueError(
                "scorecard rounds must match generated "
                "round records"
            )


@dataclass(frozen=True)
class JudgePanelScorecards:
    """Complete independently seeded three-judge panel."""

    scheduled_rounds: int
    seed: int
    judges: tuple[GeneratedJudgeScorecard, ...]

    def __post_init__(self) -> None:
        """Validate complete three-judge panel structure."""

        if type(self.scheduled_rounds) is not int:
            raise TypeError(
                "scheduled_rounds must be an integer"
            )

        if self.scheduled_rounds not in {3, 5}:
            raise ValueError(
                "scheduled_rounds must be 3 or 5"
            )

        if type(self.seed) is not int:
            raise TypeError(
                "seed must be an integer"
            )

        if self.seed < 0:
            raise ValueError(
                "seed cannot be negative"
            )

        if not isinstance(
            self.judges,
            tuple,
        ):
            raise TypeError(
                "judges must be a tuple"
            )

        if len(self.judges) != 3:
            raise ValueError(
                "judge panel must contain exactly "
                "three generated judges"
            )

        for judge in self.judges:
            if not isinstance(
                judge,
                GeneratedJudgeScorecard,
            ):
                raise TypeError(
                    "judges must contain "
                    "GeneratedJudgeScorecard values"
                )

            if (
                judge.scheduled_rounds
                != self.scheduled_rounds
            ):
                raise ValueError(
                    "all generated judges must match "
                    "scheduled_rounds"
                )

        judge_numbers = {
            judge.judge_number
            for judge in self.judges
        }

        if judge_numbers != {1, 2, 3}:
            raise ValueError(
                "judge panel must contain judge numbers "
                "1, 2, and 3 exactly once"
            )

    @property
    def scorecards(self) -> tuple[JudgeScorecard, ...]:
        """Return scorecards in generated judge order."""

        return tuple(
            judge.scorecard
            for judge in self.judges
        )


def _validate_assessments(
    assessments: tuple[RoundScoringAssessment, ...],
) -> int:
    """Validate a complete three- or five-round assessment sequence."""

    if not isinstance(
        assessments,
        tuple,
    ):
        raise TypeError(
            "assessments must be a tuple"
        )

    if len(assessments) not in {
        3,
        5,
    }:
        raise ValueError(
            "assessments must contain exactly "
            "three or five completed rounds"
        )

    for expected_round, assessment in enumerate(
        assessments,
        start=1,
    ):
        if not isinstance(
            assessment,
            RoundScoringAssessment,
        ):
            raise TypeError(
                "assessments must contain "
                "RoundScoringAssessment values"
            )

        if assessment.round_number != expected_round:
            raise ValueError(
                "assessments must be sequential "
                "starting at round one"
            )

    return len(assessments)


def generate_judge_panel_scorecards(
    assessments: tuple[RoundScoringAssessment, ...],
    *,
    seed: int,
    variability_calibration: (
        JudgeVariabilityCalibration | None
    ) = None,
    scoring_calibration: (
        RoundScoringCalibration | None
    ) = None,
) -> JudgePanelScorecards:
    """Generate three independently seeded judge scorecards."""

    scheduled_rounds = _validate_assessments(
        assessments
    )

    if type(seed) is not int:
        raise TypeError(
            "seed must be an integer"
        )

    if seed < 0:
        raise ValueError(
            "seed cannot be negative"
        )

    selected_variability = (
        variability_calibration
        if variability_calibration is not None
        else JudgeVariabilityCalibration()
    )

    if not isinstance(
        selected_variability,
        JudgeVariabilityCalibration,
    ):
        raise TypeError(
            "variability_calibration must be "
            "JudgeVariabilityCalibration"
        )

    selected_scoring = (
        scoring_calibration
        if scoring_calibration is not None
        else RoundScoringCalibration()
    )

    if not isinstance(
        selected_scoring,
        RoundScoringCalibration,
    ):
        raise TypeError(
            "scoring_calibration must be "
            "RoundScoringCalibration"
        )

    generated_judges: list[
        GeneratedJudgeScorecard
    ] = []

    for judge_number in range(
        1,
        4,
    ):
        # Judge streams are independent and stable. Adding activity,
        # transition, or finish draws elsewhere cannot perturb them.
        judge_rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    seed,
                    0x4A554447,
                    judge_number,
                ]
            )
        )

        fight_bias = float(
            judge_rng.normal(
                loc=0.0,
                scale=(
                    selected_variability
                    .fight_bias_stddev
                ),
            )
        )

        round_records: list[
            JudgeRoundScoringRecord
        ] = []

        for assessment in assessments:
            round_noise = float(
                judge_rng.normal(
                    loc=0.0,
                    scale=(
                        selected_variability
                        .round_noise_stddev
                    ),
                )
            )

            raw_adjustment = (
                fight_bias
                + round_noise
            )

            applied_adjustment = float(
                np.clip(
                    raw_adjustment,
                    -(
                        selected_variability
                        .maximum_absolute_adjustment
                    ),
                    (
                        selected_variability
                        .maximum_absolute_adjustment
                    ),
                )
            )

            adjusted_margin = (
                assessment.comparison_margin
                + applied_adjustment
            )

            score = _score_from_margin(
                round_number=assessment.round_number,
                comparison_margin=adjusted_margin,
                calibration=selected_scoring,
            )

            round_records.append(
                JudgeRoundScoringRecord(
                    judge_number=judge_number,
                    round_number=assessment.round_number,
                    base_comparison_margin=(
                        assessment.comparison_margin
                    ),
                    fight_bias=fight_bias,
                    round_noise=round_noise,
                    applied_adjustment=(
                        applied_adjustment
                    ),
                    adjusted_comparison_margin=(
                        adjusted_margin
                    ),
                    score=score,
                )
            )

        records_tuple = tuple(
            round_records
        )

        scorecard = JudgeScorecard(
            judge_number=judge_number,
            scheduled_rounds=scheduled_rounds,
            rounds=tuple(
                record.score
                for record in records_tuple
            ),
        )

        generated_judges.append(
            GeneratedJudgeScorecard(
                judge_number=judge_number,
                scheduled_rounds=scheduled_rounds,
                fight_bias=fight_bias,
                rounds=records_tuple,
                scorecard=scorecard,
            )
        )

    return JudgePanelScorecards(
        scheduled_rounds=scheduled_rounds,
        seed=seed,
        judges=tuple(
            generated_judges
        ),
    )


def generate_judge_scorecards(
    assessments: tuple[RoundScoringAssessment, ...],
    *,
    seed: int,
    variability_calibration: (
        JudgeVariabilityCalibration | None
    ) = None,
    scoring_calibration: (
        RoundScoringCalibration | None
    ) = None,
) -> tuple[JudgeScorecard, ...]:
    """Return only the three generated scorecards."""

    return generate_judge_panel_scorecards(
        assessments,
        seed=seed,
        variability_calibration=(
            variability_calibration
        ),
        scoring_calibration=scoring_calibration,
    ).scorecards
