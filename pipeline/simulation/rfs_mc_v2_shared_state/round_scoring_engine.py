"""Deterministic round-scoring engine for RFS Monte Carlo V2.

The scoring hierarchy is:

1. damage and effective offense
2. control and defensive grappling as close-round tiebreakers
3. convert the resulting evidence margin into a no-foul 10-point score

The default calibration is an explicit modeling baseline. It is not yet
empirically calibrated against historical UFC judge scorecards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
)
from pipeline.simulation.rfs_mc_v2_shared_state.round_evidence import (
    FighterRoundEvidence,
    RoundEvidence,
)
from pipeline.simulation.rfs_mc_v2_shared_state.round_scoring_contracts import (
    JudgeRoundScore,
)


CALIBRATION_FIELDS = (
    "persistent_damage_weight",
    "acute_stress_weight",
    "knockdown_weight",
    "damaging_clinch_weight",
    "distance_landed_weight",
    "clinch_landed_weight",
    "ground_landed_weight",
    "submission_attempt_weight",
    "position_advancement_weight",
    "reversal_weight",
    "control_second_weight",
    "escape_weight",
    "scramble_weight",
    "primary_close_threshold",
    "secondary_scale",
    "even_round_threshold",
    "ten_eight_threshold",
    "ten_seven_threshold",
)


@dataclass(frozen=True)
class RoundScoringCalibration:
    """Weights and thresholds for deterministic round scoring."""

    persistent_damage_weight: float = 8.0
    acute_stress_weight: float = 1.5
    knockdown_weight: float = 4.0
    damaging_clinch_weight: float = 0.75

    distance_landed_weight: float = 0.20
    clinch_landed_weight: float = 0.25
    ground_landed_weight: float = 0.30

    submission_attempt_weight: float = 1.50
    position_advancement_weight: float = 0.75
    reversal_weight: float = 0.75

    control_second_weight: float = 0.01
    escape_weight: float = 0.10
    scramble_weight: float = 0.05

    primary_close_threshold: float = 1.0
    secondary_scale: float = 1.0

    even_round_threshold: float = 0.05
    ten_eight_threshold: float = 4.0
    ten_seven_threshold: float = 8.0

    def __post_init__(self) -> None:
        """Validate finite nonnegative weights and ordered thresholds."""

        for name in CALIBRATION_FIELDS:
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

        if (
            self.primary_close_threshold
            < self.even_round_threshold
        ):
            raise ValueError(
                "primary_close_threshold must be greater "
                "than or equal to even_round_threshold"
            )

        if (
            self.ten_eight_threshold
            <= self.even_round_threshold
        ):
            raise ValueError(
                "ten_eight_threshold must be greater "
                "than even_round_threshold"
            )

        if (
            self.ten_seven_threshold
            <= self.ten_eight_threshold
        ):
            raise ValueError(
                "ten_seven_threshold must be greater "
                "than ten_eight_threshold"
            )


@dataclass(frozen=True)
class FighterRoundScoreComponents:
    """Auditable scoring components for one fighter."""

    damage_score: float
    effective_striking_score: float
    effective_grappling_score: float
    control_score: float
    defensive_grappling_score: float

    def __post_init__(self) -> None:
        """Validate finite nonnegative component values."""

        for name in (
            "damage_score",
            "effective_striking_score",
            "effective_grappling_score",
            "control_score",
            "defensive_grappling_score",
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

    @property
    def primary_score(self) -> float:
        """Return damage and effective-offense evidence."""

        return (
            self.damage_score
            + self.effective_striking_score
            + self.effective_grappling_score
        )

    @property
    def secondary_score(self) -> float:
        """Return close-round tiebreak evidence."""

        return (
            self.control_score
            + self.defensive_grappling_score
        )

    @property
    def total_descriptive_score(self) -> float:
        """Return all components for reporting, not hierarchy."""

        return (
            self.primary_score
            + self.secondary_score
        )


@dataclass(frozen=True)
class RoundScoringAssessment:
    """Complete deterministic assessment for one round."""

    round_number: int
    red: FighterRoundScoreComponents
    blue: FighterRoundScoreComponents
    primary_margin: float
    secondary_margin: float
    comparison_margin: float
    secondary_tiebreak_used: bool
    score: JudgeRoundScore

    def __post_init__(self) -> None:
        """Validate nested contracts and round consistency."""

        if type(self.round_number) is not int:
            raise TypeError(
                "round_number must be an integer"
            )

        if not 1 <= self.round_number <= 5:
            raise ValueError(
                "round_number must be between 1 and 5"
            )

        if not isinstance(
            self.red,
            FighterRoundScoreComponents,
        ):
            raise TypeError(
                "red must be FighterRoundScoreComponents"
            )

        if not isinstance(
            self.blue,
            FighterRoundScoreComponents,
        ):
            raise TypeError(
                "blue must be FighterRoundScoreComponents"
            )

        for name in (
            "primary_margin",
            "secondary_margin",
            "comparison_margin",
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

        if type(self.secondary_tiebreak_used) is not bool:
            raise TypeError(
                "secondary_tiebreak_used must be boolean"
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
                "score round_number must match assessment"
            )

    @property
    def winner(self) -> FighterSide | None:
        """Return the deterministic round winner."""

        return self.score.winner

    @property
    def point_margin(self) -> int:
        """Return the 10-point score margin."""

        return self.score.point_margin


def calculate_fighter_round_score_components(
    evidence: FighterRoundEvidence,
    calibration: RoundScoringCalibration,
) -> FighterRoundScoreComponents:
    """Convert one fighter's evidence into scoring components."""

    if not isinstance(
        evidence,
        FighterRoundEvidence,
    ):
        raise TypeError(
            "evidence must be FighterRoundEvidence"
        )

    if not isinstance(
        calibration,
        RoundScoringCalibration,
    ):
        raise TypeError(
            "calibration must be RoundScoringCalibration"
        )

    damage_score = (
        evidence.persistent_damage_inflicted
        * calibration.persistent_damage_weight
        + evidence.acute_stress_inflicted
        * calibration.acute_stress_weight
        + evidence.knockdowns
        * calibration.knockdown_weight
        + evidence.damaging_clinch_strikes
        * calibration.damaging_clinch_weight
    )

    effective_striking_score = (
        evidence.distance_strikes_landed
        * calibration.distance_landed_weight
        + evidence.clinch_strikes_landed
        * calibration.clinch_landed_weight
        + evidence.ground_strikes_landed
        * calibration.ground_landed_weight
    )

    effective_grappling_score = (
        evidence.submission_attempts
        * calibration.submission_attempt_weight
        + evidence.position_advancements
        * calibration.position_advancement_weight
        + evidence.reversal_attempts
        * calibration.reversal_weight
    )

    control_score = (
        evidence.control_seconds
        * calibration.control_second_weight
    )

    defensive_grappling_score = (
        evidence.escape_attempts
        * calibration.escape_weight
        + evidence.scramble_attempts
        * calibration.scramble_weight
    )

    return FighterRoundScoreComponents(
        damage_score=damage_score,
        effective_striking_score=effective_striking_score,
        effective_grappling_score=effective_grappling_score,
        control_score=control_score,
        defensive_grappling_score=(
            defensive_grappling_score
        ),
    )


def _score_from_margin(
    *,
    round_number: int,
    comparison_margin: float,
    calibration: RoundScoringCalibration,
) -> JudgeRoundScore:
    """Convert a signed evidence margin into a 10-point score."""

    absolute_margin = abs(
        comparison_margin
    )

    if absolute_margin <= calibration.even_round_threshold:
        return JudgeRoundScore(
            round_number=round_number,
            red_points=10,
            blue_points=10,
        )

    if absolute_margin >= calibration.ten_seven_threshold:
        point_margin = 3
    elif absolute_margin >= calibration.ten_eight_threshold:
        point_margin = 2
    else:
        point_margin = 1

    if comparison_margin > 0.0:
        return JudgeRoundScore(
            round_number=round_number,
            red_points=10,
            blue_points=10 - point_margin,
        )

    return JudgeRoundScore(
        round_number=round_number,
        red_points=10 - point_margin,
        blue_points=10,
    )


def calculate_round_scoring_assessment(
    evidence: RoundEvidence,
    calibration: RoundScoringCalibration | None = None,
) -> RoundScoringAssessment:
    """Calculate one deterministic no-foul round score."""

    if not isinstance(
        evidence,
        RoundEvidence,
    ):
        raise TypeError(
            "evidence must be RoundEvidence"
        )

    selected_calibration = (
        calibration
        if calibration is not None
        else RoundScoringCalibration()
    )

    if not isinstance(
        selected_calibration,
        RoundScoringCalibration,
    ):
        raise TypeError(
            "calibration must be RoundScoringCalibration"
        )

    red_components = (
        calculate_fighter_round_score_components(
            evidence.red,
            selected_calibration,
        )
    )
    blue_components = (
        calculate_fighter_round_score_components(
            evidence.blue,
            selected_calibration,
        )
    )

    primary_margin = (
        red_components.primary_score
        - blue_components.primary_score
    )
    secondary_margin = (
        red_components.secondary_score
        - blue_components.secondary_score
    )

    secondary_tiebreak_used = (
        abs(primary_margin)
        <= selected_calibration.primary_close_threshold
    )

    if secondary_tiebreak_used:
        comparison_margin = (
            primary_margin
            + selected_calibration.secondary_scale
            * secondary_margin
        )
    else:
        comparison_margin = primary_margin

    score = _score_from_margin(
        round_number=evidence.round_number,
        comparison_margin=comparison_margin,
        calibration=selected_calibration,
    )

    return RoundScoringAssessment(
        round_number=evidence.round_number,
        red=red_components,
        blue=blue_components,
        primary_margin=primary_margin,
        secondary_margin=secondary_margin,
        comparison_margin=comparison_margin,
        secondary_tiebreak_used=secondary_tiebreak_used,
        score=score,
    )


def score_completed_round(
    evidence: RoundEvidence,
    calibration: RoundScoringCalibration | None = None,
) -> JudgeRoundScore:
    """Return only the deterministic 10-point round score."""

    return calculate_round_scoring_assessment(
        evidence,
        calibration,
    ).score
