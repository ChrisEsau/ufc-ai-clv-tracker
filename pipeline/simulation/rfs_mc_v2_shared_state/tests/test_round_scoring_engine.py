"""Tests for the V2 deterministic round-scoring engine."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

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
from pipeline.simulation.rfs_mc_v2_shared_state.round_scoring_engine import (
    CALIBRATION_FIELDS,
    FighterRoundScoreComponents,
    RoundScoringAssessment,
    RoundScoringCalibration,
    calculate_fighter_round_score_components,
    calculate_round_scoring_assessment,
    score_completed_round,
)


EVIDENCE_INTEGER_FIELDS = (
    "distance_strikes_attempted",
    "distance_strikes_landed",
    "clinch_strikes_attempted",
    "clinch_strikes_landed",
    "ground_strikes_attempted",
    "ground_strikes_landed",
    "knockdowns",
    "damaging_clinch_strikes",
    "control_seconds",
    "submission_attempts",
    "position_advancements",
    "escape_attempts",
    "reversal_attempts",
    "scramble_attempts",
)

EVIDENCE_FLOAT_FIELDS = (
    "persistent_damage_inflicted",
    "acute_stress_inflicted",
)

COMPONENT_FIELDS = (
    "damage_score",
    "effective_striking_score",
    "effective_grappling_score",
    "control_score",
    "defensive_grappling_score",
)


def fighter_evidence(
    **overrides: int | float,
) -> FighterRoundEvidence:
    """Build valid fighter evidence with optional overrides."""

    values: dict[str, int | float] = {
        name: 0
        for name in EVIDENCE_INTEGER_FIELDS
    }
    values.update(
        {
            name: 0.0
            for name in EVIDENCE_FLOAT_FIELDS
        }
    )
    values.update(overrides)

    return FighterRoundEvidence(**values)


def round_evidence(
    *,
    round_number: int = 1,
    red: FighterRoundEvidence | None = None,
    blue: FighterRoundEvidence | None = None,
) -> RoundEvidence:
    """Build one valid round-evidence contract."""

    return RoundEvidence(
        round_number=round_number,
        red=red or fighter_evidence(),
        blue=blue or fighter_evidence(),
    )


def components(
    **overrides: float,
) -> FighterRoundScoreComponents:
    """Build valid scoring components."""

    values = {
        name: 0.0
        for name in COMPONENT_FIELDS
    }
    values.update(overrides)

    return FighterRoundScoreComponents(**values)


def isolated_margin_calibration(
    **overrides: float,
) -> RoundScoringCalibration:
    """Build calibration where damage alone controls primary margin."""

    values = {
        "persistent_damage_weight": 1.0,
        "acute_stress_weight": 0.0,
        "knockdown_weight": 0.0,
        "damaging_clinch_weight": 0.0,
        "distance_landed_weight": 0.0,
        "clinch_landed_weight": 0.0,
        "ground_landed_weight": 0.0,
        "submission_attempt_weight": 0.0,
        "position_advancement_weight": 0.0,
        "reversal_weight": 0.0,
        "control_second_weight": 0.0,
        "escape_weight": 0.0,
        "scramble_weight": 0.0,
        "primary_close_threshold": 0.05,
        "secondary_scale": 1.0,
        "even_round_threshold": 0.05,
        "ten_eight_threshold": 4.0,
        "ten_seven_threshold": 8.0,
    }
    values.update(overrides)

    return RoundScoringCalibration(**values)


def assessment(
    **overrides: object,
) -> RoundScoringAssessment:
    """Build one valid scoring assessment."""

    values: dict[str, object] = {
        "round_number": 1,
        "red": components(),
        "blue": components(),
        "primary_margin": 0.0,
        "secondary_margin": 0.0,
        "comparison_margin": 0.0,
        "secondary_tiebreak_used": True,
        "score": JudgeRoundScore(
            round_number=1,
            red_points=10,
            blue_points=10,
        ),
    }
    values.update(overrides)

    return RoundScoringAssessment(**values)


@pytest.mark.parametrize(
    "field_name",
    CALIBRATION_FIELDS,
)
def test_calibration_fields_require_numeric_values(
    field_name: str,
) -> None:
    values = {
        name: getattr(
            RoundScoringCalibration(),
            name,
        )
        for name in CALIBRATION_FIELDS
    }
    values[field_name] = True

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        RoundScoringCalibration(**values)


@pytest.mark.parametrize(
    "field_name",
    CALIBRATION_FIELDS,
)
def test_calibration_fields_must_be_finite(
    field_name: str,
) -> None:
    values = {
        name: getattr(
            RoundScoringCalibration(),
            name,
        )
        for name in CALIBRATION_FIELDS
    }
    values[field_name] = float("nan")

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        RoundScoringCalibration(**values)


@pytest.mark.parametrize(
    "field_name",
    CALIBRATION_FIELDS,
)
def test_calibration_fields_cannot_be_negative(
    field_name: str,
) -> None:
    values = {
        name: getattr(
            RoundScoringCalibration(),
            name,
        )
        for name in CALIBRATION_FIELDS
    }
    values[field_name] = -0.01

    with pytest.raises(
        ValueError,
        match=f"{field_name} cannot be negative",
    ):
        RoundScoringCalibration(**values)


def test_primary_close_threshold_cannot_be_below_even_threshold() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "primary_close_threshold must be greater "
            "than or equal to even_round_threshold"
        ),
    ):
        isolated_margin_calibration(
            primary_close_threshold=0.04,
            even_round_threshold=0.05,
        )


def test_ten_eight_threshold_must_exceed_even_threshold() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "ten_eight_threshold must be greater "
            "than even_round_threshold"
        ),
    ):
        isolated_margin_calibration(
            ten_eight_threshold=0.05,
        )


def test_ten_seven_threshold_must_exceed_ten_eight_threshold() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "ten_seven_threshold must be greater "
            "than ten_eight_threshold"
        ),
    ):
        isolated_margin_calibration(
            ten_seven_threshold=4.0,
        )


def test_calibration_is_immutable() -> None:
    selected = RoundScoringCalibration()

    with pytest.raises(FrozenInstanceError):
        selected.knockdown_weight = 2.0


@pytest.mark.parametrize(
    "field_name",
    COMPONENT_FIELDS,
)
def test_component_fields_require_numeric_values(
    field_name: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        components(
            **{
                field_name: True,
            }
        )


@pytest.mark.parametrize(
    "field_name",
    COMPONENT_FIELDS,
)
def test_component_fields_must_be_finite(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        components(
            **{
                field_name: float("inf"),
            }
        )


@pytest.mark.parametrize(
    "field_name",
    COMPONENT_FIELDS,
)
def test_component_fields_cannot_be_negative(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} cannot be negative",
    ):
        components(
            **{
                field_name: -0.01,
            }
        )


def test_component_derived_scores() -> None:
    selected = components(
        damage_score=5.0,
        effective_striking_score=3.0,
        effective_grappling_score=2.0,
        control_score=1.5,
        defensive_grappling_score=0.5,
    )

    assert selected.primary_score == pytest.approx(10.0)
    assert selected.secondary_score == pytest.approx(2.0)
    assert selected.total_descriptive_score == pytest.approx(12.0)


def test_components_are_immutable() -> None:
    selected = components()

    with pytest.raises(FrozenInstanceError):
        selected.damage_score = 1.0


def test_exact_default_component_arithmetic() -> None:
    evidence = fighter_evidence(
        distance_strikes_landed=10,
        clinch_strikes_landed=4,
        ground_strikes_landed=3,
        knockdowns=2,
        damaging_clinch_strikes=1,
        control_seconds=40,
        submission_attempts=2,
        position_advancements=3,
        escape_attempts=2,
        reversal_attempts=1,
        scramble_attempts=3,
        persistent_damage_inflicted=2.0,
        acute_stress_inflicted=3.0,
    )

    selected = calculate_fighter_round_score_components(
        evidence,
        RoundScoringCalibration(),
    )

    assert selected.damage_score == pytest.approx(29.25)
    assert selected.effective_striking_score == pytest.approx(3.90)
    assert selected.effective_grappling_score == pytest.approx(6.00)
    assert selected.control_score == pytest.approx(0.40)
    assert selected.defensive_grappling_score == pytest.approx(0.35)
    assert selected.primary_score == pytest.approx(39.15)
    assert selected.secondary_score == pytest.approx(0.75)
    assert selected.total_descriptive_score == pytest.approx(39.90)


def test_component_calculation_requires_fighter_evidence() -> None:
    with pytest.raises(
        TypeError,
        match="evidence must be FighterRoundEvidence",
    ):
        calculate_fighter_round_score_components(
            "invalid",
            RoundScoringCalibration(),
        )


def test_component_calculation_requires_calibration() -> None:
    with pytest.raises(
        TypeError,
        match="calibration must be RoundScoringCalibration",
    ):
        calculate_fighter_round_score_components(
            fighter_evidence(),
            "invalid",
        )


@pytest.mark.parametrize(
    ("margin", "expected_red", "expected_blue"),
    [
        (0.00, 10, 10),
        (0.05, 10, 10),
        (-0.05, 10, 10),
        (0.051, 10, 9),
        (-0.051, 9, 10),
        (3.999, 10, 9),
        (-3.999, 9, 10),
        (4.000, 10, 8),
        (-4.000, 8, 10),
        (7.999, 10, 8),
        (-7.999, 8, 10),
        (8.000, 10, 7),
        (-8.000, 7, 10),
    ],
)
def test_score_margin_thresholds(
    margin: float,
    expected_red: int,
    expected_blue: int,
) -> None:
    selected = calculate_round_scoring_assessment(
        round_evidence(
            red=fighter_evidence(
                persistent_damage_inflicted=max(
                    margin,
                    0.0,
                )
            ),
            blue=fighter_evidence(
                persistent_damage_inflicted=max(
                    -margin,
                    0.0,
                )
            ),
        ),
        isolated_margin_calibration(),
    )

    assert selected.comparison_margin == pytest.approx(margin)
    assert selected.score.red_points == expected_red
    assert selected.score.blue_points == expected_blue


def test_secondary_evidence_is_ignored_when_primary_margin_is_not_close() -> None:
    calibration = isolated_margin_calibration(
        control_second_weight=1.0,
        primary_close_threshold=1.0,
    )

    selected = calculate_round_scoring_assessment(
        round_evidence(
            red=fighter_evidence(
                persistent_damage_inflicted=2.0,
            ),
            blue=fighter_evidence(
                control_seconds=100,
            ),
        ),
        calibration,
    )

    assert selected.primary_margin == pytest.approx(2.0)
    assert selected.secondary_margin == pytest.approx(-100.0)
    assert selected.secondary_tiebreak_used is False
    assert selected.comparison_margin == pytest.approx(2.0)
    assert selected.winner is FighterSide.RED


def test_secondary_evidence_is_used_when_primary_margin_is_close() -> None:
    calibration = isolated_margin_calibration(
        control_second_weight=0.10,
        primary_close_threshold=1.0,
    )

    selected = calculate_round_scoring_assessment(
        round_evidence(
            red=fighter_evidence(
                persistent_damage_inflicted=0.5,
            ),
            blue=fighter_evidence(
                control_seconds=20,
            ),
        ),
        calibration,
    )

    assert selected.primary_margin == pytest.approx(0.5)
    assert selected.secondary_margin == pytest.approx(-2.0)
    assert selected.secondary_tiebreak_used is True
    assert selected.comparison_margin == pytest.approx(-1.5)
    assert selected.winner is FighterSide.BLUE


def test_primary_close_threshold_is_inclusive() -> None:
    calibration = isolated_margin_calibration(
        control_second_weight=0.10,
        primary_close_threshold=1.0,
    )

    selected = calculate_round_scoring_assessment(
        round_evidence(
            red=fighter_evidence(
                persistent_damage_inflicted=1.0,
            ),
            blue=fighter_evidence(
                control_seconds=20,
            ),
        ),
        calibration,
    )

    assert selected.primary_margin == pytest.approx(1.0)
    assert selected.secondary_tiebreak_used is True
    assert selected.comparison_margin == pytest.approx(-1.0)
    assert selected.winner is FighterSide.BLUE


def test_secondary_scale_controls_tiebreak_influence() -> None:
    calibration = isolated_margin_calibration(
        control_second_weight=0.10,
        primary_close_threshold=1.0,
        secondary_scale=0.25,
    )

    selected = calculate_round_scoring_assessment(
        round_evidence(
            red=fighter_evidence(
                persistent_damage_inflicted=0.5,
            ),
            blue=fighter_evidence(
                control_seconds=20,
            ),
        ),
        calibration,
    )

    assert selected.primary_margin == pytest.approx(0.5)
    assert selected.secondary_margin == pytest.approx(-2.0)
    assert selected.comparison_margin == pytest.approx(0.0)
    assert selected.score.is_even is True


def test_swapping_fighters_mirrors_round_result() -> None:
    calibration = isolated_margin_calibration()

    red_result = calculate_round_scoring_assessment(
        round_evidence(
            red=fighter_evidence(
                persistent_damage_inflicted=5.0,
            ),
        ),
        calibration,
    )

    blue_result = calculate_round_scoring_assessment(
        round_evidence(
            blue=fighter_evidence(
                persistent_damage_inflicted=5.0,
            ),
        ),
        calibration,
    )

    assert red_result.score.red_points == 10
    assert red_result.score.blue_points == 8
    assert blue_result.score.red_points == 8
    assert blue_result.score.blue_points == 10


def test_default_identical_evidence_produces_even_round() -> None:
    selected = calculate_round_scoring_assessment(
        round_evidence()
    )

    assert selected.primary_margin == 0.0
    assert selected.secondary_margin == 0.0
    assert selected.comparison_margin == 0.0
    assert selected.secondary_tiebreak_used is True
    assert selected.score.red_points == 10
    assert selected.score.blue_points == 10


def test_assessment_uses_round_number_from_evidence() -> None:
    selected = calculate_round_scoring_assessment(
        round_evidence(
            round_number=4,
            red=fighter_evidence(
                persistent_damage_inflicted=1.0,
            ),
        ),
        isolated_margin_calibration(),
    )

    assert selected.round_number == 4
    assert selected.score.round_number == 4


def test_score_completed_round_returns_only_score_contract() -> None:
    selected = score_completed_round(
        round_evidence(
            red=fighter_evidence(
                persistent_damage_inflicted=4.0,
            ),
        ),
        isolated_margin_calibration(),
    )

    assert isinstance(
        selected,
        JudgeRoundScore,
    )
    assert selected.red_points == 10
    assert selected.blue_points == 8


def test_assessment_requires_round_evidence() -> None:
    with pytest.raises(
        TypeError,
        match="evidence must be RoundEvidence",
    ):
        calculate_round_scoring_assessment(
            "invalid"
        )


def test_assessment_requires_valid_optional_calibration() -> None:
    with pytest.raises(
        TypeError,
        match="calibration must be RoundScoringCalibration",
    ):
        calculate_round_scoring_assessment(
            round_evidence(),
            "invalid",
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        1.0,
        True,
        "1",
    ],
)
def test_assessment_round_number_requires_exact_integer(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="round_number must be an integer",
    ):
        assessment(
            round_number=invalid_value,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        0,
        6,
    ],
)
def test_assessment_round_number_must_be_between_one_and_five(
    invalid_value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="round_number must be between 1 and 5",
    ):
        assessment(
            round_number=invalid_value,
        )


@pytest.mark.parametrize(
    "side_name",
    [
        "red",
        "blue",
    ],
)
def test_assessment_requires_component_contracts(
    side_name: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=(
            f"{side_name} must be "
            "FighterRoundScoreComponents"
        ),
    ):
        assessment(
            **{
                side_name: "invalid",
            }
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "primary_margin",
        "secondary_margin",
        "comparison_margin",
    ],
)
def test_assessment_margins_require_numeric_values(
    field_name: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        assessment(
            **{
                field_name: True,
            }
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "primary_margin",
        "secondary_margin",
        "comparison_margin",
    ],
)
def test_assessment_margins_must_be_finite(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        assessment(
            **{
                field_name: float("nan"),
            }
        )


def test_secondary_tiebreak_flag_requires_boolean() -> None:
    with pytest.raises(
        TypeError,
        match="secondary_tiebreak_used must be boolean",
    ):
        assessment(
            secondary_tiebreak_used=1,
        )


def test_assessment_requires_judge_round_score() -> None:
    with pytest.raises(
        TypeError,
        match="score must be JudgeRoundScore",
    ):
        assessment(
            score="invalid",
        )


def test_assessment_score_round_must_match() -> None:
    with pytest.raises(
        ValueError,
        match="score round_number must match assessment",
    ):
        assessment(
            round_number=1,
            score=JudgeRoundScore(
                round_number=2,
                red_points=10,
                blue_points=9,
            ),
        )


def test_assessment_result_properties() -> None:
    selected = assessment(
        score=JudgeRoundScore(
            round_number=1,
            red_points=10,
            blue_points=8,
        ),
    )

    assert selected.winner is FighterSide.RED
    assert selected.point_margin == 2


def test_assessment_is_immutable() -> None:
    selected = assessment()

    with pytest.raises(FrozenInstanceError):
        selected.primary_margin = 1.0
