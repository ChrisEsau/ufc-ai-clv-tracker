"""Tests for V2 judge-specific scorecard generation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import pipeline.simulation.rfs_mc_v2_shared_state.judge_scorecard_generator as generator_module
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
)
from pipeline.simulation.rfs_mc_v2_shared_state.decision_contracts import (
    DecisionType,
    resolve_decision,
)
from pipeline.simulation.rfs_mc_v2_shared_state.judge_scorecard_generator import (
    GeneratedJudgeScorecard,
    JudgePanelScorecards,
    JudgeRoundScoringRecord,
    JudgeVariabilityCalibration,
    generate_judge_panel_scorecards,
    generate_judge_scorecards,
)
from pipeline.simulation.rfs_mc_v2_shared_state.round_scoring_contracts import (
    JudgeRoundScore,
    JudgeScorecard,
)
from pipeline.simulation.rfs_mc_v2_shared_state.round_scoring_engine import (
    FighterRoundScoreComponents,
    RoundScoringAssessment,
    RoundScoringCalibration,
    _score_from_margin,
)


VARIABILITY_FIELDS = (
    "fight_bias_stddev",
    "round_noise_stddev",
    "maximum_absolute_adjustment",
)

MARGIN_FIELDS = (
    "base_comparison_margin",
    "fight_bias",
    "round_noise",
    "applied_adjustment",
    "adjusted_comparison_margin",
)


def zero_components() -> FighterRoundScoreComponents:
    """Build zero-valued deterministic scoring components."""

    return FighterRoundScoreComponents(
        damage_score=0.0,
        effective_striking_score=0.0,
        effective_grappling_score=0.0,
        control_score=0.0,
        defensive_grappling_score=0.0,
    )


def assessment(
    *,
    round_number: int = 1,
    margin: float = 0.0,
    scoring_calibration: RoundScoringCalibration | None = None,
) -> RoundScoringAssessment:
    """Build one valid deterministic round assessment."""

    selected_calibration = (
        scoring_calibration
        if scoring_calibration is not None
        else RoundScoringCalibration()
    )

    score = _score_from_margin(
        round_number=round_number,
        comparison_margin=margin,
        calibration=selected_calibration,
    )

    return RoundScoringAssessment(
        round_number=round_number,
        red=zero_components(),
        blue=zero_components(),
        primary_margin=margin,
        secondary_margin=0.0,
        comparison_margin=margin,
        secondary_tiebreak_used=False,
        score=score,
    )


def assessments(
    *,
    scheduled_rounds: int = 3,
    margins: tuple[float, ...] | None = None,
    scoring_calibration: RoundScoringCalibration | None = None,
) -> tuple[RoundScoringAssessment, ...]:
    """Build a sequential completed-round assessment tuple."""

    selected_margins = (
        margins
        if margins is not None
        else tuple(
            0.0
            for _ in range(scheduled_rounds)
        )
    )

    return tuple(
        assessment(
            round_number=round_number,
            margin=selected_margins[round_number - 1],
            scoring_calibration=scoring_calibration,
        )
        for round_number in range(
            1,
            scheduled_rounds + 1,
        )
    )


def zero_variability() -> JudgeVariabilityCalibration:
    """Build judge calibration with no random adjustment."""

    return JudgeVariabilityCalibration(
        fight_bias_stddev=0.0,
        round_noise_stddev=0.0,
        maximum_absolute_adjustment=0.0,
    )


def judge_round_record(
    *,
    judge_number: int = 1,
    round_number: int = 1,
    base_margin: float = 0.0,
    fight_bias: float = 0.0,
    round_noise: float = 0.0,
    applied_adjustment: float = 0.0,
    adjusted_margin: float | None = None,
    score: JudgeRoundScore | None = None,
) -> JudgeRoundScoringRecord:
    """Build one valid generated judge round record."""

    selected_adjusted_margin = (
        adjusted_margin
        if adjusted_margin is not None
        else base_margin + applied_adjustment
    )

    selected_score = (
        score
        if score is not None
        else _score_from_margin(
            round_number=round_number,
            comparison_margin=selected_adjusted_margin,
            calibration=RoundScoringCalibration(),
        )
    )

    return JudgeRoundScoringRecord(
        judge_number=judge_number,
        round_number=round_number,
        base_comparison_margin=base_margin,
        fight_bias=fight_bias,
        round_noise=round_noise,
        applied_adjustment=applied_adjustment,
        adjusted_comparison_margin=selected_adjusted_margin,
        score=selected_score,
    )


def generated_judge(
    *,
    judge_number: int = 1,
    scheduled_rounds: int = 3,
    fight_bias: float = 0.0,
) -> GeneratedJudgeScorecard:
    """Build one valid generated judge contract."""

    records = tuple(
        judge_round_record(
            judge_number=judge_number,
            round_number=round_number,
            fight_bias=fight_bias,
        )
        for round_number in range(
            1,
            scheduled_rounds + 1,
        )
    )

    scorecard = JudgeScorecard(
        judge_number=judge_number,
        scheduled_rounds=scheduled_rounds,
        rounds=tuple(
            record.score
            for record in records
        ),
    )

    return GeneratedJudgeScorecard(
        judge_number=judge_number,
        scheduled_rounds=scheduled_rounds,
        fight_bias=fight_bias,
        rounds=records,
        scorecard=scorecard,
    )


def judge_panel(
    *,
    scheduled_rounds: int = 3,
    seed: int = 2026,
) -> JudgePanelScorecards:
    """Build one valid complete judge panel."""

    return JudgePanelScorecards(
        scheduled_rounds=scheduled_rounds,
        seed=seed,
        judges=tuple(
            generated_judge(
                judge_number=judge_number,
                scheduled_rounds=scheduled_rounds,
            )
            for judge_number in range(
                1,
                4,
            )
        ),
    )


@pytest.mark.parametrize(
    "field_name",
    VARIABILITY_FIELDS,
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        True,
        "0.1",
        None,
    ],
)
def test_variability_fields_require_numeric_values(
    field_name: str,
    invalid_value: object,
) -> None:
    values = {
        "fight_bias_stddev": 0.20,
        "round_noise_stddev": 0.35,
        "maximum_absolute_adjustment": 1.50,
    }
    values[field_name] = invalid_value

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        JudgeVariabilityCalibration(**values)


@pytest.mark.parametrize(
    "field_name",
    VARIABILITY_FIELDS,
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_variability_fields_must_be_finite(
    field_name: str,
    invalid_value: float,
) -> None:
    values = {
        "fight_bias_stddev": 0.20,
        "round_noise_stddev": 0.35,
        "maximum_absolute_adjustment": 1.50,
    }
    values[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        JudgeVariabilityCalibration(**values)


@pytest.mark.parametrize(
    "field_name",
    VARIABILITY_FIELDS,
)
def test_variability_fields_cannot_be_negative(
    field_name: str,
) -> None:
    values = {
        "fight_bias_stddev": 0.20,
        "round_noise_stddev": 0.35,
        "maximum_absolute_adjustment": 1.50,
    }
    values[field_name] = -0.01

    with pytest.raises(
        ValueError,
        match=f"{field_name} cannot be negative",
    ):
        JudgeVariabilityCalibration(**values)


def test_variability_calibration_is_immutable() -> None:
    selected = JudgeVariabilityCalibration()

    with pytest.raises(FrozenInstanceError):
        selected.round_noise_stddev = 1.0


@pytest.mark.parametrize(
    "field_name",
    [
        "judge_number",
        "round_number",
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
def test_round_record_identity_requires_exact_integers(
    field_name: str,
    invalid_value: object,
) -> None:
    values = {
        "judge_number": 1,
        "round_number": 1,
    }
    values[field_name] = invalid_value

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be an integer",
    ):
        judge_round_record(**values)


@pytest.mark.parametrize(
    "invalid_value",
    [
        0,
        4,
    ],
)
def test_round_record_judge_number_range(
    invalid_value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="judge_number must be between 1 and 3",
    ):
        judge_round_record(
            judge_number=invalid_value,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        0,
        6,
    ],
)
def test_round_record_round_number_range(
    invalid_value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="round_number must be between 1 and 5",
    ):
        judge_round_record(
            round_number=invalid_value,
        )


@pytest.mark.parametrize(
    "field_name",
    MARGIN_FIELDS,
)
def test_round_record_margin_fields_require_numeric_values(
    field_name: str,
) -> None:
    values = {
        "base_margin": 0.0,
        "fight_bias": 0.0,
        "round_noise": 0.0,
        "applied_adjustment": 0.0,
        "adjusted_margin": 0.0,
    }

    argument_names = {
        "base_comparison_margin": "base_margin",
        "fight_bias": "fight_bias",
        "round_noise": "round_noise",
        "applied_adjustment": "applied_adjustment",
        "adjusted_comparison_margin": "adjusted_margin",
    }

    values[argument_names[field_name]] = True

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        judge_round_record(**values)


@pytest.mark.parametrize(
    "field_name",
    MARGIN_FIELDS,
)
def test_round_record_margin_fields_must_be_finite(
    field_name: str,
) -> None:
    values = {
        "base_margin": 0.0,
        "fight_bias": 0.0,
        "round_noise": 0.0,
        "applied_adjustment": 0.0,
        "adjusted_margin": 0.0,
    }

    argument_names = {
        "base_comparison_margin": "base_margin",
        "fight_bias": "fight_bias",
        "round_noise": "round_noise",
        "applied_adjustment": "applied_adjustment",
        "adjusted_comparison_margin": "adjusted_margin",
    }

    values[argument_names[field_name]] = float("nan")

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        judge_round_record(**values)


def test_round_record_requires_score_contract() -> None:
    with pytest.raises(
        TypeError,
        match="score must be JudgeRoundScore",
    ):
        judge_round_record(
            score="invalid",
        )


def test_round_record_score_round_must_match() -> None:
    with pytest.raises(
        ValueError,
        match="score round_number must match record",
    ):
        judge_round_record(
            round_number=1,
            score=JudgeRoundScore(
                round_number=2,
                red_points=10,
                blue_points=9,
            ),
        )


def test_adjusted_margin_must_match_base_plus_adjustment() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "adjusted_comparison_margin must equal "
            "base margin plus applied adjustment"
        ),
    ):
        judge_round_record(
            base_margin=1.0,
            applied_adjustment=0.5,
            adjusted_margin=1.6,
        )


def test_round_record_is_immutable() -> None:
    selected = judge_round_record()

    with pytest.raises(FrozenInstanceError):
        selected.applied_adjustment = 1.0


@pytest.mark.parametrize(
    "scheduled_rounds",
    [
        3,
        5,
    ],
)
def test_valid_generated_judge_contract(
    scheduled_rounds: int,
) -> None:
    selected = generated_judge(
        judge_number=2,
        scheduled_rounds=scheduled_rounds,
        fight_bias=0.25,
    )

    assert selected.judge_number == 2
    assert selected.scheduled_rounds == scheduled_rounds
    assert len(selected.rounds) == scheduled_rounds
    assert selected.scorecard.judge_number == 2


def test_generated_judge_rounds_must_be_tuple() -> None:
    selected = generated_judge()

    with pytest.raises(
        TypeError,
        match="rounds must be a tuple",
    ):
        GeneratedJudgeScorecard(
            judge_number=1,
            scheduled_rounds=3,
            fight_bias=0.0,
            rounds=list(selected.rounds),
            scorecard=selected.scorecard,
        )


def test_generated_judge_requires_one_record_per_round() -> None:
    selected = generated_judge()

    with pytest.raises(
        ValueError,
        match=(
            "generated judge must contain one record "
            "for every scheduled round"
        ),
    ):
        GeneratedJudgeScorecard(
            judge_number=1,
            scheduled_rounds=3,
            fight_bias=0.0,
            rounds=selected.rounds[:2],
            scorecard=selected.scorecard,
        )


def test_generated_judge_rounds_require_record_contracts() -> None:
    selected = generated_judge()

    with pytest.raises(
        TypeError,
        match=(
            "rounds must contain "
            "JudgeRoundScoringRecord values"
        ),
    ):
        GeneratedJudgeScorecard(
            judge_number=1,
            scheduled_rounds=3,
            fight_bias=0.0,
            rounds=(
                selected.rounds[0],
                "invalid",
                selected.rounds[2],
            ),
            scorecard=selected.scorecard,
        )


def test_generated_judge_record_judge_number_must_match() -> None:
    selected = generated_judge()

    bad_record = judge_round_record(
        judge_number=2,
        round_number=2,
    )

    with pytest.raises(
        ValueError,
        match=(
            "round record judge_number must match "
            "generated judge"
        ),
    ):
        GeneratedJudgeScorecard(
            judge_number=1,
            scheduled_rounds=3,
            fight_bias=0.0,
            rounds=(
                selected.rounds[0],
                bad_record,
                selected.rounds[2],
            ),
            scorecard=selected.scorecard,
        )


def test_generated_judge_rounds_must_be_sequential() -> None:
    selected = generated_judge()

    bad_record = judge_round_record(
        judge_number=1,
        round_number=3,
    )

    with pytest.raises(
        ValueError,
        match="judge round records must be sequential",
    ):
        GeneratedJudgeScorecard(
            judge_number=1,
            scheduled_rounds=3,
            fight_bias=0.0,
            rounds=(
                selected.rounds[0],
                bad_record,
                selected.rounds[2],
            ),
            scorecard=selected.scorecard,
        )


def test_all_round_records_must_share_fight_bias() -> None:
    selected = generated_judge(
        fight_bias=0.25,
    )

    bad_record = judge_round_record(
        judge_number=1,
        round_number=2,
        fight_bias=0.50,
    )

    with pytest.raises(
        ValueError,
        match=(
            "all round records must share the "
            "generated judge fight_bias"
        ),
    ):
        GeneratedJudgeScorecard(
            judge_number=1,
            scheduled_rounds=3,
            fight_bias=0.25,
            rounds=(
                selected.rounds[0],
                bad_record,
                selected.rounds[2],
            ),
            scorecard=selected.scorecard,
        )


def test_generated_judge_requires_scorecard_contract() -> None:
    selected = generated_judge()

    with pytest.raises(
        TypeError,
        match="scorecard must be JudgeScorecard",
    ):
        GeneratedJudgeScorecard(
            judge_number=1,
            scheduled_rounds=3,
            fight_bias=0.0,
            rounds=selected.rounds,
            scorecard="invalid",
        )


def test_generated_judge_scorecard_rounds_must_match_records() -> None:
    selected = generated_judge()

    mismatched_scorecard = JudgeScorecard(
        judge_number=1,
        scheduled_rounds=3,
        rounds=(
            JudgeRoundScore(
                round_number=1,
                red_points=10,
                blue_points=9,
            ),
            selected.rounds[1].score,
            selected.rounds[2].score,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "scorecard rounds must match generated "
            "round records"
        ),
    ):
        GeneratedJudgeScorecard(
            judge_number=1,
            scheduled_rounds=3,
            fight_bias=0.0,
            rounds=selected.rounds,
            scorecard=mismatched_scorecard,
        )


def test_generated_judge_is_immutable() -> None:
    selected = generated_judge()

    with pytest.raises(FrozenInstanceError):
        selected.fight_bias = 1.0


@pytest.mark.parametrize(
    "scheduled_rounds",
    [
        3,
        5,
    ],
)
def test_valid_judge_panel_contract(
    scheduled_rounds: int,
) -> None:
    selected = judge_panel(
        scheduled_rounds=scheduled_rounds,
    )

    assert selected.scheduled_rounds == scheduled_rounds
    assert len(selected.judges) == 3
    assert len(selected.scorecards) == 3


def test_panel_scorecards_property_preserves_judge_order() -> None:
    selected = judge_panel()

    assert selected.scorecards == tuple(
        judge.scorecard
        for judge in selected.judges
    )


def test_panel_requires_exactly_three_judges() -> None:
    selected = judge_panel()

    with pytest.raises(
        ValueError,
        match=(
            "judge panel must contain exactly "
            "three generated judges"
        ),
    ):
        JudgePanelScorecards(
            scheduled_rounds=3,
            seed=1,
            judges=selected.judges[:2],
        )


def test_panel_requires_generated_judge_contracts() -> None:
    selected = judge_panel()

    with pytest.raises(
        TypeError,
        match=(
            "judges must contain "
            "GeneratedJudgeScorecard values"
        ),
    ):
        JudgePanelScorecards(
            scheduled_rounds=3,
            seed=1,
            judges=(
                selected.judges[0],
                "invalid",
                selected.judges[2],
            ),
        )


def test_panel_judge_numbers_must_appear_exactly_once() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "judge panel must contain judge numbers "
            "1, 2, and 3 exactly once"
        ),
    ):
        JudgePanelScorecards(
            scheduled_rounds=3,
            seed=1,
            judges=(
                generated_judge(
                    judge_number=1,
                ),
                generated_judge(
                    judge_number=1,
                ),
                generated_judge(
                    judge_number=3,
                ),
            ),
        )


def test_panel_is_immutable() -> None:
    selected = judge_panel()

    with pytest.raises(FrozenInstanceError):
        selected.seed = 10


@pytest.mark.parametrize(
    "scheduled_rounds",
    [
        3,
        5,
    ],
)
def test_zero_variability_preserves_deterministic_round_scores(
    scheduled_rounds: int,
) -> None:
    selected_assessments = assessments(
        scheduled_rounds=scheduled_rounds,
        margins=tuple(
            (
                2.0
                if round_number % 2 == 1
                else -2.0
            )
            for round_number in range(
                1,
                scheduled_rounds + 1,
            )
        ),
    )

    panel = generate_judge_panel_scorecards(
        selected_assessments,
        seed=123,
        variability_calibration=zero_variability(),
    )

    expected_scores = tuple(
        selected.score
        for selected in selected_assessments
    )

    for judge in panel.judges:
        assert judge.scorecard.rounds == expected_scores
        assert judge.fight_bias == 0.0

        for record, selected_assessment in zip(
            judge.rounds,
            selected_assessments,
            strict=True,
        ):
            assert record.round_noise == 0.0
            assert record.applied_adjustment == 0.0
            assert (
                record.adjusted_comparison_margin
                == selected_assessment.comparison_margin
            )


def test_same_seed_replays_identical_panel() -> None:
    selected_assessments = assessments(
        margins=(
            0.2,
            -0.3,
            1.0,
        )
    )

    first = generate_judge_panel_scorecards(
        selected_assessments,
        seed=707,
    )
    second = generate_judge_panel_scorecards(
        selected_assessments,
        seed=707,
    )

    assert first == second


def test_different_seeds_produce_different_adjustments() -> None:
    selected_assessments = assessments()

    first = generate_judge_panel_scorecards(
        selected_assessments,
        seed=1,
    )
    second = generate_judge_panel_scorecards(
        selected_assessments,
        seed=2,
    )

    assert first != second


def test_each_judge_uses_one_constant_fight_bias() -> None:
    panel = generate_judge_panel_scorecards(
        assessments(
            scheduled_rounds=5,
        ),
        seed=919,
    )

    for judge in panel.judges:
        assert all(
            record.fight_bias == judge.fight_bias
            for record in judge.rounds
        )


def test_adjustment_arithmetic_is_preserved() -> None:
    selected_assessments = assessments(
        margins=(
            0.5,
            -0.5,
            1.0,
        )
    )

    panel = generate_judge_panel_scorecards(
        selected_assessments,
        seed=5150,
    )

    for judge in panel.judges:
        for record, selected_assessment in zip(
            judge.rounds,
            selected_assessments,
            strict=True,
        ):
            assert record.base_comparison_margin == (
                selected_assessment.comparison_margin
            )
            assert record.adjusted_comparison_margin == pytest.approx(
                record.base_comparison_margin
                + record.applied_adjustment
            )
            assert abs(record.applied_adjustment) <= 1.50


def test_bounded_adjustment_cannot_flip_dominant_rounds() -> None:
    selected_assessments = assessments(
        margins=(
            5.0,
            5.0,
            5.0,
        )
    )

    panel = generate_judge_panel_scorecards(
        selected_assessments,
        seed=111,
        variability_calibration=(
            JudgeVariabilityCalibration(
                fight_bias_stddev=100.0,
                round_noise_stddev=100.0,
                maximum_absolute_adjustment=1.0,
            )
        ),
    )

    for scorecard in panel.scorecards:
        assert scorecard.winner is FighterSide.RED


class FakeRng:
    """Deterministic normal sampler for clipping tests."""

    def __init__(
        self,
        values: tuple[float, ...],
    ) -> None:
        self._values = iter(values)

    def normal(
        self,
        *,
        loc: float,
        scale: float,
    ) -> float:
        del loc, scale

        return next(self._values)


def test_adjustments_are_clipped_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_rngs = iter(
        (
            FakeRng(
                (
                    5.0,
                    5.0,
                    -5.0,
                    0.0,
                )
            ),
            FakeRng(
                (
                    -5.0,
                    -5.0,
                    5.0,
                    0.0,
                )
            ),
            FakeRng(
                (
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            ),
        )
    )

    monkeypatch.setattr(
        generator_module.np.random,
        "default_rng",
        lambda seed_sequence: next(fake_rngs),
    )

    panel = generate_judge_panel_scorecards(
        assessments(),
        seed=10,
        variability_calibration=(
            JudgeVariabilityCalibration(
                fight_bias_stddev=1.0,
                round_noise_stddev=1.0,
                maximum_absolute_adjustment=1.5,
            )
        ),
    )

    judge_one = panel.judges[0]
    judge_two = panel.judges[1]
    judge_three = panel.judges[2]

    assert tuple(
        record.applied_adjustment
        for record in judge_one.rounds
    ) == (
        1.5,
        0.0,
        1.5,
    )

    assert tuple(
        record.applied_adjustment
        for record in judge_two.rounds
    ) == (
        -1.5,
        0.0,
        -1.5,
    )

    assert tuple(
        record.applied_adjustment
        for record in judge_three.rounds
    ) == (
        0.0,
        0.0,
        0.0,
    )


def test_judge_rng_streams_use_stable_independent_seed_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_entropy: list[object] = []

    def fake_default_rng(
        seed_sequence: np.random.SeedSequence,
    ) -> FakeRng:
        captured_entropy.append(
            seed_sequence.entropy
        )

        return FakeRng(
            (
                0.0,
                0.0,
                0.0,
                0.0,
            )
        )

    monkeypatch.setattr(
        generator_module.np.random,
        "default_rng",
        fake_default_rng,
    )

    generate_judge_panel_scorecards(
        assessments(),
        seed=2026,
    )

    assert captured_entropy == [
        [
            2026,
            0x4A554447,
            1,
        ],
        [
            2026,
            0x4A554447,
            2,
        ],
        [
            2026,
            0x4A554447,
            3,
        ],
    ]


def test_scorecard_wrapper_returns_panel_scorecards() -> None:
    selected_assessments = assessments(
        margins=(
            1.0,
            -1.0,
            1.0,
        )
    )

    panel = generate_judge_panel_scorecards(
        selected_assessments,
        seed=321,
    )
    scorecards = generate_judge_scorecards(
        selected_assessments,
        seed=321,
    )

    assert scorecards == panel.scorecards


def test_generated_scorecards_are_compatible_with_decision_resolution() -> None:
    selected_assessments = assessments(
        margins=(
            5.0,
            5.0,
            5.0,
        )
    )

    panel = generate_judge_panel_scorecards(
        selected_assessments,
        seed=404,
        variability_calibration=zero_variability(),
    )

    decision = resolve_decision(
        panel.scorecards
    )

    assert decision.winner is FighterSide.RED
    assert (
        decision.decision_type
        is DecisionType.UNANIMOUS_DECISION
    )


def test_assessments_must_be_tuple() -> None:
    with pytest.raises(
        TypeError,
        match="assessments must be a tuple",
    ):
        generate_judge_panel_scorecards(
            list(
                assessments()
            ),
            seed=1,
        )


@pytest.mark.parametrize(
    "round_count",
    [
        0,
        2,
        4,
        6,
    ],
)
def test_assessments_require_exactly_three_or_five_rounds(
    round_count: int,
) -> None:
    selected = tuple(
        assessment(
            round_number=round_number,
        )
        for round_number in range(
            1,
            min(round_count, 5) + 1,
        )
    )

    if round_count == 6:
        # Six distinct valid round numbers cannot exist because the
        # scoring contracts correctly support rounds one through five.
        # Duplicate a valid assessment to isolate length validation.
        selected = selected + (
            assessment(
                round_number=5,
            ),
        )

    with pytest.raises(
        ValueError,
        match=(
            "assessments must contain exactly "
            "three or five completed rounds"
        ),
    ):
        generate_judge_panel_scorecards(
            selected,
            seed=1,
        )


def test_assessments_require_scoring_assessment_contracts() -> None:
    selected = assessments()

    with pytest.raises(
        TypeError,
        match=(
            "assessments must contain "
            "RoundScoringAssessment values"
        ),
    ):
        generate_judge_panel_scorecards(
            (
                selected[0],
                "invalid",
                selected[2],
            ),
            seed=1,
        )


def test_assessments_must_be_sequential() -> None:
    selected = assessments()

    with pytest.raises(
        ValueError,
        match=(
            "assessments must be sequential "
            "starting at round one"
        ),
    ):
        generate_judge_panel_scorecards(
            (
                selected[0],
                assessment(
                    round_number=3,
                ),
                selected[2],
            ),
            seed=1,
        )


@pytest.mark.parametrize(
    "invalid_seed",
    [
        1.0,
        True,
        "1",
    ],
)
def test_generator_seed_requires_exact_integer(
    invalid_seed: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="seed must be an integer",
    ):
        generate_judge_panel_scorecards(
            assessments(),
            seed=invalid_seed,
        )


def test_generator_seed_cannot_be_negative() -> None:
    with pytest.raises(
        ValueError,
        match="seed cannot be negative",
    ):
        generate_judge_panel_scorecards(
            assessments(),
            seed=-1,
        )


def test_generator_requires_variability_calibration() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "variability_calibration must be "
            "JudgeVariabilityCalibration"
        ),
    ):
        generate_judge_panel_scorecards(
            assessments(),
            seed=1,
            variability_calibration="invalid",
        )


def test_generator_requires_scoring_calibration() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "scoring_calibration must be "
            "RoundScoringCalibration"
        ),
    ):
        generate_judge_panel_scorecards(
            assessments(),
            seed=1,
            scoring_calibration="invalid",
        )
