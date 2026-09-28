"""Tests for the V2 scheduled-distance result pipeline."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import pipeline.simulation.rfs_mc_v2_shared_state.finish_path_runner as finish_runner_module
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    SEGMENTS_PER_ROUND,
    FighterSide,
)
from pipeline.simulation.rfs_mc_v2_shared_state.decision_contracts import (
    DecisionType,
    resolve_decision,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_contracts import (
    FinishMethod,
    FinishResult,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_path_runner import (
    run_finish_enabled_dynamic_path,
)
from pipeline.simulation.rfs_mc_v2_shared_state.judge_scorecard_generator import (
    JudgeVariabilityCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.round_evidence import (
    RoundEvidence,
    calculate_round_evidence,
)
from pipeline.simulation.rfs_mc_v2_shared_state.round_scoring_contracts import (
    JudgeRoundScore,
    JudgeScorecard,
)
from pipeline.simulation.rfs_mc_v2_shared_state.round_scoring_engine import (
    RoundScoringCalibration,
    calculate_round_scoring_assessment,
)
from pipeline.simulation.rfs_mc_v2_shared_state.scheduled_distance_result import (
    ScheduledDistanceResult,
    ScheduledDistanceRoundResult,
    resolve_scheduled_distance_path,
)
from scripts.audit_rfs_mc_v2_finish_paths import (
    distance_only_transition_parameters,
    dynamic_parameters,
    phase_parameters,
    zero_finish_calibration,
    zero_phase_effect_calibration,
    zero_state_calibration,
    zero_transition_effect_calibration,
)


def run_full_path(
    *,
    scheduled_rounds: int = 3,
    seed: int = 2026,
):
    """Run one controlled path that reaches scheduled distance."""

    transition = distance_only_transition_parameters()
    phase = phase_parameters()
    dynamic = dynamic_parameters()

    return run_finish_enabled_dynamic_path(
        transition,
        transition,
        phase,
        phase,
        dynamic,
        dynamic,
        dynamic_state_calibration=zero_state_calibration(),
        phase_effect_calibration=zero_phase_effect_calibration(),
        transition_effect_calibration=(
            zero_transition_effect_calibration()
        ),
        finish_probability_calibration=zero_finish_calibration(),
        scheduled_rounds=scheduled_rounds,
        seed=seed,
    )


def zero_variability() -> JudgeVariabilityCalibration:
    """Return judge calibration with no random adjustment."""

    return JudgeVariabilityCalibration(
        fight_bias_stddev=0.0,
        round_noise_stddev=0.0,
        maximum_absolute_adjustment=0.0,
    )


def resolve_full_path(
    *,
    scheduled_rounds: int = 3,
    seed: int = 2026,
    scoring_calibration: RoundScoringCalibration | None = None,
    variability_calibration: JudgeVariabilityCalibration | None = None,
) -> ScheduledDistanceResult:
    """Run and resolve one controlled scheduled-distance path."""

    return resolve_scheduled_distance_path(
        run_full_path(
            scheduled_rounds=scheduled_rounds,
            seed=seed,
        ),
        scoring_calibration=scoring_calibration,
        variability_calibration=variability_calibration,
    )


def build_round_result(
    path,
    *,
    round_number: int,
    scoring_calibration: RoundScoringCalibration | None = None,
) -> ScheduledDistanceRoundResult:
    """Build one valid round result directly from a path."""

    start_index = (
        round_number - 1
    ) * SEGMENTS_PER_ROUND
    end_index = (
        start_index
        + SEGMENTS_PER_ROUND
    )

    segments = tuple(
        path.segments[
            start_index:end_index
        ]
    )
    evidence = calculate_round_evidence(
        segments
    )
    assessment = calculate_round_scoring_assessment(
        evidence,
        scoring_calibration,
    )

    return ScheduledDistanceRoundResult(
        round_number=round_number,
        segments=segments,
        evidence=evidence,
        assessment=assessment,
    )


def run_forced_finish_path(
    monkeypatch: pytest.MonkeyPatch,
    *,
    finish_call: int,
):
    """Run a path forced to finish at one selected segment."""

    call_count = 0

    def forced_sampler(
        probabilities,
        rng,
    ):
        del rng

        nonlocal call_count
        call_count += 1

        if call_count != finish_call:
            return None

        return FinishResult(
            state=probabilities.state,
            winner=FighterSide.RED,
            method=FinishMethod.KO_TKO,
            elapsed_seconds_in_segment=15,
        )

    monkeypatch.setattr(
        finish_runner_module,
        "sample_segment_finish",
        forced_sampler,
    )

    return run_full_path()


def alternate_decision(
    result: ScheduledDistanceResult,
):
    """Build a valid decision whose scorecards differ from the result."""

    scheduled_rounds = result.scheduled_rounds

    red_cards = tuple(
        JudgeScorecard(
            judge_number=judge_number,
            scheduled_rounds=scheduled_rounds,
            rounds=tuple(
                JudgeRoundScore(
                    round_number=round_number,
                    red_points=10,
                    blue_points=9,
                )
                for round_number in range(
                    1,
                    scheduled_rounds + 1,
                )
            ),
        )
        for judge_number in range(
            1,
            4,
        )
    )

    if red_cards != result.scorecards:
        return resolve_decision(
            red_cards
        )

    blue_cards = tuple(
        JudgeScorecard(
            judge_number=judge_number,
            scheduled_rounds=scheduled_rounds,
            rounds=tuple(
                JudgeRoundScore(
                    round_number=round_number,
                    red_points=9,
                    blue_points=10,
                )
                for round_number in range(
                    1,
                    scheduled_rounds + 1,
                )
            ),
        )
        for judge_number in range(
            1,
            4,
        )
    )

    return resolve_decision(
        blue_cards
    )


@pytest.mark.parametrize(
    ("scheduled_rounds", "expected_segments"),
    [
        (3, 30),
        (5, 50),
    ],
)
def test_resolves_three_and_five_round_paths(
    scheduled_rounds: int,
    expected_segments: int,
) -> None:
    selected = resolve_full_path(
        scheduled_rounds=scheduled_rounds,
    )

    assert selected.scheduled_rounds == scheduled_rounds
    assert len(selected.path.segments) == expected_segments
    assert len(selected.rounds) == scheduled_rounds
    assert len(selected.scorecards) == 3
    assert selected.path.finish is None
    assert selected.path.reached_scheduled_distance is True


def test_round_results_slice_path_exactly() -> None:
    selected = resolve_full_path(
        scheduled_rounds=5,
    )

    for round_number, round_result in enumerate(
        selected.rounds,
        start=1,
    ):
        start_index = (
            round_number - 1
        ) * SEGMENTS_PER_ROUND
        end_index = (
            start_index
            + SEGMENTS_PER_ROUND
        )

        assert round_result.round_number == round_number
        assert round_result.segments == tuple(
            selected.path.segments[
                start_index:end_index
            ]
        )


def test_round_evidence_matches_direct_calculation() -> None:
    selected = resolve_full_path()

    for round_result in selected.rounds:
        assert round_result.evidence == calculate_round_evidence(
            round_result.segments
        )


def test_round_assessment_matches_direct_calculation() -> None:
    selected = resolve_full_path()

    for round_result in selected.rounds:
        assert round_result.assessment == (
            calculate_round_scoring_assessment(
                round_result.evidence
            )
        )


def test_zero_variability_preserves_deterministic_round_scores() -> None:
    selected = resolve_full_path(
        variability_calibration=zero_variability(),
    )

    expected_scores = tuple(
        round_result.assessment.score
        for round_result in selected.rounds
    )

    for scorecard in selected.scorecards:
        assert scorecard.rounds == expected_scores


def test_judge_panel_uses_path_seed() -> None:
    selected = resolve_full_path(
        seed=919,
    )

    assert selected.seed == 919
    assert selected.judge_panel.seed == 919


def test_decision_uses_generated_panel_scorecards() -> None:
    selected = resolve_full_path()

    assert (
        selected.decision.scorecards
        == selected.judge_panel.scorecards
    )
    assert selected.scorecards == selected.judge_panel.scorecards


def test_result_properties_delegate_to_decision() -> None:
    selected = resolve_full_path()

    assert selected.winner is selected.decision.winner
    assert (
        selected.decision_type
        is selected.decision.decision_type
    )
    assert selected.is_draw is selected.decision.is_draw


def test_same_path_replays_identical_result() -> None:
    path = run_full_path(
        seed=707,
    )

    first = resolve_scheduled_distance_path(
        path
    )
    second = resolve_scheduled_distance_path(
        path
    )

    assert first == second


def test_resolver_does_not_mutate_path() -> None:
    path = run_full_path(
        seed=5150,
    )
    original_segments = path.segments

    resolve_scheduled_distance_path(
        path
    )

    assert path.segments == original_segments
    assert path.finish is None


def test_zero_scoring_weights_produce_unanimous_draw() -> None:
    calibration = RoundScoringCalibration(
        persistent_damage_weight=0.0,
        acute_stress_weight=0.0,
        knockdown_weight=0.0,
        damaging_clinch_weight=0.0,
        distance_landed_weight=0.0,
        clinch_landed_weight=0.0,
        ground_landed_weight=0.0,
        submission_attempt_weight=0.0,
        position_advancement_weight=0.0,
        reversal_weight=0.0,
        control_second_weight=0.0,
        escape_weight=0.0,
        scramble_weight=0.0,
        primary_close_threshold=0.05,
        secondary_scale=0.0,
        even_round_threshold=0.05,
        ten_eight_threshold=1.0,
        ten_seven_threshold=2.0,
    )

    selected = resolve_full_path(
        scoring_calibration=calibration,
        variability_calibration=zero_variability(),
    )

    assert selected.winner is None
    assert selected.is_draw is True
    assert (
        selected.decision_type
        is DecisionType.UNANIMOUS_DRAW
    )

    for scorecard in selected.scorecards:
        assert all(
            score.red_points == 10
            and score.blue_points == 10
            for score in scorecard.rounds
        )


def test_round_result_contract_accepts_valid_round() -> None:
    path = run_full_path()
    selected = build_round_result(
        path,
        round_number=2,
    )

    assert selected.round_number == 2
    assert len(selected.segments) == 10
    assert selected.evidence.round_number == 2
    assert selected.assessment.round_number == 2


@pytest.mark.parametrize(
    "invalid_value",
    [
        1.0,
        True,
        "1",
    ],
)
def test_round_result_round_number_requires_exact_integer(
    invalid_value: object,
) -> None:
    selected = resolve_full_path().rounds[0]

    with pytest.raises(
        TypeError,
        match="round_number must be an integer",
    ):
        ScheduledDistanceRoundResult(
            round_number=invalid_value,
            segments=selected.segments,
            evidence=selected.evidence,
            assessment=selected.assessment,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        0,
        6,
    ],
)
def test_round_result_round_number_range(
    invalid_value: int,
) -> None:
    selected = resolve_full_path().rounds[0]

    with pytest.raises(
        ValueError,
        match="round_number must be between 1 and 5",
    ):
        ScheduledDistanceRoundResult(
            round_number=invalid_value,
            segments=selected.segments,
            evidence=selected.evidence,
            assessment=selected.assessment,
        )


def test_round_result_segments_must_be_tuple() -> None:
    selected = resolve_full_path().rounds[0]

    with pytest.raises(
        TypeError,
        match="segments must be a tuple",
    ):
        ScheduledDistanceRoundResult(
            round_number=1,
            segments=list(
                selected.segments
            ),
            evidence=selected.evidence,
            assessment=selected.assessment,
        )


def test_round_result_requires_exactly_ten_segments() -> None:
    selected = resolve_full_path().rounds[0]

    with pytest.raises(
        ValueError,
        match=(
            "scheduled-distance round must contain "
            "exactly 10 segments"
        ),
    ):
        ScheduledDistanceRoundResult(
            round_number=1,
            segments=selected.segments[:-1],
            evidence=selected.evidence,
            assessment=selected.assessment,
        )


def test_round_result_requires_segment_contracts() -> None:
    selected = resolve_full_path().rounds[0]

    with pytest.raises(
        TypeError,
        match=(
            "segments must contain "
            "FinishEvaluatedPathSegment values"
        ),
    ):
        ScheduledDistanceRoundResult(
            round_number=1,
            segments=(
                "invalid",
            )
            + selected.segments[1:],
            evidence=selected.evidence,
            assessment=selected.assessment,
        )


def test_round_result_rejects_finishing_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finishing_path = run_forced_finish_path(
        monkeypatch,
        finish_call=10,
    )
    normal_round = resolve_full_path().rounds[0]

    assert len(finishing_path.segments) == 10

    with pytest.raises(
        ValueError,
        match=(
            "scheduled-distance rounds cannot "
            "contain a finish"
        ),
    ):
        ScheduledDistanceRoundResult(
            round_number=1,
            segments=finishing_path.segments,
            evidence=normal_round.evidence,
            assessment=normal_round.assessment,
        )


def test_round_result_segments_must_match_round_number() -> None:
    selected = resolve_full_path().rounds[0]

    with pytest.raises(
        ValueError,
        match="all segments must match round_number",
    ):
        ScheduledDistanceRoundResult(
            round_number=2,
            segments=selected.segments,
            evidence=RoundEvidence(
                round_number=2,
                red=selected.evidence.red,
                blue=selected.evidence.blue,
            ),
            assessment=calculate_round_scoring_assessment(
                RoundEvidence(
                    round_number=2,
                    red=selected.evidence.red,
                    blue=selected.evidence.blue,
                )
            ),
        )


def test_round_result_segments_must_be_sequential() -> None:
    selected = resolve_full_path().rounds[0]
    reordered = (
        selected.segments[1],
        selected.segments[0],
    ) + selected.segments[2:]

    with pytest.raises(
        ValueError,
        match=(
            "round segments must be sequential "
            "from one through ten"
        ),
    ):
        ScheduledDistanceRoundResult(
            round_number=1,
            segments=reordered,
            evidence=selected.evidence,
            assessment=selected.assessment,
        )


def test_round_result_requires_evidence_contract() -> None:
    selected = resolve_full_path().rounds[0]

    with pytest.raises(
        TypeError,
        match="evidence must be RoundEvidence",
    ):
        ScheduledDistanceRoundResult(
            round_number=1,
            segments=selected.segments,
            evidence="invalid",
            assessment=selected.assessment,
        )


def test_round_result_evidence_round_must_match() -> None:
    selected = resolve_full_path().rounds[0]
    wrong_evidence = RoundEvidence(
        round_number=2,
        red=selected.evidence.red,
        blue=selected.evidence.blue,
    )

    with pytest.raises(
        ValueError,
        match="evidence round_number must match result",
    ):
        ScheduledDistanceRoundResult(
            round_number=1,
            segments=selected.segments,
            evidence=wrong_evidence,
            assessment=selected.assessment,
        )


def test_round_result_requires_assessment_contract() -> None:
    selected = resolve_full_path().rounds[0]

    with pytest.raises(
        TypeError,
        match=(
            "assessment must be "
            "RoundScoringAssessment"
        ),
    ):
        ScheduledDistanceRoundResult(
            round_number=1,
            segments=selected.segments,
            evidence=selected.evidence,
            assessment="invalid",
        )


def test_round_result_assessment_round_must_match() -> None:
    selected = resolve_full_path().rounds[0]
    round_two = resolve_full_path().rounds[1]

    with pytest.raises(
        ValueError,
        match=(
            "assessment round_number must match result"
        ),
    ):
        ScheduledDistanceRoundResult(
            round_number=1,
            segments=selected.segments,
            evidence=selected.evidence,
            assessment=round_two.assessment,
        )


def test_round_result_is_immutable() -> None:
    selected = resolve_full_path().rounds[0]

    with pytest.raises(FrozenInstanceError):
        selected.round_number = 2


def test_scheduled_result_requires_path_contract() -> None:
    selected = resolve_full_path()

    with pytest.raises(
        TypeError,
        match="path must be FinishEnabledDynamicPath",
    ):
        ScheduledDistanceResult(
            path="invalid",
            rounds=selected.rounds,
            judge_panel=selected.judge_panel,
            decision=selected.decision,
        )


def test_scheduled_result_rejects_finish_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = run_forced_finish_path(
        monkeypatch,
        finish_call=1,
    )
    selected = resolve_full_path()

    with pytest.raises(
        ValueError,
        match=(
            "scheduled-distance result cannot contain "
            "a finish"
        ),
    ):
        ScheduledDistanceResult(
            path=path,
            rounds=selected.rounds,
            judge_panel=selected.judge_panel,
            decision=selected.decision,
        )


def test_scheduled_result_rounds_must_be_tuple() -> None:
    selected = resolve_full_path()

    with pytest.raises(
        TypeError,
        match="rounds must be a tuple",
    ):
        ScheduledDistanceResult(
            path=selected.path,
            rounds=list(
                selected.rounds
            ),
            judge_panel=selected.judge_panel,
            decision=selected.decision,
        )


def test_scheduled_result_requires_one_result_per_round() -> None:
    selected = resolve_full_path()

    with pytest.raises(
        ValueError,
        match=(
            "round results must contain exactly one "
            "entry per scheduled round"
        ),
    ):
        ScheduledDistanceResult(
            path=selected.path,
            rounds=selected.rounds[:2],
            judge_panel=selected.judge_panel,
            decision=selected.decision,
        )


def test_scheduled_result_requires_round_result_contracts() -> None:
    selected = resolve_full_path()

    with pytest.raises(
        TypeError,
        match=(
            "rounds must contain "
            "ScheduledDistanceRoundResult values"
        ),
    ):
        ScheduledDistanceResult(
            path=selected.path,
            rounds=(
                selected.rounds[0],
                "invalid",
                selected.rounds[2],
            ),
            judge_panel=selected.judge_panel,
            decision=selected.decision,
        )


def test_scheduled_result_rounds_must_be_sequential() -> None:
    selected = resolve_full_path()

    with pytest.raises(
        ValueError,
        match=(
            "round results must be sequential "
            "starting at round one"
        ),
    ):
        ScheduledDistanceResult(
            path=selected.path,
            rounds=(
                selected.rounds[1],
                selected.rounds[0],
                selected.rounds[2],
            ),
            judge_panel=selected.judge_panel,
            decision=selected.decision,
        )


def test_round_result_segments_must_exactly_match_path() -> None:
    first = resolve_full_path(
        seed=1,
    )
    second = resolve_full_path(
        seed=2,
    )

    with pytest.raises(
        ValueError,
        match=(
            "round-result segments must exactly match "
            "the simulation path"
        ),
    ):
        ScheduledDistanceResult(
            path=first.path,
            rounds=second.rounds,
            judge_panel=first.judge_panel,
            decision=first.decision,
        )


def test_scheduled_result_requires_judge_panel_contract() -> None:
    selected = resolve_full_path()

    with pytest.raises(
        TypeError,
        match=(
            "judge_panel must be "
            "JudgePanelScorecards"
        ),
    ):
        ScheduledDistanceResult(
            path=selected.path,
            rounds=selected.rounds,
            judge_panel="invalid",
            decision=selected.decision,
        )


def test_judge_panel_scheduled_rounds_must_match_path() -> None:
    three_round = resolve_full_path(
        scheduled_rounds=3,
    )
    five_round = resolve_full_path(
        scheduled_rounds=5,
    )

    with pytest.raises(
        ValueError,
        match=(
            "judge panel scheduled_rounds "
            "must match path"
        ),
    ):
        ScheduledDistanceResult(
            path=three_round.path,
            rounds=three_round.rounds,
            judge_panel=five_round.judge_panel,
            decision=three_round.decision,
        )


def test_judge_panel_seed_must_match_path() -> None:
    first = resolve_full_path(
        seed=1,
    )
    second = resolve_full_path(
        seed=2,
    )

    with pytest.raises(
        ValueError,
        match="judge panel seed must match path seed",
    ):
        ScheduledDistanceResult(
            path=first.path,
            rounds=first.rounds,
            judge_panel=second.judge_panel,
            decision=first.decision,
        )


def test_scheduled_result_requires_decision_contract() -> None:
    selected = resolve_full_path()

    with pytest.raises(
        TypeError,
        match="decision must be DecisionResult",
    ):
        ScheduledDistanceResult(
            path=selected.path,
            rounds=selected.rounds,
            judge_panel=selected.judge_panel,
            decision="invalid",
        )


def test_decision_scorecards_must_match_panel() -> None:
    selected = resolve_full_path()
    wrong_decision = alternate_decision(
        selected
    )

    assert (
        wrong_decision.scorecards
        != selected.judge_panel.scorecards
    )

    with pytest.raises(
        ValueError,
        match=(
            "decision scorecards must match "
            "judge panel"
        ),
    ):
        ScheduledDistanceResult(
            path=selected.path,
            rounds=selected.rounds,
            judge_panel=selected.judge_panel,
            decision=wrong_decision,
        )


def test_scheduled_result_is_immutable() -> None:
    selected = resolve_full_path()

    with pytest.raises(FrozenInstanceError):
        selected.path = run_full_path(
            seed=2,
        )


def test_resolver_requires_path_contract() -> None:
    with pytest.raises(
        TypeError,
        match="path must be FinishEnabledDynamicPath",
    ):
        resolve_scheduled_distance_path(
            "invalid"
        )


def test_resolver_rejects_finish_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = run_forced_finish_path(
        monkeypatch,
        finish_call=1,
    )

    with pytest.raises(
        ValueError,
        match="cannot score a path that ended by finish",
    ):
        resolve_scheduled_distance_path(
            path
        )


def test_resolver_requires_scoring_calibration() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "scoring_calibration must be "
            "RoundScoringCalibration"
        ),
    ):
        resolve_scheduled_distance_path(
            run_full_path(),
            scoring_calibration="invalid",
        )


def test_resolver_requires_variability_calibration() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "variability_calibration must be "
            "JudgeVariabilityCalibration"
        ),
    ):
        resolve_scheduled_distance_path(
            run_full_path(),
            variability_calibration="invalid",
        )
