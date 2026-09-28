"""Tests for the V2 unified final fight-result resolver."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import pipeline.simulation.rfs_mc_v2_shared_state.final_fight_result as result_module
import pipeline.simulation.rfs_mc_v2_shared_state.finish_path_runner as finish_runner_module
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
)
from pipeline.simulation.rfs_mc_v2_shared_state.decision_contracts import (
    DecisionType,
)
from pipeline.simulation.rfs_mc_v2_shared_state.final_fight_result import (
    FightResultBranch,
    FinalFightResult,
    resolve_final_fight_result,
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
from pipeline.simulation.rfs_mc_v2_shared_state.round_scoring_engine import (
    RoundScoringCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.scheduled_distance_result import (
    ScheduledDistanceResult,
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


def run_forced_finish_path(
    monkeypatch: pytest.MonkeyPatch,
    *,
    finish_call: int = 1,
    winner: FighterSide = FighterSide.RED,
):
    """Run a path forced to end by KO/TKO."""

    call_count = 0

    def forced_sampler(
        probabilities,
        rng,
    ) -> FinishResult | None:
        del rng

        nonlocal call_count
        call_count += 1

        if call_count != finish_call:
            return None

        return FinishResult(
            state=probabilities.state,
            winner=winner,
            method=FinishMethod.KO_TKO,
            elapsed_seconds_in_segment=15,
        )

    monkeypatch.setattr(
        finish_runner_module,
        "sample_segment_finish",
        forced_sampler,
    )

    return run_full_path()


def zero_variability() -> JudgeVariabilityCalibration:
    """Return judge calibration with no random adjustment."""

    return JudgeVariabilityCalibration(
        fight_bias_stddev=0.0,
        round_noise_stddev=0.0,
        maximum_absolute_adjustment=0.0,
    )


def zero_scoring() -> RoundScoringCalibration:
    """Return scoring calibration that makes every round even."""

    return RoundScoringCalibration(
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


def scheduled_result(
    *,
    scheduled_rounds: int = 3,
    seed: int = 2026,
) -> ScheduledDistanceResult:
    """Build one valid scheduled-distance result."""

    return resolve_scheduled_distance_path(
        run_full_path(
            scheduled_rounds=scheduled_rounds,
            seed=seed,
        )
    )


def finish_result_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> FinalFightResult:
    """Build one valid finish-branch result contract."""

    path = run_forced_finish_path(
        monkeypatch,
    )

    return FinalFightResult(
        path=path,
        branch=FightResultBranch.FINISH,
        winner=path.finish.winner,
        finish=path.finish,
        scheduled_distance=None,
    )


def distance_result_contract(
    *,
    scheduled_rounds: int = 3,
    seed: int = 2026,
) -> FinalFightResult:
    """Build one valid scheduled-distance result contract."""

    selected = scheduled_result(
        scheduled_rounds=scheduled_rounds,
        seed=seed,
    )

    return FinalFightResult(
        path=selected.path,
        branch=FightResultBranch.SCHEDULED_DISTANCE,
        winner=selected.winner,
        finish=None,
        scheduled_distance=selected,
    )


def opposite_winner(
    winner: FighterSide | None,
) -> FighterSide:
    """Return a fighter that does not match the selected winner."""

    if winner is FighterSide.RED:
        return FighterSide.BLUE

    return FighterSide.RED


def test_resolver_routes_finish_path_to_finish_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = run_forced_finish_path(
        monkeypatch,
        finish_call=7,
        winner=FighterSide.BLUE,
    )

    selected = resolve_final_fight_result(
        path
    )

    assert selected.path is path
    assert selected.branch is FightResultBranch.FINISH
    assert selected.winner is FighterSide.BLUE
    assert selected.finish == path.finish
    assert selected.scheduled_distance is None
    assert selected.is_finish is True
    assert selected.is_scheduled_distance is False
    assert selected.is_draw is False


@pytest.mark.parametrize(
    "scheduled_rounds",
    [
        3,
        5,
    ],
)
def test_resolver_routes_full_path_to_scheduled_distance_branch(
    scheduled_rounds: int,
) -> None:
    path = run_full_path(
        scheduled_rounds=scheduled_rounds,
    )

    selected = resolve_final_fight_result(
        path
    )

    assert selected.path is path
    assert (
        selected.branch
        is FightResultBranch.SCHEDULED_DISTANCE
    )
    assert selected.finish is None
    assert selected.scheduled_distance is not None
    assert (
        selected.scheduled_distance.path
        is path
    )
    assert selected.scheduled_rounds == scheduled_rounds
    assert selected.is_finish is False
    assert selected.is_scheduled_distance is True


def test_finish_branch_does_not_call_scheduled_distance_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = run_forced_finish_path(
        monkeypatch,
    )

    def unexpected_resolver(*args, **kwargs):
        raise AssertionError(
            "scheduled-distance resolver was called "
            "for a finish path"
        )

    monkeypatch.setattr(
        result_module,
        "resolve_scheduled_distance_path",
        unexpected_resolver,
    )

    selected = resolve_final_fight_result(
        path,
        scoring_calibration="ignored",
        variability_calibration="ignored",
    )

    assert selected.is_finish is True


def test_scheduled_distance_calibrations_are_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = run_full_path()
    scoring = zero_scoring()
    variability = zero_variability()

    captured: dict[str, object] = {}
    original = (
        result_module.resolve_scheduled_distance_path
    )

    def capturing_resolver(
        selected_path,
        *,
        scoring_calibration,
        variability_calibration,
    ):
        captured["path"] = selected_path
        captured["scoring"] = scoring_calibration
        captured["variability"] = variability_calibration

        return original(
            selected_path,
            scoring_calibration=scoring_calibration,
            variability_calibration=variability_calibration,
        )

    monkeypatch.setattr(
        result_module,
        "resolve_scheduled_distance_path",
        capturing_resolver,
    )

    selected = resolve_final_fight_result(
        path,
        scoring_calibration=scoring,
        variability_calibration=variability,
    )

    assert selected.is_scheduled_distance is True
    assert captured["path"] is path
    assert captured["scoring"] is scoring
    assert captured["variability"] is variability


def test_zero_scoring_resolves_unanimous_draw() -> None:
    selected = resolve_final_fight_result(
        run_full_path(),
        scoring_calibration=zero_scoring(),
        variability_calibration=zero_variability(),
    )

    assert selected.winner is None
    assert selected.is_draw is True
    assert (
        selected.decision_type
        is DecisionType.UNANIMOUS_DRAW
    )
    assert (
        selected.official_method
        is DecisionType.UNANIMOUS_DRAW
    )


def test_same_path_replays_identical_final_result() -> None:
    path = run_full_path(
        seed=707,
    )

    first = resolve_final_fight_result(
        path
    )
    second = resolve_final_fight_result(
        path
    )

    assert first == second


def test_finish_result_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = run_forced_finish_path(
        monkeypatch,
        finish_call=7,
    )

    selected = resolve_final_fight_result(
        path
    )

    assert selected.scheduled_rounds == 3
    assert selected.seed == path.seed
    assert selected.finish_method is FinishMethod.KO_TKO
    assert selected.official_method is FinishMethod.KO_TKO
    assert selected.decision_type is None
    assert selected.finish_round == 1
    assert selected.finish_segment == 7
    assert selected.elapsed_seconds_in_round == 195
    assert selected.scorecards is None


def test_scheduled_distance_result_properties() -> None:
    selected = resolve_final_fight_result(
        run_full_path()
    )

    assert selected.finish_method is None
    assert selected.finish_round is None
    assert selected.finish_segment is None
    assert selected.elapsed_seconds_in_round is None
    assert selected.decision_type is not None
    assert (
        selected.official_method
        is selected.decision_type
    )
    assert selected.scorecards is not None
    assert len(selected.scorecards) == 3


def test_final_result_requires_path_contract() -> None:
    selected = distance_result_contract()

    with pytest.raises(
        TypeError,
        match="path must be FinishEnabledDynamicPath",
    ):
        FinalFightResult(
            path="invalid",
            branch=selected.branch,
            winner=selected.winner,
            finish=None,
            scheduled_distance=selected.scheduled_distance,
        )


def test_final_result_requires_branch_enum() -> None:
    selected = distance_result_contract()

    with pytest.raises(
        TypeError,
        match="branch must be FightResultBranch",
    ):
        FinalFightResult(
            path=selected.path,
            branch="scheduled_distance",
            winner=selected.winner,
            finish=None,
            scheduled_distance=selected.scheduled_distance,
        )


@pytest.mark.parametrize(
    "invalid_winner",
    [
        "red",
        1,
    ],
)
def test_final_result_winner_requires_fighter_side_or_none(
    invalid_winner: object,
) -> None:
    selected = distance_result_contract()

    with pytest.raises(
        TypeError,
        match="winner must be FighterSide or None",
    ):
        FinalFightResult(
            path=selected.path,
            branch=selected.branch,
            winner=invalid_winner,
            finish=None,
            scheduled_distance=selected.scheduled_distance,
        )


def test_finish_payload_requires_finish_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = finish_result_contract(
        monkeypatch
    )

    with pytest.raises(
        TypeError,
        match="finish must be FinishResult or None",
    ):
        FinalFightResult(
            path=selected.path,
            branch=selected.branch,
            winner=selected.winner,
            finish="invalid",
            scheduled_distance=None,
        )


def test_scheduled_payload_requires_scheduled_result() -> None:
    selected = distance_result_contract()

    with pytest.raises(
        TypeError,
        match=(
            "scheduled_distance must be "
            "ScheduledDistanceResult or None"
        ),
    ):
        FinalFightResult(
            path=selected.path,
            branch=selected.branch,
            winner=selected.winner,
            finish=None,
            scheduled_distance="invalid",
        )


def test_finish_branch_requires_finish_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = finish_result_contract(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match="finish branch requires a finish result",
    ):
        FinalFightResult(
            path=selected.path,
            branch=FightResultBranch.FINISH,
            winner=selected.winner,
            finish=None,
            scheduled_distance=None,
        )


def test_finish_branch_rejects_scheduled_distance_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = finish_result_contract(
        monkeypatch
    )
    distance = scheduled_result(
        seed=999,
    )

    with pytest.raises(
        ValueError,
        match=(
            "finish branch cannot contain a "
            "scheduled-distance result"
        ),
    ):
        FinalFightResult(
            path=selected.path,
            branch=FightResultBranch.FINISH,
            winner=selected.winner,
            finish=selected.finish,
            scheduled_distance=distance,
        )


def test_finish_branch_requires_finished_path() -> None:
    path = run_full_path()

    artificial_finish = FinishResult(
        state=path.segments[0].state,
        winner=FighterSide.RED,
        method=FinishMethod.KO_TKO,
        elapsed_seconds_in_segment=15,
    )

    with pytest.raises(
        ValueError,
        match=(
            "finish branch requires a path that "
            "ended by finish"
        ),
    ):
        FinalFightResult(
            path=path,
            branch=FightResultBranch.FINISH,
            winner=FighterSide.RED,
            finish=artificial_finish,
            scheduled_distance=None,
        )


def test_finish_payload_must_match_path_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = finish_result_contract(
        monkeypatch
    )

    alternate = FinishResult(
        state=selected.finish.state,
        winner=selected.finish.winner,
        method=FinishMethod.KO_TKO,
        elapsed_seconds_in_segment=16,
    )

    with pytest.raises(
        ValueError,
        match=(
            "finish result must match the path finish"
        ),
    ):
        FinalFightResult(
            path=selected.path,
            branch=FightResultBranch.FINISH,
            winner=selected.winner,
            finish=alternate,
            scheduled_distance=None,
        )


def test_finish_winner_must_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = finish_result_contract(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match="winner must match the finish winner",
    ):
        FinalFightResult(
            path=selected.path,
            branch=FightResultBranch.FINISH,
            winner=selected.finish.winner.opponent,
            finish=selected.finish,
            scheduled_distance=None,
        )


def test_scheduled_branch_rejects_finish_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = distance_result_contract()
    finish_path = run_forced_finish_path(
        monkeypatch,
    )

    with pytest.raises(
        ValueError,
        match=(
            "scheduled-distance branch cannot "
            "contain a finish result"
        ),
    ):
        FinalFightResult(
            path=selected.path,
            branch=FightResultBranch.SCHEDULED_DISTANCE,
            winner=selected.winner,
            finish=finish_path.finish,
            scheduled_distance=selected.scheduled_distance,
        )


def test_scheduled_branch_requires_scheduled_result() -> None:
    path = run_full_path()

    with pytest.raises(
        ValueError,
        match=(
            "scheduled-distance branch requires a "
            "scheduled-distance result"
        ),
    ):
        FinalFightResult(
            path=path,
            branch=FightResultBranch.SCHEDULED_DISTANCE,
            winner=None,
            finish=None,
            scheduled_distance=None,
        )


def test_scheduled_branch_rejects_finished_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finish_path = run_forced_finish_path(
        monkeypatch,
    )
    distance = scheduled_result(
        seed=999,
    )

    with pytest.raises(
        ValueError,
        match=(
            "scheduled-distance branch cannot use "
            "a path that ended by finish"
        ),
    ):
        FinalFightResult(
            path=finish_path,
            branch=FightResultBranch.SCHEDULED_DISTANCE,
            winner=distance.winner,
            finish=None,
            scheduled_distance=distance,
        )


def test_scheduled_result_path_must_match_final_path() -> None:
    first_path = run_full_path(
        seed=1,
    )
    second_result = scheduled_result(
        seed=2,
    )

    with pytest.raises(
        ValueError,
        match=(
            "scheduled-distance result path must "
            "match the final-result path"
        ),
    ):
        FinalFightResult(
            path=first_path,
            branch=FightResultBranch.SCHEDULED_DISTANCE,
            winner=second_result.winner,
            finish=None,
            scheduled_distance=second_result,
        )


def test_scheduled_winner_must_match() -> None:
    selected = scheduled_result()

    with pytest.raises(
        ValueError,
        match=(
            "winner must match the scheduled-distance "
            "result winner"
        ),
    ):
        FinalFightResult(
            path=selected.path,
            branch=FightResultBranch.SCHEDULED_DISTANCE,
            winner=opposite_winner(
                selected.winner
            ),
            finish=None,
            scheduled_distance=selected,
        )


def test_resolver_requires_path_contract() -> None:
    with pytest.raises(
        TypeError,
        match="path must be FinishEnabledDynamicPath",
    ):
        resolve_final_fight_result(
            "invalid"
        )


def test_distance_branch_rejects_invalid_scoring_calibration() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "scoring_calibration must be "
            "RoundScoringCalibration"
        ),
    ):
        resolve_final_fight_result(
            run_full_path(),
            scoring_calibration="invalid",
        )


def test_distance_branch_rejects_invalid_variability_calibration() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "variability_calibration must be "
            "JudgeVariabilityCalibration"
        ),
    ):
        resolve_final_fight_result(
            run_full_path(),
            variability_calibration="invalid",
        )


def test_final_result_is_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = finish_result_contract(
        monkeypatch
    )

    with pytest.raises(FrozenInstanceError):
        selected.winner = FighterSide.BLUE
