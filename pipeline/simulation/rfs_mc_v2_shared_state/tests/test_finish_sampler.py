"""Tests for seeded V2 finish sampling."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_contracts import (
    FinishMethod,
    FinishResult,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_probability import (
    FighterSegmentFinishProbabilities,
    SegmentFinishProbabilities,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_sampler import (
    FinishCandidate,
    SampledFinishCandidate,
    build_finish_candidates,
    resolve_sampled_finish_candidates,
    sample_segment_finish,
)


def shared_state(
    *,
    phase: FightPhase = FightPhase.DISTANCE,
    owner: FighterSide | None = None,
    round_number: int = 1,
    segment_number: int = 1,
) -> SharedFightState:
    """Build one valid authoritative shared state."""

    if phase is FightPhase.DISTANCE:
        selected_owner = None
        phase_age_segments = 0
        position_quality = 0.0
    else:
        selected_owner = owner or FighterSide.RED
        phase_age_segments = 1
        position_quality = 0.50

    return SharedFightState(
        phase=phase,
        phase_owner=selected_owner,
        phase_age_segments=phase_age_segments,
        position_quality=position_quality,
        round_number=round_number,
        segment_number=segment_number,
    )


def fighter_probabilities(
    *,
    ko_tko: float = 0.0,
    submission: float = 0.0,
) -> FighterSegmentFinishProbabilities:
    """Build method-specific probabilities for one fighter."""

    return FighterSegmentFinishProbabilities(
        ko_tko_probability=ko_tko,
        submission_probability=submission,
    )


def distance_probabilities(
    *,
    red_ko_tko: float = 0.0,
    blue_ko_tko: float = 0.0,
    round_number: int = 1,
    segment_number: int = 1,
) -> SegmentFinishProbabilities:
    """Build legal distance finish probabilities."""

    return SegmentFinishProbabilities(
        state=shared_state(
            phase=FightPhase.DISTANCE,
            round_number=round_number,
            segment_number=segment_number,
        ),
        red=fighter_probabilities(
            ko_tko=red_ko_tko,
        ),
        blue=fighter_probabilities(
            ko_tko=blue_ko_tko,
        ),
    )


def ground_probabilities(
    *,
    owner: FighterSide = FighterSide.RED,
    red_ko_tko: float = 0.0,
    red_submission: float = 0.0,
    blue_ko_tko: float = 0.0,
    blue_submission: float = 0.0,
    round_number: int = 1,
    segment_number: int = 1,
) -> SegmentFinishProbabilities:
    """Build legal ground finish probabilities."""

    return SegmentFinishProbabilities(
        state=shared_state(
            phase=FightPhase.GROUND,
            owner=owner,
            round_number=round_number,
            segment_number=segment_number,
        ),
        red=fighter_probabilities(
            ko_tko=red_ko_tko,
            submission=red_submission,
        ),
        blue=fighter_probabilities(
            ko_tko=blue_ko_tko,
            submission=blue_submission,
        ),
    )


def candidate(
    *,
    state: SharedFightState | None = None,
    winner: FighterSide = FighterSide.RED,
    method: FinishMethod = FinishMethod.KO_TKO,
    probability: float = 0.50,
) -> FinishCandidate:
    """Build one valid finish candidate."""

    return FinishCandidate(
        state=state or shared_state(),
        winner=winner,
        method=method,
        probability=probability,
    )


def sampled_candidate(
    *,
    selected_candidate: FinishCandidate | None = None,
    elapsed_seconds: int = 15,
) -> SampledFinishCandidate:
    """Build one successful sampled finish candidate."""

    return SampledFinishCandidate(
        candidate=selected_candidate or candidate(),
        elapsed_seconds_in_segment=elapsed_seconds,
    )


def test_finish_candidate_retains_values() -> None:
    state = shared_state(
        round_number=2,
        segment_number=6,
    )

    selected = candidate(
        state=state,
        winner=FighterSide.BLUE,
        method=FinishMethod.KO_TKO,
        probability=0.35,
    )

    assert selected.state is state
    assert selected.winner is FighterSide.BLUE
    assert selected.method is FinishMethod.KO_TKO
    assert selected.probability == 0.35


def test_finish_candidate_probability_boundaries_are_allowed() -> None:
    low = candidate(
        probability=0.0,
    )
    high = candidate(
        probability=1.0,
    )

    assert low.probability == 0.0
    assert high.probability == 1.0


def test_finish_candidate_requires_shared_state() -> None:
    with pytest.raises(
        TypeError,
        match="state must be SharedFightState",
    ):
        FinishCandidate(
            state="invalid",
            winner=FighterSide.RED,
            method=FinishMethod.KO_TKO,
            probability=0.50,
        )


def test_finish_candidate_requires_fighter_side() -> None:
    with pytest.raises(
        TypeError,
        match="winner must be FighterSide",
    ):
        FinishCandidate(
            state=shared_state(),
            winner="red",
            method=FinishMethod.KO_TKO,
            probability=0.50,
        )


def test_finish_candidate_requires_finish_method() -> None:
    with pytest.raises(
        TypeError,
        match="method must be FinishMethod",
    ):
        FinishCandidate(
            state=shared_state(),
            winner=FighterSide.RED,
            method="ko_tko",
            probability=0.50,
        )


def test_finish_candidate_probability_must_be_numeric() -> None:
    with pytest.raises(
        TypeError,
        match="probability must be numeric",
    ):
        candidate(
            probability="invalid",
        )


@pytest.mark.parametrize(
    "invalid_probability",
    [
        float("nan"),
        float("inf"),
    ],
)
def test_finish_candidate_probability_must_be_finite(
    invalid_probability: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="probability must be finite",
    ):
        candidate(
            probability=invalid_probability,
        )


@pytest.mark.parametrize(
    "invalid_probability",
    [
        -0.01,
        1.01,
    ],
)
def test_finish_candidate_probability_must_be_in_unit_interval(
    invalid_probability: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="probability must be between 0 and 1",
    ):
        candidate(
            probability=invalid_probability,
        )


def test_finish_candidate_is_immutable() -> None:
    selected = candidate()

    with pytest.raises(FrozenInstanceError):
        selected.probability = 0.20


def test_sampled_candidate_retains_values() -> None:
    selected_candidate = candidate(
        winner=FighterSide.BLUE,
        probability=0.60,
    )

    selected = sampled_candidate(
        selected_candidate=selected_candidate,
        elapsed_seconds=12,
    )

    assert selected.candidate is selected_candidate
    assert selected.elapsed_seconds_in_segment == 12


def test_sampled_candidate_requires_finish_candidate() -> None:
    with pytest.raises(
        TypeError,
        match="candidate must be FinishCandidate",
    ):
        SampledFinishCandidate(
            candidate="invalid",
            elapsed_seconds_in_segment=15,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        1.0,
        "1",
    ],
)
def test_sampled_finish_second_must_be_integer(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match=(
            "elapsed_seconds_in_segment must be an integer"
        ),
    ):
        SampledFinishCandidate(
            candidate=candidate(),
            elapsed_seconds_in_segment=invalid_value,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        -1,
        0,
        31,
        100,
    ],
)
def test_sampled_finish_second_must_be_in_bounds(
    invalid_value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "elapsed_seconds_in_segment must be between 1 and 30"
        ),
    ):
        sampled_candidate(
            elapsed_seconds=invalid_value,
        )


def test_sampled_finish_candidate_is_immutable() -> None:
    selected = sampled_candidate()

    with pytest.raises(FrozenInstanceError):
        selected.elapsed_seconds_in_segment = 5


def test_finish_candidates_use_stable_channel_order() -> None:
    probabilities = ground_probabilities(
        owner=FighterSide.RED,
        red_ko_tko=0.10,
        red_submission=0.20,
        blue_ko_tko=0.30,
    )

    candidates = build_finish_candidates(
        probabilities
    )

    assert tuple(
        (
            selected.winner,
            selected.method,
            selected.probability,
        )
        for selected in candidates
    ) == (
        (
            FighterSide.RED,
            FinishMethod.KO_TKO,
            0.10,
        ),
        (
            FighterSide.RED,
            FinishMethod.SUBMISSION,
            0.20,
        ),
        (
            FighterSide.BLUE,
            FinishMethod.KO_TKO,
            0.30,
        ),
        (
            FighterSide.BLUE,
            FinishMethod.SUBMISSION,
            0.0,
        ),
    )


def test_finish_candidates_share_authoritative_state() -> None:
    probabilities = distance_probabilities(
        red_ko_tko=0.20,
        blue_ko_tko=0.30,
    )

    candidates = build_finish_candidates(
        probabilities
    )

    assert len(candidates) == 4

    assert all(
        selected.state is probabilities.state
        for selected in candidates
    )


def test_build_finish_candidates_requires_probability_contract() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "probabilities must be SegmentFinishProbabilities"
        ),
    ):
        build_finish_candidates(
            "invalid"
        )


def test_resolving_no_candidates_returns_none() -> None:
    result = resolve_sampled_finish_candidates(
        ()
    )

    assert result is None


def test_single_sampled_candidate_becomes_finish_result() -> None:
    state = shared_state(
        round_number=3,
        segment_number=7,
    )

    result = resolve_sampled_finish_candidates(
        (
            sampled_candidate(
                selected_candidate=candidate(
                    state=state,
                    winner=FighterSide.BLUE,
                    method=FinishMethod.KO_TKO,
                    probability=0.40,
                ),
                elapsed_seconds=11,
            ),
        )
    )

    assert isinstance(
        result,
        FinishResult,
    )
    assert result.state is state
    assert result.winner is FighterSide.BLUE
    assert result.method is FinishMethod.KO_TKO
    assert result.elapsed_seconds_in_segment == 11


def test_earliest_sampled_candidate_wins() -> None:
    early = sampled_candidate(
        selected_candidate=candidate(
            winner=FighterSide.BLUE,
            probability=0.10,
        ),
        elapsed_seconds=5,
    )
    late = sampled_candidate(
        selected_candidate=candidate(
            winner=FighterSide.RED,
            probability=0.90,
        ),
        elapsed_seconds=20,
    )

    result = resolve_sampled_finish_candidates(
        (
            late,
            early,
        )
    )

    assert result is not None
    assert result.winner is FighterSide.BLUE
    assert result.elapsed_seconds_in_segment == 5


def test_higher_probability_breaks_exact_second_tie() -> None:
    low_probability = sampled_candidate(
        selected_candidate=candidate(
            winner=FighterSide.RED,
            probability=0.20,
        ),
        elapsed_seconds=10,
    )
    high_probability = sampled_candidate(
        selected_candidate=candidate(
            winner=FighterSide.BLUE,
            probability=0.80,
        ),
        elapsed_seconds=10,
    )

    result = resolve_sampled_finish_candidates(
        (
            low_probability,
            high_probability,
        )
    )

    assert result is not None
    assert result.winner is FighterSide.BLUE


def test_knockout_breaks_equal_probability_method_tie() -> None:
    state = shared_state(
        phase=FightPhase.GROUND,
        owner=FighterSide.RED,
    )

    submission = sampled_candidate(
        selected_candidate=candidate(
            state=state,
            winner=FighterSide.RED,
            method=FinishMethod.SUBMISSION,
            probability=0.50,
        ),
        elapsed_seconds=10,
    )
    knockout = sampled_candidate(
        selected_candidate=candidate(
            state=state,
            winner=FighterSide.RED,
            method=FinishMethod.KO_TKO,
            probability=0.50,
        ),
        elapsed_seconds=10,
    )

    result = resolve_sampled_finish_candidates(
        (
            submission,
            knockout,
        )
    )

    assert result is not None
    assert result.method is FinishMethod.KO_TKO


def test_red_breaks_complete_candidate_tie() -> None:
    red = sampled_candidate(
        selected_candidate=candidate(
            winner=FighterSide.RED,
            probability=0.50,
        ),
        elapsed_seconds=10,
    )
    blue = sampled_candidate(
        selected_candidate=candidate(
            winner=FighterSide.BLUE,
            probability=0.50,
        ),
        elapsed_seconds=10,
    )

    result = resolve_sampled_finish_candidates(
        (
            blue,
            red,
        )
    )

    assert result is not None
    assert result.winner is FighterSide.RED


def test_finish_time_precedes_probability_tie_breaking() -> None:
    earlier = sampled_candidate(
        selected_candidate=candidate(
            winner=FighterSide.RED,
            probability=0.01,
        ),
        elapsed_seconds=9,
    )
    later = sampled_candidate(
        selected_candidate=candidate(
            winner=FighterSide.BLUE,
            probability=1.0,
        ),
        elapsed_seconds=10,
    )

    result = resolve_sampled_finish_candidates(
        (
            later,
            earlier,
        )
    )

    assert result is not None
    assert result.winner is FighterSide.RED
    assert result.elapsed_seconds_in_segment == 9


def test_resolver_rejects_invalid_candidate_values() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "candidates must contain "
            "SampledFinishCandidate values"
        ),
    ):
        resolve_sampled_finish_candidates(
            (
                "invalid",
            )
        )


def test_zero_probabilities_never_produce_finish() -> None:
    result = sample_segment_finish(
        distance_probabilities(),
        np.random.default_rng(2026),
    )

    assert result is None


@pytest.mark.parametrize(
    "seed",
    [
        0,
        1,
        2026,
    ],
)
def test_same_seed_produces_identical_finish_sample(
    seed: int,
) -> None:
    probabilities = ground_probabilities(
        owner=FighterSide.RED,
        red_ko_tko=0.45,
        red_submission=0.55,
        blue_ko_tko=0.35,
    )

    first = sample_segment_finish(
        probabilities,
        np.random.default_rng(seed),
    )
    second = sample_segment_finish(
        probabilities,
        np.random.default_rng(seed),
    )

    assert first == second


def test_certain_red_knockout_produces_legal_finish() -> None:
    probabilities = distance_probabilities(
        red_ko_tko=1.0,
        round_number=2,
        segment_number=4,
    )

    result = sample_segment_finish(
        probabilities,
        np.random.default_rng(17),
    )

    assert result is not None
    assert result.state is probabilities.state
    assert result.winner is FighterSide.RED
    assert result.method is FinishMethod.KO_TKO
    assert result.round_number == 2
    assert result.segment_number == 4
    assert 1 <= result.elapsed_seconds_in_segment <= 30


def test_certain_red_submission_produces_legal_finish() -> None:
    probabilities = ground_probabilities(
        owner=FighterSide.RED,
        red_submission=1.0,
    )

    result = sample_segment_finish(
        probabilities,
        np.random.default_rng(18),
    )

    assert result is not None
    assert result.winner is FighterSide.RED
    assert result.method is FinishMethod.SUBMISSION
    assert result.state.phase_owner is FighterSide.RED


def test_certain_blue_submission_produces_legal_finish() -> None:
    probabilities = ground_probabilities(
        owner=FighterSide.BLUE,
        blue_submission=1.0,
    )

    result = sample_segment_finish(
        probabilities,
        np.random.default_rng(19),
    )

    assert result is not None
    assert result.winner is FighterSide.BLUE
    assert result.method is FinishMethod.SUBMISSION
    assert result.state.phase_owner is FighterSide.BLUE


def test_sampler_matches_explicit_stable_channel_sampling() -> None:
    probabilities = ground_probabilities(
        owner=FighterSide.RED,
        red_ko_tko=1.0,
        red_submission=1.0,
        blue_ko_tko=1.0,
    )
    seed = 773

    candidates = build_finish_candidates(
        probabilities
    )

    expected_rng = np.random.default_rng(seed)
    occurrence_rolls = expected_rng.random(
        len(candidates)
    )

    successful = [
        selected
        for selected, roll in zip(
            candidates,
            occurrence_rolls,
            strict=True,
        )
        if roll < selected.probability
    ]

    sampled_seconds = expected_rng.integers(
        low=1,
        high=31,
        size=len(successful),
    )

    expected = resolve_sampled_finish_candidates(
        tuple(
            SampledFinishCandidate(
                candidate=selected,
                elapsed_seconds_in_segment=int(
                    elapsed_second
                ),
            )
            for selected, elapsed_second in zip(
                successful,
                sampled_seconds,
                strict=True,
            )
        )
    )

    actual = sample_segment_finish(
        probabilities,
        np.random.default_rng(seed),
    )

    assert actual == expected


def test_no_finish_consumes_only_four_occurrence_rolls() -> None:
    probabilities = distance_probabilities()
    seed = 811

    expected_rng = np.random.default_rng(seed)
    expected_rng.random(4)
    expected_next = expected_rng.random()

    actual_rng = np.random.default_rng(seed)

    result = sample_segment_finish(
        probabilities,
        actual_rng,
    )
    actual_next = actual_rng.random()

    assert result is None
    assert actual_next == pytest.approx(expected_next)


def test_one_finish_consumes_one_finish_second_roll() -> None:
    probabilities = distance_probabilities(
        red_ko_tko=1.0,
    )
    seed = 812

    expected_rng = np.random.default_rng(seed)
    expected_rng.random(4)
    expected_rng.integers(
        low=1,
        high=31,
        size=1,
    )
    expected_next = expected_rng.random()

    actual_rng = np.random.default_rng(seed)

    result = sample_segment_finish(
        probabilities,
        actual_rng,
    )
    actual_next = actual_rng.random()

    assert result is not None
    assert actual_next == pytest.approx(expected_next)


def test_sampler_requires_probability_contract() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "probabilities must be SegmentFinishProbabilities"
        ),
    ):
        sample_segment_finish(
            "invalid",
            np.random.default_rng(0),
        )


def test_sampler_requires_numpy_generator() -> None:
    with pytest.raises(
        TypeError,
        match="rng must be numpy.random.Generator",
    ):
        sample_segment_finish(
            distance_probabilities(),
            "invalid",
        )
