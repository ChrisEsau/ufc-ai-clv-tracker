"""Seeded finish sampling for RFS Monte Carlo V2.

This module converts deterministic segment finish probabilities into at most
one legal ``FinishResult``.

Candidate occurrence rolls use a stable channel order:

1. red KO/TKO
2. red submission
3. blue KO/TKO
4. blue submission

All four occurrence rolls are drawn before any finish-time rolls. This keeps
candidate occurrence sampling stable regardless of which earlier candidates
succeed.

Every successful candidate receives an approximate finish second sampled
uniformly from the 30-second segment. The earliest candidate wins.

Exact-second ties are resolved deterministically by:

1. higher segment probability
2. KO/TKO before submission
3. red before blue

The uniform within-segment timing assumption is intentionally isolated here so
it can later be replaced by an empirically calibrated timing model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_contracts import (
    FinishMethod,
    FinishResult,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_probability import (
    SegmentFinishProbabilities,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    SEGMENT_SECONDS,
)


def _validate_probability(
    name: str,
    value: float,
) -> None:
    """Validate one finite numeric probability in [0, 1]."""

    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be numeric"
        )

    selected = float(value)

    if not math.isfinite(selected):
        raise ValueError(
            f"{name} must be finite"
        )

    if not 0.0 <= selected <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1"
        )


@dataclass(frozen=True)
class FinishCandidate:
    """One possible method-specific finish within a segment."""

    state: SharedFightState
    winner: FighterSide
    method: FinishMethod
    probability: float

    def __post_init__(self) -> None:
        """Validate candidate identity and probability."""

        if not isinstance(
            self.state,
            SharedFightState,
        ):
            raise TypeError(
                "state must be SharedFightState"
            )

        if not isinstance(
            self.winner,
            FighterSide,
        ):
            raise TypeError(
                "winner must be FighterSide"
            )

        if not isinstance(
            self.method,
            FinishMethod,
        ):
            raise TypeError(
                "method must be FinishMethod"
            )

        _validate_probability(
            "probability",
            self.probability,
        )


@dataclass(frozen=True)
class SampledFinishCandidate:
    """One candidate whose occurrence roll succeeded."""

    candidate: FinishCandidate
    elapsed_seconds_in_segment: int

    def __post_init__(self) -> None:
        """Validate the sampled candidate and finish second."""

        if not isinstance(
            self.candidate,
            FinishCandidate,
        ):
            raise TypeError(
                "candidate must be FinishCandidate"
            )

        if not isinstance(
            self.elapsed_seconds_in_segment,
            int,
        ):
            raise TypeError(
                "elapsed_seconds_in_segment must be an integer"
            )

        if not (
            1
            <= self.elapsed_seconds_in_segment
            <= SEGMENT_SECONDS
        ):
            raise ValueError(
                "elapsed_seconds_in_segment must be between "
                f"1 and {SEGMENT_SECONDS}"
            )


def build_finish_candidates(
    probabilities: SegmentFinishProbabilities,
) -> tuple[FinishCandidate, ...]:
    """Build the four candidates in stable sampling order."""

    if not isinstance(
        probabilities,
        SegmentFinishProbabilities,
    ):
        raise TypeError(
            "probabilities must be SegmentFinishProbabilities"
        )

    return (
        FinishCandidate(
            state=probabilities.state,
            winner=FighterSide.RED,
            method=FinishMethod.KO_TKO,
            probability=(
                probabilities.red.ko_tko_probability
            ),
        ),
        FinishCandidate(
            state=probabilities.state,
            winner=FighterSide.RED,
            method=FinishMethod.SUBMISSION,
            probability=(
                probabilities.red.submission_probability
            ),
        ),
        FinishCandidate(
            state=probabilities.state,
            winner=FighterSide.BLUE,
            method=FinishMethod.KO_TKO,
            probability=(
                probabilities.blue.ko_tko_probability
            ),
        ),
        FinishCandidate(
            state=probabilities.state,
            winner=FighterSide.BLUE,
            method=FinishMethod.SUBMISSION,
            probability=(
                probabilities.blue.submission_probability
            ),
        ),
    )


def _candidate_resolution_key(
    sampled: SampledFinishCandidate,
) -> tuple[int, float, int, int]:
    """Return deterministic competing-finish resolution order."""

    method_priority = (
        0
        if sampled.candidate.method is FinishMethod.KO_TKO
        else 1
    )
    side_priority = (
        0
        if sampled.candidate.winner is FighterSide.RED
        else 1
    )

    return (
        sampled.elapsed_seconds_in_segment,
        -sampled.candidate.probability,
        method_priority,
        side_priority,
    )


def resolve_sampled_finish_candidates(
    candidates: tuple[SampledFinishCandidate, ...],
) -> FinishResult | None:
    """Resolve zero or more successful candidates into one finish."""

    for candidate in candidates:
        if not isinstance(
            candidate,
            SampledFinishCandidate,
        ):
            raise TypeError(
                "candidates must contain "
                "SampledFinishCandidate values"
            )

    if not candidates:
        return None

    selected = min(
        candidates,
        key=_candidate_resolution_key,
    )

    return FinishResult(
        state=selected.candidate.state,
        winner=selected.candidate.winner,
        method=selected.candidate.method,
        elapsed_seconds_in_segment=(
            selected.elapsed_seconds_in_segment
        ),
    )


def sample_segment_finish(
    probabilities: SegmentFinishProbabilities,
    rng: np.random.Generator,
) -> FinishResult | None:
    """Sample at most one legal finish from segment probabilities."""

    if not isinstance(
        probabilities,
        SegmentFinishProbabilities,
    ):
        raise TypeError(
            "probabilities must be SegmentFinishProbabilities"
        )

    if not isinstance(
        rng,
        np.random.Generator,
    ):
        raise TypeError(
            "rng must be numpy.random.Generator"
        )

    candidates = build_finish_candidates(
        probabilities
    )

    # Draw all candidate occurrence rolls first so candidate-channel
    # randomness does not depend on earlier occurrence outcomes.
    occurrence_rolls = rng.random(
        len(candidates)
    )

    successful_candidates = [
        candidate
        for candidate, roll in zip(
            candidates,
            occurrence_rolls,
            strict=True,
        )
        if roll < candidate.probability
    ]

    if not successful_candidates:
        return None

    sampled_seconds = rng.integers(
        low=1,
        high=SEGMENT_SECONDS + 1,
        size=len(successful_candidates),
    )

    sampled_candidates = tuple(
        SampledFinishCandidate(
            candidate=candidate,
            elapsed_seconds_in_segment=int(
                elapsed_second
            ),
        )
        for candidate, elapsed_second in zip(
            successful_candidates,
            sampled_seconds,
            strict=True,
        )
    )

    return resolve_sampled_finish_candidates(
        sampled_candidates
    )
