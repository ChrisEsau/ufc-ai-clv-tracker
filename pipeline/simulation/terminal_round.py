"""Exposure-aware handling for partial terminal rounds.

The V0 simulator samples round-level latent performance before deciding whether a
finish occurs. When a fight ends inside that round, the latent full-round line must
be thinned to the sampled exposure rather than being credited as five complete
minutes of activity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class TerminalRoundError(ValueError):
    """Raised when a terminal-round performance line is invalid."""


@dataclass(frozen=True)
class RoundPerformance:
    """One fighter's realized statistics within a simulated round."""

    sig_attempted: int
    sig_landed: int
    takedowns_attempted: int
    takedowns_landed: int
    control_seconds: float
    knockdowns: int

    def __post_init__(self) -> None:
        count_fields = (
            "sig_attempted",
            "sig_landed",
            "takedowns_attempted",
            "takedowns_landed",
            "knockdowns",
        )
        for field_name in count_fields:
            value = int(getattr(self, field_name))
            if value < 0:
                raise TerminalRoundError(f"{field_name} must be nonnegative")
        if self.sig_landed > self.sig_attempted:
            raise TerminalRoundError("sig_landed cannot exceed sig_attempted")
        if self.takedowns_landed > self.takedowns_attempted:
            raise TerminalRoundError(
                "takedowns_landed cannot exceed takedowns_attempted"
            )
        if self.control_seconds < 0:
            raise TerminalRoundError("control_seconds must be nonnegative")


def _thin_successes_and_failures(
    rng: np.random.Generator,
    attempted: int,
    landed: int,
    exposure_fraction: float,
) -> tuple[int, int]:
    """Thin landed and missed attempts separately while preserving constraints."""
    misses = attempted - landed
    partial_landed = int(rng.binomial(landed, exposure_fraction)) if landed else 0
    partial_misses = int(rng.binomial(misses, exposure_fraction)) if misses else 0
    return partial_landed + partial_misses, partial_landed


def thin_round_performance(
    rng: np.random.Generator,
    performance: RoundPerformance,
    exposure_fraction: float,
) -> RoundPerformance:
    """Return the portion of a latent full-round line realized before a finish.

    Count events are independently thinned under a uniform-within-round event-time
    approximation. Landed and missed attempts are thinned separately so the
    resulting line always satisfies landed <= attempted. Control time is scaled
    continuously because it is already measured as duration.
    """
    fraction = float(exposure_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise TerminalRoundError(
            f"exposure_fraction must be between 0 and 1; received {fraction!r}"
        )

    sig_attempted, sig_landed = _thin_successes_and_failures(
        rng,
        performance.sig_attempted,
        performance.sig_landed,
        fraction,
    )
    takedowns_attempted, takedowns_landed = _thin_successes_and_failures(
        rng,
        performance.takedowns_attempted,
        performance.takedowns_landed,
        fraction,
    )
    knockdowns = (
        int(rng.binomial(performance.knockdowns, fraction))
        if performance.knockdowns
        else 0
    )

    return RoundPerformance(
        sig_attempted=sig_attempted,
        sig_landed=sig_landed,
        takedowns_attempted=takedowns_attempted,
        takedowns_landed=takedowns_landed,
        control_seconds=float(performance.control_seconds) * fraction,
        knockdowns=knockdowns,
    )
