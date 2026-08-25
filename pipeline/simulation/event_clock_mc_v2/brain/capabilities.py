"""Final runtime capability contract for the generic causal fighter brain."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math


SUBMISSION_CAPABILITY_CENTER = 0.30
SUBMISSION_CAPABILITY_SPAN = 0.40


@dataclass(frozen=True)
class BrainCapabilities:
    standing: float = 0.0
    counter: float = 0.0
    pressure: float = 0.0
    clinch: float = 0.0
    takedown: float = 0.0
    ground_top: float = 0.0
    submission: float = 0.0
    escape: float = 0.0
    reversal: float = 0.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{field.name} must be a finite value in [0, 1]")


@dataclass(frozen=True)
class UnsupportedCapabilityPlaceholders:
    """Neutral fallbacks for callers that do not supply an approved mapping."""

    clinch: float = 0.35
    submission: float = SUBMISSION_CAPABILITY_CENTER
    escape: float = 0.40
    reversal: float = 0.30

    def __post_init__(self) -> None:
        BrainCapabilities(
            clinch=self.clinch,
            submission=self.submission,
            escape=self.escape,
            reversal=self.reversal,
        )


DEFAULT_UNSUPPORTED_CAPABILITIES = UnsupportedCapabilityPlaceholders()


def submission_capability_from_tendency_percentile(percentile: float) -> float:
    """Map submission tendency rank onto the Brain's legacy-centered capability scale.

    The historical audit validates monotonic fighter differentiation, but a direct
    0..1 percentile mapping materially overproduced submission attempts.  Keep the
    historical neutral population center at 0.30 and use a compressed 0.10..0.50
    range so tendency changes fighter allocation without raising the population
    baseline by construction.
    """
    if (
        isinstance(percentile, bool)
        or not isinstance(percentile, (int, float))
        or not math.isfinite(percentile)
        or not 0.0 <= percentile <= 1.0
    ):
        raise ValueError("submission tendency percentile must be finite in [0, 1]")
    return SUBMISSION_CAPABILITY_CENTER + SUBMISSION_CAPABILITY_SPAN * (
        float(percentile) - 0.5
    )


def capabilities_from_percentiles(
    *,
    standing_rate_percentile: float,
    standing_accuracy_percentile: float,
    takedown_rate_percentile: float,
    takedown_completion_percentile: float,
    ground_rate_percentile: float,
    ground_accuracy_percentile: float,
    submission_tendency_percentile: float | None = None,
    placeholders: UnsupportedCapabilityPlaceholders = DEFAULT_UNSUPPORTED_CAPABILITIES,
) -> BrainCapabilities:
    """Translate chronology-safe empirical ranks into Brain capabilities.

    Production Event Clock callers supply ``submission_tendency_percentile``;
    legacy diagnostics that do not have that reference retain the explicit 0.30
    fallback instead of silently inventing a proxy.
    """
    if not isinstance(placeholders, UnsupportedCapabilityPlaceholders):
        raise ValueError("placeholders must be UnsupportedCapabilityPlaceholders")
    values = (
        standing_rate_percentile,
        standing_accuracy_percentile,
        takedown_rate_percentile,
        takedown_completion_percentile,
        ground_rate_percentile,
        ground_accuracy_percentile,
    )
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError("capability percentiles must be finite values in [0, 1]")
    if submission_tendency_percentile is None:
        submission = placeholders.submission
    else:
        submission = submission_capability_from_tendency_percentile(
            submission_tendency_percentile
        )
    return BrainCapabilities(
        standing=(values[0] + values[1]) / 2.0,
        counter=values[1],
        pressure=values[0],
        clinch=placeholders.clinch,
        takedown=(values[2] + values[3]) / 2.0,
        ground_top=(values[4] + values[5]) / 2.0,
        submission=submission,
        escape=placeholders.escape,
        reversal=placeholders.reversal,
    )
