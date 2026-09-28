"""Final runtime capability contract for the generic causal fighter brain."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math


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
    """Uncalibrated neutral values retained from Standard Fighter V1 research."""

    clinch: float = 0.35
    submission: float = 0.30
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


def capabilities_from_percentiles(
    *,
    standing_rate_percentile: float,
    standing_accuracy_percentile: float,
    takedown_rate_percentile: float,
    takedown_completion_percentile: float,
    ground_rate_percentile: float,
    ground_accuracy_percentile: float,
    placeholders: UnsupportedCapabilityPlaceholders = DEFAULT_UNSUPPORTED_CAPABILITIES,
) -> BrainCapabilities:
    """Preserve the validated FSR V3 percentile-to-capability semantics.

    Population percentile construction and cold-start reporting remain upstream;
    this pure runtime boundary neither reads FSR artifacts nor alters cold starts.
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
    return BrainCapabilities(
        standing=(values[0] + values[1]) / 2.0,
        counter=values[1],
        pressure=values[0],
        clinch=placeholders.clinch,
        takedown=(values[2] + values[3]) / 2.0,
        ground_top=(values[4] + values[5]) / 2.0,
        submission=placeholders.submission,
        escape=placeholders.escape,
        reversal=placeholders.reversal,
    )
