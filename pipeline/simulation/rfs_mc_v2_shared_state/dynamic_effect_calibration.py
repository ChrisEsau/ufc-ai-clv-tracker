"""Dynamic performance-effect calibration for RFS Monte Carlo V2.

Dynamic-state accumulation and recovery are calibrated separately in
``DynamicStateCalibration``.

This module controls how accumulated fatigue, persistent damage, and acute
stress temporarily reduce fighter capabilities. It does not mutate baseline
fighter parameters.

No production-calibrated defaults are supplied. Callers must provide an
explicit calibration bundle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _validate_unit_interval(
    name: str,
    value: float,
) -> None:
    """Validate one finite normalized calibration value."""

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
class StatePenaltyWeights:
    """Maximum penalties contributed by each dynamic-state dimension.

    Each value represents the maximum capability reduction attributable to
    that state dimension when its effective burden equals one.
    """

    fatigue: float
    damage: float
    acute_stress: float

    def __post_init__(self) -> None:
        """Validate normalized penalty weights."""

        for name, value in vars(self).items():
            _validate_unit_interval(
                name,
                value,
            )


@dataclass(frozen=True)
class DynamicEffectCalibration:
    """Complete mapping from dynamic state to capability penalties.

    Capability families:

    output:
        Strike and submission attempt rates.

    accuracy:
        Distance, clinch, and ground striking accuracy.

    power:
        Knockdown probability and damaging-clinch probability.

    control:
        Expected clinch and ground control duration.

    grappling:
        Position advancement, escape, reversal, and scramble activity.

    defense:
        Submission defense. Transition-defense effects are handled separately
        when effective transition parameters are implemented.

    ``minimum_fatigue_effect_multiplier`` prevents maximum fatigue-performance
    resilience from automatically creating complete immunity to fatigue.

    ``minimum_effective_capability_multiplier`` prevents accumulated penalties
    from producing negative capabilities.
    """

    minimum_fatigue_effect_multiplier: float
    minimum_effective_capability_multiplier: float

    output: StatePenaltyWeights
    accuracy: StatePenaltyWeights
    power: StatePenaltyWeights
    control: StatePenaltyWeights
    grappling: StatePenaltyWeights
    defense: StatePenaltyWeights

    def __post_init__(self) -> None:
        """Validate scalar bounds and nested contract types."""

        _validate_unit_interval(
            "minimum_fatigue_effect_multiplier",
            self.minimum_fatigue_effect_multiplier,
        )
        _validate_unit_interval(
            "minimum_effective_capability_multiplier",
            self.minimum_effective_capability_multiplier,
        )

        nested_fields = {
            "output": self.output,
            "accuracy": self.accuracy,
            "power": self.power,
            "control": self.control,
            "grappling": self.grappling,
            "defense": self.defense,
        }

        for name, value in nested_fields.items():
            if not isinstance(
                value,
                StatePenaltyWeights,
            ):
                raise TypeError(
                    f"{name} must be StatePenaltyWeights"
                )
