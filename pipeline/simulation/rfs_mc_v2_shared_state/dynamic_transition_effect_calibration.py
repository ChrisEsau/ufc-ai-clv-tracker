"""Dynamic transition-effect calibration for RFS Monte Carlo V2.

This module controls how fatigue, persistent damage, and acute stress reduce
temporary transition capabilities.

Baseline ``FighterTransitionParameters`` remain immutable. No production
defaults are provided; callers must supply an explicit calibration bundle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_effect_calibration import (
    StatePenaltyWeights,
)


def _validate_unit_interval(
    name: str,
    value: float,
) -> None:
    """Validate one finite value constrained to [0, 1]."""

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
class DynamicTransitionEffectCalibration:
    """Map dynamic fighter state to transition-capability penalties.

    Capability families:

    entry:
        Clinch-entry and takedown-entry tendencies.

    completion:
        Takedown completion ability.

    retention:
        Distance, clinch, and ground retention.

    escape:
        Clinch-escape and ground-escape ability.

    reversal:
        Ground reversal ability.

    persistence:
        Takedown and failed-takedown persistence.

    imposition:
        General phase-imposition ability.

    resistance:
        Clinch-entry resistance, takedown resistance, and general
        phase resistance.

    ``minimum_fatigue_effect_multiplier`` prevents maximum performance
    resilience from implying complete fatigue immunity.

    ``minimum_effective_transition_multiplier`` prevents accumulated dynamic
    penalties from fully removing a nonzero baseline transition capability.
    """

    minimum_fatigue_effect_multiplier: float
    minimum_effective_transition_multiplier: float

    entry: StatePenaltyWeights
    completion: StatePenaltyWeights
    retention: StatePenaltyWeights
    escape: StatePenaltyWeights
    reversal: StatePenaltyWeights
    persistence: StatePenaltyWeights
    imposition: StatePenaltyWeights
    resistance: StatePenaltyWeights

    def __post_init__(self) -> None:
        """Validate scalar bounds and nested penalty-weight contracts."""

        _validate_unit_interval(
            "minimum_fatigue_effect_multiplier",
            self.minimum_fatigue_effect_multiplier,
        )
        _validate_unit_interval(
            "minimum_effective_transition_multiplier",
            self.minimum_effective_transition_multiplier,
        )

        nested_fields = {
            "entry": self.entry,
            "completion": self.completion,
            "retention": self.retention,
            "escape": self.escape,
            "reversal": self.reversal,
            "persistence": self.persistence,
            "imposition": self.imposition,
            "resistance": self.resistance,
        }

        for name, value in nested_fields.items():
            if not isinstance(
                value,
                StatePenaltyWeights,
            ):
                raise TypeError(
                    f"{name} must be StatePenaltyWeights"
                )
