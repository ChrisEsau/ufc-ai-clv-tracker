"""Static transition-parameter contracts for the V2 shared-state engine.

These values are dimensionless fighter strengths between 0 and 1. They are
not final transition probabilities. The transition engine will compare both
fighters and normalize their competing transition scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _validate_unit_interval(
    name: str,
    value: float,
) -> None:
    """Require a finite value between zero and one."""

    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1"
        )


@dataclass(frozen=True)
class FighterTransitionParameters:
    """One fighter's static ability to impose and resist phases."""

    # Distance-state behavior.
    distance_retention: float
    clinch_entry_tendency: float
    clinch_entry_resistance: float

    # Takedown entry and completion.
    takedown_entry_tendency: float
    takedown_completion_ability: float
    takedown_resistance: float
    takedown_persistence: float
    failed_takedown_persistence: float

    # Clinch retention and escape.
    clinch_retention: float
    clinch_escape_ability: float

    # Ground retention, escape, and ownership change.
    ground_retention: float
    ground_escape_ability: float
    reversal_ability: float

    # Broad RFS matchup traits.
    phase_imposition: float
    phase_resistance: float

    def __post_init__(self) -> None:
        """Validate all transition strengths."""

        for name, value in vars(self).items():
            _validate_unit_interval(name, value)
