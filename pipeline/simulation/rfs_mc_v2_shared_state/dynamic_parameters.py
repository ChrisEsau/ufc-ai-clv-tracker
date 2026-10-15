"""Immutable dynamic-response parameters for RFS Monte Carlo V2.

These parameters describe how a fighter responds to simulated workload,
damage, and adversity. They are separate from the mutable dynamic state and
from the fighter's baseline phase and transition capabilities.

Future RFS feature mapping will estimate these values from historical cardio,
durability, recovery, and adversity-response signals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FighterDynamicParameters:
    """One fighter's immutable response traits.

    Attributes:
        fatigue_accumulation_resistance:
            Resistance to accumulating fatigue from equivalent workload.
            Higher values mean fatigue builds more slowly.

        fatigue_performance_resilience:
            Ability to preserve performance after fatigue has accumulated.
            Higher values mean fatigue causes smaller capability reductions.

        recovery_ability:
            Ability to recover accumulated fatigue during low-work periods
            and between rounds.

        damage_resistance:
            Resistance to accumulating persistent damage from equivalent
            damaging events.

        acute_stress_resistance:
            Resistance to immediate temporary impairment after adversity.

        acute_stress_recovery:
            Ability to recover from temporary acute stress between segments
            and rounds.
    """

    fatigue_accumulation_resistance: float
    fatigue_performance_resilience: float
    recovery_ability: float
    damage_resistance: float
    acute_stress_resistance: float
    acute_stress_recovery: float

    def __post_init__(self) -> None:
        """Validate normalized response traits."""

        values = {
            "fatigue_accumulation_resistance": (
                self.fatigue_accumulation_resistance
            ),
            "fatigue_performance_resilience": (
                self.fatigue_performance_resilience
            ),
            "recovery_ability": self.recovery_ability,
            "damage_resistance": self.damage_resistance,
            "acute_stress_resistance": (
                self.acute_stress_resistance
            ),
            "acute_stress_recovery": (
                self.acute_stress_recovery
            ),
        }

        for name, value in values.items():
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name} must be numeric"
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    f"{name} must be finite"
                )

            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )
