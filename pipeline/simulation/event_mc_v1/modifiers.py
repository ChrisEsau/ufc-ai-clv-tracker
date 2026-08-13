"""Derived dynamic modifier seam; consumers do not reach into stamina internals."""

from dataclasses import dataclass

import numpy as np

from .components.profiles import FighterProfile, Side
from .state import FightState


@dataclass(frozen=True)
class DynamicModifiers:
    output_multiplier: float
    power_multiplier: float


class DynamicModifierProvider:
    def modifiers(self, profile: FighterProfile, state: FightState, side: Side) -> DynamicModifiers:
        stamina = state.red_stamina if side is Side.RED else state.blue_stamina
        resilience = float(np.clip((profile.stamina_performance_resilience - 10.0) / 80.0, 0.0, 1.0))
        output_floor = 0.25 + resilience * 0.25
        output_exponent = 1.40 - resilience * 0.60
        power_floor = 0.05 + resilience * 0.15
        power_exponent = 2.20 - resilience * 0.80
        return DynamicModifiers(
            output_multiplier=float(np.clip(output_floor + (1.0 - output_floor) * stamina**output_exponent, 0.0, 1.0)),
            power_multiplier=float(np.clip(power_floor + (1.0 - power_floor) * stamina**power_exponent, 0.0, 1.0)),
        )
