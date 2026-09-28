"""Derived dynamic modifier seam; consumers do not reach into stamina internals."""

from dataclasses import dataclass

import numpy as np

from .components.profiles import FighterProfile, Side
from .state import FightState
from .calibration import DEFAULT_CALIBRATION, EventMCCalibration


@dataclass(frozen=True)
class DynamicModifiers:
    output_multiplier: float
    power_multiplier: float


class DynamicModifierProvider:
    def __init__(self, calibration: EventMCCalibration = DEFAULT_CALIBRATION):
        self.calibration = calibration

    def modifiers(self, profile: FighterProfile, state: FightState, side: Side) -> DynamicModifiers:
        stamina = state.red_stamina if side is Side.RED else state.blue_stamina
        c = self.calibration.section("dynamic_modifiers")
        resilience = float(np.clip((profile.stamina_performance_resilience - c["resilience_rating_min"]) / c["resilience_rating_range"], 0.0, 1.0))
        output_floor = c["output_floor_low"] + resilience * (c["output_floor_high"] - c["output_floor_low"])
        output_exponent = c["output_exponent_low"] + resilience * (c["output_exponent_high"] - c["output_exponent_low"])
        power_floor = c["power_floor_low"] + resilience * (c["power_floor_high"] - c["power_floor_low"])
        power_exponent = c["power_exponent_low"] + resilience * (c["power_exponent_high"] - c["power_exponent_low"])
        return DynamicModifiers(
            output_multiplier=float(np.clip(output_floor + (1.0 - output_floor) * stamina**output_exponent, 0.0, 1.0)),
            power_multiplier=float(np.clip(power_floor + (1.0 - power_floor) * stamina**power_exponent, 0.0, 1.0)),
        )
