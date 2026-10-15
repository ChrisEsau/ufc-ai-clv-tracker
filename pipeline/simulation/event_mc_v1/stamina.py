"""Phase 4A action-driven stamina state and single round-recovery owner."""

from dataclasses import dataclass
from math import exp

import numpy as np

from .components.profiles import FighterProfile, MatchupProfiles, Side
from .state import FightState, StateDelta
from .state import Phase
from .calibration import DEFAULT_CALIBRATION, EventMCCalibration

_STAMINA = DEFAULT_CALIBRATION.section("stamina")
ACTION_COSTS = _STAMINA["action_costs"]
GLOBAL_ROUND_RECOVERY_FRACTION = _STAMINA["round_recovery_fraction"]


@dataclass(frozen=True)
class StaminaModel:
    profiles: MatchupProfiles
    enabled: bool = True
    calibration: EventMCCalibration = DEFAULT_CALIBRATION

    @staticmethod
    def fraction(state: FightState, side: Side) -> float:
        return state.red_stamina if side is Side.RED else state.blue_stamina

    def _cost_multiplier(self, profile: FighterProfile) -> float:
        config = self.calibration.section("stamina")
        raw = exp(-(profile.stamina_depletion_resistance - 50.0) / config["depletion_resistance_scale"])
        return float(np.clip(raw, config["cost_multiplier_min"], config["cost_multiplier_max"]))

    def action_delta(self, state: FightState, side: Side, action_family: str) -> StateDelta:
        costs = self.calibration.section("stamina")["action_costs"]
        if not self.enabled or action_family not in costs:
            return StateDelta()
        profile = self.profiles.fighter(side)
        normalized_cost = costs[action_family] * self._cost_multiplier(profile) / profile.stamina_capacity
        updated = float(np.clip(self.fraction(state, side) - normalized_cost, 0.0, 1.0))
        return StateDelta(**({"red_stamina": updated} if side is Side.RED else {"blue_stamina": updated}))

    def recovery_delta(self, state: FightState) -> StateDelta:
        if not self.enabled:
            return StateDelta()
        fraction = self.calibration.section("stamina")["round_recovery_fraction"]
        recover = lambda value: min(1.0, value + (1.0 - value) * fraction)
        return StateDelta(red_stamina=recover(state.red_stamina), blue_stamina=recover(state.blue_stamina))

    def positional_delta(self, state: FightState, dt_seconds: float) -> StateDelta:
        """Apply V3.2 sustained controller/resistance costs over exact engine dt."""
        if not self.enabled or state.phase is Phase.STANDING:
            return StateDelta()
        controller_value = state.ground_controller
        if controller_value is None:
            return StateDelta()
        controller = Side(controller_value)
        bottom = controller.opponent
        config = self.calibration.section("stamina")
        controller_rate = config["controller_cost_per_second"]
        resistance_rate = config["ground_resistance_cost_per_second"]
        return self._two_side_cost_delta(state, controller, controller_rate * dt_seconds, bottom, resistance_rate * dt_seconds)

    def _two_side_cost_delta(self, state, first, first_cost, second, second_cost):
        values = {Side.RED: state.red_stamina, Side.BLUE: state.blue_stamina}
        for side, base_cost in ((first, first_cost), (second, second_cost)):
            profile = self.profiles.fighter(side)
            values[side] = float(np.clip(values[side] - base_cost * self._cost_multiplier(profile) / profile.stamina_capacity, 0, 1))
        return StateDelta(red_stamina=values[Side.RED], blue_stamina=values[Side.BLUE])


@dataclass(frozen=True)
class StaminaTimeAdvanceModel:
    stamina_model: StaminaModel

    def advance(self, state, context, dt_seconds):
        return self.stamina_model.positional_delta(state, dt_seconds)
