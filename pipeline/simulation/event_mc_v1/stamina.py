"""Phase 4A action-driven stamina state and single round-recovery owner."""

from dataclasses import dataclass
from math import exp

import numpy as np

from .components.profiles import FighterProfile, MatchupProfiles, Side
from .state import FightState, StateDelta
from .state import Phase

ACTION_COSTS = {
    "strike": 0.70,
    "clinch_strike": 0.70,
    "ground_strike": 0.70,
    "takedown": 3.00,
    "clinch_takedown": 3.00,
    "clinch_entry": 1.00,
    "submission_attempt": 2.50,
    "ground_escape": 1.50,
    "ground_reversal": 2.50,
}
GLOBAL_ROUND_RECOVERY_FRACTION = 0.40


@dataclass(frozen=True)
class StaminaModel:
    profiles: MatchupProfiles
    enabled: bool = True

    @staticmethod
    def fraction(state: FightState, side: Side) -> float:
        return state.red_stamina if side is Side.RED else state.blue_stamina

    @staticmethod
    def _cost_multiplier(profile: FighterProfile) -> float:
        raw = exp(-(profile.stamina_depletion_resistance - 50.0) / 80.0)
        return float(np.clip(raw, 0.65, 1.45))

    def action_delta(self, state: FightState, side: Side, action_family: str) -> StateDelta:
        if not self.enabled or action_family not in ACTION_COSTS:
            return StateDelta()
        profile = self.profiles.fighter(side)
        normalized_cost = ACTION_COSTS[action_family] * self._cost_multiplier(profile) / profile.stamina_capacity
        updated = float(np.clip(self.fraction(state, side) - normalized_cost, 0.0, 1.0))
        return StateDelta(**({"red_stamina": updated} if side is Side.RED else {"blue_stamina": updated}))

    def recovery_delta(self, state: FightState) -> StateDelta:
        if not self.enabled:
            return StateDelta()
        recover = lambda value: min(1.0, value + (1.0 - value) * GLOBAL_ROUND_RECOVERY_FRACTION)
        return StateDelta(red_stamina=recover(state.red_stamina), blue_stamina=recover(state.blue_stamina))

    def positional_delta(self, state: FightState, dt_seconds: float) -> StateDelta:
        """Apply V3.2 sustained controller/resistance costs over exact engine dt."""
        if not self.enabled or state.phase is Phase.DISTANCE:
            return StateDelta()
        controller_value = state.clinch_controller if state.phase is Phase.CLINCH else state.ground_controller
        if controller_value is None:
            return StateDelta()
        controller = Side(controller_value)
        bottom = controller.opponent
        controller_rate = 0.025
        resistance_rate = 0.030 if state.phase is Phase.CLINCH else 0.035
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
