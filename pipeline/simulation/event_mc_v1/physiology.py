"""Phase 4B1 landed-strike impact, trauma, and nonterminal knockdowns."""

from dataclasses import dataclass
from math import exp, log

import numpy as np

from .components.actions import ActionAttempt
from .components.profiles import MatchupProfiles, Side
from .events import ConsequenceEvent
from .state import FightState, StateDelta

# Legacy Gamma means/tails retained; resistance architecture is intentionally new.
BASE_SHAPE, BASE_SCALE = 1.0, 2.0
TAIL_SHAPE, TAIL_SCALE = 1.25, 4.8
TAIL_BASE_PROBABILITY = 0.06
POWER_RATING_SCALE = 55.0
IMPACT_SCALE = 0.50
DURABILITY_SCALE = 40.0
TRAUMA_EROSION_SCALE = 80.0
ACUTE_EROSION_SCALE = 1.0
KD_SLOPE = 2.0
KD_MIDPOINT = log(8.0)
KD_ACUTE_INCREMENT = 0.50
ACUTE_HALF_LIFE_SECONDS = 30.0


@dataclass(frozen=True)
class PhysiologyOutcome:
    attacker: Side
    defender: Side
    phase: str
    impact: float
    primary_trauma: float
    current_resistance: float
    knockdown_probability: float
    knockdown: bool


@dataclass(frozen=True)
class ImpactTraumaKnockdownModel:
    profiles: MatchupProfiles

    @staticmethod
    def _sigmoid(value):
        return 1.0 / (1.0 + exp(-float(np.clip(value, -20, 20))))

    def resolve(self, state, payload, timestamp, damage_rng, kd_rng):
        if not isinstance(payload, ActionAttempt) or "strike" not in payload.action_family:
            return StateDelta(), ()
        # Outcome is notified separately; candidate payload alone cannot identify a miss.
        # Candidates mark landed status on ActionAttempt for physiology consumption.
        if not getattr(payload, "landed", False):
            return StateDelta(), ()
        attacker, defender = payload.side, payload.side.opponent
        profile, target = self.profiles.fighter(attacker), self.profiles.fighter(defender)
        power_modifier = payload.dynamic_modifiers.power_multiplier if payload.dynamic_modifiers else 1.0
        effective_power = exp((profile.striking_power - 50.0) / POWER_RATING_SCALE) * power_modifier
        severity = damage_rng.gamma(BASE_SHAPE, BASE_SCALE)
        if damage_rng.random() < TAIL_BASE_PROBABILITY:
            severity += damage_rng.gamma(TAIL_SHAPE, TAIL_SCALE)
        impact = max(1e-9, effective_power * severity * IMPACT_SCALE)
        trauma = impact * exp(-(target.damage_durability - 50.0) / DURABILITY_SCALE)
        old_trauma = getattr(state, f"{defender.value}_cumulative_trauma")
        acute = getattr(state, f"{defender.value}_acute_vulnerability")
        new_trauma = old_trauma + trauma
        resistance = max(1e-6, exp((target.knockdown_resistance - 50) / 32.0) * exp(-new_trauma / TRAUMA_EROSION_SCALE) * exp(-acute / ACUTE_EROSION_SCALE))
        p_kd = self._sigmoid(KD_SLOPE * (log(impact / resistance) - KD_MIDPOINT))
        kd = bool(kd_rng.random() < p_kd)
        values = {f"{defender.value}_cumulative_trauma": new_trauma}
        if kd:
            values[f"{defender.value}_acute_vulnerability"] = acute + KD_ACUTE_INCREMENT
        outcome = PhysiologyOutcome(attacker, defender, state.phase.value, impact, trauma, resistance, p_kd, kd)
        return StateDelta(**values), (ConsequenceEvent(timestamp, "PhysiologyOutcome", outcome),)


@dataclass(frozen=True)
class PhysiologyTimeAdvanceModel:
    stamina_model: object | None = None

    def advance(self, state, context, dt_seconds):
        stamina_delta = self.stamina_model.positional_delta(state, dt_seconds) if self.stamina_model else StateDelta()
        decay = exp(-log(2.0) * dt_seconds / ACUTE_HALF_LIFE_SECONDS)
        return StateDelta(
            red_stamina=stamina_delta.red_stamina,
            blue_stamina=stamina_delta.blue_stamina,
            red_acute_vulnerability=max(0.0, state.red_acute_vulnerability * decay),
            blue_acute_vulnerability=max(0.0, state.blue_acute_vulnerability * decay),
        )
