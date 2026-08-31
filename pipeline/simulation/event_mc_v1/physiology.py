"""Phase 4B1 landed-strike impact, trauma, and nonterminal knockdowns."""

from dataclasses import dataclass
from math import exp, log

import numpy as np

from .components.actions import ActionAttempt
from .components.profiles import MatchupProfiles, Side
from .events import ConsequenceEvent
from .state import FightState, StateDelta
from .calibration import DEFAULT_CALIBRATION, EventMCCalibration

# Legacy Gamma means/tails retained; resistance architecture is intentionally new.
_DAMAGE, _KD = DEFAULT_CALIBRATION.section("damage"), DEFAULT_CALIBRATION.section("knockdown")
BASE_SHAPE, BASE_SCALE = _DAMAGE["base_shape"], _DAMAGE["base_scale"]
TAIL_SHAPE, TAIL_SCALE = _DAMAGE["tail_shape"], _DAMAGE["tail_scale"]
TAIL_BASE_PROBABILITY = _DAMAGE["tail_probability"]
POWER_RATING_SCALE = _DAMAGE["power_rating_scale"]
IMPACT_SCALE = _DAMAGE["impact_scale"]
DURABILITY_SCALE = _DAMAGE["durability_scale"]
TRAUMA_EROSION_SCALE = _KD["trauma_erosion_scale"]
ACUTE_EROSION_SCALE = _KD["acute_erosion_scale"]
KD_SLOPE = _KD["slope"]
KD_MIDPOINT = log(_KD["midpoint_impact_ratio"])
KD_ACUTE_INCREMENT = _KD["acute_increment"]
ACUTE_HALF_LIFE_SECONDS = _KD["acute_half_life_seconds"]

# Leakage-safe historical validation of the new paired FSR V2 power trait.
# Age is matchup context, not persisted into the FSR rating.
POWER_AGE_CENTER_YEARS = 30.0
POWER_AGE_RATING_POINTS_PER_YEAR = -1.15


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
    calibration: EventMCCalibration = DEFAULT_CALIBRATION

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
        damage, kd_config = self.calibration.section("damage"), self.calibration.section("knockdown")
        power_modifier = payload.dynamic_modifiers.power_multiplier if payload.dynamic_modifiers else 1.0
        effective_power_rating = (
            profile.striking_power
            + POWER_AGE_RATING_POINTS_PER_YEAR
            * (profile.age_years - POWER_AGE_CENTER_YEARS)
        )
        effective_power = exp(
            (effective_power_rating - 50.0) / damage["power_rating_scale"]
        ) * power_modifier
        severity = damage_rng.gamma(damage["base_shape"], damage["base_scale"])
        if damage_rng.random() < damage["tail_probability"]:
            severity += damage_rng.gamma(damage["tail_shape"], damage["tail_scale"])
        impact = max(1e-9, effective_power * severity * damage["impact_scale"])
        trauma = impact * exp(-(target.damage_durability - 50.0) / damage["durability_scale"])
        old_trauma = getattr(state, f"{defender.value}_cumulative_trauma")
        acute = getattr(state, f"{defender.value}_acute_vulnerability")
        new_trauma = old_trauma + trauma
        resistance = max(1e-6, exp((target.knockdown_resistance - 50) / kd_config["resistance_scale"]) * exp(-new_trauma / kd_config["trauma_erosion_scale"]) * exp(-acute / kd_config["acute_erosion_scale"]))
        p_kd = self._sigmoid(kd_config["slope"] * (log(impact / resistance) - log(kd_config["midpoint_impact_ratio"])))
        kd = bool(kd_rng.random() < p_kd)
        values = {f"{defender.value}_cumulative_trauma": new_trauma}
        if kd:
            values[f"{defender.value}_acute_vulnerability"] = acute + kd_config["acute_increment"]
        outcome = PhysiologyOutcome(attacker, defender, state.phase.value, impact, trauma, resistance, p_kd, kd)
        return StateDelta(**values), (ConsequenceEvent(timestamp, "PhysiologyOutcome", outcome),)


@dataclass(frozen=True)
class PhysiologyTimeAdvanceModel:
    stamina_model: object | None = None
    calibration: EventMCCalibration = DEFAULT_CALIBRATION

    def advance(self, state, context, dt_seconds):
        stamina_delta = self.stamina_model.positional_delta(state, dt_seconds) if self.stamina_model else StateDelta()
        decay = exp(-log(2.0) * dt_seconds / self.calibration.section("knockdown")["acute_half_life_seconds"])
        return StateDelta(
            red_stamina=stamina_delta.red_stamina,
            blue_stamina=stamina_delta.blue_stamina,
            red_acute_vulnerability=max(0.0, state.red_acute_vulnerability * decay),
            blue_acute_vulnerability=max(0.0, state.blue_acute_vulnerability * decay),
        )
