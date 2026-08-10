"""Shadow stamina reservoir for the aligned static FSR KO/TKO Monte Carlo.

This candidate layers one dynamic stamina system on top of the current
strong-KD-collapse + damage-reservoir + between-round-recovery engine.

Design goals
------------
- Preserve the locked age adjustment and locked KD curve.
- Keep the existing damage reservoir unchanged.
- Start every fighter fully fresh; cardio traits change depletion, resilience,
  and recovery rather than starting capacity.
- Spend stamina on simulated offensive/grappling actions.
- Reduce strike output as stamina falls.
- Reduce expression of striking power substantially faster than strike output.
- Increase the full-stamina influence of the existing ``striking_power`` FSR so
  dangerous fighters are most dangerous early and lose that edge with fatigue.
- Use the existing leakage-safe cardio FSR family when present:
    * fatigue_accumulation_resistance
    * fatigue_performance_resilience
    * recovery_ability
- Fall back to neutral 50 for sparse/missing cardio values.

All numeric values in this module are SHADOW calibration candidates, not
production locks. The baseline recovery engine remains unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any

import numpy as np

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v2_round_recovery as recovery
from scripts.experimental import fsr_static_mc_v0 as base


# ---------------------------------------------------------------------------
# Stamina reservoir.
# ---------------------------------------------------------------------------
STAMINA_CAPACITY = 100.0

# Action costs are deliberately explicit so the first population sweep can tune
# them without changing the architecture.
STAMINA_COST_STRIKE_ATTEMPT = 0.70
STAMINA_COST_TD_ATTEMPT = 3.00
STAMINA_COST_TD_SUCCESS = 1.00
STAMINA_COST_CLINCH_ENTRY = 1.00
STAMINA_COST_CLINCH_CONTROL_PER_SECOND = 0.025
STAMINA_COST_GROUND_CONTROL_PER_SECOND = 0.025
STAMINA_COST_SUBMISSION_ATTEMPT = 2.50
STAMINA_COST_ESCAPE = 1.50
STAMINA_COST_REVERSAL = 2.50

# Fatigue-accumulation resistance changes the cost of equivalent work. A rating
# of 50 is neutral; high-cardio fighters spend less reservoir per action.
STAMINA_COST_RESISTANCE_SCALE = 80.0
STAMINA_COST_MULTIPLIER_MIN = 0.65
STAMINA_COST_MULTIPLIER_MAX = 1.45

# ---------------------------------------------------------------------------
# Performance suppression.
#
# Output is intentionally more resilient than explosive power. Fighters can
# continue producing activity while tired, but the upper-tail finishing threat
# deteriorates much faster.
# ---------------------------------------------------------------------------
OUTPUT_FLOOR_LOW_RESILIENCE = 0.25
OUTPUT_FLOOR_HIGH_RESILIENCE = 0.50
OUTPUT_EXPONENT_LOW_RESILIENCE = 1.40
OUTPUT_EXPONENT_HIGH_RESILIENCE = 0.80

POWER_FLOOR_LOW_RESILIENCE = 0.05
POWER_FLOOR_HIGH_RESILIENCE = 0.20
POWER_EXPONENT_LOW_RESILIENCE = 2.20
POWER_EXPONENT_HIGH_RESILIENCE = 1.40

# Stronger fresh-fighter expression of the existing FSR striking-power signal.
# Baseline Damage V1 uses rating scale 10 and tail-magnitude scale 80.
STAMINA_POWER_TAIL_RATING_SCALE = 6.50
STAMINA_TAIL_MAGNITUDE_POWER_SCALE = 55.0


@dataclass
class StaminaState:
    capacity: float = STAMINA_CAPACITY
    current: float = STAMINA_CAPACITY

    @property
    def fraction(self) -> float:
        if self.capacity <= 0.0:
            return 0.0
        return float(np.clip(self.current / self.capacity, 0.0, 1.0))


class StaticFSRMCKOTKOV3Stamina(recovery.StaticFSRMCKOTKOV2RoundRecovery):
    """Current KO/TKO engine plus action-driven stamina and power decay."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stamina_state = [StaminaState(), StaminaState()]
        self.total_stamina_spent = [0.0, 0.0]
        self.total_stamina_recovered = [0.0, 0.0]
        self.stamina_events: list[dict[str, Any]] = []
        self.stamina_round_events: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Cardio FSR translation.
    # ------------------------------------------------------------------
    def _cardio_rating(self, fighter: int, trait: str) -> float:
        return float(np.clip(base._value(self.fighters[fighter], trait, 50.0), 10.0, 90.0))

    def _resilience_unit(self, fighter: int) -> float:
        rating = self._cardio_rating(fighter, "fatigue_performance_resilience")
        return float(np.clip((rating - 10.0) / 80.0, 0.0, 1.0))

    def _stamina_cost_multiplier(self, fighter: int) -> float:
        rating = self._cardio_rating(fighter, "fatigue_accumulation_resistance")
        multiplier = exp(-(rating - 50.0) / STAMINA_COST_RESISTANCE_SCALE)
        return float(
            np.clip(
                multiplier,
                STAMINA_COST_MULTIPLIER_MIN,
                STAMINA_COST_MULTIPLIER_MAX,
            )
        )

    def stamina_output_multiplier(self, fighter: int) -> float:
        """Return stamina-scaled strike-volume multiplier."""
        s = self.stamina_state[fighter].fraction
        resilience = self._resilience_unit(fighter)
        floor = (
            OUTPUT_FLOOR_LOW_RESILIENCE
            + resilience * (OUTPUT_FLOOR_HIGH_RESILIENCE - OUTPUT_FLOOR_LOW_RESILIENCE)
        )
        exponent = (
            OUTPUT_EXPONENT_LOW_RESILIENCE
            + resilience
            * (OUTPUT_EXPONENT_HIGH_RESILIENCE - OUTPUT_EXPONENT_LOW_RESILIENCE)
        )
        return float(np.clip(floor + (1.0 - floor) * (s**exponent), 0.0, 1.0))

    def stamina_power_multiplier(self, fighter: int) -> float:
        """Return the faster-decaying multiplier for explosive power expression."""
        s = self.stamina_state[fighter].fraction
        resilience = self._resilience_unit(fighter)
        floor = (
            POWER_FLOOR_LOW_RESILIENCE
            + resilience * (POWER_FLOOR_HIGH_RESILIENCE - POWER_FLOOR_LOW_RESILIENCE)
        )
        exponent = (
            POWER_EXPONENT_LOW_RESILIENCE
            + resilience
            * (POWER_EXPONENT_HIGH_RESILIENCE - POWER_EXPONENT_LOW_RESILIENCE)
        )
        return float(np.clip(floor + (1.0 - floor) * (s**exponent), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Stamina bookkeeping.
    # ------------------------------------------------------------------
    def _spend_stamina(self, fighter: int, base_cost: float, reason: str) -> float:
        if base_cost <= 0.0:
            return 0.0
        state = self.stamina_state[fighter]
        before = state.current
        requested = float(base_cost) * self._stamina_cost_multiplier(fighter)
        state.current = max(0.0, state.current - requested)
        spent = float(before - state.current)
        self.total_stamina_spent[fighter] += spent
        self.stamina_events.append(
            {
                "fighter": fighter,
                "reason": reason,
                "base_cost": float(base_cost),
                "cost_multiplier": self._stamina_cost_multiplier(fighter),
                "spent": spent,
                "before": float(before),
                "after": float(state.current),
                "fraction_after": state.fraction,
            }
        )
        return spent

    def _apply_between_round_recovery(self, completed_round: int) -> None:
        # Preserve the existing damage-recovery mechanic exactly.
        super()._apply_between_round_recovery(completed_round)

        for fighter_index, fighter in enumerate(self.fighters):
            state = self.stamina_state[fighter_index]
            missing = max(0.0, state.capacity - state.current)
            recovery_ability = base._value(fighter, "recovery_ability")
            fraction = recovery.round_recovery_fraction(recovery_ability)
            before = float(state.current)
            restored = min(missing * fraction, missing)
            state.current = min(state.capacity, state.current + restored)
            actual_restored = float(state.current - before)
            self.total_stamina_recovered[fighter_index] += actual_restored
            self.stamina_round_events.append(
                {
                    "after_round": int(completed_round),
                    "fighter": int(fighter_index),
                    "recovery_ability": float(recovery_ability),
                    "fraction_of_missing": float(fraction),
                    "stamina_before": before,
                    "stamina_after": float(state.current),
                    "restored": actual_restored,
                }
            )

    # ------------------------------------------------------------------
    # Strike output and severity.
    # ------------------------------------------------------------------
    def _strike_attempts(
        self,
        fighter: int,
        phase: str,
        *,
        rate_multiplier: float = 1.0,
    ) -> int:
        attempts = super()._strike_attempts(
            fighter,
            phase,
            rate_multiplier=(rate_multiplier * self.stamina_output_multiplier(fighter)),
        )
        if attempts:
            self._spend_stamina(
                fighter,
                attempts * STAMINA_COST_STRIKE_ATTEMPT,
                f"{phase.lower()}_strike_attempts",
            )
        return attempts

    def _fresh_tail_probability(self, attacker: int) -> float:
        power = base._value(self.fighters[attacker], "striking_power")
        return damage._sigmoid(
            damage._logit(damage.POWER_TAIL_BASE_PROBABILITY)
            + (power - 50.0) / STAMINA_POWER_TAIL_RATING_SCALE
        )

    def _tail_probability(self, attacker: int) -> float:
        # The stronger full-stamina power signal fades rapidly with exhaustion.
        probability = self._fresh_tail_probability(attacker) * self.stamina_power_multiplier(attacker)
        return float(np.clip(probability, 0.0, 0.95))

    def _draw_strike_damage(self, attacker: int) -> float:
        """Draw strike damage with stronger fresh power and stamina decay."""
        power = base._value(self.fighters[attacker], "striking_power")
        power_expression = self.stamina_power_multiplier(attacker)

        raw_damage = float(
            self.rng.gamma(
                damage.BASE_SEVERITY_GAMMA_SHAPE,
                damage.BASE_SEVERITY_GAMMA_SCALE,
            )
        )

        if self.rng.random() < self._tail_probability(attacker):
            tail = float(
                self.rng.gamma(
                    damage.TAIL_SEVERITY_GAMMA_SHAPE,
                    damage.TAIL_SEVERITY_GAMMA_SCALE,
                )
            )
            tail *= exp((power - 50.0) / STAMINA_TAIL_MAGNITUDE_POWER_SCALE)
            # Tail magnitude deteriorates with stamina in addition to tail frequency.
            tail *= power_expression
            raw_damage += tail

        return max(0.0, raw_damage * damage.STRIKE_DAMAGE_SCALE)

    # ------------------------------------------------------------------
    # Grappling/control workload. These actions spend stamina but V1 does not
    # yet change their success probabilities as fatigue accumulates.
    # ------------------------------------------------------------------
    def _attempt_takedown(self, attacker: int, source_phase: str) -> str:
        before_landed = self.stats[attacker].td_landed
        note = super()._attempt_takedown(attacker, source_phase)
        self._spend_stamina(attacker, STAMINA_COST_TD_ATTEMPT, "takedown_attempt")
        if self.stats[attacker].td_landed > before_landed:
            self._spend_stamina(attacker, STAMINA_COST_TD_SUCCESS, "takedown_success")
        return note

    def _maybe_submission_attempt(
        self,
        fighter: int,
        *,
        rate_multiplier: float = 1.0,
    ) -> bool:
        attempted = super()._maybe_submission_attempt(
            fighter,
            rate_multiplier=rate_multiplier,
        )
        if attempted:
            self._spend_stamina(
                fighter,
                STAMINA_COST_SUBMISSION_ATTEMPT,
                "submission_attempt",
            )
        return attempted

    def _distance_transition(self) -> str:
        phase_before = self.phase
        note = super()._distance_transition()
        if phase_before == "DISTANCE" and self.phase == "CLINCH":
            controller = self.clinch_controller
            if controller is not None:
                self._spend_stamina(controller, STAMINA_COST_CLINCH_ENTRY, "clinch_entry")
        return note

    def _clinch_transition(self) -> str:
        controller = self.clinch_controller
        note = super()._clinch_transition()
        if controller is not None:
            self._spend_stamina(
                controller,
                base.SEGMENT_SECONDS * STAMINA_COST_CLINCH_CONTROL_PER_SECOND,
                "clinch_control",
            )
        return note

    def _ground_transition(self) -> str:
        controller = self.ground_controller
        bottom = self._other(controller) if controller is not None else None
        note = super()._ground_transition()

        if controller is not None:
            self._spend_stamina(
                controller,
                base.SEGMENT_SECONDS * STAMINA_COST_GROUND_CONTROL_PER_SECOND,
                "ground_control",
            )

        if bottom is not None:
            if "REVERSAL" in note:
                self._spend_stamina(bottom, STAMINA_COST_REVERSAL, "reversal")
            elif "escapes to distance" in note:
                self._spend_stamina(bottom, STAMINA_COST_ESCAPE, "ground_escape")

        return note


def stamina_summary(sim: StaticFSRMCKOTKOV3Stamina) -> list[dict[str, float | str]]:
    """Return compact inspectable stamina/cardio state for diagnostics."""
    rows: list[dict[str, float | str]] = []
    for i, name in enumerate(sim.names):
        rows.append(
            {
                "fighter": name,
                "fatigue_accumulation_resistance": sim._cardio_rating(
                    i, "fatigue_accumulation_resistance"
                ),
                "fatigue_performance_resilience": sim._cardio_rating(
                    i, "fatigue_performance_resilience"
                ),
                "recovery_ability": sim._cardio_rating(i, "recovery_ability"),
                "stamina_fraction": sim.stamina_state[i].fraction,
                "stamina_spent": sim.total_stamina_spent[i],
                "stamina_recovered": sim.total_stamina_recovered[i],
                "output_multiplier": sim.stamina_output_multiplier(i),
                "power_multiplier": sim.stamina_power_multiplier(i),
                "fresh_power_tail_probability": sim._fresh_tail_probability(i),
                "current_power_tail_probability": sim._tail_probability(i),
            }
        )
    return rows
