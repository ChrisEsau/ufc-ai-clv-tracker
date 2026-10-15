"""Shadow KO/TKO V3: FSR-defined dynamic stamina and stronger fresh power.

This candidate layers stamina onto the current strong-KD-collapse + damage
reservoir + between-round recovery engine.

Hard contract
-------------
The MC does not invent fighter-specific stamina parameters and does not reach
back into raw RFS features. Each supplied FSR profile must contain finite values
for:

- stamina_capacity
- stamina_depletion_resistance
- stamina_performance_resilience
- stamina_recovery_ability

Those fields are built by ``build_fsr_32_database.py``. In the first shadow
version the three ratings are exact aliases of the existing leakage-safe FSR
cardio traits and the starting capacity is explicitly persisted on every FSR
row. Future changes to those fighter parameters therefore belong in FSR
creation, not in this simulator.

The locked age mechanic, locked KD curve, damage reservoir, KD-collapse logic,
and damage-recovery architecture remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_32_database as fsr32
from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v2_round_recovery as recovery
from scripts.experimental import fsr_static_mc_v0 as base


FSR_PATH = fsr32.OUTPUT_PATH
REQUIRED_STAMINA_COLUMNS = set(fsr32.STAMINA_COLUMNS)

# Simulator-physics constants. Fighter-specific differences are supplied only
# by FSR-32. These values remain shadow calibration candidates.
STAMINA_COST_STRIKE_ATTEMPT = 0.70
STAMINA_COST_TD_ATTEMPT = 3.00
STAMINA_COST_TD_SUCCESS = 1.00
STAMINA_COST_CLINCH_ENTRY = 1.00
STAMINA_COST_CLINCH_CONTROL_PER_SECOND = 0.025
STAMINA_COST_GROUND_CONTROL_PER_SECOND = 0.025
STAMINA_COST_SUBMISSION_ATTEMPT = 2.50
STAMINA_COST_ESCAPE = 1.50
STAMINA_COST_REVERSAL = 2.50

STAMINA_COST_RESISTANCE_SCALE = 80.0
STAMINA_COST_MULTIPLIER_MIN = 0.65
STAMINA_COST_MULTIPLIER_MAX = 1.45

# Output is deliberately more resilient than explosive power.
OUTPUT_FLOOR_LOW_RESILIENCE = 0.25
OUTPUT_FLOOR_HIGH_RESILIENCE = 0.50
OUTPUT_EXPONENT_LOW_RESILIENCE = 1.40
OUTPUT_EXPONENT_HIGH_RESILIENCE = 0.80

POWER_FLOOR_LOW_RESILIENCE = 0.05
POWER_FLOOR_HIGH_RESILIENCE = 0.20
POWER_EXPONENT_LOW_RESILIENCE = 2.20
POWER_EXPONENT_HIGH_RESILIENCE = 1.40

# Damage V1 uses rating scale 10 / tail-magnitude scale 80. This candidate
# intentionally makes a fresh fighter's striking_power more influential, then
# lets stamina suppress that extra finishing threat as the fight progresses.
STAMINA_POWER_TAIL_RATING_SCALE = 6.50
STAMINA_TAIL_MAGNITUDE_POWER_SCALE = 55.0


@dataclass
class StaminaState:
    capacity: float
    current: float

    @property
    def fraction(self) -> float:
        if self.capacity <= 0.0:
            return 0.0
        return float(np.clip(self.current / self.capacity, 0.0, 1.0))


def _strict_profile_float(profile: pd.Series, name: str) -> float:
    """Read an explicit fighter parameter with no simulator-side fallback."""
    if name not in profile.index:
        raise ValueError(f"FSR profile missing required fighter parameter: {name}")
    raw = profile[name]
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"FSR fighter parameter {name} is not numeric: {raw!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"FSR fighter parameter {name} is not finite: {raw!r}")
    return value


class StaticFSRMCKOTKOV3Stamina(recovery.StaticFSRMCKOTKOV2RoundRecovery):
    """Current KO/TKO engine plus action-driven FSR-defined stamina."""

    def __init__(self, red: pd.Series, blue: pd.Series, *args, **kwargs) -> None:
        for profile in (red, blue):
            for column in sorted(REQUIRED_STAMINA_COLUMNS):
                _strict_profile_float(profile, column)

        super().__init__(red, blue, *args, **kwargs)

        self.stamina_state: list[StaminaState] = []
        self.total_stamina_spent = [0.0, 0.0]
        self.total_stamina_recovered = [0.0, 0.0]
        self.stamina_events: list[dict[str, Any]] = []
        self.stamina_round_events: list[dict[str, Any]] = []

        for fighter in self.fighters:
            capacity = _strict_profile_float(fighter, fsr32.STAMINA_CAPACITY)
            if capacity <= 0.0:
                raise ValueError(f"stamina_capacity must be positive, got {capacity}")
            self.stamina_state.append(
                StaminaState(capacity=float(capacity), current=float(capacity))
            )

    def _stamina_rating(self, fighter: int, trait: str) -> float:
        value = _strict_profile_float(self.fighters[fighter], trait)
        if not 10.0 <= value <= 90.0:
            raise ValueError(f"FSR stamina rating {trait} outside 10-90: {value}")
        return value

    def _resilience_unit(self, fighter: int) -> float:
        rating = self._stamina_rating(
            fighter,
            fsr32.STAMINA_PERFORMANCE_RESILIENCE,
        )
        return float(np.clip((rating - 10.0) / 80.0, 0.0, 1.0))

    def _stamina_cost_multiplier(self, fighter: int) -> float:
        rating = self._stamina_rating(
            fighter,
            fsr32.STAMINA_DEPLETION_RESISTANCE,
        )
        multiplier = exp(-(rating - 50.0) / STAMINA_COST_RESISTANCE_SCALE)
        return float(
            np.clip(
                multiplier,
                STAMINA_COST_MULTIPLIER_MIN,
                STAMINA_COST_MULTIPLIER_MAX,
            )
        )

    def stamina_output_multiplier(self, fighter: int) -> float:
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

    def _spend_stamina(self, fighter: int, base_cost: float, reason: str) -> float:
        if base_cost <= 0.0:
            return 0.0
        state = self.stamina_state[fighter]
        before = state.current
        cost_multiplier = self._stamina_cost_multiplier(fighter)
        requested = float(base_cost) * cost_multiplier
        state.current = max(0.0, state.current - requested)
        spent = float(before - state.current)
        self.total_stamina_spent[fighter] += spent
        self.stamina_events.append(
            {
                "fighter": fighter,
                "reason": reason,
                "base_cost": float(base_cost),
                "cost_multiplier": cost_multiplier,
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

        for fighter_index, _fighter in enumerate(self.fighters):
            state = self.stamina_state[fighter_index]
            missing = max(0.0, state.capacity - state.current)
            rating = self._stamina_rating(
                fighter_index,
                fsr32.STAMINA_RECOVERY_ABILITY,
            )
            fraction = recovery.round_recovery_fraction(rating)
            before = float(state.current)
            restored = min(missing * fraction, missing)
            state.current = min(state.capacity, state.current + restored)
            actual_restored = float(state.current - before)
            self.total_stamina_recovered[fighter_index] += actual_restored
            self.stamina_round_events.append(
                {
                    "after_round": int(completed_round),
                    "fighter": int(fighter_index),
                    "stamina_recovery_ability": float(rating),
                    "fraction_of_missing": float(fraction),
                    "stamina_before": before,
                    "stamina_after": float(state.current),
                    "restored": actual_restored,
                }
            )

    # Output suppression and action costs.
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

    def _td_attempt_hazard(self, attacker: int, phase: str) -> float:
        hazard = super()._td_attempt_hazard(attacker, phase)
        return float(np.clip(hazard * self.stamina_output_multiplier(attacker), 0.0, 0.70))

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
            rate_multiplier=(rate_multiplier * self.stamina_output_multiplier(fighter)),
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

    # Stronger fresh striking power, followed by aggressive fatigue decay.
    def _fresh_tail_probability(self, attacker: int) -> float:
        power = base._value(self.fighters[attacker], "striking_power")
        return damage._sigmoid(
            damage._logit(damage.POWER_TAIL_BASE_PROBABILITY)
            + (power - 50.0) / STAMINA_POWER_TAIL_RATING_SCALE
        )

    def _tail_probability(self, attacker: int) -> float:
        probability = (
            self._fresh_tail_probability(attacker)
            * self.stamina_power_multiplier(attacker)
        )
        return float(np.clip(probability, 0.0, 0.95))

    def _draw_strike_damage(self, attacker: int) -> float:
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
            tail *= power_expression
            raw_damage += tail

        return max(0.0, raw_damage * damage.STRIKE_DAMAGE_SCALE)


def load_profiles(path: Path = FSR_PATH) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = (
        set(base.REQUIRED_COLUMNS)
        | damage.REQUIRED_DAMAGE_COLUMNS
        | REQUIRED_STAMINA_COLUMNS
    )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"FSR-32 artifact missing required MC columns: {missing}")
    for column in REQUIRED_STAMINA_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"FSR-32 contains invalid {column} values")
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    return base._latest_rows(frame).reset_index(drop=True)


def stamina_summary(sim: StaticFSRMCKOTKOV3Stamina) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for i, name in enumerate(sim.names):
        rows.append(
            {
                "fighter": name,
                "stamina_capacity": sim.stamina_state[i].capacity,
                "stamina_depletion_resistance": sim._stamina_rating(
                    i, fsr32.STAMINA_DEPLETION_RESISTANCE
                ),
                "stamina_performance_resilience": sim._stamina_rating(
                    i, fsr32.STAMINA_PERFORMANCE_RESILIENCE
                ),
                "stamina_recovery_ability": sim._stamina_rating(
                    i, fsr32.STAMINA_RECOVERY_ABILITY
                ),
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
