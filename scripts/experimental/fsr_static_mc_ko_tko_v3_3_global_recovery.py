"""Shadow KO/TKO V3.3: V3.2 phase stamina with global round recovery.

V3.3 keeps the V3.2 action/position stamina costs, rolling power-only fatigue,
action-first/fatigue-after timing, stronger fresh-power mapping, damage reservoir,
KD-collapse mechanics, and locked age mechanics unchanged.

Recovery is no longer fighter-specific in this candidate:

- every surviving fighter restores 40% of missing stamina after each round;
- every surviving fighter restores 20% of missing damage reservoir after each round;
- no ``recovery_ability`` or ``stamina_recovery_ability`` FSR field is consumed.

The 20% damage-reservoir value preserves the previous rating-50 population
baseline while removing a recovery rating that had essentially no useful spread.
The 40% stamina value is the new shadow candidate intended to prevent the
population-wide over-depletion observed in V3.2.

This is a shadow calibration candidate, not a production change.
"""
from __future__ import annotations

from scripts.experimental import fsr_static_mc_ko_tko_v3_2_phase_stamina as v32


GLOBAL_STAMINA_RECOVERY_FRACTION = 0.40
GLOBAL_DAMAGE_RECOVERY_FRACTION = 0.20


class StaticFSRMCKOTKOV33GlobalRecovery(v32.StaticFSRMCKOTKOV32PhaseStamina):
    """V3.2 mechanics plus global stamina and damage between-round recovery."""

    def _apply_between_round_recovery(self, completed_round: int) -> None:
        # Damage-reservoir recovery: preserve the former average (rating=50)
        # physics without consulting a fighter-specific recovery FSR trait.
        for fighter_index, state in enumerate(self.damage_state):
            missing = max(0.0, state.reservoir_capacity - state.reservoir_current)
            before = float(state.reservoir_current)
            restored = min(
                missing * GLOBAL_DAMAGE_RECOVERY_FRACTION,
                missing,
            )
            state.reservoir_current = min(
                state.reservoir_capacity,
                state.reservoir_current + restored,
            )
            actual_restored = float(state.reservoir_current - before)
            self.total_round_recovery[fighter_index] += actual_restored
            self.round_recovery_events.append(
                {
                    "after_round": int(completed_round),
                    "fighter": int(fighter_index),
                    "recovery_mode": "global",
                    "fraction_of_missing": float(GLOBAL_DAMAGE_RECOVERY_FRACTION),
                    "reservoir_before": before,
                    "reservoir_after": float(state.reservoir_current),
                    "restored": actual_restored,
                }
            )

        # Stamina recovery: stronger global one-minute corner recovery candidate.
        for fighter_index, state in enumerate(self.stamina_state):
            missing = max(0.0, state.capacity - state.current)
            before = float(state.current)
            restored = min(
                missing * GLOBAL_STAMINA_RECOVERY_FRACTION,
                missing,
            )
            state.current = min(state.capacity, state.current + restored)
            actual_restored = float(state.current - before)
            self.total_stamina_recovered[fighter_index] += actual_restored
            self.stamina_round_events.append(
                {
                    "after_round": int(completed_round),
                    "fighter": int(fighter_index),
                    "recovery_mode": "global",
                    "fraction_of_missing": float(GLOBAL_STAMINA_RECOVERY_FRACTION),
                    "stamina_before": before,
                    "stamina_after": float(state.current),
                    "restored": actual_restored,
                }
            )


# Re-export key V3.2/V3.1 calibration constants for diagnostics.
FATIGUE_CURVE_EXPONENT = v32.FATIGUE_CURVE_EXPONENT
MAX_FATIGUE_RATING_PENALTY = v32.MAX_FATIGUE_RATING_PENALTY
FATIGUE_SENSITIVE_TRAITS = v32.FATIGUE_SENSITIVE_TRAITS
ROLLING_POWER_TAIL_RATING_SCALE = v32.ROLLING_POWER_TAIL_RATING_SCALE
ROLLING_TAIL_MAGNITUDE_POWER_SCALE = v32.ROLLING_TAIL_MAGNITUDE_POWER_SCALE
