"""Shadow KO/TKO V3.3: V3.2 phase stamina with global round recovery.

V3.3 keeps the V3.2 action/position stamina costs, rolling power-only fatigue,
action-first/fatigue-after timing, stronger fresh-power mapping, damage reservoir,
KD-collapse mechanics, and locked age mechanics unchanged.

The only physics change from V3.2 is between-round stamina recovery:

- fighter-specific stamina_recovery_ability is removed from the FSR contract;
- every surviving fighter restores 40% of missing stamina after each round;
- damage-reservoir recovery remains the existing V2 recovery mechanic.

This is a shadow calibration candidate, not a production change.
"""
from __future__ import annotations

from scripts.experimental import fsr_static_mc_ko_tko_v3_2_phase_stamina as v32


GLOBAL_STAMINA_RECOVERY_FRACTION = 0.40


class StaticFSRMCKOTKOV33GlobalRecovery(v32.StaticFSRMCKOTKOV32PhaseStamina):
    """V3.2 phase-stamina mechanics plus global between-round stamina recovery."""

    def _apply_between_round_recovery(self, completed_round: int) -> None:
        # Call the damage-recovery implementation directly. Do not call V3's
        # stamina-recovery override because that depends on the removed
        # fighter-specific stamina_recovery_ability field.
        from scripts.experimental import fsr_static_mc_ko_tko_v2_round_recovery as recovery

        recovery.StaticFSRMCKOTKOV2RoundRecovery._apply_between_round_recovery(
            self,
            completed_round,
        )

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


# Re-export the key V3.2/V3.1 calibration constants for diagnostics.
FATIGUE_CURVE_EXPONENT = v32.FATIGUE_CURVE_EXPONENT
MAX_FATIGUE_RATING_PENALTY = v32.MAX_FATIGUE_RATING_PENALTY
FATIGUE_SENSITIVE_TRAITS = v32.FATIGUE_SENSITIVE_TRAITS
ROLLING_POWER_TAIL_RATING_SCALE = v32.ROLLING_POWER_TAIL_RATING_SCALE
ROLLING_TAIL_MAGNITUDE_POWER_SCALE = v32.ROLLING_TAIL_MAGNITUDE_POWER_SCALE
