"""Shadow KO/TKO V3.1: corrected FSR-defined stamina semantics.

Changes from V3:
- A strike action uses the fighter's stamina at the start of that action; the
  stamina cost is paid after the action resolves. This prevents a combination
  from weakening its own strikes before they are thrown.
- Fatigue acts through an effective striking-power rating rather than by
  multiplying both tail probability and tail magnitude. Full stamina preserves
  the fighter's FSR striking_power; severe fatigue pulls effective power toward
  a low global physics floor.
- Fighter-specific stamina capacity, depletion resistance, performance
  resilience, and recovery remain defined exclusively by FSR-32.
"""
from __future__ import annotations

from math import exp

import numpy as np

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v2_round_recovery as recovery
from scripts.experimental import fsr_static_mc_ko_tko_v3_stamina as v3
from scripts.experimental import fsr_static_mc_v0 as base


# Global simulator physics only; no fighter-specific values live here.
FATIGUED_POWER_FLOOR_RATING = 35.0


class StaticFSRMCKOTKOV31Stamina(v3.StaticFSRMCKOTKOV3Stamina):
    """V3 stamina with corrected action ordering and non-compounded power decay."""

    def _strike_attempts(
        self,
        fighter: int,
        phase: str,
        *,
        rate_multiplier: float = 1.0,
    ) -> int:
        """Generate attempts from start-of-action stamina without spending yet."""
        return recovery.StaticFSRMCKOTKOV2RoundRecovery._strike_attempts(
            self,
            fighter,
            phase,
            rate_multiplier=(rate_multiplier * self.stamina_output_multiplier(fighter)),
        )

    def _generate_strikes_for_fighter(
        self,
        fighter: int,
        phase: str,
        *,
        rate_multiplier: float = 1.0,
    ) -> str | None:
        """Resolve the strikes first, then charge their stamina cost."""
        attempts_before = self.stats[fighter].sig_att
        note = super()._generate_strikes_for_fighter(
            fighter,
            phase,
            rate_multiplier=rate_multiplier,
        )
        attempts = self.stats[fighter].sig_att - attempts_before
        if attempts > 0:
            self._spend_stamina(
                fighter,
                attempts * v3.STAMINA_COST_STRIKE_ATTEMPT,
                f"{phase.lower()}_strike_attempts",
            )
        return note

    def effective_striking_power(self, attacker: int) -> float:
        """Translate FSR power through current stamina/resilience.

        At full stamina the FSR rating is preserved exactly. As stamina falls,
        explosive power decays toward a low global floor rather than being
        multiplied twice inside the damage tail.
        """
        power = base._value(self.fighters[attacker], "striking_power")
        expression = self.stamina_power_multiplier(attacker)
        effective = FATIGUED_POWER_FLOOR_RATING + (
            power - FATIGUED_POWER_FLOOR_RATING
        ) * expression
        return float(np.clip(effective, 10.0, 90.0))

    def _tail_probability(self, attacker: int) -> float:
        effective_power = self.effective_striking_power(attacker)
        return float(
            np.clip(
                damage._sigmoid(
                    damage._logit(damage.POWER_TAIL_BASE_PROBABILITY)
                    + (effective_power - 50.0) / v3.STAMINA_POWER_TAIL_RATING_SCALE
                ),
                0.0,
                0.95,
            )
        )

    def _draw_strike_damage(self, attacker: int) -> float:
        effective_power = self.effective_striking_power(attacker)
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
            tail *= exp(
                (effective_power - 50.0) / v3.STAMINA_TAIL_MAGNITUDE_POWER_SCALE
            )
            raw_damage += tail

        return max(0.0, raw_damage * damage.STRIKE_DAMAGE_SCALE)
