"""Shadow KO/TKO V3.2: phase-aware stamina expenditure.

This candidate preserves V3.1's rolling power-only fatigue model and 2.5
nonlinear stamina->power curve, while expanding *stamina expenditure* to account
for defensive and positional work that was previously free.

Design contract
---------------
- Stored FSR-32 remains immutable.
- Fatigue still changes striking_power only.
- Output, accuracy, defense, wrestling, control and submission FSR ratings are
  not reduced by fatigue.
- All stamina costs are queued during a 10-second segment and applied only after
  the segment resolves (action first, fatigue second).
- Existing attacker/controller costs remain unchanged.
- New costs represent opponent-side resistance/defense and are additive.
- Damage reservoir, KD resistance/durability, age, KD-collapse and between-round
  recovery mechanics are unchanged.

The relative ordering is a shadow physiology candidate, not a claim of exact
metabolic measurement:
  explosive wrestling/scramble bursts > sustained bottom resistance >
  clinch resistance ~= top positional control > ordinary strike attempts.
"""
from __future__ import annotations

from scripts.experimental import fsr_static_mc_ko_tko_v3_1_rolling_fsr as v31
from scripts.experimental import fsr_static_mc_ko_tko_v3_stamina as v3
from scripts.experimental import fsr_static_mc_v0 as base


# New opponent-side sustained costs. Existing V3 controller costs stay intact:
#   clinch controller = 0.025 / sec
#   ground controller = 0.025 / sec
STAMINA_COST_CLINCH_RESISTANCE_PER_SECOND = 0.030
STAMINA_COST_GROUND_BOTTOM_RESISTANCE_PER_SECOND = 0.035

# New defensive burst costs. These are paid whenever the defensive effort is
# required, even if the defense ultimately fails.
STAMINA_COST_TAKEDOWN_DEFENSE = 2.00
STAMINA_COST_SUBMISSION_DEFENSE = 2.00


class StaticFSRMCKOTKOV32PhaseStamina(v31.StaticFSRMCKOTKOV31RollingFSR):
    """V3.1 rolling power fatigue plus phase-aware stamina expenditure."""

    def _queue_controlled_fighter_cost(
        self,
        controller: int | None,
        per_second: float,
        reason: str,
    ) -> None:
        if controller is None:
            return
        defender = self._other(controller)
        self._spend_stamina(
            defender,
            base.SEGMENT_SECONDS * float(per_second),
            reason,
        )

    def _attempt_takedown(self, attacker: int, source_phase: str) -> str:
        # Parent V3 already charges the shooter for the attempt and adds the
        # success burst when appropriate. The opponent must spend energy
        # defending the shot whether the defense succeeds or fails.
        defender = self._other(attacker)
        note = super()._attempt_takedown(attacker, source_phase)
        self._spend_stamina(
            defender,
            STAMINA_COST_TAKEDOWN_DEFENSE,
            "takedown_defense",
        )
        return note

    def _maybe_submission_attempt(
        self,
        fighter: int,
        *,
        rate_multiplier: float = 1.0,
    ) -> bool:
        # V3.1 intentionally bypasses V3's output suppression while retaining
        # the attacker's submission-attempt stamina cost. Add the defensive
        # burst here without altering submission probability.
        attempted = super()._maybe_submission_attempt(
            fighter,
            rate_multiplier=rate_multiplier,
        )
        if attempted:
            defender = self._other(fighter)
            self._spend_stamina(
                defender,
                STAMINA_COST_SUBMISSION_DEFENSE,
                "submission_defense",
            )
        return attempted

    def _clinch_transition(self) -> str:
        # Cost is based on the position that existed for this segment, even if
        # the transition at the end of the segment changes control/phase.
        controller = self.clinch_controller
        note = super()._clinch_transition()
        self._queue_controlled_fighter_cost(
            controller,
            STAMINA_COST_CLINCH_RESISTANCE_PER_SECOND,
            "clinch_resistance",
        )
        return note

    def _ground_transition(self) -> str:
        # Parent V3 charges top control and any explicit escape/reversal burst.
        # Bottom resistance is a sustained cost paid every controlled segment;
        # a successful escape/reversal therefore remains an additional burst.
        controller = self.ground_controller
        note = super()._ground_transition()
        self._queue_controlled_fighter_cost(
            controller,
            STAMINA_COST_GROUND_BOTTOM_RESISTANCE_PER_SECOND,
            "ground_bottom_resistance",
        )
        return note


# Public aliases make the inherited/calibration constants easy to inspect from
# diagnostics without changing the V3.1 module.
FATIGUE_CURVE_EXPONENT = v31.FATIGUE_CURVE_EXPONENT
MAX_FATIGUE_RATING_PENALTY = v31.MAX_FATIGUE_RATING_PENALTY
FATIGUE_SENSITIVE_TRAITS = v31.FATIGUE_SENSITIVE_TRAITS
ROLLING_POWER_TAIL_RATING_SCALE = v31.ROLLING_POWER_TAIL_RATING_SCALE
ROLLING_TAIL_MAGNITUDE_POWER_SCALE = v31.ROLLING_TAIL_MAGNITUDE_POWER_SCALE
MIN_EFFECTIVE_FSR_RATING = v31.MIN_EFFECTIVE_FSR_RATING

# Existing costs are intentionally not changed; aliases document the full
# candidate alongside the new costs.
STAMINA_COST_STRIKE_ATTEMPT = v3.STAMINA_COST_STRIKE_ATTEMPT
STAMINA_COST_TD_ATTEMPT = v3.STAMINA_COST_TD_ATTEMPT
STAMINA_COST_TD_SUCCESS = v3.STAMINA_COST_TD_SUCCESS
STAMINA_COST_CLINCH_ENTRY = v3.STAMINA_COST_CLINCH_ENTRY
STAMINA_COST_CLINCH_CONTROL_PER_SECOND = v3.STAMINA_COST_CLINCH_CONTROL_PER_SECOND
STAMINA_COST_GROUND_CONTROL_PER_SECOND = v3.STAMINA_COST_GROUND_CONTROL_PER_SECOND
STAMINA_COST_SUBMISSION_ATTEMPT = v3.STAMINA_COST_SUBMISSION_ATTEMPT
STAMINA_COST_ESCAPE = v3.STAMINA_COST_ESCAPE
STAMINA_COST_REVERSAL = v3.STAMINA_COST_REVERSAL
