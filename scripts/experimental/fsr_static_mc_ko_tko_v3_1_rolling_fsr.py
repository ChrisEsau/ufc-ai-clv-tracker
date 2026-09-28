"""Shadow KO/TKO V3.1: rolling effective striking power driven by stamina.

Contract
--------
The stored FSR-32 row is immutable and represents the fighter fresh.
At initialization the configured fight-night age layer is applied by the lower
KO/TKO V2 engine. The resulting age-adjusted profile becomes the persistent
fight-night base state used by every segment. At the start of each 10-second
segment, the simulator derives a temporary ``effective FSR`` from that base state
and current stamina.

Every action in that segment uses the same effective profile. Only after the
segment resolves are stamina costs applied, so an action is never weakened by
the fatigue it creates itself.

For this calibration candidate fatigue changes *striking_power only*. Output,
pressure, precision, accuracy, defense, wrestling, control, and submission
ratings are not reduced by fatigue. Age modifiers are separate and externally
configured in ``config/fsr_age_modifiers.yaml``.
"""
from __future__ import annotations

from math import exp
from typing import Any

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_32_database as fsr32
from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v2 as ko
from scripts.experimental import fsr_static_mc_ko_tko_v3_stamina as v3
from scripts.experimental import fsr_static_mc_v0 as base


# Nonlinear power-only fatigue candidate.
# Neutral-resilience penalty = MAX * missing_stamina ** EXPONENT.
# The 2.5 exponent strongly protects fresh power, then accelerates degradation
# once the stamina reservoir becomes materially depleted.
MAX_FATIGUE_RATING_PENALTY = 45.0
FATIGUE_CURVE_EXPONENT = 2.50
FATIGUE_RESILIENCE_SCALE = 80.0
MIN_EFFECTIVE_FSR_RATING = 10.0

# Stronger fresh power mapping requested for this shadow experiment.
ROLLING_POWER_TAIL_RATING_SCALE = 6.50
ROLLING_TAIL_MAGNITUDE_POWER_SCALE = 55.0

# Explicitly power-only for this experiment. Do not add pressure/precision/etc.
# without separate evidence and approval.
FATIGUE_SENSITIVE_TRAITS = {"striking_power"}


class StaticFSRMCKOTKOV31RollingFSR(v3.StaticFSRMCKOTKOV3Stamina):
    """FSR-32 stamina engine using one rolling effective profile per segment."""

    def __init__(self, red: pd.Series, blue: pd.Series, *args, **kwargs) -> None:
        super().__init__(red, blue, *args, **kwargs)

        # KO/TKO V2 owns the external age layer. Reuse those fight-night base
        # profiles here instead of rebuilding from the raw stored FSR arguments;
        # otherwise segment refresh would silently erase age adjustments.
        configured = getattr(self, "age_effective_fighters", None)
        if configured is None:
            configured = [red, blue]
        self.base_fighters = [
            configured[0].copy(deep=True),
            configured[1].copy(deep=True),
        ]
        self.fighters = [
            self.base_fighters[0].copy(deep=True),
            self.base_fighters[1].copy(deep=True),
        ]
        self.pending_stamina_costs: list[list[tuple[float, str]]] = [[], []]
        self.effective_fsr_events: list[dict[str, Any]] = []

    def _stamina_rating(self, fighter: int, trait: str) -> float:
        value = v3._strict_profile_float(self.base_fighters[fighter], trait)
        if not 10.0 <= value <= 90.0:
            raise ValueError(f"FSR stamina rating {trait} outside 10-90: {value}")
        return value

    def fatigue_penalty(self, fighter: int) -> float:
        missing_fraction = float(np.clip(1.0 - self.stamina_state[fighter].fraction, 0.0, 1.0))
        resilience = self._stamina_rating(
            fighter,
            fsr32.STAMINA_PERFORMANCE_RESILIENCE,
        )
        resilience_multiplier = exp(-(resilience - 50.0) / FATIGUE_RESILIENCE_SCALE)
        nonlinear_missing = missing_fraction ** FATIGUE_CURVE_EXPONENT
        return float(
            max(
                0.0,
                nonlinear_missing
                * MAX_FATIGUE_RATING_PENALTY
                * resilience_multiplier,
            )
        )

    def _effective_profile(self, fighter: int) -> pd.Series:
        profile = self.base_fighters[fighter].copy(deep=True)
        penalty = self.fatigue_penalty(fighter)
        for trait in FATIGUE_SENSITIVE_TRAITS:
            if trait not in profile.index or pd.isna(profile[trait]):
                continue
            profile[trait] = max(
                MIN_EFFECTIVE_FSR_RATING,
                float(profile[trait]) - penalty,
            )
        return profile

    def _refresh_effective_fighters(self, round_no: int, segment_no: int) -> None:
        self.fighters = [self._effective_profile(0), self._effective_profile(1)]
        for fighter in (0, 1):
            self.effective_fsr_events.append(
                {
                    "round": int(round_no),
                    "segment": int(segment_no),
                    "fighter": int(fighter),
                    "stamina_fraction": self.stamina_state[fighter].fraction,
                    "fatigue_penalty": self.fatigue_penalty(fighter),
                    "effective_striking_power": float(
                        self.fighters[fighter].get("striking_power", np.nan)
                    ),
                }
            )

    # Queue action costs during the segment; spend only after all segment actions.
    def _spend_stamina(self, fighter: int, base_cost: float, reason: str) -> float:
        if base_cost > 0.0:
            self.pending_stamina_costs[fighter].append((float(base_cost), str(reason)))
        return 0.0

    def _flush_pending_stamina_costs(self) -> None:
        for fighter in (0, 1):
            pending = self.pending_stamina_costs[fighter]
            self.pending_stamina_costs[fighter] = []
            for base_cost, reason in pending:
                v3.StaticFSRMCKOTKOV3Stamina._spend_stamina(
                    self,
                    fighter,
                    base_cost,
                    reason,
                )

    def _strike_attempts(
        self,
        fighter: int,
        phase: str,
        *,
        rate_multiplier: float = 1.0,
    ) -> int:
        attempts = base.StaticFSRMCV0._strike_attempts(
            self,
            fighter,
            phase,
            rate_multiplier=rate_multiplier,
        )
        if attempts:
            self._spend_stamina(
                fighter,
                attempts * v3.STAMINA_COST_STRIKE_ATTEMPT,
                f"{phase.lower()}_strike_attempts",
            )
        return attempts

    def _td_attempt_hazard(self, attacker: int, phase: str) -> float:
        return base.StaticFSRMCV0._td_attempt_hazard(self, attacker, phase)

    def _maybe_submission_attempt(
        self,
        fighter: int,
        *,
        rate_multiplier: float = 1.0,
    ) -> bool:
        attempted = base.StaticFSRMCV0._maybe_submission_attempt(
            self,
            fighter,
            rate_multiplier=rate_multiplier,
        )
        if attempted:
            self._spend_stamina(
                fighter,
                v3.STAMINA_COST_SUBMISSION_ATTEMPT,
                "submission_attempt",
            )
        return attempted

    def _tail_probability(self, attacker: int) -> float:
        power = base._value(self.fighters[attacker], "striking_power")
        return damage._sigmoid(
            damage._logit(damage.POWER_TAIL_BASE_PROBABILITY)
            + (power - 50.0) / ROLLING_POWER_TAIL_RATING_SCALE
        )

    def _fresh_tail_probability(self, attacker: int) -> float:
        power = float(self.base_fighters[attacker]["striking_power"])
        return damage._sigmoid(
            damage._logit(damage.POWER_TAIL_BASE_PROBABILITY)
            + (power - 50.0) / ROLLING_POWER_TAIL_RATING_SCALE
        )

    def _draw_strike_damage(self, attacker: int) -> float:
        power = base._value(self.fighters[attacker], "striking_power")
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
            tail *= exp((power - 50.0) / ROLLING_TAIL_MAGNITUDE_POWER_SCALE)
            raw_damage += tail
        return max(0.0, raw_damage * damage.STRIKE_DAMAGE_SCALE)

    def run(self) -> ko.KOPath:
        events: list[dict[str, Any]] = []

        for round_no in range(1, self.rounds + 1):
            self.phase = "DISTANCE"
            self.ground_controller = None
            self.clinch_controller = None
            self.clinch_initiator = None

            for segment_no in range(1, base.SEGMENTS_PER_ROUND + 1):
                self._refresh_effective_fighters(round_no, segment_no)
                self.pending_stamina_costs = [[], []]

                phase_start = self.phase
                ground_controller_start = self.ground_controller
                clinch_controller_start = self.clinch_controller

                for stats in self.stats:
                    stats.phase_segments[phase_start] += 1

                strike_notes = self._generate_striking(phase_start)

                if self.finish is not None:
                    self.finish.round = round_no
                    self.finish.segment = segment_no
                    self.finish.clock_start = self._clock_start(segment_no)
                    transition_note = (
                        f"fight stopped: {self.names[self.finish.winner]} "
                        f"KO/TKO {self.names[self.finish.loser]}"
                    )
                elif phase_start == "DISTANCE":
                    transition_note = self._distance_transition()
                elif phase_start == "CLINCH":
                    transition_note = self._clinch_transition()
                else:
                    transition_note = self._ground_transition()

                self._flush_pending_stamina_costs()

                events.append(
                    {
                        "round": round_no,
                        "segment": segment_no,
                        "clock_start": self._clock_start(segment_no),
                        "phase_start": phase_start,
                        "phase_end": self.phase,
                        "top_start": (
                            self.names[ground_controller_start]
                            if ground_controller_start is not None
                            else None
                        ),
                        "top_end": (
                            self.names[self.ground_controller]
                            if self.ground_controller is not None
                            else None
                        ),
                        "clinch_controller_start": (
                            self.names[clinch_controller_start]
                            if clinch_controller_start is not None
                            else None
                        ),
                        "clinch_controller_end": (
                            self.names[self.clinch_controller]
                            if self.clinch_controller is not None
                            else None
                        ),
                        "striking": "; ".join(strike_notes) if strike_notes else "no sig attempts",
                        "transition": transition_note,
                        "finish": self.finish is not None,
                        "red_stamina_after": self.stamina_state[0].fraction,
                        "blue_stamina_after": self.stamina_state[1].fraction,
                    }
                )

                if self.finish is not None:
                    return ko.KOPath(events=events, stats=self.stats, finish=self.finish)

            if round_no < self.rounds:
                self._apply_between_round_recovery(round_no)

        return ko.KOPath(events=events, stats=self.stats, finish=None)


def rolling_fsr_summary(sim: StaticFSRMCKOTKOV31RollingFSR) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for i, name in enumerate(sim.names):
        effective = sim._effective_profile(i)
        rows.append(
            {
                "fighter": name,
                "stamina_fraction": sim.stamina_state[i].fraction,
                "fatigue_penalty": sim.fatigue_penalty(i),
                "base_striking_power": float(sim.base_fighters[i]["striking_power"]),
                "effective_striking_power": float(effective["striking_power"]),
                "fresh_power_tail_probability": sim._fresh_tail_probability(i),
                "effective_power_tail_probability": damage._sigmoid(
                    damage._logit(damage.POWER_TAIL_BASE_PROBABILITY)
                    + (float(effective["striking_power"]) - 50.0)
                    / ROLLING_POWER_TAIL_RATING_SCALE
                ),
                "stamina_spent": sim.total_stamina_spent[i],
                "stamina_recovered": sim.total_stamina_recovered[i],
            }
        )
    return rows
