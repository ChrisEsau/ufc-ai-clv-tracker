"""Shadow between-round recovery for the strong KD-collapse KO/TKO engine.

This module adds one isolated mechanic to the existing simulator: after a fighter
survives a completed round, a fraction of the missing damage reservoir is restored.
The fraction is controlled by the existing ``recovery_ability`` FSR trait.

Important boundaries
--------------------
- Recovery happens only between completed rounds (R1->R2 and R2->R3 here).
- No in-round regeneration is added.
- No recent-KD suppression is applied in this first diagnostic candidate.
- Stored FSR values and the no-recovery simulator remain unchanged.
- The numeric recovery curve is provisional and exists only to test whether
  recovery fixes the observed excess of late cumulative KO/TKO finishes.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v2 as ko
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse
from scripts.experimental import fsr_static_mc_v0 as base


# Provisional first-pass recovery curve.
# rating 10 -> 5% of missing reservoir restored
# rating 50 -> 20%
# rating 90 -> 35%
ROUND_RECOVERY_BASE_FRACTION = 0.20
ROUND_RECOVERY_PER_RATING_POINT = 0.00375
ROUND_RECOVERY_MIN_FRACTION = 0.05
ROUND_RECOVERY_MAX_FRACTION = 0.35


def round_recovery_fraction(recovery_ability: float) -> float:
    """Return fraction of missing reservoir restored during the corner break."""
    fraction = (
        ROUND_RECOVERY_BASE_FRACTION
        + ROUND_RECOVERY_PER_RATING_POINT * (float(recovery_ability) - 50.0)
    )
    return float(np.clip(fraction, ROUND_RECOVERY_MIN_FRACTION, ROUND_RECOVERY_MAX_FRACTION))


class StaticFSRMCKOTKOV2RoundRecovery(collapse.StaticFSRMCKOTKOV2KDCollapse):
    """Strong KD-collapse compatible simulator with between-round recovery."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.total_round_recovery = [0.0, 0.0]
        self.round_recovery_events: list[dict[str, Any]] = []

    def _apply_between_round_recovery(self, completed_round: int) -> None:
        """Restore a recovery-ability-scaled share of each fighter's missing reservoir."""
        for fighter_index, fighter in enumerate(self.fighters):
            state = self.damage_state[fighter_index]
            missing = max(0.0, state.reservoir_capacity - state.reservoir_current)
            recovery_ability = base._value(fighter, "recovery_ability")
            fraction = round_recovery_fraction(recovery_ability)
            restored = min(missing * fraction, missing)
            before = float(state.reservoir_current)
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
                    "recovery_ability": float(recovery_ability),
                    "fraction_of_missing": float(fraction),
                    "reservoir_before": before,
                    "reservoir_after": float(state.reservoir_current),
                    "restored": actual_restored,
                }
            )

    def run(self) -> ko.KOPath:
        """Run the fight and apply recovery after each survived non-final round."""
        events: list[dict[str, Any]] = []

        for round_no in range(1, self.rounds + 1):
            self.phase = "DISTANCE"
            self.ground_controller = None
            self.clinch_controller = None
            self.clinch_initiator = None

            for segment_no in range(1, base.SEGMENTS_PER_ROUND + 1):
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
                    }
                )

                if self.finish is not None:
                    return ko.KOPath(events=events, stats=self.stats, finish=self.finish)

            if round_no < self.rounds:
                self._apply_between_round_recovery(round_no)

        return ko.KOPath(events=events, stats=self.stats, finish=None)
