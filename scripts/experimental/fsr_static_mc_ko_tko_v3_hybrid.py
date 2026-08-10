"""Shadow KO/TKO V3 hybrid finish architecture.

Purpose
-------
Keep the useful accumulated-damage reservoir from Damage V1, but stop requiring
that reservoir exhaustion be the only route to a KO/TKO finish.

This candidate preserves the locked strike-severity, KD-probability, and age
mechanics from KO/TKO V2 while introducing three explicit finish routes:

1. acute_ko
   A confirmed knockdown can immediately become a KO when the KD-causing shock
   is severe enough. This can happen while substantial reservoir remains.

2. post_kd_tko
   Landed follow-up strikes while the defender is in the existing recent-KD
   window can trigger a TKO hazard. This models referee stoppage / inability to
   intelligently defend without inventing another hidden health meter.

3. cumulative_exhaustion
   Reservoir exhaustion remains an absolute terminal safeguard, but it is no
   longer the sole finish mechanism.

Between-round recovery is also included using the existing provisional
``recovery_ability`` curve. A recent-KD window is cleared by the one-minute
corner break because that transient state is only 30 seconds long.

Important boundaries
--------------------
- No KD-collapse reservoir bonus.
- No post-KD strike-damage multiplier.
- No consciousness, hurt, or TKO meter.
- No in-round regeneration.
- No KD suppression of between-round recovery yet.
- No phase- or target-specific strike severity yet.
- Stored FSR profiles remain untouched.
- All new finish-hazard constants below are PROVISIONAL architecture-enabling
  values, not calibration locks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v2 as ko
from scripts.experimental import fsr_static_mc_ko_tko_v2_round_recovery as recovery
from scripts.experimental import fsr_static_mc_v0 as base


# ---------------------------------------------------------------------------
# Provisional acute-KO hazard, conditional on a confirmed KD.
#
# Conditioning on the KD is deliberate: the locked KD model has already used
# strike shock, KD resistance, depletion, and recent-KD vulnerability. The KO
# hazard therefore focuses on how catastrophic that confirmed disruption was,
# while allowing accumulated damage to increase susceptibility.
# ---------------------------------------------------------------------------
ACUTE_KO_GIVEN_KD_BASE_LOGIT = -3.00
ACUTE_KO_SHOCK_COEFFICIENT = 30.0
ACUTE_KO_DEPLETION_COEFFICIENT = 1.00
ACUTE_KO_MAX_PROBABILITY = 0.85

# ---------------------------------------------------------------------------
# Provisional post-KD TKO hazard.
#
# This is evaluated only on landed FOLLOW-UP strikes when recent_kd was already
# active before the strike. Recovery ability acts here as a fighter-specific
# ability to re-stabilize after adversity. Ground top position increases the
# stoppage hazard because unanswered follow-up offense is structurally easier
# to sustain there; clinch receives a smaller bonus.
# ---------------------------------------------------------------------------
POST_KD_TKO_BASE_LOGIT = -4.25
POST_KD_TKO_SHOCK_COEFFICIENT = 18.0
POST_KD_TKO_DEPLETION_COEFFICIENT = 2.00
POST_KD_TKO_RECOVERY_SCALE = 25.0
POST_KD_TKO_GROUND_TOP_LOGIT_BONUS = 0.75
POST_KD_TKO_CLINCH_LOGIT_BONUS = 0.25
POST_KD_TKO_MAX_PROBABILITY = 0.90

REQUIRED_HYBRID_COLUMNS = {"recovery_ability"}


@dataclass
class HybridFinishResult(ko.FinishResult):
    """KO/TKO finish plus the explicit V3 finish route and event diagnostics."""

    finish_route: str = ""
    event_probability: float = 1.0
    shock_fraction: float = 0.0
    reservoir_fraction_before: float = 1.0
    reservoir_fraction_after: float = 1.0


class StaticFSRMCKOTKOV3Hybrid(ko.StaticFSRMCKOTKOV2):
    """Reservoir + acute KO + post-KD TKO + between-round recovery."""

    def __init__(self, red: pd.Series, blue: pd.Series, *args, **kwargs) -> None:
        missing = [
            col
            for col in sorted(REQUIRED_HYBRID_COLUMNS)
            if col not in red.index or col not in blue.index
        ]
        if missing:
            raise ValueError(
                f"FSR profiles missing hybrid finish traits: {sorted(set(missing))}"
            )

        super().__init__(red, blue, *args, **kwargs)
        self.total_round_recovery = [0.0, 0.0]
        self.round_recovery_events: list[dict[str, Any]] = []
        self.finish_route_counts = {
            "acute_ko": 0,
            "post_kd_tko": 0,
            "cumulative_exhaustion": 0,
        }
        self.acute_ko_checks = 0
        self.post_kd_tko_checks = 0

    def _acute_ko_probability_given_kd(
        self,
        defender: int,
        strike_damage: float,
    ) -> float:
        """Return P(immediate KO | this landed strike already caused a KD)."""
        state = self.damage_state[defender]
        shock_fraction = float(strike_damage) / state.reservoir_capacity
        depletion = 1.0 - state.reservoir_fraction
        logit_p = (
            ACUTE_KO_GIVEN_KD_BASE_LOGIT
            + ACUTE_KO_SHOCK_COEFFICIENT * shock_fraction
            + ACUTE_KO_DEPLETION_COEFFICIENT * depletion
        )
        return float(
            np.clip(
                damage._sigmoid(logit_p),
                0.0,
                ACUTE_KO_MAX_PROBABILITY,
            )
        )

    def _post_kd_tko_probability(
        self,
        attacker: int,
        defender: int,
        strike_damage: float,
    ) -> float:
        """Return TKO hazard for one landed follow-up strike during recent-KD state."""
        state = self.damage_state[defender]
        shock_fraction = float(strike_damage) / state.reservoir_capacity
        depletion = 1.0 - state.reservoir_fraction
        recovery_ability = base._value(self.fighters[defender], "recovery_ability")

        phase_bonus = 0.0
        if self.phase == "GROUND" and self.ground_controller == attacker:
            phase_bonus = POST_KD_TKO_GROUND_TOP_LOGIT_BONUS
        elif self.phase == "CLINCH":
            phase_bonus = POST_KD_TKO_CLINCH_LOGIT_BONUS

        logit_p = (
            POST_KD_TKO_BASE_LOGIT
            + POST_KD_TKO_SHOCK_COEFFICIENT * shock_fraction
            + POST_KD_TKO_DEPLETION_COEFFICIENT * depletion
            + (50.0 - recovery_ability) / POST_KD_TKO_RECOVERY_SCALE
            + phase_bonus
        )
        return float(
            np.clip(
                damage._sigmoid(logit_p),
                0.0,
                POST_KD_TKO_MAX_PROBABILITY,
            )
        )

    def _set_finish(
        self,
        *,
        attacker: int,
        defender: int,
        route: str,
        raw_damage: float,
        reservoir_before: float,
        knockdown: bool,
        recent_kd_before: bool,
        event_probability: float,
    ) -> None:
        state = self.damage_state[defender]
        capacity = state.reservoir_capacity
        self.finish = HybridFinishResult(
            winner=attacker,
            loser=defender,
            method="KO/TKO",
            raw_strike_damage=float(raw_damage),
            # V3 intentionally has no post-KD damage multiplier. Effective
            # strike damage therefore equals the actual sampled strike damage.
            effective_strike_damage=float(raw_damage),
            reservoir_before=float(reservoir_before),
            reservoir_after=float(state.reservoir_current),
            knockdown_on_strike=bool(knockdown),
            recent_kd_before=bool(recent_kd_before),
            finish_route=route,
            event_probability=float(event_probability),
            shock_fraction=float(raw_damage / capacity),
            reservoir_fraction_before=float(
                np.clip(reservoir_before / capacity, 0.0, 1.0)
            ),
            reservoir_fraction_after=float(state.reservoir_fraction),
        )
        self.finish_route_counts[route] += 1

    def _apply_landed_strikes(self, attacker: int, landed: int) -> tuple[float, int]:
        """Apply strike damage, then evaluate acute and follow-up finish hazards."""
        defender = self._other(attacker)
        total_damage = 0.0
        knockdowns = 0

        for _ in range(int(landed)):
            if self.finish is not None:
                break

            state = self.damage_state[defender]
            recent_kd_before = state.recent_knockdown
            reservoir_before = float(state.reservoir_current)

            # Keep the empirically validated Damage V1 strike-severity draw.
            # Unlike KO/TKO V2, recent KD does NOT multiply raw damage here.
            raw_damage = self._draw_strike_damage(attacker)
            state.reservoir_current = max(
                0.0,
                state.reservoir_current - raw_damage,
            )

            # Keep the locked KD probability architecture unchanged.
            p_kd = self._knockdown_probability(defender, raw_damage)
            knockdown = self.rng.random() < p_kd

            attacker_stats = self.stats[attacker]
            defender_stats = self.stats[defender]
            assert isinstance(attacker_stats, damage.DamageFighterStats)
            assert isinstance(defender_stats, damage.DamageFighterStats)

            attacker_stats.damage_dealt += raw_damage
            defender_stats.damage_absorbed += raw_damage
            attacker_stats.max_single_strike_damage = max(
                attacker_stats.max_single_strike_damage,
                raw_damage,
            )
            defender_stats.max_single_strike_damage = max(
                defender_stats.max_single_strike_damage,
                raw_damage,
            )
            total_damage += raw_damage

            if knockdown:
                attacker_stats.knockdowns_scored += 1
                defender_stats.knockdowns_absorbed += 1
                state.recent_knockdown_segments = max(
                    state.recent_knockdown_segments,
                    damage.RECENT_KD_SEGMENTS,
                )
                knockdowns += 1

                # Route 1: a confirmed KD can be a fight-ending acute event even
                # when the accumulated-damage reservoir is nowhere near zero.
                self.acute_ko_checks += 1
                p_acute_ko = self._acute_ko_probability_given_kd(
                    defender,
                    raw_damage,
                )
                if self.rng.random() < p_acute_ko:
                    self._set_finish(
                        attacker=attacker,
                        defender=defender,
                        route="acute_ko",
                        raw_damage=raw_damage,
                        reservoir_before=reservoir_before,
                        knockdown=True,
                        recent_kd_before=recent_kd_before,
                        event_probability=p_acute_ko,
                    )
                    break

            # Route 2: only strikes that land while the fighter was ALREADY in
            # recent-KD state can trigger the post-KD TKO hazard. The KD-causing
            # strike itself never gets reclassified as a follow-up strike.
            if recent_kd_before:
                self.post_kd_tko_checks += 1
                p_tko = self._post_kd_tko_probability(
                    attacker,
                    defender,
                    raw_damage,
                )
                if self.rng.random() < p_tko:
                    self._set_finish(
                        attacker=attacker,
                        defender=defender,
                        route="post_kd_tko",
                        raw_damage=raw_damage,
                        reservoir_before=reservoir_before,
                        knockdown=knockdown,
                        recent_kd_before=True,
                        event_probability=p_tko,
                    )
                    break

            # Route 3: retain reservoir exhaustion only as an absolute safeguard.
            if state.reservoir_current <= ko.RESERVOIR_FINISH_EPSILON:
                self._set_finish(
                    attacker=attacker,
                    defender=defender,
                    route="cumulative_exhaustion",
                    raw_damage=raw_damage,
                    reservoir_before=reservoir_before,
                    knockdown=knockdown,
                    recent_kd_before=recent_kd_before,
                    event_probability=1.0,
                )
                break

        return total_damage, knockdowns

    def _apply_between_round_recovery(self, completed_round: int) -> None:
        """Restore reservoir between rounds and expire the 30-second KD state."""
        for fighter_index, fighter in enumerate(self.fighters):
            state = self.damage_state[fighter_index]
            missing = max(0.0, state.reservoir_capacity - state.reservoir_current)
            recovery_ability = base._value(fighter, "recovery_ability")
            fraction = recovery.round_recovery_fraction(recovery_ability)
            restored = min(missing * fraction, missing)
            before = float(state.reservoir_current)
            recent_kd_before_break = bool(state.recent_knockdown)

            state.reservoir_current = min(
                state.reservoir_capacity,
                state.reservoir_current + restored,
            )
            # A one-minute round break is longer than the 30-second recent-KD
            # vulnerability window, so the transient KD state cannot carry over.
            state.recent_knockdown_segments = 0

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
                    "recent_kd_before_break": recent_kd_before_break,
                    "recent_kd_after_break": False,
                }
            )

    def run(self) -> ko.KOPath:
        """Run one fight path with hybrid finish hazards and round recovery."""
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
                    route = getattr(self.finish, "finish_route", "unknown")
                    transition_note = (
                        f"fight stopped: {self.names[self.finish.winner]} "
                        f"KO/TKO {self.names[self.finish.loser]} via {route}"
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
                        "striking": "; ".join(strike_notes)
                        if strike_notes
                        else "no sig attempts",
                        "transition": transition_note,
                        "finish": self.finish is not None,
                        "finish_route": (
                            getattr(self.finish, "finish_route", None)
                            if self.finish is not None
                            else None
                        ),
                    }
                )

                if self.finish is not None:
                    return ko.KOPath(events=events, stats=self.stats, finish=self.finish)

            if round_no < self.rounds:
                self._apply_between_round_recovery(round_no)

        return ko.KOPath(events=events, stats=self.stats, finish=None)
