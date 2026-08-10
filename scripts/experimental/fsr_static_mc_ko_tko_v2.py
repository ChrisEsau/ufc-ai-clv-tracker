"""Shock-driven KO/TKO V2 shadow layer for the FSR static Monte Carlo.

This replaces the *mechanical idea* used by the rejected KO/TKO V1 experiment
without deleting or rewriting that file. V1 allowed a small generic KO hazard on
every landed significant strike; repeated ordinary-strike opportunities then
accumulated into too many high-reservoir/non-KD finishes.

V2 keeps the locked Damage Reservoir V1 + KD model intact and makes acute shock
the primary KO/TKO signal. The finish curve is intentionally parameterized: no
single numeric KO calibration is locked in this module. Use the companion
population audit to compare candidate curves before selecting constants.

The KO logit uses a nonlinear shock term::

    shock_signal = shock_fraction + shock_curvature * shock_fraction**2

so ordinary shocks remain very weak while large acute shocks rise rapidly.
Reservoir depletion, a KD on the current strike, and recent-KD state act as
secondary susceptibility modifiers.

Important boundaries
--------------------
- no deterministic reservoir-zero finish rule;
- no bonus reservoir damage when a KD occurs;
- no hidden consciousness/hurt meter;
- no generic defender-side per-strike damage-resistance trait;
- KO constants remain research-only until population calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Any

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_v0 as base


@dataclass(frozen=True)
class KOParameters:
    """Research-only KO/TKO curve parameters."""

    name: str
    base_logit: float
    shock_coefficient: float
    shock_curvature: float
    depletion_coefficient: float
    current_kd_logit_bonus: float
    recent_kd_logit_bonus: float
    max_strike_probability: float = 0.95


@dataclass
class FinishResult:
    winner: int
    loser: int
    method: str
    probability: float
    strike_damage: float
    shock_fraction: float
    reservoir_fraction_before: float
    reservoir_fraction_after: float
    knockdown_on_strike: bool
    recent_kd_before: bool
    round: int | None = None
    segment: int | None = None
    clock_start: str | None = None


@dataclass
class KOPath(base.FightPath):
    finish: FinishResult | None = None


class StaticFSRMCKOTKOV2(damage.StaticFSRMCDamageV1):
    """Locked Damage V1/KD mechanics plus parameterized shock-driven KO/TKO."""

    def __init__(
        self,
        red: pd.Series,
        blue: pd.Series,
        *,
        ko_params: KOParameters,
        rounds: int = base.DEFAULT_ROUNDS,
        seed: int = 7,
    ) -> None:
        super().__init__(red, blue, rounds=rounds, seed=seed)
        self.ko_params = ko_params
        self.finish: FinishResult | None = None
        self._segment_finish_candidates: list[dict[str, Any]] = []

    def _ko_probability(
        self,
        defender: int,
        strike_damage: float,
        *,
        reservoir_fraction_before: float,
        knockdown_on_strike: bool,
        recent_kd_before: bool,
    ) -> float:
        """Return strike-level KO/TKO probability for the configured curve."""
        state = self.damage_state[defender]
        shock_fraction = strike_damage / state.reservoir_capacity
        depletion = 1.0 - state.reservoir_fraction
        shock_signal = (
            shock_fraction
            + self.ko_params.shock_curvature * shock_fraction * shock_fraction
        )

        logit_p = (
            self.ko_params.base_logit
            + self.ko_params.shock_coefficient * shock_signal
            + self.ko_params.depletion_coefficient * depletion
            + (
                self.ko_params.current_kd_logit_bonus
                if knockdown_on_strike
                else 0.0
            )
            + (
                self.ko_params.recent_kd_logit_bonus
                if recent_kd_before
                else 0.0
            )
        )
        return float(
            np.clip(
                damage._sigmoid(logit_p),
                0.0,
                self.ko_params.max_strike_probability,
            )
        )

    def _apply_landed_strikes(self, attacker: int, landed: int) -> tuple[float, int]:
        """Apply locked damage/KD mechanics and collect V2 finish candidates."""
        defender = self._other(attacker)
        total_damage = 0.0
        knockdowns = 0

        for _ in range(int(landed)):
            state = self.damage_state[defender]
            reservoir_fraction_before = state.reservoir_fraction
            recent_kd_before = state.recent_knockdown
            strike_damage = self._draw_strike_damage(attacker)

            # Apply reservoir loss once, exactly as Damage V1 does.
            state.reservoir_current = max(
                0.0,
                state.reservoir_current - strike_damage,
            )
            p_kd = self._knockdown_probability(defender, strike_damage)
            knockdown = self.rng.random() < p_kd

            attacker_stats = self.stats[attacker]
            defender_stats = self.stats[defender]
            assert isinstance(attacker_stats, damage.DamageFighterStats)
            assert isinstance(defender_stats, damage.DamageFighterStats)

            attacker_stats.damage_dealt += strike_damage
            defender_stats.damage_absorbed += strike_damage
            attacker_stats.max_single_strike_damage = max(
                attacker_stats.max_single_strike_damage,
                strike_damage,
            )
            defender_stats.max_single_strike_damage = max(
                defender_stats.max_single_strike_damage,
                strike_damage,
            )
            total_damage += strike_damage

            if knockdown:
                attacker_stats.knockdowns_scored += 1
                defender_stats.knockdowns_absorbed += 1
                state.recent_knockdown_segments = max(
                    state.recent_knockdown_segments,
                    damage.RECENT_KD_SEGMENTS,
                )
                knockdowns += 1

            p_finish = self._ko_probability(
                defender,
                strike_damage,
                reservoir_fraction_before=reservoir_fraction_before,
                knockdown_on_strike=knockdown,
                recent_kd_before=recent_kd_before,
            )
            self._segment_finish_candidates.append(
                {
                    "attacker": attacker,
                    "defender": defender,
                    "probability": p_finish,
                    "strike_damage": strike_damage,
                    "shock_fraction": strike_damage / state.reservoir_capacity,
                    "reservoir_fraction_before": reservoir_fraction_before,
                    "reservoir_fraction_after": state.reservoir_fraction,
                    "knockdown_on_strike": knockdown,
                    "recent_kd_before": recent_kd_before,
                }
            )

        return total_damage, knockdowns

    @staticmethod
    def _probability_to_rate(probability: float) -> float:
        p = float(np.clip(probability, 0.0, 1.0 - 1e-12))
        return -log(1.0 - p)

    def _resolve_segment_finish(self) -> FinishResult | None:
        """Resolve at most one KO/TKO from all strike hazards in the segment."""
        positive: list[tuple[dict[str, Any], float]] = []
        for candidate in self._segment_finish_candidates:
            rate = self._probability_to_rate(float(candidate["probability"]))
            if rate > 0.0:
                positive.append((candidate, rate))

        if not positive:
            return None

        total_rate = sum(rate for _, rate in positive)
        if self.rng.random() >= 1.0 - exp(-total_rate):
            return None

        draw = self.rng.random() * total_rate
        running = 0.0
        selected = positive[-1][0]
        for candidate, rate in positive:
            running += rate
            if draw <= running:
                selected = candidate
                break

        return FinishResult(
            winner=int(selected["attacker"]),
            loser=int(selected["defender"]),
            method="KO/TKO",
            probability=float(selected["probability"]),
            strike_damage=float(selected["strike_damage"]),
            shock_fraction=float(selected["shock_fraction"]),
            reservoir_fraction_before=float(selected["reservoir_fraction_before"]),
            reservoir_fraction_after=float(selected["reservoir_fraction_after"]),
            knockdown_on_strike=bool(selected["knockdown_on_strike"]),
            recent_kd_before=bool(selected["recent_kd_before"]),
        )

    def _generate_striking(self, phase: str) -> list[str]:
        # Advance KD timer exactly once per segment, then generate both fighters'
        # striking using the V0 sequence while this subclass records KO hazards.
        self._segment_finish_candidates = []
        self._advance_damage_timers()
        notes = base.StaticFSRMCV0._generate_striking(self, phase)
        self.finish = self._resolve_segment_finish()
        if self.finish is not None:
            notes.append(
                f"STOPPAGE: {self.names[self.finish.winner]} defeats "
                f"{self.names[self.finish.loser]} by KO/TKO"
            )
        return notes

    def run(self) -> KOPath:
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
                    return KOPath(events=events, stats=self.stats, finish=self.finish)

        return KOPath(events=events, stats=self.stats, finish=None)
