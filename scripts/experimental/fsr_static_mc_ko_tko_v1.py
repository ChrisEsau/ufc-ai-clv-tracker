"""Shadow KO/TKO finish layer for Static FSR MC Damage Reservoir V1.

This module keeps the calibrated Damage Reservoir V1 engine intact and adds the
next research step from ``docs/FSR_KO_DAMAGE_RESERVOIR_V1.md``: a probabilistic
KO/TKO stoppage hazard.

Architecture
------------
Each landed significant strike still:
1. draws stochastic damage from attacker ``striking_power``;
2. depletes defender reservoir;
3. may produce a knockdown through the calibrated Damage V1 KD model.

The same strike also creates a provisional KO/TKO hazard from:
- acute shock fraction;
- remaining reservoir condition;
- whether that strike produced a knockdown;
- whether the defender was already in the short-lived recent-KD state.

All strike-level stoppage hazards generated inside a 10-second segment are
resolved as competing hazards after both fighters' striking has been generated.
That avoids a fixed red-first/blue-second finish bias.

Important boundaries
--------------------
- reservoir exhaustion is NOT an automatic finish;
- catastrophic acute finishes may occur above zero reservoir;
- a KD does NOT subtract bonus reservoir damage;
- no new hidden consciousness/hurt meter is introduced;
- numeric KO/TKO constants are intentionally provisional and require population
  audit/calibration before any promotion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import exp, log
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_v0 as base


FSR_PATH = damage.FSR_PATH

# ---------------------------------------------------------------------------
# PROVISIONAL KO/TKO mechanics constants.
# These make the architecture executable for shadow audits only.
# They are deliberately not calibrated finish-rate locks.
# ---------------------------------------------------------------------------
KO_BASE_LOGIT = -8.50
KO_SHOCK_COEFFICIENT = 18.0
KO_DEPLETION_COEFFICIENT = 6.0
KO_CURRENT_KD_LOGIT_BONUS = 2.00
KO_RECENT_KD_LOGIT_BONUS = 1.00
KO_MAX_STRIKE_PROBABILITY = 0.95


@dataclass
class FinishResult:
    winner: int
    loser: int
    method: str
    probability: float
    strike_damage: float
    shock_fraction: float
    reservoir_fraction_after: float
    knockdown_on_strike: bool
    recent_kd_before: bool
    round: int | None = None
    segment: int | None = None
    clock_start: str | None = None


@dataclass
class KOPath(base.FightPath):
    finish: FinishResult | None = None


class StaticFSRMCKOTKOV1(damage.StaticFSRMCDamageV1):
    """Damage Reservoir V1 plus provisional probabilistic KO/TKO stoppages."""

    def __init__(
        self,
        red: pd.Series,
        blue: pd.Series,
        *,
        rounds: int = base.DEFAULT_ROUNDS,
        seed: int = 7,
    ) -> None:
        super().__init__(red, blue, rounds=rounds, seed=seed)
        self.finish: FinishResult | None = None
        self._segment_finish_candidates: list[dict[str, Any]] = []

    def _ko_probability(
        self,
        defender: int,
        strike_damage: float,
        *,
        knockdown_on_strike: bool,
        recent_kd_before: bool,
    ) -> float:
        """Return provisional strike-level KO/TKO stoppage probability."""
        state = self.damage_state[defender]
        shock_fraction = strike_damage / state.reservoir_capacity
        depletion = 1.0 - state.reservoir_fraction

        logit_p = (
            KO_BASE_LOGIT
            + KO_SHOCK_COEFFICIENT * shock_fraction
            + KO_DEPLETION_COEFFICIENT * depletion
            + (KO_CURRENT_KD_LOGIT_BONUS if knockdown_on_strike else 0.0)
            + (KO_RECENT_KD_LOGIT_BONUS if recent_kd_before else 0.0)
        )
        return float(
            np.clip(
                damage._sigmoid(logit_p),
                0.0,
                KO_MAX_STRIKE_PROBABILITY,
            )
        )

    def _apply_landed_strikes(self, attacker: int, landed: int) -> tuple[float, int]:
        """Apply Damage V1 mechanics and collect strike-level finish hazards."""
        defender = self._other(attacker)
        total_damage = 0.0
        knockdowns = 0

        for _ in range(int(landed)):
            state = self.damage_state[defender]
            recent_kd_before = state.recent_knockdown
            strike_damage = self._draw_strike_damage(attacker)

            # Strike damage is applied exactly once. A KD never subtracts a
            # second arbitrary damage block.
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

            shock_fraction = strike_damage / state.reservoir_capacity
            p_finish = self._ko_probability(
                defender,
                strike_damage,
                knockdown_on_strike=knockdown,
                recent_kd_before=recent_kd_before,
            )
            self._segment_finish_candidates.append(
                {
                    "attacker": attacker,
                    "defender": defender,
                    "probability": p_finish,
                    "strike_damage": strike_damage,
                    "shock_fraction": shock_fraction,
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
            reservoir_fraction_after=float(selected["reservoir_fraction_after"]),
            knockdown_on_strike=bool(selected["knockdown_on_strike"]),
            recent_kd_before=bool(selected["recent_kd_before"]),
        )

    def _generate_striking(self, phase: str) -> list[str]:
        # Reset candidates once per segment, then let Damage V1 generate both
        # fighters' striking and all strike-level damage/KD events.
        self._segment_finish_candidates = []
        notes = super()._generate_striking(phase)
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
                        "striking": (
                            "; ".join(strike_notes)
                            if strike_notes
                            else "no sig attempts"
                        ),
                        "transition": transition_note,
                        "finish": self.finish is not None,
                    }
                )

                if self.finish is not None:
                    return KOPath(events=events, stats=self.stats, finish=self.finish)

        return KOPath(events=events, stats=self.stats, finish=None)


def print_ko_summary(sim: StaticFSRMCKOTKOV1, path: KOPath) -> None:
    damage.print_damage_summary(sim)
    print("\nKO/TKO V1 FINISH SUMMARY")
    print("-" * 100)
    if path.finish is None:
        print("No KO/TKO stoppage drawn.")
    else:
        finish = path.finish
        print(
            f"{sim.names[finish.winner]} def. {sim.names[finish.loser]} "
            f"by {finish.method} | R{finish.round} {finish.clock_start} | "
            f"strike_damage={finish.strike_damage:.2f} | "
            f"shock={finish.shock_fraction:.3f} | "
            f"loser_reservoir={finish.reservoir_fraction_after:.1%} | "
            f"KD_on_strike={finish.knockdown_on_strike} | "
            f"recent_KD_before={finish.recent_kd_before} | "
            f"candidate_p={finish.probability:.3%}"
        )
    print(
        "\nKO/TKO V1 NOTE: finish constants are provisional shadow mechanics. "
        "Population calibration is required before interpreting finish rates."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run shadow Static FSR MC KO/TKO V1"
    )
    parser.add_argument("--red", help="fighter name or fighter_id")
    parser.add_argument("--blue", help="fighter name or fighter_id")
    parser.add_argument("--rounds", type=int, default=base.DEFAULT_ROUNDS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    args = parser.parse_args()

    print(f"[FSR MC KO/TKO V1] loading profiles from {args.fsr_path}", flush=True)
    profiles = damage.load_profiles(args.fsr_path)
    print(f"[FSR MC KO/TKO V1] latest fighter profiles: {len(profiles):,}", flush=True)

    if args.red and args.blue:
        red = base.find_profile(profiles, args.red)
        blue = base.find_profile(profiles, args.blue)
    elif args.red or args.blue:
        raise SystemExit("Provide both --red and --blue, or neither.")
    else:
        red, blue = base._default_matchup(profiles)
        print(
            "[FSR MC KO/TKO V1] no matchup supplied; using first two profiles.",
            flush=True,
        )

    sim = StaticFSRMCKOTKOV1(red, blue, rounds=args.rounds, seed=args.seed)
    path = sim.run()
    base.print_path(path, sim.names)
    print_ko_summary(sim, path)


if __name__ == "__main__":
    main()
