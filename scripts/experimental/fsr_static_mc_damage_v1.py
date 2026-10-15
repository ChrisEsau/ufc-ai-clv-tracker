"""Shadow static Monte Carlo with Damage Reservoir V1 mechanics.

This module preserves ``fsr_static_mc_v0.py`` as the frozen decision-only
baseline and layers the research-backed damage architecture on top of it.

Implemented here
----------------
- fighter-specific reservoir capacity from ``damage_durability``;
- stochastic strike-level damage draws;
- ``striking_power`` acting primarily through the upper severity tail;
- acute knockdown probability from strike shock, ``knockdown_resistance``,
  reservoir condition, and recent-KD vulnerability;
- short-lived post-KD vulnerability state;
- deterministic seeding inherited from the V0 path simulator.

Not implemented yet
-------------------
- KO/TKO stoppage hazard;
- damage recovery;
- stamina/fatigue interactions;
- calibrated finish rates.

The damage/KD constants below are SHADOW CALIBRATION locks selected from the V1
mechanics sweeps and full-path finalist validation. They are not production
promotion locks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import exp, log
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_v0 as base


FSR_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_28_shadow/"
    "fsr_28_prefight_snapshots.parquet"
)

REQUIRED_DAMAGE_COLUMNS = {
    "striking_power",
    "knockdown_resistance",
    "damage_durability",
}

# ---------------------------------------------------------------------------
# Reservoir architecture.
# ---------------------------------------------------------------------------
AVERAGE_RESERVOIR_CAPACITY = 100.0
CAPACITY_UNITS_PER_DURABILITY_POINT = 0.50
MIN_RESERVOIR_CAPACITY = 80.0
MAX_RESERVOIR_CAPACITY = 120.0

# ---------------------------------------------------------------------------
# Strike-severity architecture. ``STRIKE_DAMAGE_SCALE`` was selected from the
# reservoir-consumption sweep. Power continues to act primarily on the upper
# tail rather than multiplying every ordinary strike.
#
# Shadow gamma revisions (2026-08-10):
# - Base severity changed from Gamma(1.60, 1.25) to Gamma(1.00, 2.00).
#   Both have raw mean 2.0, so the change preserves average base severity while
#   increasing right-tail mass.
# - Power-tail severity changed from Gamma(2.00, 3.00) to Gamma(1.25, 4.80).
#   Both have raw mean 6.0. The lower shape/higher scale makes power-tail events
#   more variable and more right-tailed without increasing their expected mean.
# These are explicit shadow calibration changes, not production locks.
# ---------------------------------------------------------------------------
STRIKE_DAMAGE_SCALE = 0.50
BASE_SEVERITY_GAMMA_SHAPE = 1.00
BASE_SEVERITY_GAMMA_SCALE = 2.00
POWER_TAIL_BASE_PROBABILITY = 0.06
POWER_TAIL_RATING_SCALE = 10.0
TAIL_SEVERITY_GAMMA_SHAPE = 1.25
TAIL_SEVERITY_GAMMA_SCALE = 4.80
TAIL_MAGNITUDE_POWER_SCALE = 80.0

# ---------------------------------------------------------------------------
# Shadow KD calibration lock selected from the full-path finalist audit:
# - shock coefficient 80 makes KD primarily an acute-shock event;
# - fitted baseline preserves the observed aggregate KD-per-strike target;
# - resistance scale preserves historical power-vs-resistance separation;
# - depletion and recent-KD terms remain secondary susceptibility modifiers.
# ---------------------------------------------------------------------------
KD_BASE_LOGIT = -8.635900
KD_SHOCK_COEFFICIENT = 80.0
KD_RESISTANCE_SCALE = 32.0
KD_DEPLETION_COEFFICIENT = 1.50
KD_RECENT_KD_LOGIT_BONUS = 0.50
RECENT_KD_SEGMENTS = 3


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-9, 1.0 - 1e-9))
    return log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return 1.0 / (1.0 + exp(-x))


def reservoir_capacity_from_durability(durability: float) -> float:
    """Map FSR durability to a bounded reservoir centered on 100 units."""
    value = (
        AVERAGE_RESERVOIR_CAPACITY
        + CAPACITY_UNITS_PER_DURABILITY_POINT * (float(durability) - 50.0)
    )
    return float(np.clip(value, MIN_RESERVOIR_CAPACITY, MAX_RESERVOIR_CAPACITY))


@dataclass
class DamageState:
    reservoir_capacity: float
    reservoir_current: float
    recent_knockdown_segments: int = 0

    @property
    def reservoir_fraction(self) -> float:
        if self.reservoir_capacity <= 0:
            return 0.0
        return float(np.clip(self.reservoir_current / self.reservoir_capacity, 0.0, 1.0))

    @property
    def recent_knockdown(self) -> bool:
        return self.recent_knockdown_segments > 0


@dataclass
class DamageFighterStats(base.FighterStats):
    damage_dealt: float = 0.0
    damage_absorbed: float = 0.0
    knockdowns_scored: int = 0
    knockdowns_absorbed: int = 0
    max_single_strike_damage: float = 0.0


class StaticFSRMCDamageV1(base.StaticFSRMCV0):
    """Static FSR path simulator with isolated Damage Reservoir V1 state."""

    def __init__(
        self,
        red: pd.Series,
        blue: pd.Series,
        *,
        rounds: int = base.DEFAULT_ROUNDS,
        seed: int = 7,
    ) -> None:
        missing = [
            col
            for col in sorted(REQUIRED_DAMAGE_COLUMNS)
            if col not in red.index or col not in blue.index
        ]
        if missing:
            raise ValueError(f"FSR-28 profiles missing damage traits: {sorted(set(missing))}")

        super().__init__(red, blue, rounds=rounds, seed=seed)
        self.stats = [DamageFighterStats(), DamageFighterStats()]
        self.damage_state: list[DamageState] = []
        for fighter in self.fighters:
            durability = base._value(fighter, "damage_durability")
            capacity = reservoir_capacity_from_durability(durability)
            self.damage_state.append(
                DamageState(
                    reservoir_capacity=capacity,
                    reservoir_current=capacity,
                )
            )

    def _advance_damage_timers(self) -> None:
        """Advance short-lived KD vulnerability once per 10-second segment."""
        for state in self.damage_state:
            if state.recent_knockdown_segments > 0:
                state.recent_knockdown_segments -= 1

    def _tail_probability(self, attacker: int) -> float:
        power = base._value(self.fighters[attacker], "striking_power")
        return _sigmoid(
            _logit(POWER_TAIL_BASE_PROBABILITY)
            + (power - 50.0) / POWER_TAIL_RATING_SCALE
        )

    def _draw_strike_damage(self, attacker: int) -> float:
        """Draw one landed-strike damage value with a power-sensitive upper tail."""
        power = base._value(self.fighters[attacker], "striking_power")
        raw_damage = float(
            self.rng.gamma(BASE_SEVERITY_GAMMA_SHAPE, BASE_SEVERITY_GAMMA_SCALE)
        )

        # Power primarily changes how often/magnificently the rare damaging tail
        # appears; it does not multiply every ordinary strike by the same factor.
        if self.rng.random() < self._tail_probability(attacker):
            tail = float(
                self.rng.gamma(TAIL_SEVERITY_GAMMA_SHAPE, TAIL_SEVERITY_GAMMA_SCALE)
            )
            tail *= exp((power - 50.0) / TAIL_MAGNITUDE_POWER_SCALE)
            raw_damage += tail

        return max(0.0, raw_damage * STRIKE_DAMAGE_SCALE)

    def _knockdown_probability(self, defender: int, strike_damage: float) -> float:
        """Return shadow-locked acute KD probability for one landed strike."""
        state = self.damage_state[defender]
        resistance = base._value(self.fighters[defender], "knockdown_resistance")
        shock_fraction = strike_damage / state.reservoir_capacity
        depletion = 1.0 - state.reservoir_fraction

        logit_p = (
            KD_BASE_LOGIT
            + KD_SHOCK_COEFFICIENT * shock_fraction
            + (50.0 - resistance) / KD_RESISTANCE_SCALE
            + KD_DEPLETION_COEFFICIENT * depletion
            + (KD_RECENT_KD_LOGIT_BONUS if state.recent_knockdown else 0.0)
        )
        return float(np.clip(_sigmoid(logit_p), 0.0, 0.95))

    def _apply_landed_strikes(self, attacker: int, landed: int) -> tuple[float, int]:
        defender = self._other(attacker)
        total_damage = 0.0
        knockdowns = 0

        for _ in range(int(landed)):
            damage = self._draw_strike_damage(attacker)
            state = self.damage_state[defender]

            # The strike itself depletes the reservoir. Low reservoir does NOT
            # increase raw strike damage in V1; it changes susceptibility below.
            state.reservoir_current = max(0.0, state.reservoir_current - damage)
            p_kd = self._knockdown_probability(defender, damage)

            attacker_stats = self.stats[attacker]
            defender_stats = self.stats[defender]
            assert isinstance(attacker_stats, DamageFighterStats)
            assert isinstance(defender_stats, DamageFighterStats)

            attacker_stats.damage_dealt += damage
            defender_stats.damage_absorbed += damage
            attacker_stats.max_single_strike_damage = max(
                attacker_stats.max_single_strike_damage, damage
            )
            defender_stats.max_single_strike_damage = max(
                defender_stats.max_single_strike_damage, damage
            )
            total_damage += damage

            if self.rng.random() < p_kd:
                attacker_stats.knockdowns_scored += 1
                defender_stats.knockdowns_absorbed += 1
                state.recent_knockdown_segments = max(
                    state.recent_knockdown_segments,
                    RECENT_KD_SEGMENTS,
                )
                knockdowns += 1

        return total_damage, knockdowns

    def _generate_strikes_for_fighter(
        self,
        fighter: int,
        phase: str,
        *,
        rate_multiplier: float = 1.0,
    ) -> str | None:
        before = self.stats[fighter].sig_landed
        note = super()._generate_strikes_for_fighter(
            fighter,
            phase,
            rate_multiplier=rate_multiplier,
        )
        landed = self.stats[fighter].sig_landed - before
        if landed <= 0:
            return note

        damage, knockdowns = self._apply_landed_strikes(fighter, landed)
        defender = self._other(fighter)
        reservoir_pct = 100.0 * self.damage_state[defender].reservoir_fraction
        suffix = f" dmg={damage:.1f}, opp_res={reservoir_pct:.0f}%"
        if knockdowns:
            suffix += f", KD={knockdowns}"
        return f"{note}{suffix}" if note else suffix.strip()

    def _generate_striking(self, phase: str) -> list[str]:
        # Called exactly once for each simulated 10-second segment by V0.run().
        self._advance_damage_timers()
        return super()._generate_striking(phase)


def load_profiles(path: Path = FSR_PATH) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = set(base.REQUIRED_COLUMNS) | REQUIRED_DAMAGE_COLUMNS
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"FSR-28 artifact missing required columns: {missing}")
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    return base._latest_rows(frame).reset_index(drop=True)


def print_damage_summary(sim: StaticFSRMCDamageV1) -> None:
    print("\nDAMAGE RESERVOIR V1 SUMMARY")
    print("-" * 100)
    for i, name in enumerate(sim.names):
        stats = sim.stats[i]
        state = sim.damage_state[i]
        assert isinstance(stats, DamageFighterStats)
        print(
            f"{name}: capacity={state.reservoir_capacity:.1f}, "
            f"remaining={state.reservoir_current:.1f} ({state.reservoir_fraction:.1%}), "
            f"damage absorbed={stats.damage_absorbed:.1f}, "
            f"KD absorbed={stats.knockdowns_absorbed}, KD scored={stats.knockdowns_scored}"
        )
    print(
        "\nV1 NOTE: reservoir/KD mechanics use shadow calibration locks; "
        "KO/TKO stoppages, recovery, fatigue, and finish calibration remain disabled."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shadow FSR static MC Damage Reservoir V1")
    parser.add_argument("--red", help="fighter name or fighter_id")
    parser.add_argument("--blue", help="fighter name or fighter_id")
    parser.add_argument("--rounds", type=int, default=base.DEFAULT_ROUNDS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    args = parser.parse_args()

    print(f"[FSR MC Damage V1] loading profiles from {args.fsr_path}", flush=True)
    profiles = load_profiles(args.fsr_path)
    print(f"[FSR MC Damage V1] latest fighter profiles: {len(profiles):,}", flush=True)

    if args.red and args.blue:
        red = base.find_profile(profiles, args.red)
        blue = base.find_profile(profiles, args.blue)
    elif args.red or args.blue:
        raise SystemExit("Provide both --red and --blue, or neither.")
    else:
        red, blue = base._default_matchup(profiles)
        print("[FSR MC Damage V1] no matchup supplied; using first two latest profiles.", flush=True)

    sim = StaticFSRMCDamageV1(red, blue, rounds=args.rounds, seed=args.seed)
    path = sim.run()
    base.print_path(path, sim.names)
    print_damage_summary(sim)


if __name__ == "__main__":
    main()
