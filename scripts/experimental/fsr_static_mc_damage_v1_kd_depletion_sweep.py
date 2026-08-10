"""Sweep the Damage V1 reservoir-depletion KD coefficient.

Purpose
-------
Hold the current provisional damage/KD settings fixed and vary only how much
accumulated reservoir depletion increases knockdown susceptibility.

Fixed during this study:
- strike damage scale = 0.50
- KD base logit = -6.40
- KD resistance scale = 32
- recent-KD bonus = current Damage V1 value
- power-tail architecture unchanged

The goal is directional calibration only: depleted fighters should become easier
to knock down, but the 0-25% reservoir state should not create an implausibly
large step-up relative to fresh fighters.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_v0 as base


FSR_PATH = damage.FSR_PATH
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_kd_depletion_sweep.parquet"
)

DAMAGE_SCALE = 0.50
KD_BASE_LOGIT = -6.40
KD_RESISTANCE_SCALE = 32.0
DEPLETION_COEFFICIENTS = [0.0, 0.5, 1.0, 1.5, 1.8, 2.2]

DEFAULT_MATCHUPS = 300
DEFAULT_PATHS_PER_MATCHUP = 10
DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260809


class SweepSim(damage.StaticFSRMCDamageV1):
    def __init__(self, *args: Any, depletion_coefficient: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.depletion_coefficient = float(depletion_coefficient)
        self.strike_records: list[dict[str, Any]] = []

    def _draw_strike_damage(self, attacker: int) -> float:
        return DAMAGE_SCALE * super()._draw_strike_damage(attacker)

    def _knockdown_probability(self, defender: int, strike_damage: float) -> float:
        state = self.damage_state[defender]
        resistance = base._value(self.fighters[defender], "knockdown_resistance")
        shock_fraction = strike_damage / state.reservoir_capacity
        depletion = 1.0 - state.reservoir_fraction
        logit_p = (
            KD_BASE_LOGIT
            + damage.KD_SHOCK_COEFFICIENT * shock_fraction
            + (50.0 - resistance) / KD_RESISTANCE_SCALE
            + self.depletion_coefficient * depletion
            + (
                damage.KD_RECENT_KD_LOGIT_BONUS
                if state.recent_knockdown
                else 0.0
            )
        )
        return float(np.clip(damage._sigmoid(logit_p), 0.0, 0.95))

    def _apply_landed_strikes(self, attacker: int, landed: int) -> tuple[float, int]:
        defender = self._other(attacker)
        total_damage = 0.0
        knockdowns = 0
        for _ in range(int(landed)):
            state = self.damage_state[defender]
            reservoir_before = state.reservoir_fraction
            recent_kd_before = state.recent_knockdown
            damage_value = self._draw_strike_damage(attacker)

            state.reservoir_current = max(0.0, state.reservoir_current - damage_value)
            p_kd = self._knockdown_probability(defender, damage_value)

            attacker_stats = self.stats[attacker]
            defender_stats = self.stats[defender]
            assert isinstance(attacker_stats, damage.DamageFighterStats)
            assert isinstance(defender_stats, damage.DamageFighterStats)
            attacker_stats.damage_dealt += damage_value
            defender_stats.damage_absorbed += damage_value
            total_damage += damage_value

            kd = self.rng.random() < p_kd
            if kd:
                attacker_stats.knockdowns_scored += 1
                defender_stats.knockdowns_absorbed += 1
                state.recent_knockdown_segments = max(
                    state.recent_knockdown_segments,
                    damage.RECENT_KD_SEGMENTS,
                )
                knockdowns += 1

            self.strike_records.append(
                {
                    "reservoir_fraction_before": reservoir_before,
                    "recent_kd_before": int(recent_kd_before),
                    "knockdown": int(kd),
                    "kd_probability": p_kd,
                }
            )
        return total_damage, knockdowns


def _rank_matchups(profiles: pd.DataFrame, n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    pairs = []
    for _ in range(n):
        a, b = rng.choice(len(profiles), size=2, replace=False)
        pairs.append((int(a), int(b)))
    return pairs


def _condition_label(x: float) -> str:
    if x <= 0.25:
        return "0-25%"
    if x <= 0.50:
        return "25-50%"
    if x <= 0.75:
        return "50-75%"
    return "75-100%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Damage V1 KD depletion coefficient")
    parser.add_argument("--matchups", type=int, default=DEFAULT_MATCHUPS)
    parser.add_argument("--paths-per-matchup", type=int, default=DEFAULT_PATHS_PER_MATCHUP)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    args = parser.parse_args()

    profiles = damage.load_profiles(args.fsr_path)
    rng = np.random.default_rng(args.seed)
    matchups = _rank_matchups(profiles, args.matchups, rng)
    path_seeds = [
        int(rng.integers(0, 2**31 - 1))
        for _ in range(args.matchups * args.paths_per_matchup)
    ]

    all_rows: list[dict[str, Any]] = []
    total_paths = args.matchups * args.paths_per_matchup

    for coefficient in DEPLETION_COEFFICIENTS:
        strike_rows: list[dict[str, Any]] = []
        path_counter = 0
        seed_i = 0
        for red_i, blue_i in matchups:
            red = profiles.iloc[red_i]
            blue = profiles.iloc[blue_i]
            for _ in range(args.paths_per_matchup):
                sim = SweepSim(
                    red,
                    blue,
                    rounds=args.rounds,
                    seed=path_seeds[seed_i],
                    depletion_coefficient=coefficient,
                )
                seed_i += 1
                sim.run()
                strike_rows.extend(sim.strike_records)
                path_counter += 1
                if path_counter % 1000 == 0 or path_counter == total_paths:
                    print(
                        f"[KD depletion sweep] coeff={coefficient:.2f} "
                        f"paths {path_counter:,}/{total_paths:,}; strikes={len(strike_rows):,}",
                        flush=True,
                    )

        strikes = pd.DataFrame(strike_rows)
        strikes["reservoir_condition"] = strikes["reservoir_fraction_before"].map(_condition_label)
        overall = {
            "depletion_coefficient": coefficient,
            "landed_strikes": len(strikes),
            "overall_kd_per_strike": strikes["knockdown"].mean(),
        }
        for label in ["75-100%", "50-75%", "25-50%", "0-25%"]:
            g = strikes[strikes["reservoir_condition"] == label]
            overall[f"kd_rate_{label}"] = g["knockdown"].mean() if len(g) else np.nan
            overall[f"strike_count_{label}"] = len(g)
        fresh = overall["kd_rate_75-100%"]
        depleted = overall["kd_rate_0-25%"]
        overall["depleted_to_fresh_ratio"] = (
            depleted / fresh if pd.notna(fresh) and fresh > 0 else np.nan
        )
        all_rows.append(overall)

    out = pd.DataFrame(all_rows)
    print("\n" + "=" * 120)
    print("DAMAGE RESERVOIR V1 — KD DEPLETION COEFFICIENT SWEEP")
    print("=" * 120)
    print(f"fixed damage scale: {DAMAGE_SCALE:.2f}")
    print(f"fixed KD base logit: {KD_BASE_LOGIT:.2f}")
    print(f"fixed KD resistance scale: {KD_RESISTANCE_SCALE:.1f}")
    print()
    print(out.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n[KD depletion sweep] wrote {OUTPUT_PATH}")
    print(
        "\nCALIBRATION BOUNDARY: choose depletion sensitivity from the change in KD rate "
        "from fresh to depleted reservoir states. Recent-KD bonus remains fixed."
    )


if __name__ == "__main__":
    main()
