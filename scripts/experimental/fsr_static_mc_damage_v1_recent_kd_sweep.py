"""Sweep recent-knockdown vulnerability for Damage Reservoir V1.

Purpose
-------
Hold the provisional damage/KD settings fixed and vary only the temporary
recent-KD logit bonus. The goal is to choose a short-lived follow-up effect
that increases immediate knockdown susceptibility without creating runaway
multi-KD clustering.

Fixed provisional settings in this study:
- strike-damage scale: 0.50
- KD base logit: -6.40
- KD resistance scale: 32.0
- KD depletion coefficient: 1.50

This is a shadow calibration study only. It does not modify the engine.
"""

from __future__ import annotations

import argparse
from math import exp
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_v0 as base


FSR_PATH = damage.FSR_PATH
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_recent_kd_sweep.parquet"
)

DEFAULT_MATCHUPS = 300
DEFAULT_PATHS_PER_MATCHUP = 10
DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260809
DAMAGE_SCALE = 0.50
KD_BASE_LOGIT = -6.40
KD_RESISTANCE_SCALE = 32.0
KD_DEPLETION_COEFFICIENT = 1.50
RECENT_KD_BONUSES = [0.0, 0.25, 0.50, 0.75, 1.00]


class RecentKDSweepSim(damage.StaticFSRMCDamageV1):
    def __init__(
        self,
        *args: Any,
        recent_kd_bonus: float,
        **kwargs: Any,
    ) -> None:
        self._recent_kd_bonus = float(recent_kd_bonus)
        self.strike_records: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)

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
            + KD_DEPLETION_COEFFICIENT * depletion
            + (self._recent_kd_bonus if state.recent_knockdown else 0.0)
        )
        return float(np.clip(damage._sigmoid(logit_p), 0.0, 0.95))

    def _apply_landed_strikes(self, attacker: int, landed: int) -> tuple[float, int]:
        defender = self._other(attacker)
        total_damage = 0.0
        knockdowns = 0

        for _ in range(int(landed)):
            state = self.damage_state[defender]
            recent_before = state.recent_knockdown
            reservoir_before = state.reservoir_fraction
            damage_value = self._draw_strike_damage(attacker)
            p_kd = self._knockdown_probability(defender, damage_value)

            state.reservoir_current = max(0.0, state.reservoir_current - damage_value)

            attacker_stats = self.stats[attacker]
            defender_stats = self.stats[defender]
            assert isinstance(attacker_stats, damage.DamageFighterStats)
            assert isinstance(defender_stats, damage.DamageFighterStats)

            attacker_stats.damage_dealt += damage_value
            defender_stats.damage_absorbed += damage_value
            attacker_stats.max_single_strike_damage = max(
                attacker_stats.max_single_strike_damage, damage_value
            )
            defender_stats.max_single_strike_damage = max(
                defender_stats.max_single_strike_damage, damage_value
            )
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
                    "recent_kd_before": int(recent_before),
                    "reservoir_fraction_before": reservoir_before,
                    "kd_probability": p_kd,
                    "knockdown": int(kd),
                }
            )

        return total_damage, knockdowns


def _latest_profiles(path: Path) -> pd.DataFrame:
    return damage.load_profiles(path).reset_index(drop=True)


def _choose_matchups(
    profiles: pd.DataFrame,
    matchup_count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for _ in range(matchup_count):
        a, b = rng.choice(len(profiles), size=2, replace=False)
        pairs.append((int(a), int(b)))
    return pairs


def _run_sweep(
    profiles: pd.DataFrame,
    matchup_count: int,
    paths_per_matchup: int,
    rounds: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    matchups = _choose_matchups(profiles, matchup_count, rng)
    path_seeds = rng.integers(
        0,
        2**31 - 1,
        size=(matchup_count, paths_per_matchup),
        dtype=np.int64,
    )

    rows: list[dict[str, Any]] = []

    for bonus in RECENT_KD_BONUSES:
        print(f"[recent KD sweep] bonus={bonus:.2f}", flush=True)
        strike_rows: list[dict[str, Any]] = []
        fighter_kds: list[int] = []

        total_paths = matchup_count * paths_per_matchup
        counter = 0
        for matchup_index, (red_i, blue_i) in enumerate(matchups):
            red = profiles.iloc[red_i]
            blue = profiles.iloc[blue_i]
            for path_index in range(paths_per_matchup):
                sim = RecentKDSweepSim(
                    red,
                    blue,
                    rounds=rounds,
                    seed=int(path_seeds[matchup_index, path_index]),
                    recent_kd_bonus=bonus,
                )
                sim.run()
                fighter_kds.extend(
                    [
                        int(sim.stats[0].knockdowns_scored),
                        int(sim.stats[1].knockdowns_scored),
                    ]
                )
                strike_rows.extend(sim.strike_records)
                counter += 1
                if counter % 1000 == 0 or counter == total_paths:
                    print(
                        f"[recent KD sweep] bonus={bonus:.2f} paths "
                        f"{counter:,}/{total_paths:,}",
                        flush=True,
                    )

        frame = pd.DataFrame(strike_rows)
        normal = frame[frame["recent_kd_before"] == 0]
        recent = frame[frame["recent_kd_before"] == 1]
        normal_rate = float(normal["knockdown"].mean()) if len(normal) else np.nan
        recent_rate = float(recent["knockdown"].mean()) if len(recent) else np.nan
        ratio = recent_rate / normal_rate if normal_rate and normal_rate > 0 else np.nan
        kd_array = np.asarray(fighter_kds, dtype=float)

        rows.append(
            {
                "recent_kd_logit_bonus": bonus,
                "landed_strikes": len(frame),
                "normal_strikes": len(normal),
                "recent_kd_strikes": len(recent),
                "normal_kd_per_strike": normal_rate,
                "recent_kd_kd_per_strike": recent_rate,
                "recent_to_normal_ratio": ratio,
                "overall_kd_per_strike": float(frame["knockdown"].mean()),
                "mean_kd_scored_per_fighter": float(kd_array.mean()),
                "one_plus_kd_probability": float((kd_array >= 1).mean()),
                "two_plus_kd_probability": float((kd_array >= 2).mean()),
                "three_plus_kd_probability": float((kd_array >= 3).mean()),
                "mean_reservoir_recent_kd_before": (
                    float(recent["reservoir_fraction_before"].mean())
                    if len(recent)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep temporary recent-KD vulnerability for Damage Reservoir V1"
    )
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--matchups", type=int, default=DEFAULT_MATCHUPS)
    parser.add_argument("--paths-per-matchup", type=int, default=DEFAULT_PATHS_PER_MATCHUP)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    print(f"[recent KD sweep] loading {args.fsr_path}", flush=True)
    profiles = _latest_profiles(args.fsr_path)
    print(f"[recent KD sweep] latest profiles={len(profiles):,}", flush=True)

    result = _run_sweep(
        profiles,
        matchup_count=args.matchups,
        paths_per_matchup=args.paths_per_matchup,
        rounds=args.rounds,
        seed=args.seed,
    )

    print("\n" + "=" * 120)
    print("DAMAGE RESERVOIR V1 — RECENT-KD VULNERABILITY SWEEP")
    print("=" * 120)
    print(f"fixed damage scale: {DAMAGE_SCALE:.2f}")
    print(f"fixed KD base logit: {KD_BASE_LOGIT:.2f}")
    print(f"fixed KD resistance scale: {KD_RESISTANCE_SCALE:.1f}")
    print(f"fixed KD depletion coefficient: {KD_DEPLETION_COEFFICIENT:.2f}\n")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n[recent KD sweep] wrote {OUTPUT_PATH}")
    print(
        "\nCALIBRATION BOUNDARY: choose only the temporary post-KD amplification here. "
        "Do not modify damage scale, baseline KD rate, resistance sensitivity, or "
        "depletion sensitivity from this sweep."
    )


if __name__ == "__main__":
    main()
