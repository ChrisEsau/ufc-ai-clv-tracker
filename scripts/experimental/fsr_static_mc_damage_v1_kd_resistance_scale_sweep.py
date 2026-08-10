"""Shadow sweep for knockdown-resistance sensitivity in Damage Reservoir V1.

Purpose
-------
The current KD model has a reasonable overall KD rate near KD_BASE_LOGIT=-6.40,
but the power-vs-resistance edge separates extreme matchups too strongly. This
study changes only the denominator applied to knockdown_resistance while keeping
all other provisional mechanics fixed.

No production behavior is changed by this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_v0 as base

FSR_PATH = damage.FSR_PATH
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_kd_resistance_scale_sweep.parquet"
)

DAMAGE_SCALE = 0.50
KD_BASE_LOGIT = -6.40
RESISTANCE_SCALES = (14.0, 20.0, 26.0, 32.0, 40.0)
DEFAULT_MATCHUPS = 300
DEFAULT_PATHS_PER_MATCHUP = 10
DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260809


class SweepSim(damage.StaticFSRMCDamageV1):
    def __init__(self, *args: Any, resistance_scale: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.resistance_scale = float(resistance_scale)

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
            + (50.0 - resistance) / self.resistance_scale
            + damage.KD_DEPLETION_COEFFICIENT * depletion
            + (damage.KD_RECENT_KD_LOGIT_BONUS if state.recent_knockdown else 0.0)
        )
        return float(np.clip(damage._sigmoid(logit_p), 0.0, 0.95))


def _rank_bucket(series: pd.Series) -> pd.Series:
    rank = pd.to_numeric(series, errors="coerce").rank(method="first", pct=True)
    return pd.cut(
        rank,
        bins=np.linspace(0.0, 1.0, 6),
        labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
        include_lowest=True,
    )


def _choose_matchups(profiles: pd.DataFrame, n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    return [tuple(map(int, rng.choice(len(profiles), size=2, replace=False))) for _ in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep KD resistance sensitivity")
    parser.add_argument("--matchups", type=int, default=DEFAULT_MATCHUPS)
    parser.add_argument("--paths-per-matchup", type=int, default=DEFAULT_PATHS_PER_MATCHUP)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    args = parser.parse_args()

    profiles = damage.load_profiles(args.fsr_path)
    rng = np.random.default_rng(args.seed)
    matchups = _choose_matchups(profiles, args.matchups, rng)
    path_seeds = rng.integers(
        0, 2**31 - 1, size=(args.matchups, args.paths_per_matchup), dtype=np.int64
    )

    rows: list[dict[str, Any]] = []
    total_per_scale = args.matchups * args.paths_per_matchup

    for resistance_scale in RESISTANCE_SCALES:
        done = 0
        for matchup_i, (red_i, blue_i) in enumerate(matchups):
            red = profiles.iloc[red_i]
            blue = profiles.iloc[blue_i]
            for path_i in range(args.paths_per_matchup):
                sim = SweepSim(
                    red,
                    blue,
                    rounds=args.rounds,
                    seed=int(path_seeds[matchup_i, path_i]),
                    resistance_scale=resistance_scale,
                )
                sim.run()
                for fighter in (0, 1):
                    opponent = 1 - fighter
                    stats = sim.stats[fighter]
                    rows.append(
                        {
                            "resistance_scale": resistance_scale,
                            "fighter_id": str(sim.fighters[fighter]["fighter_id"]),
                            "sig_landed": stats.sig_landed,
                            "kd_scored": stats.knockdowns_scored,
                            "power_minus_opponent_kd_resistance": (
                                base._value(sim.fighters[fighter], "striking_power")
                                - base._value(sim.fighters[opponent], "knockdown_resistance")
                            ),
                        }
                    )
                done += 1
                if done % 1000 == 0 or done == total_per_scale:
                    print(
                        f"[KD resistance sweep] scale={resistance_scale:.0f} "
                        f"paths {done:,}/{total_per_scale:,}",
                        flush=True,
                    )

    frame = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT_PATH, index=False)

    print("\n" + "=" * 120)
    print("DAMAGE RESERVOIR V1 — KD RESISTANCE SCALE SWEEP")
    print("=" * 120)
    print(f"fixed damage scale: {DAMAGE_SCALE:.2f}")
    print(f"fixed KD base logit: {KD_BASE_LOGIT:.2f}")

    overall_rows = []
    edge_rows = []
    for resistance_scale, g in frame.groupby("resistance_scale", sort=True):
        landed = g["sig_landed"].sum()
        overall_rows.append(
            {
                "resistance_scale": resistance_scale,
                "fighter_paths": len(g),
                "mean_kd_scored": g["kd_scored"].mean(),
                "kd_scored_probability": (g["kd_scored"] >= 1).mean(),
                "two_plus_kd_scored_probability": (g["kd_scored"] >= 2).mean(),
                "pooled_kd_per_sig_landed": g["kd_scored"].sum() / landed if landed else np.nan,
            }
        )

        temp = g.copy()
        temp["edge_bucket"] = _rank_bucket(temp["power_minus_opponent_kd_resistance"])
        bucket_stats: dict[str, float] = {}
        for bucket, bg in temp.groupby("edge_bucket", observed=True, sort=False):
            bucket_landed = bg["sig_landed"].sum()
            kd_per_sig = bg["kd_scored"].sum() / bucket_landed if bucket_landed else np.nan
            bucket_stats[str(bucket)] = kd_per_sig
            edge_rows.append(
                {
                    "resistance_scale": resistance_scale,
                    "edge_bucket": str(bucket),
                    "fighter_paths": len(bg),
                    "mean_edge": bg["power_minus_opponent_kd_resistance"].mean(),
                    "pooled_kd_per_sig_landed": kd_per_sig,
                }
            )
        q1 = bucket_stats.get("Q1 lowest", np.nan)
        q5 = bucket_stats.get("Q5 highest", np.nan)
        overall_rows[-1]["q5_q1_kd_per_sig_ratio"] = q5 / q1 if q1 and np.isfinite(q1) else np.nan

    print("\nOVERALL")
    print(pd.DataFrame(overall_rows).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nPOWER - OPPONENT KD RESISTANCE EDGE QUINTILES")
    print(pd.DataFrame(edge_rows).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\n[KD resistance sweep] wrote {OUTPUT_PATH}")
    print(
        "\nCALIBRATION BOUNDARY: choose the resistance sensitivity from matchup separation "
        "while confirming the overall KD rate remains reasonable. Depletion and recent-KD "
        "effects are still unchanged."
    )


if __name__ == "__main__":
    main()
