"""Damage Reservoir V1 scale calibration sweep.

Purpose
-------
Isolate the reservoir-consumption problem discovered by the first Damage V1
population audit.  This script changes ONLY a multiplicative strike-damage
scale while keeping:

- FSR-28 fighter profiles;
- durability -> capacity mapping;
- strike-severity distribution shape;
- power-tail logic;
- KD probability model;
- recent-KD state;
- phase / transition mechanics;
- sampled matchups and path seeds

identical across candidates.

This is a shadow calibration study, not a production lock.  The primary outputs
are reservoir remaining / exhaustion and their separation by durability.  KD
statistics are printed only as a side effect of changing strike shock; they are
NOT targets in this sweep.
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
DEFAULT_SCALES = (1.00, 0.75, 0.60, 0.50, 0.40)
DEFAULT_MATCHUPS = 300
DEFAULT_PATHS_PER_MATCHUP = 10
DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260809
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_scale_sweep.parquet"
)


class ScaledDamageV1(damage.StaticFSRMCDamageV1):
    """Damage V1 with one explicit research-only damage multiplier."""

    def __init__(self, *args: Any, damage_scale: float, **kwargs: Any) -> None:
        if damage_scale <= 0:
            raise ValueError("damage_scale must be positive")
        self.damage_scale = float(damage_scale)
        super().__init__(*args, **kwargs)

    def _draw_strike_damage(self, attacker: int) -> float:
        # Scale the final strike-damage draw without changing the distribution
        # shape or the attacker's power-tail process.
        return float(super()._draw_strike_damage(attacker) * self.damage_scale)


@dataclass(frozen=True)
class PathSpec:
    matchup_index: int
    path_index: int
    red_index: int
    blue_index: int
    path_seed: int


def _build_path_specs(
    profile_count: int,
    matchup_count: int,
    paths_per_matchup: int,
    seed: int,
) -> list[PathSpec]:
    """Create one fixed matchup/seed schedule reused for every scale."""
    if profile_count < 2:
        raise ValueError("Need at least two latest fighter profiles")

    rng = np.random.default_rng(seed)
    specs: list[PathSpec] = []
    for matchup_index in range(1, matchup_count + 1):
        red_i, blue_i = rng.choice(profile_count, size=2, replace=False)
        for path_index in range(paths_per_matchup):
            specs.append(
                PathSpec(
                    matchup_index=matchup_index,
                    path_index=path_index,
                    red_index=int(red_i),
                    blue_index=int(blue_i),
                    path_seed=int(rng.integers(0, 2**31 - 1)),
                )
            )
    return specs


def _run_scale(
    profiles: pd.DataFrame,
    specs: list[PathSpec],
    *,
    damage_scale: float,
    rounds: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(specs)

    for counter, spec in enumerate(specs, start=1):
        red = profiles.iloc[spec.red_index]
        blue = profiles.iloc[spec.blue_index]
        sim = ScaledDamageV1(
            red,
            blue,
            rounds=rounds,
            seed=spec.path_seed,
            damage_scale=damage_scale,
        )
        sim.run()

        for fighter in (0, 1):
            opponent = 1 - fighter
            stats = sim.stats[fighter]
            state = sim.damage_state[fighter]
            assert isinstance(stats, damage.DamageFighterStats)

            rows.append(
                {
                    "damage_scale": damage_scale,
                    "matchup_index": spec.matchup_index,
                    "path_index": spec.path_index,
                    "path_seed": spec.path_seed,
                    "fighter_id": str(sim.fighters[fighter]["fighter_id"]),
                    "opponent_id": str(sim.fighters[opponent]["fighter_id"]),
                    "damage_durability": base._value(
                        sim.fighters[fighter], "damage_durability"
                    ),
                    "reservoir_capacity": state.reservoir_capacity,
                    "reservoir_remaining": state.reservoir_current,
                    "reservoir_fraction": state.reservoir_fraction,
                    "sig_absorbed": sim.stats[opponent].sig_landed,
                    "damage_absorbed": stats.damage_absorbed,
                    "knockdowns_absorbed": stats.knockdowns_absorbed,
                }
            )

        if counter % 1000 == 0 or counter == total:
            print(
                f"[damage scale {damage_scale:.2f}] "
                f"paths {counter:,}/{total:,}; fighter_rows={len(rows):,}",
                flush=True,
            )

    return pd.DataFrame(rows)


def _quintile_summary(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    rank = work["damage_durability"].rank(method="first", pct=True)
    work["durability_bucket"] = pd.cut(
        rank,
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
        include_lowest=True,
    )
    rows = []
    for bucket, group in work.groupby("durability_bucket", observed=True, sort=False):
        rows.append(
            {
                "damage_scale": float(group["damage_scale"].iloc[0]),
                "durability_bucket": str(bucket),
                "fighter_paths": len(group),
                "mean_durability": group["damage_durability"].mean(),
                "mean_capacity": group["reservoir_capacity"].mean(),
                "mean_sig_absorbed": group["sig_absorbed"].mean(),
                "mean_damage_absorbed": group["damage_absorbed"].mean(),
                "mean_reservoir_fraction": group["reservoir_fraction"].mean(),
                "reservoir_exhausted_probability": (
                    group["reservoir_fraction"] <= 0.0
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def _overall_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scale, group in frame.groupby("damage_scale", sort=False):
        rows.append(
            {
                "damage_scale": float(scale),
                "fighter_paths": len(group),
                "mean_sig_absorbed": group["sig_absorbed"].mean(),
                "mean_damage_absorbed": group["damage_absorbed"].mean(),
                "mean_reservoir_fraction": group["reservoir_fraction"].mean(),
                "median_reservoir_fraction": group["reservoir_fraction"].median(),
                "p10_reservoir_fraction": group["reservoir_fraction"].quantile(0.10),
                "p25_reservoir_fraction": group["reservoir_fraction"].quantile(0.25),
                "reservoir_exhausted_probability": (
                    group["reservoir_fraction"] <= 0.0
                ).mean(),
                "mean_kd_absorbed": group["knockdowns_absorbed"].mean(),
                "kd_absorbed_probability": (group["knockdowns_absorbed"] >= 1).mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep Damage Reservoir V1 strike-damage scale"
    )
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--matchups", type=int, default=DEFAULT_MATCHUPS)
    parser.add_argument(
        "--paths-per-matchup", type=int, default=DEFAULT_PATHS_PER_MATCHUP
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=list(DEFAULT_SCALES),
        help="Research-only multiplicative strike-damage scales",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    print(f"[damage scale sweep] loading profiles: {args.fsr_path}", flush=True)
    profiles = damage.load_profiles(args.fsr_path)
    print(f"[damage scale sweep] latest fighter profiles: {len(profiles):,}", flush=True)

    specs = _build_path_specs(
        len(profiles),
        args.matchups,
        args.paths_per_matchup,
        args.seed,
    )
    print(
        f"[damage scale sweep] fixed path schedule: {len(specs):,} paths/scale; "
        f"scales={','.join(f'{x:.2f}' for x in args.scales)}",
        flush=True,
    )

    frames = []
    for scale in args.scales:
        frames.append(
            _run_scale(
                profiles,
                specs,
                damage_scale=float(scale),
                rounds=args.rounds,
            )
        )

    all_rows = pd.concat(frames, ignore_index=True)
    overall = _overall_summary(all_rows)
    quintiles = pd.concat(
        [_quintile_summary(g) for _, g in all_rows.groupby("damage_scale", sort=False)],
        ignore_index=True,
    )

    print("\n" + "=" * 130)
    print("DAMAGE RESERVOIR V1 — STRIKE-DAMAGE SCALE SWEEP")
    print("=" * 130)
    print("\nOVERALL")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nDURABILITY QUINTILES BY SCALE")
    print(quintiles.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_rows.to_parquet(args.output, index=False)
    print(f"\n[damage scale sweep] wrote {args.output}")
    print(
        "\nCALIBRATION BOUNDARY: choose damage scale from reservoir-consumption "
        "plausibility only. KD values move because strike shock changes, but KD "
        "constants are not being calibrated in this sweep."
    )


if __name__ == "__main__":
    main()
