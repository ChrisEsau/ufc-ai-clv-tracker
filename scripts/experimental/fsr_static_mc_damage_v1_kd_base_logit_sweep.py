"""KD baseline-logit sweep for shadow Static FSR MC Damage Reservoir V1.

Purpose
-------
Calibrate only the fresh-state KD baseline while holding the rest of the current
Damage V1 architecture fixed:

- strike-damage scale fixed at 0.50;
- reservoir-capacity mapping fixed;
- striking-power tail mechanics fixed;
- knockdown-resistance coefficient fixed;
- reservoir-depletion coefficient fixed;
- recent-knockdown bonus fixed.

The sweep uses the same sampled matchups and path seeds at every candidate
baseline so differences are attributable to ``KD_BASE_LOGIT`` rather than
Monte Carlo noise.

This is a shadow calibration study only. It does not modify production or the
frozen V0 simulator.
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
    "fsr_static_mc_damage_v1_kd_base_logit_sweep.parquet"
)

DAMAGE_SCALE = 0.50
DEFAULT_MATCHUPS = 300
DEFAULT_PATHS_PER_MATCHUP = 10
DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260809

# Current provisional value is -4.80. Sweep progressively lower baselines while
# leaving every other KD term unchanged.
CANDIDATE_KD_BASE_LOGITS = (-4.80, -5.20, -5.60, -6.00, -6.40, -6.80)


@dataclass(frozen=True)
class PathSpec:
    matchup_index: int
    path_index: int
    red_index: int
    blue_index: int
    path_seed: int


class KDBaselineSweepSim(damage.StaticFSRMCDamageV1):
    def __init__(
        self,
        *args: Any,
        kd_base_logit: float,
        damage_scale: float = DAMAGE_SCALE,
        **kwargs: Any,
    ) -> None:
        self.kd_base_logit = float(kd_base_logit)
        self.damage_scale = float(damage_scale)
        super().__init__(*args, **kwargs)

    def _draw_strike_damage(self, attacker: int) -> float:
        return self.damage_scale * super()._draw_strike_damage(attacker)

    def _knockdown_probability(self, defender: int, strike_damage: float) -> float:
        state = self.damage_state[defender]
        resistance = base._value(self.fighters[defender], "knockdown_resistance")
        shock_fraction = strike_damage / state.reservoir_capacity
        depletion = 1.0 - state.reservoir_fraction

        logit_p = (
            self.kd_base_logit
            + damage.KD_SHOCK_COEFFICIENT * shock_fraction
            + (50.0 - resistance) / damage.KD_RESISTANCE_SCALE
            + damage.KD_DEPLETION_COEFFICIENT * depletion
            + (
                damage.KD_RECENT_KD_LOGIT_BONUS
                if state.recent_knockdown
                else 0.0
            )
        )
        return float(np.clip(damage._sigmoid(logit_p), 0.0, 0.95))


def _latest_profiles(path: Path) -> pd.DataFrame:
    profiles = damage.load_profiles(path).copy()
    return profiles.reset_index(drop=True)


def _build_path_specs(
    profiles: pd.DataFrame,
    matchup_count: int,
    paths_per_matchup: int,
    seed: int,
) -> list[PathSpec]:
    rng = np.random.default_rng(seed)
    specs: list[PathSpec] = []

    for matchup_index in range(1, matchup_count + 1):
        red_index, blue_index = rng.choice(len(profiles), size=2, replace=False)
        for path_index in range(paths_per_matchup):
            specs.append(
                PathSpec(
                    matchup_index=matchup_index,
                    path_index=path_index,
                    red_index=int(red_index),
                    blue_index=int(blue_index),
                    path_seed=int(rng.integers(0, 2**31 - 1)),
                )
            )
    return specs


def _run_candidate(
    profiles: pd.DataFrame,
    specs: list[PathSpec],
    *,
    kd_base_logit: float,
    rounds: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(specs)

    for completed, spec in enumerate(specs, start=1):
        red = profiles.iloc[spec.red_index]
        blue = profiles.iloc[spec.blue_index]
        sim = KDBaselineSweepSim(
            red,
            blue,
            rounds=rounds,
            seed=spec.path_seed,
            kd_base_logit=kd_base_logit,
        )
        sim.run()

        for fighter in (0, 1):
            opponent = 1 - fighter
            stats = sim.stats[fighter]
            opp_stats = sim.stats[opponent]
            assert isinstance(stats, damage.DamageFighterStats)
            assert isinstance(opp_stats, damage.DamageFighterStats)

            rows.append(
                {
                    "kd_base_logit": kd_base_logit,
                    "damage_scale": DAMAGE_SCALE,
                    "matchup_index": spec.matchup_index,
                    "path_index": spec.path_index,
                    "fighter_id": str(sim.fighters[fighter]["fighter_id"]),
                    "opponent_id": str(sim.fighters[opponent]["fighter_id"]),
                    "sig_landed": stats.sig_landed,
                    "sig_absorbed": opp_stats.sig_landed,
                    "knockdowns_scored": stats.knockdowns_scored,
                    "knockdowns_absorbed": stats.knockdowns_absorbed,
                    "striking_power": base._value(sim.fighters[fighter], "striking_power"),
                    "opponent_knockdown_resistance": base._value(
                        sim.fighters[opponent], "knockdown_resistance"
                    ),
                    "power_minus_opponent_kd_resistance": (
                        base._value(sim.fighters[fighter], "striking_power")
                        - base._value(
                            sim.fighters[opponent], "knockdown_resistance"
                        )
                    ),
                    "reservoir_fraction": sim.damage_state[fighter].reservoir_fraction,
                }
            )

        if completed % 1000 == 0 or completed == total:
            print(
                f"[KD base sweep] logit={kd_base_logit:.2f} "
                f"paths {completed:,}/{total:,}",
                flush=True,
            )

    return pd.DataFrame(rows)


def _rank_bucket(series: pd.Series) -> pd.Series:
    rank = pd.to_numeric(series, errors="coerce").rank(method="first", pct=True)
    return pd.cut(
        rank,
        bins=np.linspace(0.0, 1.0, 6),
        labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
        include_lowest=True,
    )


def _summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    for kd_base_logit, g in frame.groupby("kd_base_logit", sort=False):
        total_landed = g["sig_landed"].sum()
        overall_rows.append(
            {
                "kd_base_logit": kd_base_logit,
                "fighter_paths": len(g),
                "mean_sig_landed": g["sig_landed"].mean(),
                "mean_kd_scored": g["knockdowns_scored"].mean(),
                "kd_scored_probability": (g["knockdowns_scored"] >= 1).mean(),
                "two_plus_kd_scored_probability": (g["knockdowns_scored"] >= 2).mean(),
                "pooled_kd_per_sig_landed": (
                    g["knockdowns_scored"].sum() / total_landed
                    if total_landed > 0
                    else np.nan
                ),
                "mean_reservoir_fraction": g["reservoir_fraction"].mean(),
            }
        )

        edge = g.copy()
        edge["bucket"] = _rank_bucket(edge["power_minus_opponent_kd_resistance"])
        for bucket, b in edge.groupby("bucket", observed=True, sort=False):
            landed = b["sig_landed"].sum()
            edge_rows.append(
                {
                    "kd_base_logit": kd_base_logit,
                    "edge_bucket": str(bucket),
                    "fighter_paths": len(b),
                    "mean_edge": b["power_minus_opponent_kd_resistance"].mean(),
                    "kd_scored_probability": (b["knockdowns_scored"] >= 1).mean(),
                    "pooled_kd_per_sig_landed": (
                        b["knockdowns_scored"].sum() / landed
                        if landed > 0
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(overall_rows), pd.DataFrame(edge_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Damage V1 KD baseline logit")
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--matchups", type=int, default=DEFAULT_MATCHUPS)
    parser.add_argument("--paths-per-matchup", type=int, default=DEFAULT_PATHS_PER_MATCHUP)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    print(f"[KD base sweep] loading FSR-28 profiles: {args.fsr_path}", flush=True)
    profiles = _latest_profiles(args.fsr_path)
    specs = _build_path_specs(
        profiles,
        matchup_count=args.matchups,
        paths_per_matchup=args.paths_per_matchup,
        seed=args.seed,
    )
    print(
        f"[KD base sweep] profiles={len(profiles):,}; matchups={args.matchups:,}; "
        f"paths/candidate={len(specs):,}; damage_scale={DAMAGE_SCALE:.2f}",
        flush=True,
    )

    frames = []
    for candidate in CANDIDATE_KD_BASE_LOGITS:
        frames.append(
            _run_candidate(
                profiles,
                specs,
                kd_base_logit=candidate,
                rounds=args.rounds,
            )
        )

    all_rows = pd.concat(frames, ignore_index=True)
    overall, edge = _summarize(all_rows)

    print("\n" + "=" * 120)
    print("DAMAGE RESERVOIR V1 — KD BASELINE LOGIT SWEEP")
    print("=" * 120)
    print("\nOVERALL")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nPOWER - OPPONENT KD RESISTANCE EDGE QUINTILES")
    print(edge.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_rows.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n[KD base sweep] wrote {OUTPUT_PATH}")
    print(
        "\nCALIBRATION BOUNDARY: choose the baseline from overall KD frequency "
        "only. Power/resistance, depletion, and recent-KD coefficients remain fixed."
    )


if __name__ == "__main__":
    main()
