"""Population audit for shadow Static FSR MC Damage Reservoir V1.

Purpose
-------
Exercise the executable reservoir/KD mechanics across many random FSR-28
matchups before any KO/TKO stoppage logic or finish-rate calibration is added.

This audit is descriptive. It does NOT tune constants or promote mechanics.
It asks whether the provisional architecture behaves directionally as intended:

- damage_durability -> larger reservoir capacity and more remaining reservoir;
- striking_power - opponent knockdown_resistance -> higher KD production;
- lower reservoir condition -> higher strike-level KD probability;
- recent knockdown -> higher follow-up KD probability;
- damage / KD outcomes do not collapse into pathological extremes.

The audit instruments individual landed strikes by subclassing Damage V1. The
frozen V0 baseline and the Damage V1 implementation are not modified here.
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
    "fsr_static_mc_damage_v1_population_audit.parquet"
)
STRIKE_OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_strike_audit.parquet"
)

DEFAULT_MATCHUPS = 500
DEFAULT_PATHS_PER_MATCHUP = 20
DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260809


class InstrumentedDamageV1(damage.StaticFSRMCDamageV1):
    """Damage V1 with strike-level audit records only."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.strike_records: list[dict[str, Any]] = []

    def _apply_landed_strikes(self, attacker: int, landed: int) -> tuple[float, int]:
        defender = self._other(attacker)
        total_damage = 0.0
        knockdowns = 0

        for _ in range(int(landed)):
            state = self.damage_state[defender]
            reservoir_before = state.reservoir_fraction
            recent_kd_before = state.recent_knockdown
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
                    "attacker": attacker,
                    "defender": defender,
                    "attacker_power": base._value(self.fighters[attacker], "striking_power"),
                    "defender_knockdown_resistance": base._value(
                        self.fighters[defender], "knockdown_resistance"
                    ),
                    "defender_damage_durability": base._value(
                        self.fighters[defender], "damage_durability"
                    ),
                    "reservoir_fraction_before": reservoir_before,
                    "reservoir_fraction_after": state.reservoir_fraction,
                    "recent_kd_before": int(recent_kd_before),
                    "strike_damage": damage_value,
                    "shock_fraction": damage_value / state.reservoir_capacity,
                    "kd_probability": p_kd,
                    "knockdown": int(kd),
                }
            )

        return total_damage, knockdowns


def _safe_spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return float("nan")
    return float(pd.DataFrame({"x": x[mask], "y": y[mask]}).corr(method="spearman").iloc[0, 1])


def _rank_bucket(series: pd.Series, labels: list[str]) -> pd.Series:
    rank = pd.to_numeric(series, errors="coerce").rank(method="first", pct=True)
    return pd.cut(
        rank,
        bins=np.linspace(0.0, 1.0, len(labels) + 1),
        labels=labels,
        include_lowest=True,
    )


def _latest_profiles(path: Path) -> pd.DataFrame:
    profiles = damage.load_profiles(path).copy()
    needed = {
        "fighter_id",
        "striking_power",
        "knockdown_resistance",
        "damage_durability",
    }
    missing = sorted(needed - set(profiles.columns))
    if missing:
        raise ValueError(f"latest FSR-28 profiles missing audit columns: {missing}")
    return profiles.reset_index(drop=True)


def _choose_matchups(
    profiles: pd.DataFrame,
    matchup_count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if len(profiles) < 2:
        raise ValueError("Need at least two latest fighter profiles")
    pairs: list[tuple[int, int]] = []
    for _ in range(matchup_count):
        a, b = rng.choice(len(profiles), size=2, replace=False)
        pairs.append((int(a), int(b)))
    return pairs


def _run_population(
    profiles: pd.DataFrame,
    matchup_count: int,
    paths_per_matchup: int,
    rounds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    matchups = _choose_matchups(profiles, matchup_count, rng)
    fighter_rows: list[dict[str, Any]] = []
    strike_rows: list[dict[str, Any]] = []

    total_paths = matchup_count * paths_per_matchup
    path_counter = 0

    for matchup_index, (red_i, blue_i) in enumerate(matchups, start=1):
        red = profiles.iloc[red_i]
        blue = profiles.iloc[blue_i]

        for path_index in range(paths_per_matchup):
            path_seed = int(rng.integers(0, 2**31 - 1))
            sim = InstrumentedDamageV1(red, blue, rounds=rounds, seed=path_seed)
            sim.run()

            for fighter in (0, 1):
                opponent = 1 - fighter
                stats = sim.stats[fighter]
                state = sim.damage_state[fighter]
                assert isinstance(stats, damage.DamageFighterStats)

                fighter_rows.append(
                    {
                        "matchup_index": matchup_index,
                        "path_index": path_index,
                        "path_seed": path_seed,
                        "fighter_id": str(sim.fighters[fighter]["fighter_id"]),
                        "opponent_id": str(sim.fighters[opponent]["fighter_id"]),
                        "striking_power": base._value(sim.fighters[fighter], "striking_power"),
                        "opponent_knockdown_resistance": base._value(
                            sim.fighters[opponent], "knockdown_resistance"
                        ),
                        "damage_durability": base._value(
                            sim.fighters[fighter], "damage_durability"
                        ),
                        "reservoir_capacity": state.reservoir_capacity,
                        "reservoir_remaining": state.reservoir_current,
                        "reservoir_fraction": state.reservoir_fraction,
                        "sig_landed": stats.sig_landed,
                        "sig_absorbed": sim.stats[opponent].sig_landed,
                        "damage_dealt": stats.damage_dealt,
                        "damage_absorbed": stats.damage_absorbed,
                        "knockdowns_scored": stats.knockdowns_scored,
                        "knockdowns_absorbed": stats.knockdowns_absorbed,
                        "max_single_strike_damage": stats.max_single_strike_damage,
                        "power_minus_opponent_kd_resistance": (
                            base._value(sim.fighters[fighter], "striking_power")
                            - base._value(sim.fighters[opponent], "knockdown_resistance")
                        ),
                    }
                )

            for record in sim.strike_records:
                attacker = int(record["attacker"])
                defender = int(record["defender"])
                strike_rows.append(
                    {
                        "matchup_index": matchup_index,
                        "path_index": path_index,
                        "path_seed": path_seed,
                        "attacker_id": str(sim.fighters[attacker]["fighter_id"]),
                        "defender_id": str(sim.fighters[defender]["fighter_id"]),
                        **{k: v for k, v in record.items() if k not in {"attacker", "defender"}},
                    }
                )

            path_counter += 1
            if path_counter % 1000 == 0 or path_counter == total_paths:
                print(
                    f"[damage V1 audit] paths {path_counter:,}/{total_paths:,}; "
                    f"fighter_rows={len(fighter_rows):,}; strikes={len(strike_rows):,}",
                    flush=True,
                )

    return pd.DataFrame(fighter_rows), pd.DataFrame(strike_rows)


def _print_fighter_summary(frame: pd.DataFrame) -> None:
    print("\n" + "=" * 120)
    print("DAMAGE RESERVOIR V1 — PATH POPULATION SUMMARY")
    print("=" * 120)
    print(f"fighter-path rows: {len(frame):,}")
    print(f"mean sig absorbed: {frame['sig_absorbed'].mean():.3f}")
    print(f"mean damage absorbed: {frame['damage_absorbed'].mean():.3f}")
    print(f"mean reservoir remaining: {frame['reservoir_fraction'].mean():.3%}")
    print(f"median reservoir remaining: {frame['reservoir_fraction'].median():.3%}")
    print(f"p10 reservoir remaining: {frame['reservoir_fraction'].quantile(0.10):.3%}")
    print(f"reservoir exhausted: {(frame['reservoir_fraction'] <= 0).mean():.3%}")
    print(f"mean KD absorbed: {frame['knockdowns_absorbed'].mean():.4f}")
    print(f">=1 KD absorbed: {(frame['knockdowns_absorbed'] >= 1).mean():.3%}")
    print(f">=2 KD absorbed: {(frame['knockdowns_absorbed'] >= 2).mean():.3%}")

    print("\nDURABILITY QUINTILES")
    work = frame.copy()
    work["bucket"] = _rank_bucket(
        work["damage_durability"],
        ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
    )
    rows = []
    for bucket, g in work.groupby("bucket", observed=True, sort=False):
        rows.append(
            {
                "bucket": str(bucket),
                "fighter_paths": len(g),
                "mean_durability": g["damage_durability"].mean(),
                "mean_capacity": g["reservoir_capacity"].mean(),
                "mean_sig_absorbed": g["sig_absorbed"].mean(),
                "mean_damage_absorbed": g["damage_absorbed"].mean(),
                "mean_reservoir_fraction": g["reservoir_fraction"].mean(),
                "reservoir_exhausted_probability": (g["reservoir_fraction"] <= 0).mean(),
                "kd_absorbed_probability": (g["knockdowns_absorbed"] >= 1).mean(),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nPOWER - OPPONENT KD RESISTANCE EDGE QUINTILES")
    edge = frame.copy()
    edge["bucket"] = _rank_bucket(
        edge["power_minus_opponent_kd_resistance"],
        ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
    )
    rows = []
    for bucket, g in edge.groupby("bucket", observed=True, sort=False):
        landed = g["sig_landed"].sum()
        rows.append(
            {
                "bucket": str(bucket),
                "fighter_paths": len(g),
                "mean_edge": g["power_minus_opponent_kd_resistance"].mean(),
                "mean_sig_landed": g["sig_landed"].mean(),
                "kd_scored_probability": (g["knockdowns_scored"] >= 1).mean(),
                "mean_kd_scored": g["knockdowns_scored"].mean(),
                "pooled_kd_per_sig_landed": (
                    g["knockdowns_scored"].sum() / landed if landed > 0 else np.nan
                ),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.5f}"))


def _print_strike_summary(frame: pd.DataFrame) -> None:
    print("\n" + "=" * 120)
    print("STRIKE-LEVEL DAMAGE / KD SUMMARY")
    print("=" * 120)
    print(f"landed strikes audited: {len(frame):,}")
    print(f"mean strike damage: {frame['strike_damage'].mean():.4f}")
    print(f"median strike damage: {frame['strike_damage'].median():.4f}")
    print(f"p90 strike damage: {frame['strike_damage'].quantile(0.90):.4f}")
    print(f"p99 strike damage: {frame['strike_damage'].quantile(0.99):.4f}")
    print(f"max strike damage: {frame['strike_damage'].max():.4f}")
    print(f"overall KD per landed strike: {frame['knockdown'].mean():.5f}")
    print(
        "Spearman attacker power vs strike damage: "
        f"{_safe_spearman(frame['attacker_power'], frame['strike_damage']):.5f}"
    )
    print(
        "Spearman power-resistance edge vs KD: "
        f"{_safe_spearman(frame['attacker_power'] - frame['defender_knockdown_resistance'], frame['knockdown']):.5f}"
    )
    print(
        "Spearman reservoir fraction before strike vs KD: "
        f"{_safe_spearman(frame['reservoir_fraction_before'], frame['knockdown']):.5f}"
    )

    print("\nKD BY RESERVOIR CONDITION")
    work = frame.copy()
    work["condition"] = pd.cut(
        work["reservoir_fraction_before"],
        bins=[-1e-9, 0.25, 0.50, 0.75, 1.0000001],
        labels=["0-25%", "25-50%", "50-75%", "75-100%"],
        include_lowest=True,
    )
    rows = []
    for condition, g in work.groupby("condition", observed=True, sort=False):
        rows.append(
            {
                "reservoir_condition": str(condition),
                "landed_strikes": len(g),
                "mean_strike_damage": g["strike_damage"].mean(),
                "mean_model_kd_probability": g["kd_probability"].mean(),
                "realized_kd_probability": g["knockdown"].mean(),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nRECENT-KD FOLLOW-UP")
    rows = []
    for recent in (0, 1):
        g = frame[frame["recent_kd_before"] == recent]
        if g.empty:
            continue
        rows.append(
            {
                "recent_kd_before": recent,
                "landed_strikes": len(g),
                "mean_reservoir_before": g["reservoir_fraction_before"].mean(),
                "mean_strike_damage": g["strike_damage"].mean(),
                "mean_model_kd_probability": g["kd_probability"].mean(),
                "realized_kd_probability": g["knockdown"].mean(),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.5f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Population audit for Damage Reservoir V1")
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--matchups", type=int, default=DEFAULT_MATCHUPS)
    parser.add_argument("--paths-per-matchup", type=int, default=DEFAULT_PATHS_PER_MATCHUP)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--strike-output", type=Path, default=STRIKE_OUTPUT_PATH)
    args = parser.parse_args()

    print(f"[damage V1 audit] loading latest FSR-28 profiles: {args.fsr_path}", flush=True)
    profiles = _latest_profiles(args.fsr_path)
    print(f"[damage V1 audit] latest fighters: {len(profiles):,}", flush=True)
    print(
        f"[damage V1 audit] matchups={args.matchups:,}, "
        f"paths/matchup={args.paths_per_matchup:,}, rounds={args.rounds}, seed={args.seed}",
        flush=True,
    )

    fighters, strikes = _run_population(
        profiles,
        matchup_count=args.matchups,
        paths_per_matchup=args.paths_per_matchup,
        rounds=args.rounds,
        seed=args.seed,
    )

    _print_fighter_summary(fighters)
    _print_strike_summary(strikes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fighters.to_parquet(args.output, index=False)
    strikes.to_parquet(args.strike_output, index=False)
    print(f"\n[damage V1 audit] wrote {args.output}", flush=True)
    print(f"[damage V1 audit] wrote {args.strike_output}", flush=True)
    print(
        "\nAUDIT BOUNDARY: inspect directions and path plausibility only. "
        "Do not interpret these provisional KD/damage rates as calibrated targets.",
        flush=True,
    )


if __name__ == "__main__":
    main()
