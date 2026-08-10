"""Strike-level shock audit for calibrated Static FSR MC Damage Reservoir V1.

Purpose
-------
Follow the documented Damage Reservoir V0/V1 research path before defining any
KO/TKO probability curve.

The design documents define the acute strike quantity as::

    shock_fraction = reservoir_delta / reservoir_capacity

This script measures that quantity under the calibrated Damage V1 strike model.
It does NOT implement or tune KO/TKO mechanics.

Questions answered
------------------
- What is the unconditional shock distribution for landed significant strikes?
- What shock values occur on simulated KD versus non-KD strikes?
- How does realized KD rate change across empirical shock percentiles?
- How does reservoir condition shift the shock/KD relationship?
- How does attacker striking power change the upper shock tail?

Important boundary
------------------
UFCStats does not provide exact strike-level KO timing/severity, so this audit
cannot directly estimate a historical p(KO | exact shock) curve. Its job is to
characterize the simulator's calibrated shock scale first. Any later KO/TKO
curve must be designed and validated separately against historical finish
constraints.
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
    "fsr_static_mc_damage_v1_shock_audit.parquet"
)

DEFAULT_MATCHUPS = 500
DEFAULT_PATHS_PER_MATCHUP = 20
DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260809

SHOCK_PERCENTILES = [0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999]


class ShockAuditSim(damage.StaticFSRMCDamageV1):
    """Calibrated Damage V1 with strike-level observation only.

    The application order intentionally matches the active Damage V1 engine:
    draw damage -> deplete reservoir -> calculate KD probability -> draw KD.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.strike_records: list[dict[str, Any]] = []

    def _apply_landed_strikes(self, attacker: int, landed: int) -> tuple[float, int]:
        defender = self._other(attacker)
        total_damage = 0.0
        knockdowns = 0

        for _ in range(int(landed)):
            state = self.damage_state[defender]
            reservoir_fraction_before = state.reservoir_fraction
            recent_kd_before = state.recent_knockdown
            strike_damage = self._draw_strike_damage(attacker)
            shock_fraction = strike_damage / state.reservoir_capacity

            # Match the active Damage V1 ordering exactly.
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

            self.strike_records.append(
                {
                    "attacker": attacker,
                    "defender": defender,
                    "attacker_power": base._value(
                        self.fighters[attacker], "striking_power"
                    ),
                    "defender_knockdown_resistance": base._value(
                        self.fighters[defender], "knockdown_resistance"
                    ),
                    "defender_damage_durability": base._value(
                        self.fighters[defender], "damage_durability"
                    ),
                    "reservoir_capacity": state.reservoir_capacity,
                    "reservoir_fraction_before": reservoir_fraction_before,
                    "reservoir_fraction_after": state.reservoir_fraction,
                    "recent_kd_before": int(recent_kd_before),
                    "strike_damage": strike_damage,
                    "shock_fraction": shock_fraction,
                    "kd_probability": p_kd,
                    "knockdown": int(knockdown),
                }
            )

        return total_damage, knockdowns


def _rank_bucket(series: pd.Series, labels: list[str]) -> pd.Series:
    rank = pd.to_numeric(series, errors="coerce").rank(method="first", pct=True)
    return pd.cut(
        rank,
        bins=np.linspace(0.0, 1.0, len(labels) + 1),
        labels=labels,
        include_lowest=True,
    )


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


def _run_audit(
    profiles: pd.DataFrame,
    matchup_count: int,
    paths_per_matchup: int,
    rounds: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    matchups = _choose_matchups(profiles, matchup_count, rng)
    strike_rows: list[dict[str, Any]] = []

    total_paths = matchup_count * paths_per_matchup
    path_counter = 0

    for matchup_index, (red_i, blue_i) in enumerate(matchups, start=1):
        red = profiles.iloc[red_i]
        blue = profiles.iloc[blue_i]

        for path_index in range(paths_per_matchup):
            path_seed = int(rng.integers(0, 2**31 - 1))
            sim = ShockAuditSim(red, blue, rounds=rounds, seed=path_seed)
            sim.run()

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
                        **{
                            key: value
                            for key, value in record.items()
                            if key not in {"attacker", "defender"}
                        },
                    }
                )

            path_counter += 1
            if path_counter % 1000 == 0 or path_counter == total_paths:
                print(
                    f"[shock audit] paths {path_counter:,}/{total_paths:,}; "
                    f"landed_strikes={len(strike_rows):,}",
                    flush=True,
                )

    return pd.DataFrame(strike_rows)


def _print_percentiles(frame: pd.DataFrame) -> None:
    shock = frame["shock_fraction"]
    print("\nSHOCK DISTRIBUTION — ALL LANDED SIGNIFICANT STRIKES")
    rows = []
    for q in SHOCK_PERCENTILES:
        rows.append(
            {
                "percentile": f"p{100*q:g}",
                "shock_fraction": shock.quantile(q),
                "shock_percent_capacity": 100.0 * shock.quantile(q),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"mean shock fraction: {shock.mean():.6f}")
    print(f"max shock fraction: {shock.max():.6f}")


def _print_kd_vs_non_kd(frame: pd.DataFrame) -> None:
    print("\nSHOCK ON KD VS NON-KD STRIKES")
    rows = []
    for kd_value, label in ((0, "non-KD"), (1, "KD")):
        g = frame[frame["knockdown"] == kd_value]
        row: dict[str, Any] = {
            "strike_type": label,
            "strikes": len(g),
            "mean_shock": g["shock_fraction"].mean(),
            "median_shock": g["shock_fraction"].median(),
        }
        for q in (0.75, 0.90, 0.95, 0.99):
            row[f"p{100*q:g}_shock"] = g["shock_fraction"].quantile(q)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def _print_empirical_shock_buckets(frame: pd.DataFrame) -> None:
    print("\nKD RATE ACROSS EMPIRICAL SHOCK DECILES")
    work = frame.copy()
    work["shock_bucket"] = _rank_bucket(
        work["shock_fraction"],
        [
            "D1 lowest", "D2", "D3", "D4", "D5",
            "D6", "D7", "D8", "D9", "D10 highest",
        ],
    )
    rows = []
    for bucket, g in work.groupby("shock_bucket", observed=True, sort=False):
        rows.append(
            {
                "shock_bucket": str(bucket),
                "strikes": len(g),
                "mean_shock": g["shock_fraction"].mean(),
                "min_shock": g["shock_fraction"].min(),
                "max_shock": g["shock_fraction"].max(),
                "mean_model_kd_probability": g["kd_probability"].mean(),
                "realized_kd_per_strike": g["knockdown"].mean(),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nUPPER-TAIL SHOCK THRESHOLDS")
    rows = []
    for q in (0.90, 0.95, 0.99, 0.995, 0.999):
        threshold = work["shock_fraction"].quantile(q)
        g = work[work["shock_fraction"] >= threshold]
        rows.append(
            {
                "tail": f">=p{100*q:g}",
                "shock_threshold": threshold,
                "strikes": len(g),
                "realized_kd_per_strike": g["knockdown"].mean(),
                "mean_reservoir_before": g["reservoir_fraction_before"].mean(),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def _print_reservoir_interaction(frame: pd.DataFrame) -> None:
    print("\nSHOCK / KD BY RESERVOIR CONDITION")
    work = frame.copy()
    work["reservoir_bucket"] = pd.cut(
        work["reservoir_fraction_before"],
        bins=[-1e-9, 0.25, 0.50, 0.75, 1.0000001],
        labels=["0-25%", "25-50%", "50-75%", "75-100%"],
        include_lowest=True,
    )
    rows = []
    for bucket, g in work.groupby("reservoir_bucket", observed=True, sort=False):
        kd = g[g["knockdown"] == 1]
        rows.append(
            {
                "reservoir_bucket": str(bucket),
                "strikes": len(g),
                "mean_shock": g["shock_fraction"].mean(),
                "p95_shock": g["shock_fraction"].quantile(0.95),
                "kd_per_strike": g["knockdown"].mean(),
                "median_shock_on_KD": (
                    kd["shock_fraction"].median() if len(kd) else np.nan
                ),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def _print_power_tail(frame: pd.DataFrame) -> None:
    print("\nSHOCK DISTRIBUTION BY ATTACKER POWER QUINTILE")
    work = frame.copy()
    work["power_bucket"] = _rank_bucket(
        work["attacker_power"],
        ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
    )
    rows = []
    for bucket, g in work.groupby("power_bucket", observed=True, sort=False):
        rows.append(
            {
                "power_bucket": str(bucket),
                "strikes": len(g),
                "mean_power": g["attacker_power"].mean(),
                "median_shock": g["shock_fraction"].median(),
                "p90_shock": g["shock_fraction"].quantile(0.90),
                "p95_shock": g["shock_fraction"].quantile(0.95),
                "p99_shock": g["shock_fraction"].quantile(0.99),
                "p99_5_shock": g["shock_fraction"].quantile(0.995),
                "kd_per_strike": g["knockdown"].mean(),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def _print_summary(frame: pd.DataFrame) -> None:
    print("\n" + "=" * 120)
    print("DAMAGE RESERVOIR V1 — STRIKE SHOCK AUDIT")
    print("=" * 120)
    print(f"landed significant strikes: {len(frame):,}")
    print(f"overall KD per landed strike: {frame['knockdown'].mean():.6f}")
    print(
        "definition: shock_fraction = strike_damage / defender_reservoir_capacity"
    )
    print(
        "boundary: this audit characterizes shock only; it does not define or "
        "calibrate a KO/TKO curve."
    )

    _print_percentiles(frame)
    _print_kd_vs_non_kd(frame)
    _print_empirical_shock_buckets(frame)
    _print_reservoir_interaction(frame)
    _print_power_tail(frame)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit calibrated Damage Reservoir V1 strike shock distribution"
    )
    parser.add_argument("--matchups", type=int, default=DEFAULT_MATCHUPS)
    parser.add_argument(
        "--paths-per-matchup", type=int, default=DEFAULT_PATHS_PER_MATCHUP
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    print(f"[shock audit] loading profiles from {args.fsr_path}", flush=True)
    profiles = damage.load_profiles(args.fsr_path)
    print(f"[shock audit] latest fighter profiles: {len(profiles):,}", flush=True)
    print(
        f"[shock audit] matchups={args.matchups:,}; "
        f"paths_per_matchup={args.paths_per_matchup:,}; rounds={args.rounds}",
        flush=True,
    )

    strikes = _run_audit(
        profiles,
        matchup_count=args.matchups,
        paths_per_matchup=args.paths_per_matchup,
        rounds=args.rounds,
        seed=args.seed,
    )
    if strikes.empty:
        raise RuntimeError("Shock audit produced no landed strikes")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    strikes.to_parquet(args.output, index=False)
    _print_summary(strikes)

    print(f"\n[shock audit] wrote {args.output}", flush=True)
    print(
        "\nRESEARCH BOUNDARY: inspect the calibrated shock scale and its KD "
        "relationship. Do not choose KO/TKO mechanics from this audit alone.",
        flush=True,
    )


if __name__ == "__main__":
    main()
