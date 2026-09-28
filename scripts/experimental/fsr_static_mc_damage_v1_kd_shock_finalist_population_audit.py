"""Full-path population validation for Damage Reservoir V1 KD shock finalists.

Purpose
-------
Validate the narrow shock-response finalists selected by the offline curve sweep
through the actual Monte Carlo path engine, including dynamic reservoir depletion
and recent-knockdown feedback.

This script does NOT modify the active Damage V1 constants. Each candidate is
implemented in an isolated subclass and uses the same matchup/path seed schedule
for an apples-to-apples comparison.

Research sequence
-----------------
1. Load the previously generated strike-level shock audit.
2. Fit a base logit for each candidate shock coefficient so the offline strike
   population preserves the current overall KD-per-landed-strike target.
3. Run full 3-round Monte Carlo paths for coefficients 60/70/80/90/100.
4. Compare actual realized path behavior:
   - KD per landed significant strike;
   - >=1, >=2, >=3 KDs per fighter path;
   - shock distribution on KD strikes;
   - share of KDs produced by upper-tail shocks;
   - power - opponent KD-resistance separation;
   - fresh vs depleted KD rate;
   - recent-KD vs normal-strike KD rate.

Boundary
--------
This is still KD calibration only. It does not implement or calibrate KO/TKO.
Do not promote a candidate from this script without reviewing the population
results against the documented historical constraints.
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
SHOCK_AUDIT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_shock_audit.parquet"
)
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_kd_shock_finalist_population_audit.parquet"
)
STRIKE_OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_kd_shock_finalist_strikes.parquet"
)

FINALIST_SHOCK_COEFFICIENTS = (60.0, 70.0, 80.0, 90.0, 100.0)
DEFAULT_MATCHUPS = 300
DEFAULT_PATHS_PER_MATCHUP = 20
DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260809


def _sigmoid_array(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def _fit_base_logit(
    frame: pd.DataFrame,
    shock_coefficient: float,
    target_rate: float,
) -> float:
    """Fit only the intercept while preserving all other active KD terms."""
    shock = pd.to_numeric(frame["shock_fraction"], errors="coerce").to_numpy(float)
    resistance = pd.to_numeric(
        frame["defender_knockdown_resistance"], errors="coerce"
    ).to_numpy(float)
    reservoir_after = pd.to_numeric(
        frame["reservoir_fraction_after"], errors="coerce"
    ).to_numpy(float)
    recent = pd.to_numeric(frame["recent_kd_before"], errors="coerce").to_numpy(float)

    fixed = (
        float(shock_coefficient) * shock
        + (50.0 - resistance) / damage.KD_RESISTANCE_SCALE
        + damage.KD_DEPLETION_COEFFICIENT * (1.0 - reservoir_after)
        + damage.KD_RECENT_KD_LOGIT_BONUS * recent
    )

    lo, hi = -25.0, 5.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        rate = float(_sigmoid_array(mid + fixed).mean())
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)


@dataclass(frozen=True)
class Candidate:
    shock_coefficient: float
    base_logit: float


class FinalistDamageV1(damage.StaticFSRMCDamageV1):
    """Damage V1 with candidate KD curve and strike-level instrumentation."""

    def __init__(
        self,
        *args: Any,
        shock_coefficient: float,
        base_logit: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.audit_shock_coefficient = float(shock_coefficient)
        self.audit_base_logit = float(base_logit)
        self.strike_records: list[dict[str, Any]] = []

    def _candidate_kd_probability(self, defender: int, strike_damage: float) -> float:
        state = self.damage_state[defender]
        resistance = base._value(self.fighters[defender], "knockdown_resistance")
        shock_fraction = strike_damage / state.reservoir_capacity
        depletion = 1.0 - state.reservoir_fraction

        logit_p = (
            self.audit_base_logit
            + self.audit_shock_coefficient * shock_fraction
            + (50.0 - resistance) / damage.KD_RESISTANCE_SCALE
            + damage.KD_DEPLETION_COEFFICIENT * depletion
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
            strike_damage = self._draw_strike_damage(attacker)
            shock_fraction = strike_damage / state.reservoir_capacity

            # Preserve the active Damage V1 ordering exactly:
            # draw damage -> deplete reservoir -> evaluate KD -> draw KD.
            state.reservoir_current = max(
                0.0,
                state.reservoir_current - strike_damage,
            )
            p_kd = self._candidate_kd_probability(defender, strike_damage)
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
                    "reservoir_fraction_before": reservoir_before,
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
        red_i, blue_i = rng.choice(len(profiles), size=2, replace=False)
        pairs.append((int(red_i), int(blue_i)))
    return pairs


def _build_path_schedule(
    profiles: pd.DataFrame,
    matchup_count: int,
    paths_per_matchup: int,
    seed: int,
) -> list[tuple[int, int, int, int, int]]:
    """Build one shared matchup/seed schedule used by every finalist."""
    rng = np.random.default_rng(seed)
    matchups = _choose_matchups(profiles, matchup_count, rng)
    schedule: list[tuple[int, int, int, int, int]] = []
    for matchup_index, (red_i, blue_i) in enumerate(matchups, start=1):
        for path_index in range(paths_per_matchup):
            path_seed = int(rng.integers(0, 2**31 - 1))
            schedule.append(
                (matchup_index, path_index, red_i, blue_i, path_seed)
            )
    return schedule


def _run_candidate(
    profiles: pd.DataFrame,
    schedule: list[tuple[int, int, int, int, int]],
    candidate: Candidate,
    rounds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fighter_rows: list[dict[str, Any]] = []
    strike_rows: list[dict[str, Any]] = []
    total_paths = len(schedule)

    for path_counter, (
        matchup_index,
        path_index,
        red_i,
        blue_i,
        path_seed,
    ) in enumerate(schedule, start=1):
        red = profiles.iloc[red_i]
        blue = profiles.iloc[blue_i]
        sim = FinalistDamageV1(
            red,
            blue,
            rounds=rounds,
            seed=path_seed,
            shock_coefficient=candidate.shock_coefficient,
            base_logit=candidate.base_logit,
        )
        sim.run()

        for fighter in (0, 1):
            opponent = 1 - fighter
            stats = sim.stats[fighter]
            assert isinstance(stats, damage.DamageFighterStats)
            fighter_rows.append(
                {
                    "shock_coefficient": candidate.shock_coefficient,
                    "base_logit": candidate.base_logit,
                    "matchup_index": matchup_index,
                    "path_index": path_index,
                    "path_seed": path_seed,
                    "fighter_id": str(sim.fighters[fighter]["fighter_id"]),
                    "opponent_id": str(sim.fighters[opponent]["fighter_id"]),
                    "striking_power": base._value(
                        sim.fighters[fighter], "striking_power"
                    ),
                    "opponent_knockdown_resistance": base._value(
                        sim.fighters[opponent], "knockdown_resistance"
                    ),
                    "sig_landed": stats.sig_landed,
                    "knockdowns_scored": stats.knockdowns_scored,
                    "knockdowns_absorbed": stats.knockdowns_absorbed,
                    "power_minus_opponent_kd_resistance": (
                        base._value(sim.fighters[fighter], "striking_power")
                        - base._value(
                            sim.fighters[opponent], "knockdown_resistance"
                        )
                    ),
                }
            )

        for record in sim.strike_records:
            attacker = int(record["attacker"])
            defender = int(record["defender"])
            strike_rows.append(
                {
                    "shock_coefficient": candidate.shock_coefficient,
                    "base_logit": candidate.base_logit,
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

        if path_counter % 1000 == 0 or path_counter == total_paths:
            print(
                f"[KD finalist {candidate.shock_coefficient:g}] "
                f"paths {path_counter:,}/{total_paths:,}; "
                f"fighter_rows={len(fighter_rows):,}; "
                f"strikes={len(strike_rows):,}",
                flush=True,
            )

    return pd.DataFrame(fighter_rows), pd.DataFrame(strike_rows)


def _summarize_candidate(
    fighters: pd.DataFrame,
    strikes: pd.DataFrame,
    shock_reference: pd.Series,
) -> dict[str, Any]:
    coefficient = float(strikes["shock_coefficient"].iloc[0])
    base_logit = float(strikes["base_logit"].iloc[0])
    landed = len(strikes)
    kds = int(strikes["knockdown"].sum())
    kd_strikes = strikes[strikes["knockdown"] == 1]

    p99 = float(shock_reference.quantile(0.99))
    p995 = float(shock_reference.quantile(0.995))
    p999 = float(shock_reference.quantile(0.999))

    edge = fighters.copy()
    edge["edge_bucket"] = _rank_bucket(
        edge["power_minus_opponent_kd_resistance"],
        ["Q1", "Q2", "Q3", "Q4", "Q5"],
    )
    edge_rates: dict[str, float] = {}
    for bucket, group in edge.groupby("edge_bucket", observed=True, sort=False):
        denominator = float(group["sig_landed"].sum())
        edge_rates[str(bucket)] = (
            float(group["knockdowns_scored"].sum()) / denominator
            if denominator > 0
            else float("nan")
        )
    q1 = edge_rates.get("Q1", float("nan"))
    q5 = edge_rates.get("Q5", float("nan"))

    fresh = strikes[strikes["reservoir_fraction_before"] >= 0.75]
    depleted = strikes[strikes["reservoir_fraction_before"] <= 0.25]
    normal = strikes[strikes["recent_kd_before"] == 0]
    recent = strikes[strikes["recent_kd_before"] == 1]

    return {
        "shock_coefficient": coefficient,
        "base_logit": base_logit,
        "fighter_paths": len(fighters),
        "landed_strikes": landed,
        "knockdowns": kds,
        "kd_per_landed_strike": kds / landed if landed else np.nan,
        "fighter_path_ge1_kd": (fighters["knockdowns_scored"] >= 1).mean(),
        "fighter_path_ge2_kd": (fighters["knockdowns_scored"] >= 2).mean(),
        "fighter_path_ge3_kd": (fighters["knockdowns_scored"] >= 3).mean(),
        "mean_kd_per_fighter_path": fighters["knockdowns_scored"].mean(),
        "median_shock_on_kd": (
            kd_strikes["shock_fraction"].median() if len(kd_strikes) else np.nan
        ),
        "p90_shock_on_kd": (
            kd_strikes["shock_fraction"].quantile(0.90)
            if len(kd_strikes)
            else np.nan
        ),
        "kd_share_ge_p99_shock": (
            (kd_strikes["shock_fraction"] >= p99).mean()
            if len(kd_strikes)
            else np.nan
        ),
        "kd_share_ge_p99_5_shock": (
            (kd_strikes["shock_fraction"] >= p995).mean()
            if len(kd_strikes)
            else np.nan
        ),
        "kd_share_ge_p99_9_shock": (
            (kd_strikes["shock_fraction"] >= p999).mean()
            if len(kd_strikes)
            else np.nan
        ),
        "q1_edge_kd_per_sig": q1,
        "q5_edge_kd_per_sig": q5,
        "q5_q1_edge_ratio": q5 / q1 if q1 and np.isfinite(q1) else np.nan,
        "fresh_kd_per_strike": fresh["knockdown"].mean() if len(fresh) else np.nan,
        "depleted_kd_per_strike": (
            depleted["knockdown"].mean() if len(depleted) else np.nan
        ),
        "depleted_fresh_ratio": (
            depleted["knockdown"].mean() / fresh["knockdown"].mean()
            if len(depleted) and len(fresh) and fresh["knockdown"].mean() > 0
            else np.nan
        ),
        "normal_kd_per_strike": normal["knockdown"].mean() if len(normal) else np.nan,
        "recent_kd_per_strike": recent["knockdown"].mean() if len(recent) else np.nan,
        "recent_normal_ratio": (
            recent["knockdown"].mean() / normal["knockdown"].mean()
            if len(recent) and len(normal) and normal["knockdown"].mean() > 0
            else np.nan
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full-path KD shock finalist population validation"
    )
    parser.add_argument("--matchups", type=int, default=DEFAULT_MATCHUPS)
    parser.add_argument(
        "--paths-per-matchup", type=int, default=DEFAULT_PATHS_PER_MATCHUP
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--shock-audit", type=Path, default=SHOCK_AUDIT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--strike-output", type=Path, default=STRIKE_OUTPUT_PATH)
    args = parser.parse_args()

    print(f"[KD finalist audit] loading profiles from {args.fsr_path}", flush=True)
    profiles = damage.load_profiles(args.fsr_path)
    print(f"[KD finalist audit] latest fighter profiles: {len(profiles):,}", flush=True)

    print(f"[KD finalist audit] loading shock audit from {args.shock_audit}", flush=True)
    shock_audit = pd.read_parquet(args.shock_audit)
    required = {
        "shock_fraction",
        "defender_knockdown_resistance",
        "reservoir_fraction_after",
        "recent_kd_before",
        "knockdown",
    }
    missing = sorted(required - set(shock_audit.columns))
    if missing:
        raise ValueError(f"shock audit missing required columns: {missing}")

    target_rate = float(shock_audit["knockdown"].mean())
    print(
        f"[KD finalist audit] target overall KD/strike={target_rate:.6f}; "
        f"coefficients={FINALIST_SHOCK_COEFFICIENTS}",
        flush=True,
    )

    candidates = [
        Candidate(
            shock_coefficient=coefficient,
            base_logit=_fit_base_logit(shock_audit, coefficient, target_rate),
        )
        for coefficient in FINALIST_SHOCK_COEFFICIENTS
    ]
    print("[KD finalist audit] fitted candidate intercepts:", flush=True)
    for candidate in candidates:
        print(
            f"  shock={candidate.shock_coefficient:g}; "
            f"base_logit={candidate.base_logit:.6f}",
            flush=True,
        )

    schedule = _build_path_schedule(
        profiles,
        args.matchups,
        args.paths_per_matchup,
        args.seed,
    )
    print(
        f"[KD finalist audit] shared schedule: {len(schedule):,} paths/candidate; "
        f"rounds={args.rounds}",
        flush=True,
    )

    all_fighters: list[pd.DataFrame] = []
    all_strikes: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    shock_reference = pd.to_numeric(shock_audit["shock_fraction"], errors="coerce")

    for candidate in candidates:
        print(
            "\n" + "=" * 120 + "\n"
            f"RUNNING FULL-PATH FINALIST: shock_coefficient="
            f"{candidate.shock_coefficient:g}, base_logit={candidate.base_logit:.6f}\n"
            + "=" * 120,
            flush=True,
        )
        fighters, strikes = _run_candidate(
            profiles,
            schedule,
            candidate,
            args.rounds,
        )
        all_fighters.append(fighters)
        all_strikes.append(strikes)
        summaries.append(_summarize_candidate(fighters, strikes, shock_reference))

    summary = pd.DataFrame(summaries).sort_values("shock_coefficient")
    combined_fighters = pd.concat(all_fighters, ignore_index=True)
    combined_strikes = pd.concat(all_strikes, ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(args.output, index=False)
    combined_strikes.to_parquet(args.strike_output, index=False)

    print("\n" + "=" * 160)
    print("DAMAGE RESERVOIR V1 — FULL-PATH KD SHOCK FINALIST SUMMARY")
    print("=" * 160)
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nREFERENCE TARGETS / INTERPRETATION BOUNDARY")
    print(f"offline target overall KD per landed strike: {target_rate:.6f}")
    print("historical power - opponent KD-resistance extreme separation: ~2.94x")
    print(
        "Choose no coefficient from one metric alone. Review KD frequency, "
        "multi-KD chains, shock concentration, fighter separation, depletion, "
        "and recent-KD feedback together."
    )
    print("KO/TKO mechanics remain out of scope for this audit.")
    print(f"\n[KD finalist audit] wrote {args.output}")
    print(f"[KD finalist audit] wrote {args.strike_output}")


if __name__ == "__main__":
    main()
