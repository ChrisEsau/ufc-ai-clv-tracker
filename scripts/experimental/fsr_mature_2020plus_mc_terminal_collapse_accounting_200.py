"""Research-only 200-bout audit for terminal-collapse KO-only accounting.

Definition under test
---------------------
- A confirmed knockdown whose collapse leaves reservoir > 0 is recorded as a KD.
- A confirmed knockdown whose collapse exhausts the reservoir is recorded only as
  a KO/TKO; it does NOT increment knockdown statistics.
- A strike that directly exhausts the reservoir remains an immediate KO/TKO.

Working strike model is fixed for this audit:
- mean-1 lognormal contact quality, sigma 0.80
- fresh/effective FSR striking_power changes magnitude only
- power magnitude scale 75
- action/landing frequency unchanged by striking_power

KD shape candidate for this first accounting audit:
- base logit -9.15
- shock coefficient 100
- depletion coefficient 0
- resistance and recent-KD terms unchanged

Strong KD collapse remains fixed at (scale=5.0, curvature=2.0).

The script runs 200 bouts x 10 paths by default and writes path-level results so
Codespaces restarts do not erase the evidence.
"""
from __future__ import annotations

import argparse
from math import exp
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v2 as ko
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse
from scripts.experimental import fsr_static_mc_ko_tko_v3_3_global_recovery as v33
from scripts.experimental import fsr_static_mc_v0 as base

DEFAULT_BOUTS = 200
DEFAULT_PATHS = 10
DEFAULT_SEED = 20260810

CONTACT_SIGMA = 0.80
POWER_MAGNITUDE_SCALE = 75.0
BASE_DAMAGE_AT_POWER_50 = 1.18
KD_BASE_LOGIT = -9.15
KD_SHOCK_COEFFICIENT = 100.0
KD_DEPLETION_COEFFICIENT = 0.0
STRONG_COLLAPSE = collapse.CollapseCandidate("strong", 5.0, 2.0)

OUTPUT_PATH = Path("data/experimental/terminal_collapse_accounting_200.csv")
SUMMARY_PATH = Path("data/experimental/terminal_collapse_accounting_200_summary.csv")

HISTORICAL_MEAN_R1_KD = 0.2281
HISTORICAL_ANY_R1_KD_RATE = 0.2000
HISTORICAL_MEAN_KD = 0.4364
HISTORICAL_ANY_KD_RATE = 0.3578
HISTORICAL_ANY_KO_RATE = 0.3144
HISTORICAL_R1_KO_RATE = 0.1406
HISTORICAL_MEAN_KO_ROUND = 1.835


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return 1.0 / (1.0 + exp(-x))


def power_multiplier(power: float) -> float:
    return float(exp((float(power) - 50.0) / POWER_MAGNITUDE_SCALE))


class TerminalCollapseAccountingV33(v33.StaticFSRMCKOTKOV33GlobalRecovery):
    """Current V3.3 physics with new contact/power model and KO-only terminal collapse."""

    def _draw_contact_quality(self) -> float:
        sigma = CONTACT_SIGMA
        return float(self.rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma))

    def _draw_strike_damage(self, attacker: int) -> float:
        effective_power = base._value(self.fighters[attacker], "striking_power")
        return max(
            0.0,
            BASE_DAMAGE_AT_POWER_50
            * self._draw_contact_quality()
            * power_multiplier(effective_power),
        )

    def _knockdown_probability(self, defender: int, strike_damage: float) -> float:
        state = self.damage_state[defender]
        resistance = base._value(self.fighters[defender], "knockdown_resistance")
        shock_fraction = strike_damage / state.reservoir_capacity
        logit_p = (
            KD_BASE_LOGIT
            + KD_SHOCK_COEFFICIENT * shock_fraction
            + (50.0 - resistance) / damage.KD_RESISTANCE_SCALE
            + KD_DEPLETION_COEFFICIENT * (1.0 - state.reservoir_fraction)
            + (damage.KD_RECENT_KD_LOGIT_BONUS if state.recent_knockdown else 0.0)
        )
        return float(np.clip(_sigmoid(logit_p), 0.0, 0.95))

    def _apply_landed_strikes(self, attacker: int, landed: int) -> tuple[float, int]:
        defender = self._other(attacker)
        total_damage = 0.0
        knockdowns = 0

        for _ in range(int(landed)):
            if self.finish is not None:
                break

            state = self.damage_state[defender]
            recent_kd_before = state.recent_knockdown
            reservoir_before = float(state.reservoir_current)

            raw_damage = self._draw_strike_damage(attacker)
            effective_damage = raw_damage
            if recent_kd_before:
                effective_damage *= ko.POST_KD_FOLLOWUP_DAMAGE_MULTIPLIER

            # First apply the landed strike itself.
            state.reservoir_current = max(0.0, state.reservoir_current - effective_damage)

            attacker_stats = self.stats[attacker]
            defender_stats = self.stats[defender]
            assert isinstance(attacker_stats, damage.DamageFighterStats)
            assert isinstance(defender_stats, damage.DamageFighterStats)

            attacker_stats.damage_dealt += effective_damage
            defender_stats.damage_absorbed += effective_damage
            attacker_stats.max_single_strike_damage = max(
                attacker_stats.max_single_strike_damage, effective_damage
            )
            defender_stats.max_single_strike_damage = max(
                defender_stats.max_single_strike_damage, effective_damage
            )
            total_damage += effective_damage

            # A strike can directly end the fight before any KD classification.
            if state.reservoir_current <= ko.RESERVOIR_FINISH_EPSILON:
                self.finish = ko.FinishResult(
                    winner=attacker,
                    loser=defender,
                    method="KO/TKO",
                    raw_strike_damage=float(raw_damage),
                    effective_strike_damage=float(effective_damage),
                    reservoir_before=reservoir_before,
                    reservoir_after=float(state.reservoir_current),
                    knockdown_on_strike=False,
                    recent_kd_before=bool(recent_kd_before),
                )
                break

            # Only surviving strikes proceed to a KD check.
            p_kd = self._knockdown_probability(defender, effective_damage)
            knockdown_event = self.rng.random() < p_kd

            if knockdown_event:
                shock_fraction = effective_damage / state.reservoir_capacity
                collapse_fraction = self._kd_collapse_fraction(shock_fraction)
                collapse_damage = collapse_fraction * state.reservoir_capacity
                collapse_damage = min(collapse_damage, state.reservoir_current)
                state.reservoir_current = max(
                    0.0, state.reservoir_current - collapse_damage
                )
                self.kd_collapse_damage_dealt[attacker] += collapse_damage
                attacker_stats.damage_dealt += collapse_damage
                defender_stats.damage_absorbed += collapse_damage
                total_damage += collapse_damage

                # Classification contract: terminal collapse is KO/TKO ONLY.
                if state.reservoir_current <= ko.RESERVOIR_FINISH_EPSILON:
                    self.finish = ko.FinishResult(
                        winner=attacker,
                        loser=defender,
                        method="KO/TKO",
                        raw_strike_damage=float(raw_damage),
                        effective_strike_damage=float(effective_damage),
                        reservoir_before=reservoir_before,
                        reservoir_after=float(state.reservoir_current),
                        knockdown_on_strike=True,
                        recent_kd_before=bool(recent_kd_before),
                    )
                    break

                # Non-terminal collapse is a true KD because the fight continues.
                attacker_stats.knockdowns_scored += 1
                defender_stats.knockdowns_absorbed += 1
                state.recent_knockdown_segments = max(
                    state.recent_knockdown_segments,
                    damage.RECENT_KD_SEGMENTS,
                )
                knockdowns += 1

        return total_damage, knockdowns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bouts", type=int, default=DEFAULT_BOUTS)
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def _run_prefix(red, blue, *, rounds: int, seed: int, r_age, b_age):
    sim = TerminalCollapseAccountingV33(
        red,
        blue,
        collapse=STRONG_COLLAPSE,
        rounds=rounds,
        seed=seed,
        red_age=r_age,
        blue_age=b_age,
    )
    path = sim.run()
    kd = int(sim.stats[0].knockdowns_scored) + int(sim.stats[1].knockdowns_scored)
    return path, kd


def main() -> None:
    args = parse_args()
    if args.bouts <= 0 or args.paths <= 0:
        raise ValueError("--bouts and --paths must be positive")

    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.head(args.bouts).reset_index(drop=True)
    total_paths = len(cohort) * args.paths

    rng = np.random.default_rng(args.seed)
    seed_matrix = rng.integers(
        0, 2**31 - 1, size=(len(cohort), args.paths), dtype=np.int64
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []

    print("\n" + "=" * 132)
    print("TERMINAL COLLAPSE ACCOUNTING — 200-BOUT x 10-PATH AUDIT")
    print("=" * 132)
    print(f"bouts: {len(cohort):,}; paths/bout: {args.paths}; total paths: {total_paths:,}")
    print(f"contact sigma={CONTACT_SIGMA:.2f}; power scale={POWER_MAGNITUDE_SCALE:.0f}")
    print(f"KD base={KD_BASE_LOGIT:.2f}; shock={KD_SHOCK_COEFFICIENT:.0f}; depletion={KD_DEPLETION_COEFFICIENT:.2f}")
    print("classification: terminal collapse=KO/TKO only; surviving collapse=KD")
    print(f"path CSV: {OUTPUT_PATH}")

    completed = 0
    for bout_idx, (_, bout) in enumerate(cohort.iterrows()):
        red, blue = pairs[str(bout["bout_id"])]
        r_age = float(bout["r_age"]) if not np.isnan(bout["r_age"]) else None
        b_age = float(bout["b_age"]) if not np.isnan(bout["b_age"]) else None

        for seed_value in seed_matrix[bout_idx]:
            seed = int(seed_value)
            full_path, kd3 = _run_prefix(
                red, blue, rounds=3, seed=seed, r_age=r_age, b_age=b_age
            )
            p1, kd1 = _run_prefix(
                red, blue, rounds=1, seed=seed, r_age=r_age, b_age=b_age
            )
            p2, kd2_cum = _run_prefix(
                red, blue, rounds=2, seed=seed, r_age=r_age, b_age=b_age
            )

            r2_kd = max(0, kd2_cum - kd1)
            r3_kd = max(0, kd3 - kd2_cum)
            finish = full_path.finish
            finish_round = int(getattr(finish, "round", 0) or 0) if finish is not None else 0

            rows.append(
                {
                    "bout_id": str(bout["bout_id"]),
                    "seed": seed,
                    "r1_kd": kd1,
                    "r2_kd": r2_kd,
                    "r3_kd": r3_kd,
                    "total_kd": kd3,
                    "any_kd": int(kd3 > 0),
                    "ko": int(finish is not None),
                    "ko_round": finish_round,
                    "r1_ko": int(finish_round == 1),
                    "r2_ko": int(finish_round == 2),
                    "r3_ko": int(finish_round == 3),
                    "terminal_collapse_ko": int(
                        finish is not None and bool(finish.knockdown_on_strike)
                    ),
                    "direct_strike_ko": int(
                        finish is not None and not bool(finish.knockdown_on_strike)
                    ),
                }
            )

        completed += args.paths
        if completed % 500 == 0 or bout_idx + 1 == len(cohort):
            pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
            print(
                f"paths {completed:,}/{total_paths:,}; bouts {bout_idx + 1:,}/{len(cohort):,}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    ko_rows = df.loc[df["ko"] == 1]
    summary = {
        "bouts": len(cohort),
        "paths_per_bout": args.paths,
        "paths": len(df),
        "mean_r1_kd": df["r1_kd"].mean(),
        "mean_r2_kd": df["r2_kd"].mean(),
        "mean_r3_kd": df["r3_kd"].mean(),
        "mean_total_kd": df["total_kd"].mean(),
        "any_kd": df["any_kd"].mean(),
        "r1_ko": df["r1_ko"].mean(),
        "r2_ko": df["r2_ko"].mean(),
        "r3_ko": df["r3_ko"].mean(),
        "any_ko": df["ko"].mean(),
        "mean_ko_round": ko_rows["ko_round"].mean() if len(ko_rows) else np.nan,
        "terminal_collapse_ko": df["terminal_collapse_ko"].mean(),
        "direct_strike_ko": df["direct_strike_ko"].mean(),
    }
    pd.DataFrame([summary]).to_csv(SUMMARY_PATH, index=False)

    print("\nHISTORICAL REFERENCES")
    print(
        f"R1 KD mean={HISTORICAL_MEAN_R1_KD:.4f}; any R1 KD={HISTORICAL_ANY_R1_KD_RATE:.2%}; "
        f"total KD mean={HISTORICAL_MEAN_KD:.4f}; any KD={HISTORICAL_ANY_KD_RATE:.2%}"
    )
    print(
        f"R1 KO={HISTORICAL_R1_KO_RATE:.2%}; total KO={HISTORICAL_ANY_KO_RATE:.2%}; "
        f"mean KO round={HISTORICAL_MEAN_KO_ROUND:.3f}"
    )

    print("\nSIMULATED ROUND BREAKDOWN")
    print(f"R1 KD/fight: {summary['mean_r1_kd']:.4f}")
    print(f"R2 KD/fight: {summary['mean_r2_kd']:.4f}")
    print(f"R3 KD/fight: {summary['mean_r3_kd']:.4f}")
    print(f"TOTAL KD/fight: {summary['mean_total_kd']:.4f}; any KD: {summary['any_kd']:.2%}")
    print(f"R1 KO: {summary['r1_ko']:.2%}")
    print(f"R2 KO: {summary['r2_ko']:.2%}")
    print(f"R3 KO: {summary['r3_ko']:.2%}")
    print(f"TOTAL KO: {summary['any_ko']:.2%}; mean KO round: {summary['mean_ko_round']:.3f}")
    print(f"terminal-collapse KO share of all paths: {summary['terminal_collapse_ko']:.2%}")
    print(f"direct-strike KO share of all paths: {summary['direct_strike_ko']:.2%}")
    print(f"\nSaved path CSV: {OUTPUT_PATH}")
    print(f"Saved summary CSV: {SUMMARY_PATH}")
    print("Research only: no production simulator or FSR artifact modified.")


if __name__ == "__main__":
    main()
