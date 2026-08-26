"""Single research audit: KD base -8.70, collapse scale 2.0, curvature 20.0.

Fixed architecture:
- 200 mature 2020+ bouts x 10 paths
- contact sigma 0.80
- power magnitude scale 75
- base damage at power 50 = 1.18
- KD shock coefficient 100
- KD depletion coefficient 0
- collapse scale 2.0
- collapse curvature 20.0
- terminal collapse = KO/TKO only
- surviving knockdown = KD

No production simulator or FSR artifact is modified.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v2 as ko
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse_mod
from scripts.experimental import fsr_static_mc_ko_tko_v3_3_global_recovery as v33
from scripts.experimental import fsr_static_mc_v0 as base

DEFAULT_BOUTS = 200
DEFAULT_PATHS = 10
DEFAULT_SEED = 20260810

CONTACT_SIGMA = 0.80
POWER_MAGNITUDE_SCALE = 75.0
BASE_DAMAGE_AT_POWER_50 = 1.18
KD_BASE_LOGIT = -8.70
KD_SHOCK_COEFFICIENT = 100.0
KD_DEPLETION_COEFFICIENT = 0.0
COLLAPSE_SCALE = 2.0
COLLAPSE_CURVATURE = 20.0

OUTPUT_PATH = Path("data/experimental/kdbase87_curve20_scale2_200.csv")

HIST_R1_KD_MEAN = 0.2281
HIST_TOTAL_KD_MEAN = 0.4364
HIST_ANY_KD = 0.3578
HIST_R1_KO = 0.1406
HIST_TOTAL_KO = 0.3144
HIST_MEAN_KO_ROUND = 1.835

COLLAPSE = collapse_mod.CollapseCandidate(
    "scale2.0_curve20.0", COLLAPSE_SCALE, COLLAPSE_CURVATURE
)


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return 1.0 / (1.0 + exp(-x))


def _power_multiplier(power: float) -> float:
    return float(exp((float(power) - 50.0) / POWER_MAGNITUDE_SCALE))


class AuditSim(v33.StaticFSRMCKOTKOV33GlobalRecovery):
    def __init__(self, *args, **kwargs) -> None:
        self.terminal_collapse_finishes = 0
        self.direct_strike_finishes = 0
        super().__init__(*args, collapse=COLLAPSE, **kwargs)

    def _draw_contact_quality(self) -> float:
        sigma = CONTACT_SIGMA
        return float(self.rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma))

    def _draw_strike_damage(self, attacker: int) -> float:
        effective_power = base._value(self.fighters[attacker], "striking_power")
        q = self._draw_contact_quality()
        return max(0.0, BASE_DAMAGE_AT_POWER_50 * q * _power_multiplier(effective_power))

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
        surviving_kds = 0

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

            state.reservoir_current = max(0.0, state.reservoir_current - effective_damage)

            attacker_stats = self.stats[attacker]
            defender_stats = self.stats[defender]
            assert isinstance(attacker_stats, damage.DamageFighterStats)
            assert isinstance(defender_stats, damage.DamageFighterStats)

            attacker_stats.damage_dealt += effective_damage
            defender_stats.damage_absorbed += effective_damage
            attacker_stats.max_single_strike_damage = max(attacker_stats.max_single_strike_damage, effective_damage)
            defender_stats.max_single_strike_damage = max(defender_stats.max_single_strike_damage, effective_damage)
            total_damage += effective_damage

            if state.reservoir_current <= ko.RESERVOIR_FINISH_EPSILON:
                self.direct_strike_finishes += 1
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

            knockdown = self.rng.random() < self._knockdown_probability(defender, effective_damage)
            if not knockdown:
                continue

            shock_fraction = effective_damage / state.reservoir_capacity
            collapse_fraction = self._kd_collapse_fraction(shock_fraction)
            collapse_damage = min(collapse_fraction * state.reservoir_capacity, state.reservoir_current)
            state.reservoir_current = max(0.0, state.reservoir_current - collapse_damage)

            attacker_stats.damage_dealt += collapse_damage
            defender_stats.damage_absorbed += collapse_damage
            total_damage += collapse_damage

            if state.reservoir_current <= ko.RESERVOIR_FINISH_EPSILON:
                self.terminal_collapse_finishes += 1
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

            attacker_stats.knockdowns_scored += 1
            defender_stats.knockdowns_absorbed += 1
            state.recent_knockdown_segments = max(state.recent_knockdown_segments, damage.RECENT_KD_SEGMENTS)
            surviving_kds += 1

        return total_damage, surviving_kds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bouts", type=int, default=DEFAULT_BOUTS)
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def _run_prefix(red, blue, *, rounds: int, seed: int, red_age, blue_age):
    sim = AuditSim(red, blue, rounds=rounds, seed=seed, red_age=red_age, blue_age=blue_age)
    path = sim.run()
    kd = int(sim.stats[0].knockdowns_scored) + int(sim.stats[1].knockdowns_scored)
    finish_round = int(getattr(path.finish, "round", 0) or 0) if path.finish is not None else 0
    return sim, path, kd, finish_round


def main() -> None:
    args = parse_args()
    if args.bouts <= 0 or args.paths <= 0:
        raise ValueError("--bouts and --paths must be positive")

    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.head(args.bouts).reset_index(drop=True)
    total_paths = len(cohort) * args.paths

    rng = np.random.default_rng(args.seed)
    seed_matrix = rng.integers(0, 2**31 - 1, size=(len(cohort), args.paths), dtype=np.int64)

    r1_kd = r2_kd = r3_kd = 0
    any_kd = 0
    r1_ko = r2_ko = r3_ko = 0
    ko_total = 0
    ko_round_sum = 0
    terminal_ko = 0
    direct_ko = 0
    completed = 0

    print("\n" + "=" * 150)
    print("KD BASE -8.70 + COLLAPSE CURVE 20 — SCALE 2.0 — 200 BOUTS x 10 PATHS")
    print("=" * 150)
    print(f"contact sigma={CONTACT_SIGMA:.2f}; power scale={POWER_MAGNITUDE_SCALE:.0f}")
    print(f"KD base={KD_BASE_LOGIT:.2f}; shock={KD_SHOCK_COEFFICIENT:.0f}; depletion={KD_DEPLETION_COEFFICIENT:.2f}")
    print(f"collapse scale={COLLAPSE_SCALE:.1f}; curvature={COLLAPSE_CURVATURE:.1f}")
    print("terminal collapse = KO/TKO only; surviving knockdown = KD")
    print(f"CSV: {OUTPUT_PATH}")

    for bout_idx, (_, bout) in enumerate(cohort.iterrows()):
        red, blue = pairs[str(bout["bout_id"])]
        r_age = float(bout["r_age"]) if not np.isnan(bout["r_age"]) else None
        b_age = float(bout["b_age"]) if not np.isnan(bout["b_age"]) else None

        for seed in seed_matrix[bout_idx]:
            _, _, kd1, _ = _run_prefix(red, blue, rounds=1, seed=int(seed), red_age=r_age, blue_age=b_age)
            _, _, kd2, _ = _run_prefix(red, blue, rounds=2, seed=int(seed), red_age=r_age, blue_age=b_age)
            sim3, path3, kd3, finish_round = _run_prefix(red, blue, rounds=3, seed=int(seed), red_age=r_age, blue_age=b_age)

            r1_kd += kd1
            r2_kd += max(0, kd2 - kd1)
            r3_kd += max(0, kd3 - kd2)
            any_kd += int(kd3 > 0)

            if path3.finish is not None:
                ko_total += 1
                ko_round_sum += finish_round
                r1_ko += int(finish_round == 1)
                r2_ko += int(finish_round == 2)
                r3_ko += int(finish_round == 3)
                terminal_ko += int(sim3.terminal_collapse_finishes > 0)
                direct_ko += int(sim3.direct_strike_finishes > 0)

        completed += args.paths
        if completed % 500 == 0 or bout_idx + 1 == len(cohort):
            print(f"paths {completed:,}/{total_paths:,}", flush=True)

    total_kd = r1_kd + r2_kd + r3_kd
    row: dict[str, Any] = {
        "kd_base_logit": KD_BASE_LOGIT,
        "kd_shock": KD_SHOCK_COEFFICIENT,
        "collapse_scale": COLLAPSE_SCALE,
        "curvature": COLLAPSE_CURVATURE,
        "r1_kd_mean": r1_kd / total_paths,
        "r2_kd_mean": r2_kd / total_paths,
        "r3_kd_mean": r3_kd / total_paths,
        "total_kd_mean": total_kd / total_paths,
        "any_kd": any_kd / total_paths,
        "r1_ko": r1_ko / total_paths,
        "r2_ko": r2_ko / total_paths,
        "r3_ko": r3_ko / total_paths,
        "total_ko": ko_total / total_paths,
        "mean_ko_round": ko_round_sum / ko_total if ko_total else float("nan"),
        "terminal_collapse_ko": terminal_ko / total_paths,
        "direct_strike_ko": direct_ko / total_paths,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(OUTPUT_PATH, index=False)

    print("\nSIMULATED")
    print(f"R1/R2/R3 KD={row['r1_kd_mean']:.4f}/{row['r2_kd_mean']:.4f}/{row['r3_kd_mean']:.4f}")
    print(f"total KD={row['total_kd_mean']:.4f}; any KD={row['any_kd']:.2%}")
    print(f"R1/R2/R3 KO={row['r1_ko']:.2%}/{row['r2_ko']:.2%}/{row['r3_ko']:.2%}")
    print(f"total KO={row['total_ko']:.2%}; mean KO round={row['mean_ko_round']:.3f}")
    print(f"terminal-collapse KO={row['terminal_collapse_ko']:.2%}; direct-strike KO={row['direct_strike_ko']:.2%}")

    print("\nHISTORICAL REFERENCES")
    print(f"R1 KD mean={HIST_R1_KD_MEAN:.4f}; total KD mean={HIST_TOTAL_KD_MEAN:.4f}; any KD={HIST_ANY_KD:.2%}")
    print(f"R1 KO={HIST_R1_KO:.2%}; total KO={HIST_TOTAL_KO:.2%}; mean KO round={HIST_MEAN_KO_ROUND:.3f}")
    print(f"\nSaved CSV: {OUTPUT_PATH}")
    print("Research only: no production simulator or FSR artifact modified.")


if __name__ == "__main__":
    main()
