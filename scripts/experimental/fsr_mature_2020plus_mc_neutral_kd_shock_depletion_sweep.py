"""Screen KD shock and depletion sensitivity with striking_power neutralized.

Research only. No simulator or FSR artifact is modified.

Why this sweep
--------------
The full-cohort neutral-power baseline showed opposite errors:
- Round-1 KD incidence is slightly too low.
- Total-fight KD incidence is materially too high.

A single global KD-strength knob cannot fix both. This screen therefore varies:
1. KD shock coefficient: acute response to a single damaging strike.
2. KD depletion coefficient: extra KD susceptibility as the damage reservoir is depleted.

All fighter-specific striking_power effects on severity are disabled so KD physics can
be calibrated independently of the new power trait.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import exp

import numpy as np

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse
from scripts.experimental import fsr_static_mc_ko_tko_v3_3_global_recovery as v33
from scripts.experimental import fsr_static_mc_v0 as base

DEFAULT_BOUTS = 200
DEFAULT_PATHS = 10
DEFAULT_SEED = 20260810
SHOCK_VALUES = (90.0, 95.0, 100.0)
DEPLETION_VALUES = (0.0, 0.5, 1.0, 1.5)
STRONG_COLLAPSE = collapse.CollapseCandidate("strong", 5.0, 2.0)

HISTORICAL_MEAN_R1_KD = 0.2281
HISTORICAL_ANY_R1_KD = 0.2000
HISTORICAL_MEAN_KD = 0.4364
HISTORICAL_ANY_KD = 0.3578
HISTORICAL_R1_KO = 0.1406
HISTORICAL_ANY_KO = 0.3144


@dataclass(frozen=True)
class Candidate:
    shock: float
    depletion: float


CANDIDATES = tuple(
    Candidate(shock, depletion)
    for shock in SHOCK_VALUES
    for depletion in DEPLETION_VALUES
)


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return 1.0 / (1.0 + exp(-x))


class NeutralKDCalibrationV33(v33.StaticFSRMCKOTKOV33GlobalRecovery):
    """Neutral strike severity with candidate KD shock/depletion coefficients."""

    def __init__(self, *args, candidate: Candidate, **kwargs) -> None:
        self.kd_candidate = candidate
        super().__init__(*args, **kwargs)

    def _tail_probability(self, attacker: int) -> float:
        return float(damage.POWER_TAIL_BASE_PROBABILITY)

    def _draw_strike_damage(self, attacker: int) -> float:
        raw_damage = float(
            self.rng.gamma(
                damage.BASE_SEVERITY_GAMMA_SHAPE,
                damage.BASE_SEVERITY_GAMMA_SCALE,
            )
        )
        if self.rng.random() < self._tail_probability(attacker):
            raw_damage += float(
                self.rng.gamma(
                    damage.TAIL_SEVERITY_GAMMA_SHAPE,
                    damage.TAIL_SEVERITY_GAMMA_SCALE,
                )
            )
        return max(0.0, raw_damage * damage.STRIKE_DAMAGE_SCALE)

    def _knockdown_probability(self, defender: int, strike_damage: float) -> float:
        state = self.damage_state[defender]
        resistance = base._value(self.fighters[defender], "knockdown_resistance")
        shock_fraction = strike_damage / state.reservoir_capacity
        depletion = 1.0 - state.reservoir_fraction
        logit_p = (
            damage.KD_BASE_LOGIT
            + self.kd_candidate.shock * shock_fraction
            + (50.0 - resistance) / damage.KD_RESISTANCE_SCALE
            + self.kd_candidate.depletion * depletion
            + (damage.KD_RECENT_KD_LOGIT_BONUS if state.recent_knockdown else 0.0)
        )
        return float(np.clip(_sigmoid(logit_p), 0.0, 0.95))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bouts", type=int, default=DEFAULT_BOUTS)
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.bouts <= 0 or args.paths <= 0:
        raise ValueError("--bouts and --paths must be positive")

    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.head(args.bouts).reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    seed_matrix = rng.integers(
        0,
        2**31 - 1,
        size=(len(cohort), args.paths),
        dtype=np.int64,
    )

    rows = []
    for candidate in CANDIDATES:
        total_paths = len(cohort) * args.paths
        kd_total = 0
        any_kd_count = 0
        r1_kd_total = 0
        any_r1_kd_count = 0
        ko_count = 0
        r1_ko_count = 0
        completed = 0

        for bout_idx, (_, bout) in enumerate(cohort.iterrows()):
            red, blue = pairs[str(bout["bout_id"])]
            r_age = float(bout["r_age"]) if not np.isnan(bout["r_age"]) else None
            b_age = float(bout["b_age"]) if not np.isnan(bout["b_age"]) else None

            for seed in seed_matrix[bout_idx]:
                kwargs = dict(
                    collapse=STRONG_COLLAPSE,
                    seed=int(seed),
                    red_age=r_age,
                    blue_age=b_age,
                    candidate=candidate,
                )

                sim = NeutralKDCalibrationV33(red, blue, rounds=3, **kwargs)
                result = sim.run()
                path_kd = int(sim.stats[0].knockdowns_scored) + int(sim.stats[1].knockdowns_scored)
                kd_total += path_kd
                any_kd_count += int(path_kd > 0)
                if result.finish is not None:
                    ko_count += 1
                    r1_ko_count += int(int(getattr(result.finish, "round", 0) or 0) == 1)

                # Exact Round-1 prefix replay using the same path seed.
                sim_r1 = NeutralKDCalibrationV33(red, blue, rounds=1, **kwargs)
                result_r1 = sim_r1.run()
                path_r1_kd = int(sim_r1.stats[0].knockdowns_scored) + int(sim_r1.stats[1].knockdowns_scored)
                r1_kd_total += path_r1_kd
                any_r1_kd_count += int(path_r1_kd > 0)

            completed += args.paths
            if completed % 500 == 0 or bout_idx + 1 == len(cohort):
                print(
                    f"[shock={candidate.shock:.0f}, dep={candidate.depletion:.1f}] "
                    f"paths {completed:,}/{total_paths:,}",
                    flush=True,
                )

        mean_r1_kd = r1_kd_total / total_paths
        any_r1_kd = any_r1_kd_count / total_paths
        mean_kd = kd_total / total_paths
        any_kd = any_kd_count / total_paths
        r1_ko = r1_ko_count / total_paths
        any_ko = ko_count / total_paths

        # Simple KD-only screening score. Normalize each error by its historical target
        # so mean and incidence metrics contribute on similar relative scales.
        score = (
            abs(mean_r1_kd - HISTORICAL_MEAN_R1_KD) / HISTORICAL_MEAN_R1_KD
            + abs(any_r1_kd - HISTORICAL_ANY_R1_KD) / HISTORICAL_ANY_R1_KD
            + abs(mean_kd - HISTORICAL_MEAN_KD) / HISTORICAL_MEAN_KD
            + abs(any_kd - HISTORICAL_ANY_KD) / HISTORICAL_ANY_KD
        )

        rows.append(
            dict(
                shock=candidate.shock,
                depletion=candidate.depletion,
                mean_r1_kd=mean_r1_kd,
                any_r1_kd=any_r1_kd,
                mean_kd=mean_kd,
                any_kd=any_kd,
                r1_ko=r1_ko,
                any_ko=any_ko,
                score=score,
            )
        )

    rows.sort(key=lambda r: r["score"])

    print("\n" + "=" * 124)
    print("NEUTRAL-POWER KD CALIBRATION — SHOCK x DEPLETION 200-BOUT SCREEN")
    print("=" * 124)
    print(f"bouts: {len(cohort):,}; paths/bout: {args.paths}; seed: {args.seed}")
    print("striking_power severity mapping: OFF")
    print("collapse: strong (5.0, 2.0); recent-KD bonus unchanged")
    print("\nKD HISTORICAL TARGETS")
    print(
        f"mean R1 KD={HISTORICAL_MEAN_R1_KD:.4f}; any R1 KD={HISTORICAL_ANY_R1_KD:.2%}; "
        f"mean KD={HISTORICAL_MEAN_KD:.4f}; any KD={HISTORICAL_ANY_KD:.2%}"
    )
    print("\nRESULTS — sorted by KD target error")
    print(
        f"{'shock':>6} {'dep':>5} {'meanR1KD':>10} {'anyR1KD':>9} "
        f"{'meanKD':>9} {'anyKD':>9} {'R1KO':>8} {'anyKO':>8} {'score':>8}"
    )
    for row in rows:
        print(
            f"{row['shock']:6.0f} {row['depletion']:5.1f} "
            f"{row['mean_r1_kd']:10.4f} {row['any_r1_kd']:9.2%} "
            f"{row['mean_kd']:9.4f} {row['any_kd']:9.2%} "
            f"{row['r1_ko']:8.2%} {row['any_ko']:8.2%} {row['score']:8.3f}"
        )

    print("\nUse this only to identify promising KD physics. Confirm finalists on the full 1,565-bout cohort before locking.")
    print("Research only: no simulator or FSR artifact modified.")


if __name__ == "__main__":
    main()
