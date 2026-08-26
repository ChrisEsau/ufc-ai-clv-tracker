"""Focused Round-1 KD shock-coefficient sweep on 200 mature 2020+ bouts.

Only KD_SHOCK_COEFFICIENT changes. Everything else remains locked to the current
V3.3 configuration, including the base KD logit, resistance/depletion terms,
strong KD collapse, FSR inputs, one-round horizon, and deterministic seeds.

Candidates: 90, 100, 110. The first 200 bouts from the established aligned
mature 2020+ cohort are used for a fast calibration screen.
"""
from __future__ import annotations

import argparse
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
SHOCK_COEFFICIENTS = (90.0, 100.0, 110.0)
STRONG_COLLAPSE = collapse.CollapseCandidate("strong", 5.0, 2.0)

HISTORICAL_MEAN_R1_KD_PER_FIGHT = 0.2281
HISTORICAL_ANY_R1_KD_RATE = 0.2000
HISTORICAL_R1_KO_RATE = 0.1406


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return 1.0 / (1.0 + exp(-x))


class ShockSweepV33(v33.StaticFSRMCKOTKOV33GlobalRecovery):
    """V3.3 with only the KD shock coefficient overridden for this audit."""

    def __init__(self, *args, kd_shock_coefficient: float, **kwargs) -> None:
        self.kd_shock_coefficient = float(kd_shock_coefficient)
        super().__init__(*args, **kwargs)

    def _knockdown_probability(self, defender: int, strike_damage: float) -> float:
        state = self.damage_state[defender]
        resistance = base._value(self.fighters[defender], "knockdown_resistance")
        shock_fraction = strike_damage / state.reservoir_capacity
        depletion = 1.0 - state.reservoir_fraction

        logit_p = (
            damage.KD_BASE_LOGIT
            + self.kd_shock_coefficient * shock_fraction
            + (50.0 - resistance) / damage.KD_RESISTANCE_SCALE
            + damage.KD_DEPLETION_COEFFICIENT * depletion
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
    if args.bouts <= 0:
        raise ValueError("--bouts must be positive")
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.head(args.bouts).reset_index(drop=True)

    # Pre-generate one shared seed matrix so every coefficient sees exactly the
    # same bouts and path seeds. This makes candidate deltas paired and clean.
    rng = np.random.default_rng(args.seed)
    seed_matrix = rng.integers(
        0,
        2**31 - 1,
        size=(len(cohort), args.paths),
        dtype=np.int64,
    )

    rows = []
    for coefficient in SHOCK_COEFFICIENTS:
        total_paths = len(cohort) * args.paths
        total_kds = 0
        paths_with_kd = 0
        r1_kos = 0
        red_kds = 0
        blue_kds = 0
        completed = 0

        for bout_idx, (_, bout) in enumerate(cohort.iterrows()):
            red, blue = pairs[str(bout["bout_id"])]
            r_age = float(bout["r_age"]) if not np.isnan(bout["r_age"]) else None
            b_age = float(bout["b_age"]) if not np.isnan(bout["b_age"]) else None

            for seed in seed_matrix[bout_idx]:
                sim = ShockSweepV33(
                    red,
                    blue,
                    collapse=STRONG_COLLAPSE,
                    rounds=1,
                    seed=int(seed),
                    red_age=r_age,
                    blue_age=b_age,
                    kd_shock_coefficient=coefficient,
                )
                result = sim.run()

                r_kd = int(sim.stats[0].knockdowns_scored)
                b_kd = int(sim.stats[1].knockdowns_scored)
                red_kds += r_kd
                blue_kds += b_kd
                total_kds += r_kd + b_kd
                if (r_kd + b_kd) > 0:
                    paths_with_kd += 1
                if result.finish is not None:
                    r1_kos += 1

            completed += args.paths
            if completed % 500 == 0 or bout_idx + 1 == len(cohort):
                print(
                    f"[shock={coefficient:.0f}] paths {completed:,}/{total_paths:,}; "
                    f"bouts {bout_idx + 1:,}/{len(cohort):,}",
                    flush=True,
                )

        mean_kd = total_kds / total_paths
        any_kd = paths_with_kd / total_paths
        ko_rate = r1_kos / total_paths
        rows.append((coefficient, mean_kd, any_kd, ko_rate, red_kds / total_paths, blue_kds / total_paths))

    print("\n" + "=" * 108)
    print("V3.3 ROUND-1 KD SHOCK COEFFICIENT SWEEP — 200-BOUT SCREEN")
    print("=" * 108)
    print(f"bouts: {len(cohort):,}")
    print(f"paths per bout: {args.paths}")
    print(f"paths per candidate: {len(cohort) * args.paths:,}")
    print(f"shared seed: {args.seed}")
    print(f"locked KD base logit: {damage.KD_BASE_LOGIT:.6f}")
    print("locked KD collapse: strong (scale=5.0, curvature=2.0)")
    print("\nHistorical full-cohort reference targets:")
    print(f"  mean R1 KDs/fight: {HISTORICAL_MEAN_R1_KD_PER_FIGHT:.4f}")
    print(f"  fights with >=1 R1 KD: {HISTORICAL_ANY_R1_KD_RATE:.2%}")
    print(f"  R1 KO/TKO rate: {HISTORICAL_R1_KO_RATE:.2%}")

    print("\nRESULTS")
    print(
        f"{'shock':>7} {'mean_R1_KD':>12} {'any_R1_KD':>11} "
        f"{'R1_KO':>9} {'red_KD':>10} {'blue_KD':>10}"
    )
    for coefficient, mean_kd, any_kd, ko_rate, red_mean, blue_mean in rows:
        print(
            f"{coefficient:7.0f} {mean_kd:12.4f} {any_kd:11.2%} "
            f"{ko_rate:9.2%} {red_mean:10.4f} {blue_mean:10.4f}"
        )

    print("\nNOTE: This is a 200-bout screening sweep. Historical targets above are from the full 1,565-bout cohort.")
    print("Use this to identify the promising shock coefficient; confirm the finalist on the full cohort before locking it.")


if __name__ == "__main__":
    main()
