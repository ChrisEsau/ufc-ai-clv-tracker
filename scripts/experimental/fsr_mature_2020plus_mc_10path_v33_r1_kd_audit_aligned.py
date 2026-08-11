"""Exact Round-1 KD audit for the aligned mature 2020+ V3.3 cohort.

Runs the same 1,565-bout aligned cohort and the same deterministic seed schedule
as the existing V3.3 population audits, but with a one-round horizon. Because
Round 1 is identical regardless of later-round horizon, this gives an exact
Round-1 knockdown count without depending on event-log instrumentation.
"""
from __future__ import annotations

import argparse

import numpy as np

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse
from scripts.experimental import fsr_static_mc_ko_tko_v3_3_global_recovery as v33

DEFAULT_PATHS = 10
DEFAULT_SEED = 20260810
STRONG_COLLAPSE = collapse.CollapseCandidate("strong", 5.0, 2.0)

HISTORICAL_MEAN_R1_KD_PER_FIGHT = 0.2281
HISTORICAL_ANY_R1_KD_RATE = 0.2000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    cohort, pairs = cohort32.build_aligned_cohort()
    rng = np.random.default_rng(args.seed)

    total_paths = len(cohort) * args.paths
    paths_with_kd = 0
    total_kds = 0
    red_kds = 0
    blue_kds = 0
    completed = 0

    for bout_no, (_, bout) in enumerate(cohort.iterrows(), start=1):
        red, blue = pairs[str(bout["bout_id"])]
        r_age = float(bout["r_age"]) if not np.isnan(bout["r_age"]) else None
        b_age = float(bout["b_age"]) if not np.isnan(bout["b_age"]) else None
        seeds = rng.integers(0, 2**31 - 1, size=args.paths, dtype=np.int64)

        for seed in seeds:
            sim = v33.StaticFSRMCKOTKOV33GlobalRecovery(
                red,
                blue,
                collapse=STRONG_COLLAPSE,
                rounds=1,
                seed=int(seed),
                red_age=r_age,
                blue_age=b_age,
            )
            sim.run()

            r_kd = int(sim.stats[0].knockdowns_scored)
            b_kd = int(sim.stats[1].knockdowns_scored)
            red_kds += r_kd
            blue_kds += b_kd
            total_kds += r_kd + b_kd
            if (r_kd + b_kd) > 0:
                paths_with_kd += 1

        completed += args.paths
        if completed % 1000 == 0 or bout_no == len(cohort):
            print(
                f"[V3.3 R1 KD audit] paths {completed:,}/{total_paths:,}; "
                f"bouts {bout_no:,}/{len(cohort):,}",
                flush=True,
            )

    sim_any_r1_kd_rate = paths_with_kd / total_paths
    sim_mean_r1_kd = total_kds / total_paths

    print("\n" + "=" * 88)
    print("ROUND 1 KD AUDIT — EXACT MATURE 2020+ V3.3 MC COHORT")
    print("=" * 88)
    print(f"cohort fights: {len(cohort):,}")
    print(f"paths per bout: {args.paths}")
    print(f"total Round-1 paths: {total_paths:,}")

    print("\nHISTORICAL TARGETS")
    print(f"historical mean R1 KDs/fight: {HISTORICAL_MEAN_R1_KD_PER_FIGHT:.4f}")
    print(f"historical fights with >=1 R1 KD: {HISTORICAL_ANY_R1_KD_RATE:.2%}")

    print("\nV3.3 ROUND 1 KDS")
    print(f"simulated total R1 KDs: {total_kds:,}")
    print(f"simulated mean R1 KDs/fight: {sim_mean_r1_kd:.4f}")
    print(f"simulated paths with >=1 R1 KD: {sim_any_r1_kd_rate:.2%}")
    print(f"simulated mean red R1 KDs/fight: {red_kds / total_paths:.4f}")
    print(f"simulated mean blue R1 KDs/fight: {blue_kds / total_paths:.4f}")

    print("\nGAP VS HISTORICAL")
    print(
        "mean R1 KD gap: "
        f"{sim_mean_r1_kd - HISTORICAL_MEAN_R1_KD_PER_FIGHT:+.4f} "
        f"({sim_mean_r1_kd / HISTORICAL_MEAN_R1_KD_PER_FIGHT - 1:+.1%})"
    )
    print(
        "any-R1-KD rate gap: "
        f"{sim_any_r1_kd_rate - HISTORICAL_ANY_R1_KD_RATE:+.2%}"
    )


if __name__ == "__main__":
    main()
