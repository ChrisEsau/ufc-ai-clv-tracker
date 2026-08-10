"""Quick Round-1 knockdown diagnostic for the 2020+ mature-fighter cohort.

Uses the same cohort and strong-collapse simulator as the modern KO validation,
but explicitly traces whether each Monte Carlo path produces at least one
Round-1 knockdown. This is diagnostic only; no simulator constants are changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.experimental import fsr_static_mc_ko_tko_v2_2020plus_mature_actual_validation as modern_mc
from scripts.experimental import fsr_static_mc_ko_tko_v2_actual_ko_timing_diagnostic as timing
from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature as modern


OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_ko_tko_v2_2020plus_mature_r1_kd_diagnostic.parquet"
)
DEFAULT_PATHS_PER_BOUT = 10
DEFAULT_SEED = 20260810
HEARTBEAT_PATHS = 1000


def _safe_auc(y: pd.Series, p: pd.Series) -> float:
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y.astype(int), p.astype(float)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace R1 knockdowns on the 2020+ mature-fighter cohort")
    parser.add_argument("--master", type=Path, default=modern.MASTER_PATH)
    parser.add_argument("--fsr-path", type=Path, default=modern.FSR_PATH)
    parser.add_argument("--paths-per-bout", type=int, default=DEFAULT_PATHS_PER_BOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if args.paths_per_bout <= 0:
        raise ValueError("--paths-per-bout must be positive")

    master = modern._load_master(args.master)
    candidate = modern._build_outcome_cohort(master)
    cohort, pairs = modern._load_fsr_pairs_for_cohort(args.fsr_path, candidate)
    cohort = modern_mc._attach_actual_metadata(cohort, master)

    rng = np.random.default_rng(args.seed)
    total_paths = len(cohort) * args.paths_per_bout
    counter = 0
    rows: list[dict[str, object]] = []

    print(
        f"[2020+ mature R1 KD] bouts={len(cohort):,}; paths_per_bout={args.paths_per_bout}; "
        f"total_paths={total_paths:,}",
        flush=True,
    )

    for bout_index, (_, bout) in enumerate(cohort.iterrows(), start=1):
        bout_id = str(bout["bout_id"])
        red, blue = pairs[bout_id]
        rounds = modern_mc._scheduled_rounds(bout.get("total_rounds"))

        bout_r1_kd = 0
        bout_r1_total_kd = 0
        bout_r1_ko = 0
        bout_any_ko = 0

        for _ in range(args.paths_per_bout):
            path_seed = int(rng.integers(0, 2**31 - 1))
            sim = timing.TracedStrongKOSim(red, blue, rounds=rounds, seed=path_seed)
            path = sim.run()
            finish = path.finish

            bout_r1_kd += int(sim.r1_kd > 0)
            bout_r1_total_kd += int(sim.r1_kd)
            bout_any_ko += int(finish is not None)
            bout_r1_ko += int(finish is not None and finish.round == 1)

            counter += 1
            if counter % HEARTBEAT_PATHS == 0 or counter == total_paths:
                recent = rows[-100:] if rows else []
                print(
                    f"[2020+ mature R1 KD] paths {counter:,}/{total_paths:,}; "
                    f"bouts_started={bout_index:,}/{len(cohort):,}",
                    flush=True,
                )

        rows.append(
            {
                "bout_id": bout_id,
                "event_date": bout["event_date"],
                "actual_ko_tko": int(bout["actual_ko_tko"]),
                "actual_r1_ko": int(bout["actual_r1_ko"]),
                "mc_paths": args.paths_per_bout,
                "mc_p_r1_kd": bout_r1_kd / args.paths_per_bout,
                "mc_mean_r1_kd": bout_r1_total_kd / args.paths_per_bout,
                "mc_p_r1_ko": bout_r1_ko / args.paths_per_bout,
                "mc_p_any_ko": bout_any_ko / args.paths_per_bout,
            }
        )

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    actual_r1 = out[out["actual_r1_ko"].eq(1)]
    non_r1 = out[out["actual_r1_ko"].eq(0)]

    print("\n" + "=" * 100)
    print("2020+ MATURE-FIGHTER ROUND-1 KNOCKDOWN DIAGNOSTIC")
    print("=" * 100)
    print(f"bouts: {len(out):,}")
    print(f"actual R1 KO bouts: {len(actual_r1):,} ({len(actual_r1)/len(out):.2%})")
    print(f"mean MC P(R1 KD), all bouts: {out['mc_p_r1_kd'].mean():.2%}")
    print(f"mean MC P(R1 KD), actual R1 KO bouts: {actual_r1['mc_p_r1_kd'].mean():.2%}")
    print(f"mean MC P(R1 KD), other bouts: {non_r1['mc_p_r1_kd'].mean():.2%}")
    print(f"R1 KO discrimination using MC P(R1 KD): {_safe_auc(out['actual_r1_ko'], out['mc_p_r1_kd']):.4f}")
    print(f"mean MC total R1 KD per path: {out['mc_mean_r1_kd'].mean():.4f}")
    print(f"mean MC P(R1 KO): {out['mc_p_r1_ko'].mean():.2%}")
    print(f"mean MC P(any KO): {out['mc_p_any_ko'].mean():.2%}")

    print("\nINTERPRETATION")
    print("- Low P(R1 KD) even in actual R1-KO bouts -> early shock/KD generation is the bottleneck.")
    print("- Reasonable P(R1 KD) but very low P(R1 KO) -> post-KD collapse/follow-up lethality is the bottleneck.")
    print("- This script changes no simulator constants or FSR values.")
    print(f"\n[2020+ mature R1 KD] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
