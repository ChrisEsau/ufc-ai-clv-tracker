"""Focused replay of the strongest age-decay candidate on actual R1 KO/TKO bouts.

This is a fast diagnostic companion to the full multi-variant replay.
It runs only the best curve from the prior age-decay search:

    linear_on30_s2 = subtract 2 effective trait points per year after age 30

The curve is applied independently to knockdown_resistance and
 damage_durability. Stored leakage-safe FSR values are never modified.

Default scope:
- mature 2020+ cohort
- actual R1 KO/TKO bouts only
- 50 Monte Carlo paths per bout

No controls and no alternative variants are simulated here. The goal is to get
an inexpensive first read on whether this candidate improves R1-KO direction,
especially against older actual losers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_age_decay_curve_search_kd_durability_2020plus_mature as curves
from scripts.experimental import fsr_r1_ko_age_curve_mc_replay_2020plus_mature as replay

OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "r1_ko_age_curve_single_candidate_replay.csv"
)
DEFAULT_PATHS = 50
DEFAULT_SEED = 20260810

BEST_VARIANT = replay.Variant(
    "linear_on30_s2",
    curves.Curve("linear", onset=30.0, slope=2.0),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths",
        type=int,
        default=DEFAULT_PATHS,
        help="Monte Carlo paths per actual R1 KO/TKO bout (default: 50)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def _print_summary(results: pd.DataFrame) -> None:
    results = results.copy()
    results["loser_age_band"] = pd.cut(
        results["loser_age"],
        bins=replay.AGE_BINS,
        labels=replay.AGE_LABELS,
    )

    print("\n" + "=" * 118)
    print("ACTUAL R1 KO/TKO — BEST AGE-CURVE SINGLE-CANDIDATE REPLAY")
    print("=" * 118)
    print(f"variant: {BEST_VARIANT.label}")
    print(f"bouts: {len(results):,}")
    print(f"mean P(any R1 KO): {results['p_any_r1_ko'].mean():.4f}")
    print(
        "mean P(actual winner R1 KO): "
        f"{results['p_actual_winner_r1_ko'].mean():.4f}"
    )
    print(
        "mean P(actual loser R1 KO):  "
        f"{results['p_actual_loser_r1_ko'].mean():.4f}"
    )
    print(
        "winner-direction hit rate:  "
        f"{results['winner_direction_hit'].mean():.4f}"
    )
    print(
        "direction tie rate:         "
        f"{results['winner_direction_tie'].mean():.4f}"
    )

    print("\nBY ACTUAL LOSER AGE")
    by_age = (
        results.groupby("loser_age_band", observed=True)
        .agg(
            bouts=("bout_id", "size"),
            mean_loser_age=("loser_age", "mean"),
            mean_p_any_r1_ko=("p_any_r1_ko", "mean"),
            mean_p_actual_winner_r1_ko=("p_actual_winner_r1_ko", "mean"),
            mean_p_actual_loser_r1_ko=("p_actual_loser_r1_ko", "mean"),
            winner_direction_hit_rate=("winner_direction_hit", "mean"),
        )
        .reset_index()
    )
    print(by_age.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Specifically expose the older-loser slice that motivated this study.
    older = results.loc[results["loser_age"].ge(37.0)]
    if not older.empty:
        print("\nACTUAL LOSER AGE >=37")
        print(f"bouts: {len(older):,}")
        print(f"mean P(any R1 KO): {older['p_any_r1_ko'].mean():.4f}")
        print(
            "mean P(actual winner R1 KO): "
            f"{older['p_actual_winner_r1_ko'].mean():.4f}"
        )
        print(
            "winner-direction hit rate:  "
            f"{older['winner_direction_hit'].mean():.4f}"
        )


def main() -> None:
    args = _parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    cohort, pairs = replay._build_cohort()
    positives = cohort.loc[cohort["actual_r1_ko"].eq(1)].copy().reset_index(drop=True)
    positives["sample_class"] = "actual_r1_ko"

    print(f"mature 2020+ cohort: {len(cohort):,} bouts")
    print(f"actual R1 KO/TKO bouts: {len(positives):,}")
    print(f"variant: {BEST_VARIANT.label}")
    print(f"paths per bout: {args.paths:,}")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    total_paths = len(positives) * args.paths
    completed_paths = 0

    for _, bout in positives.iterrows():
        bout_id = str(bout["bout_id"])
        path_seeds = rng.integers(0, 2**31 - 1, size=args.paths, dtype=np.int64)
        rows.append(
            replay._simulate_bout(
                bout,
                pairs[bout_id],
                BEST_VARIANT,
                path_seeds,
            )
        )
        completed_paths += args.paths
        if completed_paths % 1000 == 0 or completed_paths == total_paths:
            print(
                f"[single age replay] paths {completed_paths:,}/{total_paths:,}; "
                f"bouts {len(rows):,}/{len(positives):,}",
                flush=True,
            )

    results = pd.DataFrame(rows)
    _print_summary(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nWrote {len(results):,} bout rows to {args.output}")
    print("No stored FSR values or simulator constants were changed.")


if __name__ == "__main__":
    main()
