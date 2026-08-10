"""Replay five reproducibly random actual Round-1 KO/TKO bouts.

Selection contract
------------------
- Start from the existing 2020+ mature-fighter Round-1 severity decomposition.
- Restrict to the 220 actual Round-1 KO/TKO bouts.
- Draw five bouts without replacement using a fixed seed.
- Resolve leakage-safe pre-fight FSR pairs.
- Freshly re-simulate Round 1 under the current shadow simulator.
- Print pre-fight FSR traits, aggregate R1 MC behavior, and one representative path.

This is a diagnostic/replay tool only. It changes no FSR values or simulator constants.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_r1_ko_hit_miss_replay as replay


DEFAULT_INPUT = replay.DEFAULT_INPUT
DEFAULT_SAMPLE_SIZE = 5
DEFAULT_FRESH_PATHS = 100
DEFAULT_SEED = 20260810


def _select_random_bouts(summary: pd.DataFrame, *, n: int, seed: int) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("--sample-size must be positive")
    if n > len(summary):
        raise ValueError(f"Requested {n} bouts, but only {len(summary)} are available")
    return summary.sample(n=n, random_state=seed, replace=False).reset_index(drop=True)


def _print_case(
    case_no: int,
    selected: pd.Series,
    cohort: pd.DataFrame,
    pairs: dict[str, tuple[pd.Series, pd.Series]],
    *,
    fresh_paths: int,
    seed: int,
) -> None:
    bout_id = str(selected["bout_id"])
    if bout_id not in pairs:
        raise KeyError(f"Selected bout {bout_id} missing from current FSR pairs.")

    red, blue = pairs[bout_id]
    bout_rows = cohort[cohort["bout_id"].eq(bout_id)]
    bout = bout_rows.iloc[0] if len(bout_rows) else pd.Series(dtype=object)

    print("\n" + "=" * 120)
    print(f"RANDOM R1-KO CASE {case_no}: {replay._fighter_name(red)} vs {replay._fighter_name(blue)}")
    print("=" * 120)
    print(f"bout_id: {bout_id}")
    if "event_date" in bout.index and pd.notna(bout.get("event_date")):
        print(f"event_date: {pd.Timestamp(bout['event_date']).date()}")
    print("actual outcome: Round-1 KO/TKO")
    print(f"actual winner: {replay._maybe_winner_name(bout, red, blue)}")
    print(
        "existing diagnostic MC: "
        f"P(R1 KO)={float(selected['mc_p_r1_ko']):.2%}; "
        f"P(R1 KD)={float(selected['mc_p_r1_kd']):.2%}; "
        f"mean KD={float(selected['mean_r1_kd']):.3f}; "
        f"mean sig landed={float(selected['mean_r1_sig_landed']):.2f}; "
        f"mean max shock={float(selected['mean_r1_max_shock']):.4f}; "
        f"paths={int(selected['diagnostic_paths'])}"
    )

    print("\nPRE-FIGHT FSR TRAITS")
    replay._print_trait_table(red, blue)

    batch, sims = replay._run_fresh_batch(
        red,
        blue,
        paths=fresh_paths,
        seed=seed,
    )

    print("\nFRESH CURRENT-MC REPLAY")
    print(
        f"paths={len(batch)}; "
        f"P(R1 KO)={batch['r1_ko'].mean():.2%}; "
        f"P(R1 KD)={batch['r1_any_kd'].mean():.2%}; "
        f"mean KD={batch['r1_kd'].mean():.3f}; "
        f"mean sig attempts={batch['r1_sig_att'].mean():.2f}; "
        f"mean sig landed={batch['r1_sig_landed'].mean():.2f}; "
        f"mean max shock={batch['r1_max_shock'].mean():.4f}"
    )

    # For random cases, prefer a finish path when one exists so we can inspect how
    # the current engine expresses an R1 KO. If none exists, print a representative miss.
    prefer_finish = bool(batch['r1_ko'].any())
    representative = replay._representative_sim(
        batch,
        sims,
        prefer_finish=prefer_finish,
    )
    replay._print_path(representative[1], representative[2], representative[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay five reproducibly random actual R1 KO/TKO bouts"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--master", type=Path, default=replay.severity.modern.MASTER_PATH)
    parser.add_argument("--fsr-path", type=Path, default=replay.severity.modern.FSR_PATH)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--fresh-paths", type=int, default=DEFAULT_FRESH_PATHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.fresh_paths <= 0:
        raise ValueError("--fresh-paths must be positive")

    diagnostic = replay._load_path_diagnostic(args.input)
    summary = replay._rank_actual_r1_ko_bouts(diagnostic)
    selected = _select_random_bouts(summary, n=args.sample_size, seed=args.seed)
    cohort, pairs = replay._resolve_current_cohort_and_pairs(args.fsr_path, args.master)

    print("=" * 120)
    print("ACTUAL ROUND-1 KO/TKO — FIVE RANDOM CURRENT-MC REPLAYS")
    print("=" * 120)
    print(
        f"actual R1 KO bouts available: {len(summary):,}; "
        f"sample_size={len(selected)}; seed={args.seed}; "
        f"fresh_paths_per_bout={args.fresh_paths}"
    )
    print("selected bout_ids: " + ", ".join(selected['bout_id'].astype(str)))

    for i, (_, row) in enumerate(selected.iterrows(), start=1):
        _print_case(
            i,
            row,
            cohort,
            pairs,
            fresh_paths=args.fresh_paths,
            seed=args.seed + i,
        )

    print("\nNo simulator constants or FSR values were changed.")


if __name__ == "__main__":
    main()
