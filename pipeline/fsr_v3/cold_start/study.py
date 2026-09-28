"""CLI for the measurement-only FSR V3 cold-start historical study."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .mma_global import load_mma_global_fighter_bouts
from .validation import ColdStartSplit, validate_standing_and_takedown, write_results

DEFAULT_OUT = Path("data/diagnostics/fsr_v3_cold_start")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mma-global-duckdb", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-start", default="2012-01-01")
    parser.add_argument("--calibration-start", default="2022-01-01")
    parser.add_argument("--test-start", default="2024-01-01")
    parser.add_argument("--test-end", default="2025-12-31")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 128)
    print("FSR V3 COLD-START EXTERNAL-EVIDENCE STUDY — MEASUREMENT ONLY")
    print("=" * 128)
    print(f"source: {args.mma_global_duckdb}")
    print(
        f"split: train {args.train_start}..< {args.calibration_start} | "
        f"calibration {args.calibration_start}..< {args.test_start} | "
        f"test {args.test_start}..{args.test_end}"
    )
    print("Loading longitudinal external MMA history and building leakage-safe non-UFC Elo...")
    external = load_mma_global_fighter_bouts(args.mma_global_duckdb)
    print(
        f"external fighter-bout rows={len(external):,} | fights={external['fight_id'].nunique():,} | "
        f"fighters={external['fighter_key'].nunique():,} | dates={external['event_date'].min().date()}..{external['event_date'].max().date()}"
    )

    split = ColdStartSplit(
        train_start=args.train_start,
        calibration_start=args.calibration_start,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    results = validate_standing_and_takedown(external, split=split)
    write_results(results, args.output_dir)

    headline = []
    for name, result in results.items():
        print()
        print("-" * 128)
        print(name.upper())
        print("chosen external equivalent evidence seconds:", result.chosen_extra_seconds)
        print(result.summary.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        if not result.bootstrap.empty:
            print("\nFight-cluster bootstrap:")
            print(result.bootstrap.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        h = result.summary[
            (result.summary["coverage"] == "HAS_EXTERNAL")
            & (
                result.summary["ufc_bucket"].isin(["0", "1"])
                | result.summary["ufc_bucket"].eq("EARLY_0_1")
            )
        ].copy()
        headline.append(h)
    if headline:
        pd.concat(headline, ignore_index=True).to_csv(args.output_dir / "headline.csv", index=False)
    print()
    print(f"Outputs: {args.output_dir}")
    print("DONE — no production FSR settings or Event Clock mechanics changed.")


if __name__ == "__main__":
    main()
