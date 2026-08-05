"""Run the evaluation-only low-experience historical replay audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.common.paths import MODEL_LAB_DIR
from pipeline.simulation.historical_experience_threshold_diagnostics import (
    audit_experience_threshold,
)


REPLAY_DIR = (
    MODEL_LAB_DIR
    / "simulation"
    / "historical_replay_v0"
    / "hierarchical_finish"
)
DEFAULT_PREDICTIONS = REPLAY_DIR / "survival_finish_hazard_provider_predictions.parquet"
OUTPUT_DIR = REPLAY_DIR / "experience_threshold"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flag fights where either fighter has limited prior history"
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--minimum-prior-fights", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.predictions.exists():
        raise FileNotFoundError(
            f"Historical replay predictions not found: {args.predictions}"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_parquet(args.predictions)
    result = audit_experience_threshold(
        predictions,
        minimum_prior_fights=args.minimum_prior_fights,
    )
    result.flagged_predictions.to_parquet(
        OUTPUT_DIR / "flagged_predictions.parquet", index=False
    )
    result.flagged_predictions.to_csv(
        OUTPUT_DIR / "flagged_predictions.csv", index=False
    )
    result.metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("=" * 80)
    print("HISTORICAL EXPERIENCE THRESHOLD AUDIT")
    print("=" * 80)
    print(f"Flag rule: {result.summary['flag_rule']}")
    print(result.metrics.to_string(index=False))
    print(f"\nSummary: {OUTPUT_DIR / 'summary.json'}")
    print("Evaluation only. No probabilities or simulator mechanics changed.")


if __name__ == "__main__":
    main()
