"""Compare blocked simulator candidates on the experienced-only fight cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.common.paths import MODEL_LAB_DIR
from pipeline.simulation.historical_experienced_model_comparison import (
    REFERENCE_VARIANT,
    compare_experienced_model_candidates,
)


SURVIVAL_DIR = (
    MODEL_LAB_DIR / "simulation" / "historical_replay_v0" / "survival_components"
)
HIERARCHICAL_DIR = (
    MODEL_LAB_DIR / "simulation" / "historical_replay_v0" / "hierarchical_finish"
)
OUTPUT_DIR = (
    MODEL_LAB_DIR
    / "simulation"
    / "historical_replay_v0"
    / "experienced_model_comparison"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate simulator candidates only when both fighters have sufficient history"
    )
    parser.add_argument("--minimum-prior-fights", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=211)
    return parser


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required prediction artifact not found: {path}")
    return pd.read_parquet(path)


def main() -> None:
    args = build_parser().parse_args()
    predictions = {
        REFERENCE_VARIANT: _read(
            SURVIVAL_DIR / "survival_finish_hazard_provider_predictions.parquet"
        ),
        "class_finish_hazard_provider": _read(
            SURVIVAL_DIR / "class_finish_hazard_provider_predictions.parquet"
        ),
        "strike_and_survival_finish_providers": _read(
            SURVIVAL_DIR / "strike_and_survival_finish_providers_predictions.parquet"
        ),
        "dynamic_strike_and_survival_finish_providers": _read(
            SURVIVAL_DIR
            / "dynamic_strike_and_survival_finish_providers_predictions.parquet"
        ),
        "hierarchical_class_finish_hazard_provider": _read(
            HIERARCHICAL_DIR
            / "hierarchical_class_finish_hazard_provider_predictions.parquet"
        ),
        "hierarchical_survival_finish_hazard_provider": _read(
            HIERARCHICAL_DIR
            / "hierarchical_survival_finish_hazard_provider_predictions.parquet"
        ),
        "heuristic_simulator": _read(
            SURVIVAL_DIR / "heuristic_simulator_predictions.parquet"
        ),
        "historical_baseline": _read(
            SURVIVAL_DIR / "historical_baseline_predictions.parquet"
        ),
    }
    result = compare_experienced_model_candidates(
        predictions,
        minimum_prior_fights=args.minimum_prior_fights,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        candidate_variants=[
            "class_finish_hazard_provider",
            "strike_and_survival_finish_providers",
            "dynamic_strike_and_survival_finish_providers",
            "hierarchical_class_finish_hazard_provider",
            "hierarchical_survival_finish_hazard_provider",
            "heuristic_simulator",
            "historical_baseline",
        ],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    result.paired_deltas.to_csv(OUTPUT_DIR / "paired_deltas.csv", index=False)
    result.eligible_fights.to_csv(OUTPUT_DIR / "eligible_fights.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("=" * 80)
    print("EXPERIENCED-ONLY SIMULATOR MODEL COMPARISON")
    print("=" * 80)
    print(
        f"Eligibility: both fighters have at least {args.minimum_prior_fights} prior fights"
    )
    print(f"Eligible fights: {result.summary['eligible_fights']}")
    print(result.metrics.to_string(index=False))
    print(f"\nSummary: {OUTPUT_DIR / 'summary.json'}")
    print("Evaluation-only. No production or simulator probabilities changed.")


if __name__ == "__main__":
    main()
