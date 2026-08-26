"""Run hierarchical pre-fight finish-hazard benchmarks in shadow mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.common.paths import MODEL_LAB_DIR
from pipeline.simulation.artifacts import SIMULATION_TRAINING_DATASET_PATH
from pipeline.simulation.finish_hazard_model import DEFAULT_TEST_YEARS, FINISH_CLASSES
from pipeline.simulation.hierarchical_finish_hazard_model import (
    HIERARCHICAL_MODELS,
    walk_forward_hierarchical_finish_benchmark,
)


OUTPUT_DIR = (
    MODEL_LAB_DIR / "simulation" / "models" / "finish_hazard_hierarchical_v0"
)
CALIBRATION = "sequential_hierarchical_calibrated"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train leakage-safe hierarchical pre-fight finish hazards"
    )
    parser.add_argument("--training", type=Path, default=SIMULATION_TRAINING_DATASET_PATH)
    parser.add_argument(
        "--test-years",
        type=int,
        nargs="+",
        default=list(DEFAULT_TEST_YEARS),
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _metric_row(
    frame: pd.DataFrame,
    model_name: str,
    test_year: int | str,
) -> pd.Series:
    match = frame.loc[
        frame["model_name"].eq(model_name)
        & frame["calibration"].eq(CALIBRATION)
        & frame["test_year"].astype(str).eq(str(test_year))
    ]
    if len(match) != 1:
        raise RuntimeError(
            f"Hierarchical metric row not found: {model_name}/{test_year}"
        )
    return match.iloc[0]


def main() -> None:
    args = build_parser().parse_args()
    if not args.training.exists():
        raise FileNotFoundError(
            f"Training table not found: {args.training}. Run the simulator builder first."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training = pd.read_parquet(args.training)
    result = walk_forward_hierarchical_finish_benchmark(
        training,
        test_years=args.test_years,
        seed=args.seed,
    )

    result.raw_predictions.to_parquet(
        OUTPUT_DIR / "raw_walk_forward_predictions.parquet", index=False
    )
    result.calibrated_predictions.to_parquet(
        OUTPUT_DIR / "calibrated_walk_forward_predictions.parquet", index=False
    )
    result.fold_metrics.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    result.aggregate_metrics.to_csv(OUTPUT_DIR / "aggregate_metrics.csv", index=False)
    result.stage_metrics.to_csv(OUTPUT_DIR / "stage_metrics.csv", index=False)
    result.calibration_schedule.to_csv(
        OUTPUT_DIR / "calibration_schedule.csv", index=False
    )
    result.feature_importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv", index=False
    )
    result.feature_manifest.to_csv(
        OUTPUT_DIR / "feature_manifest.csv", index=False
    )

    calibrated = result.aggregate_metrics.loc[
        result.aggregate_metrics["calibration"].eq(CALIBRATION)
    ].sort_values("log_loss")
    best = calibrated.iloc[0]
    latest_year = max(int(year) for year in args.test_years)
    latest = _metric_row(result.fold_metrics, str(best["model_name"]), latest_year)

    class_rates = [
        {
            "class": name,
            "actual_rate": float(best[f"actual_rate_{name}"]),
            "predicted_rate": float(best[f"predicted_rate_{name}"]),
            "error": float(
                best[f"predicted_rate_{name}"] - best[f"actual_rate_{name}"]
            ),
        }
        for name in FINISH_CLASSES
    ]
    feature_counts = (
        result.feature_manifest.groupby(["model_name", "stage"])["feature"]
        .nunique()
        .astype(int)
        .reset_index(name="features")
    )
    selected_stage_metrics = result.stage_metrics.loc[
        result.stage_metrics["model_name"].eq(str(best["model_name"]))
        & result.stage_metrics["calibration"].eq(CALIBRATION)
    ]

    summary = {
        "status": "shadow_only",
        "architecture": "hierarchical_finish_method_side_survival",
        "test_years": [int(year) for year in args.test_years],
        "candidate_models": list(HIERARCHICAL_MODELS),
        "best_candidate": str(best["model_name"]),
        "best_candidate_log_loss": float(best["log_loss"]),
        "best_candidate_multiclass_brier": float(best["multiclass_brier"]),
        "latest_year": latest_year,
        "latest_candidate_log_loss": float(latest["log_loss"]),
        "latest_candidate_multiclass_brier": float(latest["multiclass_brier"]),
        "class_rates": class_rates,
        "stage_metrics": selected_stage_metrics.to_dict(orient="records"),
        "feature_counts": feature_counts.to_dict(orient="records"),
        "aggregate_metrics": calibrated.to_dict(orient="records"),
        "promotion_status": "not_evaluated_until_full_simulator_replay",
        "artifacts": {
            "raw_predictions": str(
                OUTPUT_DIR / "raw_walk_forward_predictions.parquet"
            ),
            "calibrated_predictions": str(
                OUTPUT_DIR / "calibrated_walk_forward_predictions.parquet"
            ),
            "fold_metrics": str(OUTPUT_DIR / "fold_metrics.csv"),
            "aggregate_metrics": str(OUTPUT_DIR / "aggregate_metrics.csv"),
            "stage_metrics": str(OUTPUT_DIR / "stage_metrics.csv"),
            "calibration_schedule": str(OUTPUT_DIR / "calibration_schedule.csv"),
            "feature_importance": str(OUTPUT_DIR / "feature_importance.csv"),
            "feature_manifest": str(OUTPUT_DIR / "feature_manifest.csv"),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("=" * 80)
    print("HIERARCHICAL PRE-FIGHT FINISH HAZARD BENCHMARK")
    print("=" * 80)
    print(calibrated.to_string(index=False))
    print("\nBest candidate stage metrics:")
    print(selected_stage_metrics.to_string(index=False))
    print("\nBest candidate class rates:")
    print(pd.DataFrame(class_rates).to_string(index=False))
    print(f"\nSummary: {OUTPUT_DIR / 'summary.json'}")
    print("Shadow-only component. No simulator or production artifact was promoted.")


if __name__ == "__main__":
    main()
