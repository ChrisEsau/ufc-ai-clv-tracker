"""Run pre-fight competing-risk finish hazard benchmarks in shadow mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.common.paths import MODEL_LAB_DIR
from pipeline.simulation.artifacts import SIMULATION_TRAINING_DATASET_PATH
from pipeline.simulation.finish_hazard_model import (
    DEFAULT_TEST_YEARS,
    FINISH_CLASSES,
    walk_forward_finish_hazard_benchmark,
)


OUTPUT_DIR = MODEL_LAB_DIR / "simulation" / "models" / "finish_hazard_prefight_v0"
CANDIDATE_MODELS = ("xgb_prefight_context", "xgb_prefight_context_rfs")
BASELINE_MODEL = "round_frequency_baseline"
CALIBRATION = "sequential_class_calibrated"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train leakage-safe pre-fight competing-risk finish hazards"
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


def _row(
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
            f"Metric row not found: {model_name}/{CALIBRATION}/{test_year}"
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
    result = walk_forward_finish_hazard_benchmark(
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
    candidate_rows = calibrated.loc[
        calibrated["model_name"].isin(CANDIDATE_MODELS)
    ]
    best = candidate_rows.iloc[0]
    baseline = calibrated.loc[
        calibrated["model_name"].eq(BASELINE_MODEL)
    ].iloc[0]
    latest_year = max(int(year) for year in args.test_years)
    latest_best = _row(result.fold_metrics, str(best["model_name"]), latest_year)
    latest_baseline = _row(result.fold_metrics, BASELINE_MODEL, latest_year)

    aggregate_improvement = (
        float(baseline["log_loss"]) - float(best["log_loss"])
    ) / float(baseline["log_loss"])
    latest_improvement = (
        float(latest_baseline["log_loss"]) - float(latest_best["log_loss"])
    ) / float(latest_baseline["log_loss"])

    class_rates = []
    for name in FINISH_CLASSES:
        class_rates.append(
            {
                "class": name,
                "actual_rate": float(best[f"actual_rate_{name}"]),
                "predicted_rate": float(best[f"predicted_rate_{name}"]),
                "error": float(
                    best[f"predicted_rate_{name}"] - best[f"actual_rate_{name}"]
                ),
            }
        )

    gate_status = "pass" if aggregate_improvement > 0 and latest_improvement > 0 else "blocked"
    blocking_reasons = []
    if aggregate_improvement <= 0:
        blocking_reasons.append("no_aggregate_log_loss_improvement")
    if latest_improvement <= 0:
        blocking_reasons.append("no_latest_year_log_loss_improvement")

    feature_counts = (
        result.feature_manifest.groupby("model_name")["feature"]
        .nunique()
        .astype(int)
        .to_dict()
    )
    summary = {
        "status": "shadow_only",
        "test_years": [int(year) for year in args.test_years],
        "best_candidate": str(best["model_name"]),
        "best_candidate_log_loss": float(best["log_loss"]),
        "baseline_log_loss": float(baseline["log_loss"]),
        "aggregate_log_loss_improvement": aggregate_improvement,
        "latest_year": latest_year,
        "latest_candidate_log_loss": float(latest_best["log_loss"]),
        "latest_baseline_log_loss": float(latest_baseline["log_loss"]),
        "latest_log_loss_improvement": latest_improvement,
        "gate_status": gate_status,
        "blocking_reasons": blocking_reasons,
        "class_rates": class_rates,
        "feature_counts": feature_counts,
        "aggregate_metrics": calibrated.to_dict(orient="records"),
        "artifacts": {
            "raw_predictions": str(
                OUTPUT_DIR / "raw_walk_forward_predictions.parquet"
            ),
            "calibrated_predictions": str(
                OUTPUT_DIR / "calibrated_walk_forward_predictions.parquet"
            ),
            "fold_metrics": str(OUTPUT_DIR / "fold_metrics.csv"),
            "aggregate_metrics": str(OUTPUT_DIR / "aggregate_metrics.csv"),
            "calibration_schedule": str(OUTPUT_DIR / "calibration_schedule.csv"),
            "feature_importance": str(OUTPUT_DIR / "feature_importance.csv"),
            "feature_manifest": str(OUTPUT_DIR / "feature_manifest.csv"),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("=" * 80)
    print("PRE-FIGHT COMPETING-RISK FINISH HAZARD BENCHMARK")
    print("=" * 80)
    print(calibrated.to_string(index=False))
    print("\nBest candidate class rates:")
    print(pd.DataFrame(class_rates).to_string(index=False))
    print(f"\nGate: {gate_status}")
    if blocking_reasons:
        print("Blocking reasons: " + ", ".join(blocking_reasons))
    print(f"Summary: {OUTPUT_DIR / 'summary.json'}")
    print("Shadow-only component. No simulator or production artifact was promoted.")


if __name__ == "__main__":
    main()
