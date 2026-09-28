"""Run compact-RFS significant-strike stability experiments in shadow mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.simulation.artifacts import (
    SIG_ATTEMPT_PREDICTIONS_PATH,
    SIG_ATTEMPT_STABILITY_AGGREGATE_METRICS_PATH,
    SIG_ATTEMPT_STABILITY_CALIBRATED_PREDICTIONS_PATH,
    SIG_ATTEMPT_STABILITY_DIR,
    SIG_ATTEMPT_STABILITY_FEATURE_IMPORTANCE_PATH,
    SIG_ATTEMPT_STABILITY_FEATURE_MANIFEST_PATH,
    SIG_ATTEMPT_STABILITY_FOLD_METRICS_PATH,
    SIG_ATTEMPT_STABILITY_GATES_PATH,
    SIG_ATTEMPT_STABILITY_RAW_PREDICTIONS_PATH,
    SIG_ATTEMPT_STABILITY_SUBGROUP_METRICS_PATH,
    SIG_ATTEMPT_STABILITY_SUMMARY_PATH,
    SIMULATION_TRAINING_DATASET_PATH,
    ensure_simulation_dirs,
)
from pipeline.simulation.sig_attempt_model import DEFAULT_TEST_YEARS
from pipeline.simulation.sig_attempt_stability import (
    COMPACT_MODEL_NAMES,
    walk_forward_sig_attempt_stability,
)


class SigAttemptStabilityRunnerError(RuntimeError):
    """Raised when stability artifacts cannot be read or written."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run compact-RFS strike-pace stability experiments"
    )
    parser.add_argument(
        "--training", type=Path, default=SIMULATION_TRAINING_DATASET_PATH
    )
    parser.add_argument(
        "--reference-predictions", type=Path, default=SIG_ATTEMPT_PREDICTIONS_PATH
    )
    parser.add_argument(
        "--test-years",
        type=int,
        nargs="+",
        default=list(DEFAULT_TEST_YEARS),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--minimum-prior-rows", type=int, default=1_000)
    parser.add_argument("--minimum-subgroup-rows", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_simulation_dirs()
    SIG_ATTEMPT_STABILITY_DIR.mkdir(parents=True, exist_ok=True)

    for path, label in (
        (args.training, "Simulator training table"),
        (args.reference_predictions, "Reference walk-forward predictions"),
    ):
        if not path.exists():
            raise SigAttemptStabilityRunnerError(f"{label} not found: {path}")

    training = pd.read_parquet(args.training)
    reference_predictions = pd.read_parquet(args.reference_predictions)
    result = walk_forward_sig_attempt_stability(
        training_df=training,
        reference_predictions=reference_predictions,
        test_years=args.test_years,
        seed=args.seed,
        minimum_prior_rows=args.minimum_prior_rows,
        minimum_subgroup_rows=args.minimum_subgroup_rows,
    )

    result.raw_predictions.to_parquet(
        SIG_ATTEMPT_STABILITY_RAW_PREDICTIONS_PATH, index=False
    )
    result.calibrated_predictions.to_parquet(
        SIG_ATTEMPT_STABILITY_CALIBRATED_PREDICTIONS_PATH, index=False
    )
    result.fold_metrics.to_csv(SIG_ATTEMPT_STABILITY_FOLD_METRICS_PATH, index=False)
    result.aggregate_metrics.to_csv(
        SIG_ATTEMPT_STABILITY_AGGREGATE_METRICS_PATH, index=False
    )
    result.subgroup_metrics.to_csv(
        SIG_ATTEMPT_STABILITY_SUBGROUP_METRICS_PATH, index=False
    )
    result.feature_importance.to_csv(
        SIG_ATTEMPT_STABILITY_FEATURE_IMPORTANCE_PATH, index=False
    )
    result.feature_manifest.to_csv(
        SIG_ATTEMPT_STABILITY_FEATURE_MANIFEST_PATH, index=False
    )
    result.gates.to_csv(SIG_ATTEMPT_STABILITY_GATES_PATH, index=False)

    calibrated = result.aggregate_metrics.loc[
        result.aggregate_metrics["calibration"].eq("sequential_mean_calibrated")
    ].sort_values("count_poisson_deviance")
    compact = calibrated.loc[calibrated["model_name"].isin(COMPACT_MODEL_NAMES)]
    best_compact = compact.iloc[0].to_dict()
    best_gate = result.gates.loc[
        result.gates["candidate_model"].eq(best_compact["model_name"])
    ].iloc[0].to_dict()
    feature_counts = (
        result.feature_manifest.groupby("model_name")["feature"]
        .nunique()
        .astype(int)
        .to_dict()
    )

    summary = {
        "status": "shadow_only",
        "test_years": [int(year) for year in args.test_years],
        "best_compact_model": best_compact["model_name"],
        "best_compact_poisson_deviance": float(
            best_compact["count_poisson_deviance"]
        ),
        "best_compact_gate_status": best_gate["gate_status"],
        "best_compact_blocking_reasons": best_gate["blocking_reasons"],
        "feature_counts": feature_counts,
        "aggregate_metrics": result.aggregate_metrics.to_dict(orient="records"),
        "gates": result.gates.to_dict(orient="records"),
        "artifacts": {
            "raw_predictions": str(SIG_ATTEMPT_STABILITY_RAW_PREDICTIONS_PATH),
            "calibrated_predictions": str(
                SIG_ATTEMPT_STABILITY_CALIBRATED_PREDICTIONS_PATH
            ),
            "fold_metrics": str(SIG_ATTEMPT_STABILITY_FOLD_METRICS_PATH),
            "aggregate_metrics": str(
                SIG_ATTEMPT_STABILITY_AGGREGATE_METRICS_PATH
            ),
            "subgroup_metrics": str(SIG_ATTEMPT_STABILITY_SUBGROUP_METRICS_PATH),
            "feature_importance": str(
                SIG_ATTEMPT_STABILITY_FEATURE_IMPORTANCE_PATH
            ),
            "feature_manifest": str(SIG_ATTEMPT_STABILITY_FEATURE_MANIFEST_PATH),
            "gates": str(SIG_ATTEMPT_STABILITY_GATES_PATH),
        },
    }
    SIG_ATTEMPT_STABILITY_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("=" * 80)
    print("SIGNIFICANT-STRIKE COMPACT RFS STABILITY")
    print("=" * 80)
    print(calibrated.to_string(index=False))
    print("\nStability gates:")
    print(result.gates.to_string(index=False))
    print(f"Summary: {SIG_ATTEMPT_STABILITY_SUMMARY_PATH}")
    print("Shadow-only experiment. No simulator component was promoted.")


if __name__ == "__main__":
    main()
