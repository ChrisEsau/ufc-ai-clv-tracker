"""Calibrate walk-forward significant-strike pace predictions in shadow mode.

Run after the benchmark artifacts exist:

    python -m pipeline.simulation.run_calibrate_sig_attempt_model

The output defines mean-correction and gamma-Poisson dispersion contracts for
research use. It does not promote a production simulator component.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.simulation.artifacts import (
    SIG_ATTEMPT_CALIBRATED_PREDICTIONS_PATH,
    SIG_ATTEMPT_CALIBRATION_METRICS_PATH,
    SIG_ATTEMPT_CALIBRATION_SCHEDULE_PATH,
    SIG_ATTEMPT_CALIBRATION_SUMMARY_PATH,
    SIG_ATTEMPT_FINAL_PARAMETERS_PATH,
    SIG_ATTEMPT_PREDICTIONS_PATH,
    ensure_simulation_dirs,
)
from pipeline.simulation.sig_attempt_calibration import (
    calibrate_walk_forward_predictions,
)


class SigAttemptCalibrationRunnerError(RuntimeError):
    """Raised when calibration artifacts cannot be read or written."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequentially calibrate simulator strike-pace predictions"
    )
    parser.add_argument("--input", type=Path, default=SIG_ATTEMPT_PREDICTIONS_PATH)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgb_context", "xgb_context_rfs"],
    )
    parser.add_argument("--minimum-prior-rows", type=int, default=1_000)
    parser.add_argument("--default-overdispersion", type=float, default=0.35)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_simulation_dirs()

    if not args.input.exists():
        raise SigAttemptCalibrationRunnerError(
            f"Walk-forward predictions not found: {args.input}. "
            "Run python -m pipeline.simulation.run_train_sig_attempt_model first."
        )

    predictions = pd.read_parquet(args.input)
    result = calibrate_walk_forward_predictions(
        predictions=predictions,
        model_names=args.models,
        minimum_prior_rows=args.minimum_prior_rows,
        default_overdispersion=args.default_overdispersion,
    )

    result.schedule.to_csv(SIG_ATTEMPT_CALIBRATION_SCHEDULE_PATH, index=False)
    result.predictions.to_parquet(
        SIG_ATTEMPT_CALIBRATED_PREDICTIONS_PATH,
        index=False,
    )
    result.metrics.to_csv(SIG_ATTEMPT_CALIBRATION_METRICS_PATH, index=False)
    result.final_parameters.to_csv(SIG_ATTEMPT_FINAL_PARAMETERS_PATH, index=False)

    calibrated_metrics = result.metrics.loc[
        result.metrics["calibration"].eq("sequential_mean_calibrated")
    ].sort_values("count_poisson_deviance")
    best = calibrated_metrics.iloc[0].to_dict()
    summary = {
        "status": "shadow_only",
        "models": list(args.models),
        "minimum_prior_rows": int(args.minimum_prior_rows),
        "default_overdispersion": float(args.default_overdispersion),
        "best_calibrated_model": best["model_name"],
        "best_calibrated_count_poisson_deviance": float(
            best["count_poisson_deviance"]
        ),
        "metrics": result.metrics.to_dict(orient="records"),
        "sequential_schedule": result.schedule.to_dict(orient="records"),
        "final_distribution_parameters": result.final_parameters.to_dict(
            orient="records"
        ),
        "artifacts": {
            "schedule": str(SIG_ATTEMPT_CALIBRATION_SCHEDULE_PATH),
            "predictions": str(SIG_ATTEMPT_CALIBRATED_PREDICTIONS_PATH),
            "metrics": str(SIG_ATTEMPT_CALIBRATION_METRICS_PATH),
            "final_parameters": str(SIG_ATTEMPT_FINAL_PARAMETERS_PATH),
        },
    }
    SIG_ATTEMPT_CALIBRATION_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("=" * 80)
    print("SIGNIFICANT-STRIKE PACE CALIBRATION")
    print("=" * 80)
    print(result.metrics.to_string(index=False))
    print("\nFinal distribution parameters:")
    print(result.final_parameters.to_string(index=False))
    print(f"Summary: {SIG_ATTEMPT_CALIBRATION_SUMMARY_PATH}")
    print("Shadow-only calibration. No production model was promoted.")


if __name__ == "__main__":
    main()
