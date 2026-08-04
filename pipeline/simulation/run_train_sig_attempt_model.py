"""Train and evaluate the first simulator component benchmark in shadow mode.

Run from the repository root after building the fighter-round table:

    python -m pipeline.simulation.run_train_sig_attempt_model

The command writes only model-lab artifacts. It does not promote a production
model or alter prediction/betting outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.simulation.artifacts import (
    SIG_ATTEMPT_AGGREGATE_METRICS_PATH,
    SIG_ATTEMPT_FEATURE_IMPORTANCE_PATH,
    SIG_ATTEMPT_FOLD_METRICS_PATH,
    SIG_ATTEMPT_MODEL_BUNDLE_PATH,
    SIG_ATTEMPT_MODEL_CARD_PATH,
    SIG_ATTEMPT_PREDICTIONS_PATH,
    SIG_ATTEMPT_SUMMARY_PATH,
    SIMULATION_TRAINING_DATASET_PATH,
    ensure_simulation_dirs,
)
from pipeline.simulation.sig_attempt_model import (
    DEFAULT_TEST_YEARS,
    SIG_ATTEMPT_MODEL_VERSION,
    save_model_bundle,
    walk_forward_sig_attempt_benchmark,
)


class SigAttemptRunnerError(RuntimeError):
    """Raised when the benchmark input or output cannot be processed."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run walk-forward significant-strike pace benchmark"
    )
    parser.add_argument("--input", type=Path, default=SIMULATION_TRAINING_DATASET_PATH)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--test-years",
        type=int,
        nargs="+",
        default=list(DEFAULT_TEST_YEARS),
    )
    return parser


def _write_model_card(summary: dict[str, object], path: Path) -> None:
    aggregate = pd.DataFrame(summary["aggregate_metrics"])
    lines = [
        "# Significant-Strike Pace Model V0",
        "",
        "**Status: shadow-only benchmark; not calibrated or approved for wagering.**",
        "",
        "## Target",
        "",
        "Exposure-adjusted significant-strike attempts per minute. Predictions are ",
        "converted back to expected counts using the observed exposure only for evaluation.",
        "",
        "## Walk-forward design",
        "",
        f"Test years: `{summary['test_years']}`",
        "",
        "Models compared:",
        "",
        "- round-specific historical mean;",
        "- leakage-safe, shrinkage-adjusted fighter prior pace;",
        "- XGBoost using fight context and prior-round context;",
        "- XGBoost adding point-in-time RFS features.",
        "",
        "## Aggregate results",
        "",
        "| Model | Poisson deviance | Count MAE | Count RMSE | Improvement vs fighter history |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        improvement = row.get("poisson_improvement_vs_fighter_history")
        improvement_text = "" if pd.isna(improvement) else f"{float(improvement):.2%}"
        lines.append(
            "| {model_name} | {count_poisson_deviance:.4f} | {count_mae:.3f} | "
            "{count_rmse:.3f} | {improvement} |".format(
                model_name=row["model_name"],
                count_poisson_deviance=float(row["count_poisson_deviance"]),
                count_mae=float(row["count_mae"]),
                count_rmse=float(row["count_rmse"]),
                improvement=improvement_text,
            )
        )
    lines.extend(
        [
            "",
            "## Promotion rule",
            "",
            "This component remains research-only. RFS inclusion is retained only if it ",
            "improves recent walk-forward performance and remains stable by year and round.",
            "Full simulator integration requires separate distribution calibration.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    ensure_simulation_dirs()

    if not args.input.exists():
        raise SigAttemptRunnerError(
            f"Simulator training table not found: {args.input}. "
            "Run python -m pipeline.simulation.run_build_training_dataset first."
        )

    training = pd.read_parquet(args.input)
    result = walk_forward_sig_attempt_benchmark(
        training_df=training,
        test_years=args.test_years,
        seed=args.seed,
    )

    result.fold_metrics.to_csv(SIG_ATTEMPT_FOLD_METRICS_PATH, index=False)
    result.aggregate_metrics.to_csv(SIG_ATTEMPT_AGGREGATE_METRICS_PATH, index=False)
    result.predictions.to_parquet(SIG_ATTEMPT_PREDICTIONS_PATH, index=False)
    result.feature_importance.to_csv(SIG_ATTEMPT_FEATURE_IMPORTANCE_PATH, index=False)
    save_model_bundle(result.final_bundle, str(SIG_ATTEMPT_MODEL_BUNDLE_PATH))

    best_row = result.aggregate_metrics.iloc[0].to_dict()
    rfs_match = result.aggregate_metrics.loc[
        result.aggregate_metrics["model_name"].eq("xgb_context_rfs")
    ]
    context_match = result.aggregate_metrics.loc[
        result.aggregate_metrics["model_name"].eq("xgb_context")
    ]
    rfs_incremental = None
    if not rfs_match.empty and not context_match.empty:
        context_deviance = float(context_match.iloc[0]["count_poisson_deviance"])
        rfs_deviance = float(rfs_match.iloc[0]["count_poisson_deviance"])
        rfs_incremental = (context_deviance - rfs_deviance) / context_deviance

    summary: dict[str, object] = {
        "model_version": SIG_ATTEMPT_MODEL_VERSION,
        "status": "shadow_only",
        "training_rows": int(len(training)),
        "training_fights": int(training["fight_id"].nunique()),
        "test_years": [int(year) for year in args.test_years],
        "best_model": best_row["model_name"],
        "best_count_poisson_deviance": float(best_row["count_poisson_deviance"]),
        "rfs_incremental_poisson_improvement_vs_context": rfs_incremental,
        "aggregate_metrics": result.aggregate_metrics.to_dict(orient="records"),
        "fold_metrics": result.fold_metrics.to_dict(orient="records"),
        "artifacts": {
            "fold_metrics": str(SIG_ATTEMPT_FOLD_METRICS_PATH),
            "aggregate_metrics": str(SIG_ATTEMPT_AGGREGATE_METRICS_PATH),
            "predictions": str(SIG_ATTEMPT_PREDICTIONS_PATH),
            "feature_importance": str(SIG_ATTEMPT_FEATURE_IMPORTANCE_PATH),
            "model_bundle": str(SIG_ATTEMPT_MODEL_BUNDLE_PATH),
        },
    }
    SIG_ATTEMPT_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_model_card(summary, SIG_ATTEMPT_MODEL_CARD_PATH)

    print("=" * 80)
    print("SIGNIFICANT-STRIKE PACE BENCHMARK V0")
    print("=" * 80)
    print(result.aggregate_metrics.to_string(index=False))
    print(f"Best model: {summary['best_model']}")
    if rfs_incremental is not None:
        print(f"RFS incremental Poisson improvement vs context: {rfs_incremental:.3%}")
    print(f"Summary: {SIG_ATTEMPT_SUMMARY_PATH}")
    print("Shadow-only benchmark. No production model was promoted.")


if __name__ == "__main__":
    main()
