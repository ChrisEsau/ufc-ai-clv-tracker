"""Replay shared fight-round dependence for calibrated strike marginals.

Run after calibrated walk-forward strike predictions exist:

    python -m pipeline.simulation.run_replay_joint_sig_attempt_distribution

The command compares independent negative-binomial draws with a sequential
Gaussian copula. It writes only model-lab research artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.simulation.artifacts import (
    SIG_ATTEMPT_CALIBRATED_PREDICTIONS_PATH,
    SIG_ATTEMPT_JOINT_CORRELATION_PATH,
    SIG_ATTEMPT_JOINT_FINAL_DEPENDENCE_PATH,
    SIG_ATTEMPT_JOINT_INTERVAL_PATH,
    SIG_ATTEMPT_JOINT_SCHEDULE_PATH,
    SIG_ATTEMPT_JOINT_SUMMARY_PATH,
    ensure_simulation_dirs,
)
from pipeline.simulation.sig_attempt_joint_replay import (
    sequential_joint_strike_replay,
)


class JointStrikeReplayRunnerError(RuntimeError):
    """Raised when joint replay artifacts cannot be processed."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay joint significant-strike attempt distributions"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SIG_ATTEMPT_CALIBRATED_PREDICTIONS_PATH,
    )
    parser.add_argument("--model", default="xgb_context_rfs")
    parser.add_argument("--minimum-prior-pairs", type=int, default=500)
    parser.add_argument("--simulations", type=int, default=750)
    parser.add_argument("--seed", type=int, default=71)
    return parser


def _weighted_average(frame: pd.DataFrame, value_column: str) -> float:
    weights = frame["pairs"].to_numpy(dtype=float)
    values = frame[value_column].to_numpy(dtype=float)
    return float((values * weights).sum() / weights.sum())


def main() -> None:
    args = build_parser().parse_args()
    ensure_simulation_dirs()

    if not args.input.exists():
        raise JointStrikeReplayRunnerError(
            f"Calibrated strike predictions not found: {args.input}. "
            "Run python -m pipeline.simulation.run_calibrate_sig_attempt_model first."
        )

    predictions = pd.read_parquet(args.input)
    result = sequential_joint_strike_replay(
        calibrated_predictions=predictions,
        model_name=args.model,
        minimum_prior_pairs=args.minimum_prior_pairs,
        simulations=args.simulations,
        seed=args.seed,
    )

    result.dependence_schedule.to_csv(SIG_ATTEMPT_JOINT_SCHEDULE_PATH, index=False)
    result.correlation_metrics.to_csv(
        SIG_ATTEMPT_JOINT_CORRELATION_PATH,
        index=False,
    )
    result.total_interval_coverage.to_csv(
        SIG_ATTEMPT_JOINT_INTERVAL_PATH,
        index=False,
    )
    result.final_dependence.to_csv(
        SIG_ATTEMPT_JOINT_FINAL_DEPENDENCE_PATH,
        index=False,
    )

    independent = result.correlation_metrics.loc[
        result.correlation_metrics["joint_model"].eq("independent")
    ]
    copula = result.correlation_metrics.loc[
        result.correlation_metrics["joint_model"].eq("gaussian_copula")
    ]
    independent_error = _weighted_average(
        independent,
        "absolute_correlation_error",
    )
    copula_error = _weighted_average(copula, "absolute_correlation_error")

    interval_summary: list[dict[str, object]] = []
    for (joint_model, nominal), group in result.total_interval_coverage.groupby(
        ["joint_model", "nominal_coverage"]
    ):
        interval_summary.append(
            {
                "joint_model": joint_model,
                "nominal_coverage": float(nominal),
                "weighted_empirical_coverage": _weighted_average(
                    group,
                    "empirical_coverage",
                ),
                "weighted_coverage_error": _weighted_average(
                    group,
                    "coverage_error",
                ),
                "weighted_mean_interval_width": _weighted_average(
                    group,
                    "mean_interval_width",
                ),
            }
        )

    final_row = result.final_dependence.iloc[0].to_dict()
    summary = {
        "status": "shadow_only",
        "model_name": args.model,
        "simulations_per_year": int(args.simulations),
        "minimum_prior_pairs": int(args.minimum_prior_pairs),
        "final_gaussian_copula_rho": float(
            final_row["gaussian_copula_rho"]
        ),
        "pairs": int(final_row["pairs"]),
        "fights": int(final_row["fights"]),
        "independent_weighted_absolute_correlation_error": independent_error,
        "copula_weighted_absolute_correlation_error": copula_error,
        "correlation_error_improvement": float(
            (independent_error - copula_error) / independent_error
        ),
        "dependence_schedule": result.dependence_schedule.to_dict(
            orient="records"
        ),
        "correlation_metrics": result.correlation_metrics.to_dict(
            orient="records"
        ),
        "total_interval_summary": interval_summary,
        "artifacts": {
            "dependence_schedule": str(SIG_ATTEMPT_JOINT_SCHEDULE_PATH),
            "correlation_metrics": str(SIG_ATTEMPT_JOINT_CORRELATION_PATH),
            "total_interval_coverage": str(SIG_ATTEMPT_JOINT_INTERVAL_PATH),
            "final_dependence": str(SIG_ATTEMPT_JOINT_FINAL_DEPENDENCE_PATH),
        },
    }
    SIG_ATTEMPT_JOINT_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("=" * 80)
    print("JOINT SIGNIFICANT-STRIKE DISTRIBUTION REPLAY")
    print("=" * 80)
    print("Dependence schedule:")
    print(result.dependence_schedule.to_string(index=False))
    print("\nCorrelation metrics:")
    print(result.correlation_metrics.to_string(index=False))
    print("\nFight-round total interval coverage:")
    print(result.total_interval_coverage.to_string(index=False))
    print(f"\nFinal copula rho: {summary['final_gaussian_copula_rho']:.4f}")
    print(
        "Correlation error improvement: "
        f"{summary['correlation_error_improvement']:.2%}"
    )
    print(f"Summary: {SIG_ATTEMPT_JOINT_SUMMARY_PATH}")
    print("Shadow-only joint replay. No live simulator component was promoted.")


if __name__ == "__main__":
    main()
