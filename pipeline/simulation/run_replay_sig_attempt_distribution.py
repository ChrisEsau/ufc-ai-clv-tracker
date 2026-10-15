"""Replay calibrated significant-strike distributions on historical holdouts.

Run after the strike benchmark and calibration artifacts exist:

    python -m pipeline.simulation.run_replay_sig_attempt_distribution

This command scores out-of-fold distributions only. It does not run the full fight
simulator and does not alter production prediction or betting artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.simulation.artifacts import (
    SIG_ATTEMPT_CALIBRATED_PREDICTIONS_PATH,
    SIG_ATTEMPT_REPLAY_AGGREGATE_PATH,
    SIG_ATTEMPT_REPLAY_BY_ROUND_PATH,
    SIG_ATTEMPT_REPLAY_BY_YEAR_PATH,
    SIG_ATTEMPT_REPLAY_DECILES_PATH,
    SIG_ATTEMPT_REPLAY_INTERVAL_PATH,
    SIG_ATTEMPT_REPLAY_PAIR_DIAGNOSTICS_PATH,
    SIG_ATTEMPT_REPLAY_ROWS_PATH,
    SIG_ATTEMPT_REPLAY_SUMMARY_PATH,
    ensure_simulation_dirs,
)
from pipeline.simulation.round_parameter_provider import (
    HistoricalSignificantStrikeProvider,
    RoundParameterKey,
)
from pipeline.simulation.sig_attempt_replay import (
    replay_calibrated_strike_distribution,
)


class SigAttemptReplayRunnerError(RuntimeError):
    """Raised when replay input or output cannot be processed."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay calibrated significant-strike distributions"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SIG_ATTEMPT_CALIBRATED_PREDICTIONS_PATH,
    )
    parser.add_argument("--model", default="xgb_context_rfs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_simulation_dirs()

    if not args.input.exists():
        raise SigAttemptReplayRunnerError(
            f"Calibrated strike predictions not found: {args.input}. "
            "Run python -m pipeline.simulation.run_calibrate_sig_attempt_model first."
        )

    predictions = pd.read_parquet(args.input)
    provider = HistoricalSignificantStrikeProvider(
        predictions,
        model_name=args.model,
    )
    result = replay_calibrated_strike_distribution(
        predictions,
        model_name=args.model,
    )

    # Exercise the public provider boundary against a real replay row. The replay
    # metrics are vectorized, but future simulator integration will consume this
    # typed interface one fighter-round at a time.
    first = result.replay_rows.iloc[0]
    first_parameters = provider.significant_strike_attempts(
        RoundParameterKey(
            fight_id=str(first["fight_id"]),
            fighter_id=str(first["fighter_id"]),
            round=int(first["round"]),
        )
    )
    expected_at_observed_exposure = first_parameters.expected_count(
        float(first["round_exposure_seconds"])
    )
    if abs(
        expected_at_observed_exposure
        - float(first["calibrated_count_at_actual_exposure"])
    ) > 1e-8:
        raise SigAttemptReplayRunnerError(
            "Provider expected count does not match calibrated replay mean"
        )

    result.aggregate_metrics.to_csv(SIG_ATTEMPT_REPLAY_AGGREGATE_PATH, index=False)
    result.interval_coverage.to_csv(SIG_ATTEMPT_REPLAY_INTERVAL_PATH, index=False)
    result.metrics_by_year.to_csv(SIG_ATTEMPT_REPLAY_BY_YEAR_PATH, index=False)
    result.metrics_by_round.to_csv(SIG_ATTEMPT_REPLAY_BY_ROUND_PATH, index=False)
    result.calibration_deciles.to_csv(SIG_ATTEMPT_REPLAY_DECILES_PATH, index=False)
    result.pair_diagnostics.to_csv(
        SIG_ATTEMPT_REPLAY_PAIR_DIAGNOSTICS_PATH,
        index=False,
    )
    result.replay_rows.to_parquet(SIG_ATTEMPT_REPLAY_ROWS_PATH, index=False)

    best = result.aggregate_metrics.iloc[0].to_dict()
    gamma = result.aggregate_metrics.loc[
        result.aggregate_metrics["distribution"].eq(
            "calibrated_gamma_poisson"
        )
    ].iloc[0]
    poisson = result.aggregate_metrics.loc[
        result.aggregate_metrics["distribution"].eq("calibrated_poisson")
    ].iloc[0]
    pair_values = {
        row["diagnostic"]: row["value"]
        for row in result.pair_diagnostics.to_dict(orient="records")
    }
    summary = {
        "status": "shadow_only",
        "model_name": args.model,
        "provider_rows": len(provider),
        "replay_rows": int(len(result.replay_rows)),
        "replay_fights": int(result.replay_rows["fight_id"].nunique()),
        "best_distribution_by_nll": best["distribution"],
        "best_mean_negative_log_likelihood": float(
            best["mean_negative_log_likelihood"]
        ),
        "gamma_poisson_nll_improvement_vs_calibrated_poisson": float(
            (
                float(poisson["mean_negative_log_likelihood"])
                - float(gamma["mean_negative_log_likelihood"])
            )
            / float(poisson["mean_negative_log_likelihood"])
        ),
        "paired_actual_count_correlation": pair_values.get(
            "paired_actual_count_correlation"
        ),
        "paired_predicted_mean_correlation": pair_values.get(
            "paired_predicted_mean_correlation"
        ),
        "paired_residual_correlation": pair_values.get(
            "paired_residual_correlation"
        ),
        "aggregate_metrics": result.aggregate_metrics.to_dict(orient="records"),
        "interval_coverage": result.interval_coverage.to_dict(orient="records"),
        "pair_diagnostics": result.pair_diagnostics.to_dict(orient="records"),
        "artifacts": {
            "aggregate_metrics": str(SIG_ATTEMPT_REPLAY_AGGREGATE_PATH),
            "interval_coverage": str(SIG_ATTEMPT_REPLAY_INTERVAL_PATH),
            "metrics_by_year": str(SIG_ATTEMPT_REPLAY_BY_YEAR_PATH),
            "metrics_by_round": str(SIG_ATTEMPT_REPLAY_BY_ROUND_PATH),
            "calibration_deciles": str(SIG_ATTEMPT_REPLAY_DECILES_PATH),
            "pair_diagnostics": str(SIG_ATTEMPT_REPLAY_PAIR_DIAGNOSTICS_PATH),
            "replay_rows": str(SIG_ATTEMPT_REPLAY_ROWS_PATH),
        },
    }
    SIG_ATTEMPT_REPLAY_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("=" * 80)
    print("SIGNIFICANT-STRIKE DISTRIBUTION REPLAY")
    print("=" * 80)
    print(result.aggregate_metrics.to_string(index=False))
    print("\nInterval coverage:")
    print(result.interval_coverage.to_string(index=False))
    print("\nPaired diagnostics:")
    print(result.pair_diagnostics.to_string(index=False))
    print(f"Summary: {SIG_ATTEMPT_REPLAY_SUMMARY_PATH}")
    print("Shadow-only replay. No live simulator component was promoted.")


if __name__ == "__main__":
    main()
