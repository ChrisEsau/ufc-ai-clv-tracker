"""Compare the heuristic simulator with an absolute strike-provider ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.common.paths import MASTER_PATH, MODEL_LAB_DIR
from pipeline.simulation.artifacts import SIMULATION_TRAINING_DATASET_PATH
from pipeline.simulation.historical_simulator_replay import (
    metric_lookup,
    run_historical_simulator_replay,
)
from pipeline.simulation.historical_strike_provider_replay import (
    run_historical_strike_provider_replay,
)
from pipeline.simulation.run_historical_simulator_replay import (
    _attach_scoring_labels,
)


OUTPUT_DIR = MODEL_LAB_DIR / "simulation" / "historical_replay_v0" / "strike_provider"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare heuristic and absolute-provider strike mechanics"
    )
    parser.add_argument("--input", type=Path, default=SIMULATION_TRAINING_DATASET_PATH)
    parser.add_argument("--master", type=Path, default=MASTER_PATH)
    parser.add_argument("--test-year", type=int, default=2026)
    parser.add_argument("--simulations-per-fight", type=int, default=500)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--max-fights", type=int, default=None)
    return parser


def _metric_table(heuristic: pd.DataFrame, provider: pd.DataFrame) -> pd.DataFrame:
    heuristic_rows = heuristic.copy()
    heuristic_rows["model"] = heuristic_rows["model"].replace(
        {"simulator": "heuristic_simulator"}
    )
    provider_rows = provider.loc[provider["model"].eq("simulator")].copy()
    provider_rows["model"] = "absolute_strike_provider"
    baseline_rows = provider.loc[
        provider["model"].eq("historical_baseline")
    ].copy()
    return pd.concat(
        [heuristic_rows.loc[heuristic_rows["model"].ne("historical_baseline")], provider_rows, baseline_rows],
        ignore_index=True,
    ).sort_values(["task", "metric", "model"]).reset_index(drop=True)


def _aggregate_table(
    heuristic: pd.DataFrame,
    provider: pd.DataFrame,
) -> pd.DataFrame:
    left = heuristic[["quantity", "actual", "simulator"]].rename(
        columns={"simulator": "heuristic_simulator"}
    )
    right = provider[["quantity", "simulator"]].rename(
        columns={"simulator": "absolute_strike_provider"}
    )
    out = left.merge(right, on="quantity", how="inner", validate="one_to_one")
    out["provider_error"] = out["absolute_strike_provider"] - out["actual"]
    out["heuristic_error"] = out["heuristic_simulator"] - out["actual"]
    out["absolute_error_improvement"] = (
        out["heuristic_error"].abs() - out["provider_error"].abs()
    )
    return out


def _lookup(metrics: pd.DataFrame, task: str, model: str, metric: str) -> float:
    match = metrics.loc[
        metrics["task"].eq(task)
        & metrics["model"].eq(model)
        & metrics["metric"].eq(metric),
        "value",
    ]
    if len(match) != 1:
        raise RuntimeError(f"Metric lookup failed: {task}/{model}/{metric}")
    return float(match.iloc[0])


def main() -> None:
    args = build_parser().parse_args()
    for path, label in (
        (args.input, "Simulator training table"),
        (args.master, "Master fight table"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training = pd.read_parquet(args.input)
    master = pd.read_parquet(args.master)
    labeled_training = _attach_scoring_labels(training, master)

    heuristic = run_historical_simulator_replay(
        labeled_training,
        test_year=args.test_year,
        simulations_per_fight=args.simulations_per_fight,
        seed=args.seed,
        max_fights=args.max_fights,
    )
    provider = run_historical_strike_provider_replay(
        labeled_training,
        test_year=args.test_year,
        simulations_per_fight=args.simulations_per_fight,
        seed=args.seed,
        max_fights=args.max_fights,
    )

    metrics = _metric_table(heuristic.metrics, provider.metrics)
    aggregate = _aggregate_table(
        heuristic.aggregate_comparison,
        provider.aggregate_comparison,
    )

    heuristic.fight_predictions.to_parquet(
        OUTPUT_DIR / "heuristic_fight_predictions.parquet", index=False
    )
    provider.fight_predictions.to_parquet(
        OUTPUT_DIR / "provider_fight_predictions.parquet", index=False
    )
    metrics.to_csv(OUTPUT_DIR / "comparison_metrics.csv", index=False)
    aggregate.to_csv(OUTPUT_DIR / "aggregate_comparison.csv", index=False)
    provider.calibration.to_csv(OUTPUT_DIR / "provider_calibration.csv", index=False)

    provider_strike_mae = _lookup(
        metrics,
        "fighter_sig_attempted",
        "absolute_strike_provider",
        "mae",
    )
    heuristic_strike_mae = _lookup(
        metrics,
        "fighter_sig_attempted",
        "heuristic_simulator",
        "mae",
    )
    baseline_strike_mae = _lookup(
        metrics,
        "fighter_sig_attempted",
        "historical_baseline",
        "mae",
    )
    provider_method_loss = _lookup(
        metrics,
        "method",
        "absolute_strike_provider",
        "log_loss",
    )
    heuristic_method_loss = _lookup(
        metrics,
        "method",
        "heuristic_simulator",
        "log_loss",
    )
    provider_winner_brier = _lookup(
        metrics,
        "winner",
        "absolute_strike_provider",
        "brier",
    )
    heuristic_winner_brier = _lookup(
        metrics,
        "winner",
        "heuristic_simulator",
        "brier",
    )

    summary = {
        "status": "shadow_only",
        "test_year": int(args.test_year),
        "fights": int(len(provider.fight_predictions)),
        "simulations_per_fight": int(args.simulations_per_fight),
        "strike_calibration": {
            "rows": provider.strike_calibration.rows,
            "fights": provider.strike_calibration.fights,
            "raw_predicted_mean": provider.strike_calibration.raw_predicted_mean,
            "actual_mean": provider.strike_calibration.actual_mean,
            "mean_calibration_factor": (
                provider.strike_calibration.mean_calibration_factor
            ),
            "gamma_poisson_overdispersion": (
                provider.strike_calibration.gamma_poisson_overdispersion
            ),
        },
        "fighter_sig_attempt_mae": {
            "absolute_strike_provider": provider_strike_mae,
            "heuristic_simulator": heuristic_strike_mae,
            "historical_baseline": baseline_strike_mae,
            "provider_improvement_vs_heuristic": (
                heuristic_strike_mae - provider_strike_mae
            )
            / heuristic_strike_mae,
        },
        "method_log_loss": {
            "absolute_strike_provider": provider_method_loss,
            "heuristic_simulator": heuristic_method_loss,
            "provider_improvement_vs_heuristic": (
                heuristic_method_loss - provider_method_loss
            )
            / heuristic_method_loss,
        },
        "winner_brier": {
            "absolute_strike_provider": provider_winner_brier,
            "heuristic_simulator": heuristic_winner_brier,
            "provider_improvement_vs_heuristic": (
                heuristic_winner_brier - provider_winner_brier
            )
            / heuristic_winner_brier,
        },
        "aggregate_comparison": aggregate.to_dict(orient="records"),
        "artifacts": {
            "comparison_metrics": str(OUTPUT_DIR / "comparison_metrics.csv"),
            "aggregate_comparison": str(OUTPUT_DIR / "aggregate_comparison.csv"),
            "heuristic_predictions": str(
                OUTPUT_DIR / "heuristic_fight_predictions.parquet"
            ),
            "provider_predictions": str(
                OUTPUT_DIR / "provider_fight_predictions.parquet"
            ),
        },
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("=" * 80)
    print("HISTORICAL ABSOLUTE STRIKE-PROVIDER ABLATION")
    print("=" * 80)
    print(metrics.to_string(index=False))
    print("\nAggregate comparison:")
    print(aggregate.to_string(index=False))
    print(f"\nSummary: {summary_path}")
    print("Shadow-only ablation. No production artifact was changed.")


if __name__ == "__main__":
    main()
