"""Run four-way historical simulator component comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.common.paths import MASTER_PATH, MODEL_LAB_DIR
from pipeline.simulation.artifacts import SIMULATION_TRAINING_DATASET_PATH
from pipeline.simulation.historical_component_provider_replay import (
    VARIANTS,
    run_historical_component_provider_replay,
)
from pipeline.simulation.run_historical_simulator_replay import (
    _attach_scoring_labels,
)


FINISH_MODEL_DIR = (
    MODEL_LAB_DIR / "simulation" / "models" / "finish_hazard_prefight_v0"
)
DEFAULT_FINISH_SCHEDULE = FINISH_MODEL_DIR / "calibration_schedule.csv"
OUTPUT_DIR = MODEL_LAB_DIR / "simulation" / "historical_replay_v0" / "components"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare heuristic, strike, finish, and combined simulator paths"
    )
    parser.add_argument("--input", type=Path, default=SIMULATION_TRAINING_DATASET_PATH)
    parser.add_argument("--master", type=Path, default=MASTER_PATH)
    parser.add_argument(
        "--finish-calibration-schedule",
        type=Path,
        default=DEFAULT_FINISH_SCHEDULE,
    )
    parser.add_argument("--finish-model", default="xgb_prefight_context")
    parser.add_argument("--test-year", type=int, default=2026)
    parser.add_argument("--simulations-per-fight", type=int, default=500)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--max-fights", type=int, default=None)
    return parser


def _lookup(
    metrics: pd.DataFrame,
    task: str,
    model: str,
    metric: str,
) -> float:
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
        (args.finish_calibration_schedule, "Finish calibration schedule"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training = pd.read_parquet(args.input)
    master = pd.read_parquet(args.master)
    labeled_training = _attach_scoring_labels(training, master)
    schedule = pd.read_csv(args.finish_calibration_schedule)

    result = run_historical_component_provider_replay(
        labeled_training,
        schedule,
        test_year=args.test_year,
        simulations_per_fight=args.simulations_per_fight,
        seed=args.seed,
        max_fights=args.max_fights,
        finish_model_name=args.finish_model,
    )

    for variant, frame in result.fight_predictions.items():
        frame.to_parquet(OUTPUT_DIR / f"{variant}_predictions.parquet", index=False)
    result.metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    result.aggregate_comparison.to_csv(
        OUTPUT_DIR / "aggregate_comparison.csv", index=False
    )
    result.finish_predictions.predictions.to_parquet(
        OUTPUT_DIR / "counterfactual_finish_predictions.parquet", index=False
    )

    metric_summary: dict[str, dict[str, float]] = {}
    for variant in (*VARIANTS, "historical_baseline"):
        metric_summary[variant] = {
            "winner_brier": _lookup(result.metrics, "winner", variant, "brier"),
            "winner_accuracy": _lookup(
                result.metrics, "winner", variant, "accuracy"
            ),
            "method_log_loss": _lookup(
                result.metrics, "method", variant, "log_loss"
            ),
            "method_accuracy": _lookup(
                result.metrics, "method", variant, "accuracy"
            ),
            "goes_distance_brier": _lookup(
                result.metrics, "goes_distance", variant, "brier"
            ),
            "fight_time_mae": _lookup(
                result.metrics, "fight_time_seconds", variant, "mae"
            ),
            "fighter_sig_attempt_mae": _lookup(
                result.metrics, "fighter_sig_attempted", variant, "mae"
            ),
        }

    best_method = min(VARIANTS, key=lambda name: metric_summary[name]["method_log_loss"])
    best_winner = min(VARIANTS, key=lambda name: metric_summary[name]["winner_brier"])
    best_time = min(VARIANTS, key=lambda name: metric_summary[name]["fight_time_mae"])
    best_strikes = min(
        VARIANTS,
        key=lambda name: metric_summary[name]["fighter_sig_attempt_mae"],
    )

    summary = {
        "status": "shadow_only",
        "test_year": int(args.test_year),
        "fights": int(len(result.fight_predictions[VARIANTS[0]])),
        "simulations_per_fight": int(args.simulations_per_fight),
        "finish_model_name": args.finish_model,
        "finish_model_seed": result.finish_predictions.model_seed,
        "finish_calibration_source": (
            result.finish_predictions.calibration_source
        ),
        "counterfactual_finish_rows": int(
            len(result.finish_predictions.predictions)
        ),
        "strike_calibration": {
            "rows": result.strike_calibration.rows,
            "fights": result.strike_calibration.fights,
            "mean_calibration_factor": (
                result.strike_calibration.mean_calibration_factor
            ),
            "gamma_poisson_overdispersion": (
                result.strike_calibration.gamma_poisson_overdispersion
            ),
        },
        "best_variants": {
            "method_log_loss": best_method,
            "winner_brier": best_winner,
            "fight_time_mae": best_time,
            "fighter_sig_attempt_mae": best_strikes,
        },
        "metrics": metric_summary,
        "aggregate_comparison": result.aggregate_comparison.to_dict(
            orient="records"
        ),
        "artifacts": {
            "metrics": str(OUTPUT_DIR / "metrics.csv"),
            "aggregate_comparison": str(
                OUTPUT_DIR / "aggregate_comparison.csv"
            ),
            "counterfactual_finish_predictions": str(
                OUTPUT_DIR / "counterfactual_finish_predictions.parquet"
            ),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("=" * 80)
    print("HISTORICAL SIMULATOR COMPONENT REPLAY")
    print("=" * 80)
    print(result.metrics.to_string(index=False))
    print("\nAggregate comparison:")
    print(result.aggregate_comparison.to_string(index=False))
    print(f"\nSummary: {OUTPUT_DIR / 'summary.json'}")
    print("Shadow-only comparison. No production artifact was changed.")


if __name__ == "__main__":
    main()
