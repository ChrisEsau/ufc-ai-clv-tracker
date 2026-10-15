"""Run historical round-survival calibrated finish-provider comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.common.paths import MASTER_PATH, MODEL_LAB_DIR
from pipeline.simulation.artifacts import SIMULATION_TRAINING_DATASET_PATH
from pipeline.simulation.historical_replay_evaluation import (
    RECOMMENDED_VARIANT,
    evaluate_historical_replay_cohort,
)
from pipeline.simulation.historical_survival_provider_replay import (
    SURVIVAL_VARIANTS,
    run_historical_survival_provider_replay,
)
from pipeline.simulation.run_historical_simulator_replay import (
    _attach_scoring_labels,
)


FINISH_MODEL_DIR = (
    MODEL_LAB_DIR / "simulation" / "models" / "finish_hazard_prefight_v0"
)
DEFAULT_CLASS_SCHEDULE = FINISH_MODEL_DIR / "calibration_schedule.csv"
DEFAULT_OOF_PREDICTIONS = (
    FINISH_MODEL_DIR / "calibrated_walk_forward_predictions.parquet"
)
OUTPUT_DIR = (
    MODEL_LAB_DIR / "simulation" / "historical_replay_v0" / "survival_components"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare class-only and round-survival finish providers"
    )
    parser.add_argument("--input", type=Path, default=SIMULATION_TRAINING_DATASET_PATH)
    parser.add_argument("--master", type=Path, default=MASTER_PATH)
    parser.add_argument(
        "--finish-class-calibration-schedule",
        type=Path,
        default=DEFAULT_CLASS_SCHEDULE,
    )
    parser.add_argument(
        "--finish-walk-forward-predictions",
        type=Path,
        default=DEFAULT_OOF_PREDICTIONS,
    )
    parser.add_argument("--finish-model", default="xgb_prefight_context")
    parser.add_argument("--test-year", type=int, default=2026)
    parser.add_argument("--simulations-per-fight", type=int, default=500)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--max-fights", type=int, default=None)
    parser.add_argument("--group-prior-rows", type=float, default=200.0)
    parser.add_argument("--minimum-subgroup-fights", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
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
        (args.finish_class_calibration_schedule, "Finish class schedule"),
        (args.finish_walk_forward_predictions, "Finish walk-forward predictions"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training = pd.read_parquet(args.input)
    master = pd.read_parquet(args.master)
    labeled_training = _attach_scoring_labels(training, master)
    class_schedule = pd.read_csv(args.finish_class_calibration_schedule)
    walk_forward_predictions = pd.read_parquet(
        args.finish_walk_forward_predictions
    )

    result = run_historical_survival_provider_replay(
        labeled_training,
        class_schedule,
        walk_forward_predictions,
        test_year=args.test_year,
        simulations_per_fight=args.simulations_per_fight,
        seed=args.seed,
        max_fights=args.max_fights,
        finish_model_name=args.finish_model,
        group_prior_rows=args.group_prior_rows,
    )
    evaluation = evaluate_historical_replay_cohort(
        result.fight_predictions,
        labeled_training,
        recommended_variant=RECOMMENDED_VARIANT,
        minimum_group_size=args.minimum_subgroup_fights,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    for variant, frame in evaluation.enriched_predictions.items():
        frame.to_parquet(OUTPUT_DIR / f"{variant}_predictions.parquet", index=False)
    result.metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    result.aggregate_comparison.to_csv(
        OUTPUT_DIR / "aggregate_comparison.csv", index=False
    )
    evaluation.subgroup_metrics.to_csv(
        OUTPUT_DIR / "subgroup_metrics.csv", index=False
    )
    evaluation.calibration.to_csv(
        OUTPUT_DIR / "calibration_diagnostics.csv", index=False
    )
    evaluation.bootstrap_metrics.to_csv(
        OUTPUT_DIR / "bootstrap_metrics.csv", index=False
    )
    evaluation.paired_variant_deltas.to_csv(
        OUTPUT_DIR / "paired_variant_deltas.csv", index=False
    )
    evaluation.stability_metrics.to_csv(
        OUTPUT_DIR / "stability_metrics.csv", index=False
    )
    (OUTPUT_DIR / "evaluation_summary.json").write_text(
        json.dumps(evaluation.summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result.class_finish_predictions.predictions.to_parquet(
        OUTPUT_DIR / "class_counterfactual_finish_predictions.parquet", index=False
    )
    result.survival_finish_predictions.predictions.to_parquet(
        OUTPUT_DIR / "survival_counterfactual_finish_predictions.parquet",
        index=False,
    )
    result.survival_finish_predictions.schedule.to_csv(
        OUTPUT_DIR / "survival_calibration_schedule.csv", index=False
    )

    metric_summary: dict[str, dict[str, float]] = {}
    for variant in (*SURVIVAL_VARIANTS, "historical_baseline"):
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
            "fight_time_bias": _lookup(
                result.metrics, "fight_time_seconds", variant, "bias"
            ),
            "fighter_sig_attempt_mae": _lookup(
                result.metrics, "fighter_sig_attempted", variant, "mae"
            ),
        }

    best_method = min(
        SURVIVAL_VARIANTS,
        key=lambda name: metric_summary[name]["method_log_loss"],
    )
    best_winner = min(
        SURVIVAL_VARIANTS,
        key=lambda name: metric_summary[name]["winner_brier"],
    )
    best_time = min(
        SURVIVAL_VARIANTS,
        key=lambda name: metric_summary[name]["fight_time_mae"],
    )
    best_strikes = min(
        SURVIVAL_VARIANTS,
        key=lambda name: metric_summary[name]["fighter_sig_attempt_mae"],
    )

    summary = {
        "status": "shadow_only",
        "decision_unit": "large_historical_cohort",
        "single_card_role": "smoke_test_only",
        "recommended_variant": RECOMMENDED_VARIANT,
        "test_year": int(args.test_year),
        "fights": int(len(result.fight_predictions[SURVIVAL_VARIANTS[0]])),
        "simulations_per_fight": int(args.simulations_per_fight),
        "finish_model_name": args.finish_model,
        "group_prior_rows": float(args.group_prior_rows),
        "minimum_subgroup_fights": int(args.minimum_subgroup_fights),
        "bootstrap_samples": int(args.bootstrap_samples),
        "survival_calibration_source": "prior_walk_forward_round_survival",
        "survival_schedule_rows": int(
            len(result.survival_finish_predictions.schedule)
        ),
        "metrics": metric_summary,
        "best_variants": {
            "method_log_loss": best_method,
            "winner_brier": best_winner,
            "fight_time_mae": best_time,
            "fighter_sig_attempt_mae": best_strikes,
        },
        "large_cohort_evaluation": evaluation.summary,
        "aggregate_comparison": result.aggregate_comparison.to_dict(
            orient="records"
        ),
        "survival_schedule": result.survival_finish_predictions.schedule.to_dict(
            orient="records"
        ),
        "artifacts": {
            "metrics": str(OUTPUT_DIR / "metrics.csv"),
            "aggregate_comparison": str(
                OUTPUT_DIR / "aggregate_comparison.csv"
            ),
            "subgroup_metrics": str(OUTPUT_DIR / "subgroup_metrics.csv"),
            "calibration": str(OUTPUT_DIR / "calibration_diagnostics.csv"),
            "bootstrap_metrics": str(OUTPUT_DIR / "bootstrap_metrics.csv"),
            "paired_variant_deltas": str(
                OUTPUT_DIR / "paired_variant_deltas.csv"
            ),
            "stability_metrics": str(OUTPUT_DIR / "stability_metrics.csv"),
            "evaluation_summary": str(OUTPUT_DIR / "evaluation_summary.json"),
            "survival_schedule": str(
                OUTPUT_DIR / "survival_calibration_schedule.csv"
            ),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("=" * 80)
    print("HISTORICAL ROUND-SURVIVAL FINISH REPLAY")
    print("=" * 80)
    print(result.metrics.to_string(index=False))
    print("\nAggregate comparison:")
    print(result.aggregate_comparison.to_string(index=False))
    print("\nLarge-cohort bootstrap metrics:")
    print(evaluation.bootstrap_metrics.to_string(index=False))
    print("\nPaired deltas against the recommended survival variant:")
    print(evaluation.paired_variant_deltas.to_string(index=False))
    print("\nSurvival schedule:")
    print(result.survival_finish_predictions.schedule.to_string(index=False))
    print(f"\nSummary: {OUTPUT_DIR / 'summary.json'}")
    print("Shadow-only comparison. No production artifact was changed.")


if __name__ == "__main__":
    main()
