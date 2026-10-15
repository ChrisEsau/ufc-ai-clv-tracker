"""Run the hierarchical finish-provider ablation on the historical cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.common.paths import MASTER_PATH, MODEL_LAB_DIR
from pipeline.simulation.artifacts import SIMULATION_TRAINING_DATASET_PATH
from pipeline.simulation.hierarchical_finish_hazard_model import HIERARCHICAL_MODELS
from pipeline.simulation.historical_hierarchical_finish_replay import (
    HIERARCHICAL_REPLAY_VARIANTS,
    run_historical_hierarchical_finish_replay,
)
from pipeline.simulation.historical_replay_evaluation import (
    evaluate_historical_replay_cohort,
)
from pipeline.simulation.historical_submission_diagnostics import (
    audit_submission_failures,
)
from pipeline.simulation.run_historical_simulator_replay import (
    _attach_scoring_labels,
)


CURRENT_FINISH_MODEL_DIR = (
    MODEL_LAB_DIR / "simulation" / "models" / "finish_hazard_prefight_v0"
)
HIERARCHICAL_FINISH_MODEL_DIR = (
    MODEL_LAB_DIR / "simulation" / "models" / "finish_hazard_hierarchical_v0"
)
OUTPUT_DIR = (
    MODEL_LAB_DIR / "simulation" / "historical_replay_v0" / "hierarchical_finish"
)
CURRENT_REFERENCE_VARIANT = "survival_finish_hazard_provider"
HIERARCHICAL_VARIANT = "hierarchical_survival_finish_hazard_provider"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare flat and hierarchical finish-provider architectures"
    )
    parser.add_argument("--input", type=Path, default=SIMULATION_TRAINING_DATASET_PATH)
    parser.add_argument("--master", type=Path, default=MASTER_PATH)
    parser.add_argument(
        "--current-class-calibration-schedule",
        type=Path,
        default=CURRENT_FINISH_MODEL_DIR / "calibration_schedule.csv",
    )
    parser.add_argument(
        "--current-walk-forward-predictions",
        type=Path,
        default=CURRENT_FINISH_MODEL_DIR / "calibrated_walk_forward_predictions.parquet",
    )
    parser.add_argument(
        "--hierarchical-calibration-schedule",
        type=Path,
        default=HIERARCHICAL_FINISH_MODEL_DIR / "calibration_schedule.csv",
    )
    parser.add_argument(
        "--hierarchical-walk-forward-predictions",
        type=Path,
        default=HIERARCHICAL_FINISH_MODEL_DIR
        / "calibrated_walk_forward_predictions.parquet",
    )
    parser.add_argument("--current-model", default="xgb_prefight_context")
    parser.add_argument("--hierarchical-model", default=HIERARCHICAL_MODELS[0])
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


def _variant_metrics(metrics: pd.DataFrame, variant: str) -> dict[str, float]:
    return {
        "winner_brier": _lookup(metrics, "winner", variant, "brier"),
        "winner_accuracy": _lookup(metrics, "winner", variant, "accuracy"),
        "method_log_loss": _lookup(metrics, "method", variant, "log_loss"),
        "method_accuracy": _lookup(metrics, "method", variant, "accuracy"),
        "goes_distance_brier": _lookup(
            metrics, "goes_distance", variant, "brier"
        ),
        "fight_time_mae": _lookup(
            metrics, "fight_time_seconds", variant, "mae"
        ),
        "fighter_sig_attempt_mae": _lookup(
            metrics, "fighter_sig_attempted", variant, "mae"
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    required = (
        (args.input, "Simulator training table"),
        (args.master, "Master fight table"),
        (args.current_class_calibration_schedule, "Current class schedule"),
        (args.current_walk_forward_predictions, "Current walk-forward predictions"),
        (args.hierarchical_calibration_schedule, "Hierarchical calibration schedule"),
        (
            args.hierarchical_walk_forward_predictions,
            "Hierarchical walk-forward predictions",
        ),
    )
    for path, label in required:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training = pd.read_parquet(args.input)
    master = pd.read_parquet(args.master)
    labeled_training = _attach_scoring_labels(training, master)
    current_schedule = pd.read_csv(args.current_class_calibration_schedule)
    current_walk_forward = pd.read_parquet(args.current_walk_forward_predictions)
    hierarchical_schedule = pd.read_csv(args.hierarchical_calibration_schedule)
    hierarchical_walk_forward = pd.read_parquet(
        args.hierarchical_walk_forward_predictions
    )

    result = run_historical_hierarchical_finish_replay(
        labeled_training,
        current_schedule,
        current_walk_forward,
        hierarchical_schedule,
        hierarchical_walk_forward,
        test_year=args.test_year,
        simulations_per_fight=args.simulations_per_fight,
        seed=args.seed,
        max_fights=args.max_fights,
        current_model_name=args.current_model,
        hierarchical_model_name=args.hierarchical_model,
        group_prior_rows=args.group_prior_rows,
    )
    evaluation = evaluate_historical_replay_cohort(
        result.fight_predictions,
        labeled_training,
        recommended_variant=CURRENT_REFERENCE_VARIANT,
        minimum_group_size=args.minimum_subgroup_fights,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    current_audit = audit_submission_failures(
        result.fight_predictions[CURRENT_REFERENCE_VARIANT],
        result.current_survival_predictions.predictions,
        labeled_training,
        test_year=args.test_year,
        minimum_group_size=args.minimum_subgroup_fights,
    )
    hierarchical_audit = audit_submission_failures(
        result.fight_predictions[HIERARCHICAL_VARIANT],
        result.hierarchical_survival_predictions.predictions,
        labeled_training,
        test_year=args.test_year,
        minimum_group_size=args.minimum_subgroup_fights,
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

    result.current_survival_predictions.predictions.to_parquet(
        OUTPUT_DIR / "current_survival_finish_predictions.parquet", index=False
    )
    result.hierarchical_class_predictions.predictions.to_parquet(
        OUTPUT_DIR / "hierarchical_class_finish_predictions.parquet", index=False
    )
    result.hierarchical_survival_predictions.predictions.to_parquet(
        OUTPUT_DIR / "hierarchical_survival_finish_predictions.parquet", index=False
    )
    result.hierarchical_survival_predictions.schedule.to_csv(
        OUTPUT_DIR / "hierarchical_survival_schedule.csv", index=False
    )

    for label, audit in (
        ("current", current_audit),
        ("hierarchical", hierarchical_audit),
    ):
        audit.fight_diagnostics.to_parquet(
            OUTPUT_DIR / f"{label}_submission_fight_diagnostics.parquet",
            index=False,
        )
        audit.error_classes.to_csv(
            OUTPUT_DIR / f"{label}_submission_error_classes.csv", index=False
        )
        audit.calibration.to_csv(
            OUTPUT_DIR / f"{label}_submission_calibration.csv", index=False
        )
        audit.subgroup_metrics.to_csv(
            OUTPUT_DIR / f"{label}_submission_subgroups.csv", index=False
        )
        (OUTPUT_DIR / f"{label}_submission_summary.json").write_text(
            json.dumps(audit.summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    metrics_by_variant = {
        variant: _variant_metrics(result.metrics, variant)
        for variant in (*HIERARCHICAL_REPLAY_VARIANTS, "historical_baseline")
    }
    current_metrics = metrics_by_variant[CURRENT_REFERENCE_VARIANT]
    hierarchical_metrics = metrics_by_variant[HIERARCHICAL_VARIANT]
    current_submission = current_audit.summary["metrics"]
    hierarchical_submission = hierarchical_audit.summary["metrics"]

    gate_checks = {
        "method_log_loss_improves": (
            hierarchical_metrics["method_log_loss"]
            < current_metrics["method_log_loss"]
        ),
        "winner_brier_not_worse": (
            hierarchical_metrics["winner_brier"]
            <= current_metrics["winner_brier"]
        ),
        "submission_detection_improves_5pp": (
            float(hierarchical_submission["submission_method_detection_rate"])
            >= float(current_submission["submission_method_detection_rate"]) + 0.05
        ),
        "submission_side_accuracy_not_worse_2pp": (
            float(hierarchical_submission["submission_side_accuracy"])
            >= float(current_submission["submission_side_accuracy"]) - 0.02
        ),
    }
    gate_pass = all(gate_checks.values())
    selected_variant = (
        HIERARCHICAL_VARIANT if gate_pass else CURRENT_REFERENCE_VARIANT
    )

    summary = {
        "status": "shadow_only",
        "decision_unit": "large_historical_cohort",
        "architecture_test": "hierarchical_finish_method_side_survival",
        "test_year": int(args.test_year),
        "fights": int(len(result.fight_predictions[CURRENT_REFERENCE_VARIANT])),
        "simulations_per_fight": int(args.simulations_per_fight),
        "bootstrap_samples": int(args.bootstrap_samples),
        "current_reference_variant": CURRENT_REFERENCE_VARIANT,
        "hierarchical_variant": HIERARCHICAL_VARIANT,
        "selected_variant_after_ablation": selected_variant,
        "research_gate_status": "pass" if gate_pass else "blocked",
        "gate_checks": gate_checks,
        "metrics": metrics_by_variant,
        "current_submission_audit": current_audit.summary,
        "hierarchical_submission_audit": hierarchical_audit.summary,
        "paired_evaluation": evaluation.summary,
        "aggregate_comparison": result.aggregate_comparison.to_dict(
            orient="records"
        ),
        "production_impact": "none",
        "artifacts": {
            "metrics": str(OUTPUT_DIR / "metrics.csv"),
            "aggregate_comparison": str(
                OUTPUT_DIR / "aggregate_comparison.csv"
            ),
            "bootstrap_metrics": str(OUTPUT_DIR / "bootstrap_metrics.csv"),
            "paired_variant_deltas": str(
                OUTPUT_DIR / "paired_variant_deltas.csv"
            ),
            "current_submission_summary": str(
                OUTPUT_DIR / "current_submission_summary.json"
            ),
            "hierarchical_submission_summary": str(
                OUTPUT_DIR / "hierarchical_submission_summary.json"
            ),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("=" * 80)
    print("HISTORICAL HIERARCHICAL FINISH-PROVIDER ABLATION")
    print("=" * 80)
    print(result.metrics.to_string(index=False))
    print("\nAggregate comparison:")
    print(result.aggregate_comparison.to_string(index=False))
    print("\nPaired deltas against current survival finish:")
    print(evaluation.paired_variant_deltas.to_string(index=False))
    print("\nSubmission audit comparison:")
    print(
        pd.DataFrame(
            [
                {"variant": "current", **current_submission},
                {"variant": "hierarchical", **hierarchical_submission},
            ]
        ).to_string(index=False)
    )
    print(f"\nResearch gate: {'pass' if gate_pass else 'blocked'}")
    print(f"Selected variant after ablation: {selected_variant}")
    print(f"Summary: {OUTPUT_DIR / 'summary.json'}")
    print("Shadow-only comparison. No production artifact was changed.")


if __name__ == "__main__":
    main()
