"""Run the evaluation-only historical submission and grappling failure audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.common.paths import MASTER_PATH, MODEL_LAB_DIR
from pipeline.simulation.artifacts import SIMULATION_TRAINING_DATASET_PATH
from pipeline.simulation.historical_replay_evaluation import RECOMMENDED_VARIANT
from pipeline.simulation.historical_submission_diagnostics import (
    audit_submission_failures,
)
from pipeline.simulation.run_historical_simulator_replay import (
    _attach_scoring_labels,
)


OUTPUT_DIR = (
    MODEL_LAB_DIR / "simulation" / "historical_replay_v0" / "survival_components"
)
DEFAULT_SIMULATOR_PREDICTIONS = (
    OUTPUT_DIR / f"{RECOMMENDED_VARIANT}_predictions.parquet"
)
DEFAULT_FINISH_PREDICTIONS = (
    OUTPUT_DIR / "survival_counterfactual_finish_predictions.parquet"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit historical submission detection, side allocation, and grappling state"
    )
    parser.add_argument("--input", type=Path, default=SIMULATION_TRAINING_DATASET_PATH)
    parser.add_argument("--master", type=Path, default=MASTER_PATH)
    parser.add_argument(
        "--simulator-predictions",
        type=Path,
        default=DEFAULT_SIMULATOR_PREDICTIONS,
    )
    parser.add_argument(
        "--finish-predictions",
        type=Path,
        default=DEFAULT_FINISH_PREDICTIONS,
    )
    parser.add_argument("--test-year", type=int, default=2026)
    parser.add_argument("--minimum-subgroup-fights", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for path, label in (
        (args.input, "Simulator training table"),
        (args.master, "Master fight table"),
        (args.simulator_predictions, "Survival simulator predictions"),
        (args.finish_predictions, "Survival finish predictions"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    training = pd.read_parquet(args.input)
    master = pd.read_parquet(args.master)
    labeled_training = _attach_scoring_labels(training, master)
    simulator_predictions = pd.read_parquet(args.simulator_predictions)
    finish_predictions = pd.read_parquet(args.finish_predictions)

    result = audit_submission_failures(
        simulator_predictions,
        finish_predictions,
        labeled_training,
        test_year=args.test_year,
        minimum_group_size=args.minimum_subgroup_fights,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.fight_diagnostics.to_parquet(
        OUTPUT_DIR / "submission_fight_diagnostics.parquet",
        index=False,
    )
    result.error_classes.to_csv(
        OUTPUT_DIR / "submission_error_classes.csv",
        index=False,
    )
    result.calibration.to_csv(
        OUTPUT_DIR / "submission_calibration.csv",
        index=False,
    )
    result.subgroup_metrics.to_csv(
        OUTPUT_DIR / "submission_subgroup_metrics.csv",
        index=False,
    )

    summary = dict(result.summary)
    summary["simulator_variant"] = RECOMMENDED_VARIANT
    summary["minimum_subgroup_fights"] = int(args.minimum_subgroup_fights)
    summary["artifacts"] = {
        "fight_diagnostics": str(
            OUTPUT_DIR / "submission_fight_diagnostics.parquet"
        ),
        "error_classes": str(OUTPUT_DIR / "submission_error_classes.csv"),
        "calibration": str(OUTPUT_DIR / "submission_calibration.csv"),
        "subgroup_metrics": str(
            OUTPUT_DIR / "submission_subgroup_metrics.csv"
        ),
    }
    (OUTPUT_DIR / "submission_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("=" * 80)
    print("HISTORICAL SUBMISSION AND GRAPPLING FAILURE AUDIT")
    print("=" * 80)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nError classes:")
    print(result.error_classes.to_string(index=False))
    print("\nSubmission calibration:")
    print(result.calibration.to_string(index=False))
    print("\nSubmission subgroup metrics:")
    print(result.subgroup_metrics.to_string(index=False))
    print("\nEvaluation-only audit. No simulator or production artifact was changed.")


if __name__ == "__main__":
    main()
