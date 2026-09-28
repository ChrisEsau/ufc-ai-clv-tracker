"""Compare heuristic, strike-only, and trained-finish simulator replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.common.paths import MASTER_PATH, MODEL_LAB_DIR
from pipeline.simulation.artifacts import SIMULATION_TRAINING_DATASET_PATH
from pipeline.simulation.finish_hazard_holdout import (
    build_counterfactual_finish_predictions,
)
from pipeline.simulation.historical_simulator_replay import (
    metric_lookup,
    run_historical_simulator_replay,
)
from pipeline.simulation.historical_strike_provider_replay import (
    run_historical_strike_provider_replay,
)
from pipeline.simulation.historical_trained_component_replay import (
    run_historical_trained_component_replay,
)
from pipeline.simulation.run_historical_simulator_replay import (
    _attach_scoring_labels,
)


OUTPUT_DIR = MODEL_LAB_DIR / "simulation" / "historical_trained_component_replay_v0"
DEFAULT_FINISH_DIR = (
    MODEL_LAB_DIR / "simulation" / "models" / "finish_hazard_prefight_v0"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay trained strike and finish components on historical fights"
    )
    parser.add_argument("--input", type=Path, default=SIMULATION_TRAINING_DATASET_PATH)
    parser.add_argument("--master", type=Path, default=MASTER_PATH)
    parser.add_argument(
        "--finish-calibration-schedule",
        type=Path,
        default=DEFAULT_FINISH_DIR / "calibration_schedule.csv",
    )
    parser.add_argument(
        "--finish-model",
        type=str,
        default="xgb_prefight_context",
    )
    parser.add_argument("--test-year", type=int, default=2026)
    parser.add_argument("--simulations-per-fight", type=int, default=500)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--max-fights", type=int, default=None)
    return parser


def _variant_metric_rows(result, variant: str) -> pd.DataFrame:
    frame = result.metrics.copy()
    frame.insert(0, "variant", variant)
    return frame


def _variant_aggregate_rows(result, variant: str) -> pd.DataFrame:
    frame = result.aggregate_comparison.copy()
    frame.insert(0, "variant", variant)
    return frame


def _metric(result, task: str, metric: str) -> float:
    return metric_lookup(result.metrics, task, "simulator", metric)


def _aggregate(result, quantity: str) -> tuple[float, float]:
    row = result.aggregate_comparison.loc[
        result.aggregate_comparison["quantity"].eq(quantity)
    ]
    if len(row) != 1:
        raise RuntimeError(f"Aggregate quantity not found: {quantity}")
    return float(row.iloc[0]["actual"]), float(row.iloc[0]["simulator"])


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
    labeled = _attach_scoring_labels(training, master)
    schedule = pd.read_csv(args.finish_calibration_schedule)
    counterfactual = build_counterfactual_finish_predictions(
        training_df=labeled,
        calibration_schedule=schedule,
        test_year=args.test_year,
        model_name=args.finish_model,
        seed=7,
    )

    common = {
        "test_year": args.test_year,
        "simulations_per_fight": args.simulations_per_fight,
        "seed": args.seed,
        "max_fights": args.max_fights,
    }
    heuristic = run_historical_simulator_replay(labeled, **common)
    strike_only = run_historical_strike_provider_replay(labeled, **common)
    trained = run_historical_trained_component_replay(
        labeled,
        finish_predictions=counterfactual.predictions,
        finish_model_name=args.finish_model,
        **common,
    )

    variants = {
        "heuristic": heuristic,
        "absolute_strike_only": strike_only,
        "absolute_strike_trained_finish": trained,
    }
    prediction_frames = []
    metric_frames = []
    aggregate_frames = []
    calibration_frames = []
    for name, result in variants.items():
        predictions = result.fight_predictions.copy()
        predictions.insert(0, "variant", name)
        prediction_frames.append(predictions)
        metric_frames.append(_variant_metric_rows(result, name))
        aggregate_frames.append(_variant_aggregate_rows(result, name))
        calibration = result.calibration.copy()
        calibration.insert(0, "variant", name)
        calibration_frames.append(calibration)

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_aggregate = pd.concat(aggregate_frames, ignore_index=True)
    all_calibration = pd.concat(calibration_frames, ignore_index=True)

    all_predictions.to_parquet(OUTPUT_DIR / "fight_predictions.parquet", index=False)
    all_metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    all_aggregate.to_csv(OUTPUT_DIR / "aggregate_comparison.csv", index=False)
    all_calibration.to_csv(OUTPUT_DIR / "calibration.csv", index=False)
    counterfactual.predictions.to_parquet(
        OUTPUT_DIR / "counterfactual_finish_predictions.parquet",
        index=False,
    )

    comparison_rows = []
    for name, result in variants.items():
        actual_decision, predicted_decision = _aggregate(result, "decision_rate")
        actual_ko, predicted_ko = _aggregate(result, "ko_tko_rate")
        actual_sub, predicted_sub = _aggregate(result, "submission_rate")
        actual_strikes, predicted_strikes = _aggregate(
            result, "fighter_sig_attempted"
        )
        comparison_rows.append(
            {
                "variant": name,
                "winner_brier": _metric(result, "winner", "brier"),
                "winner_log_loss": _metric(result, "winner", "log_loss"),
                "method_log_loss": _metric(result, "method", "log_loss"),
                "goes_distance_brier": _metric(
                    result, "goes_distance", "brier"
                ),
                "fight_time_mae": _metric(
                    result, "fight_time_seconds", "mae"
                ),
                "strike_attempt_mae": _metric(
                    result, "fighter_sig_attempted", "mae"
                ),
                "actual_decision_rate": actual_decision,
                "predicted_decision_rate": predicted_decision,
                "actual_ko_tko_rate": actual_ko,
                "predicted_ko_tko_rate": predicted_ko,
                "actual_submission_rate": actual_sub,
                "predicted_submission_rate": predicted_sub,
                "actual_fighter_sig_attempted": actual_strikes,
                "predicted_fighter_sig_attempted": predicted_strikes,
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_DIR / "variant_comparison.csv", index=False)

    best = comparison.loc[
        comparison["variant"].eq("absolute_strike_trained_finish")
    ].iloc[0]
    heuristic_row = comparison.loc[
        comparison["variant"].eq("heuristic")
    ].iloc[0]
    baseline_method = metric_lookup(
        heuristic.metrics,
        "method",
        "historical_baseline",
        "log_loss",
    )
    baseline_distance = metric_lookup(
        heuristic.metrics,
        "goes_distance",
        "historical_baseline",
        "brier",
    )
    baseline_winner = metric_lookup(
        heuristic.metrics,
        "winner",
        "historical_baseline",
        "brier",
    )

    summary = {
        "status": "shadow_only",
        "test_year": int(args.test_year),
        "fights": int(len(trained.fight_predictions)),
        "simulations_per_fight": int(args.simulations_per_fight),
        "finish_model": args.finish_model,
        "finish_model_seed": int(counterfactual.model_seed),
        "finish_calibration_source": counterfactual.calibration_source,
        "variant_comparison": comparison.to_dict(orient="records"),
        "baseline_metrics": {
            "winner_brier": baseline_winner,
            "method_log_loss": baseline_method,
            "goes_distance_brier": baseline_distance,
        },
        "trained_finish_improvement_vs_heuristic": {
            "winner_brier": float(
                (heuristic_row["winner_brier"] - best["winner_brier"])
                / heuristic_row["winner_brier"]
            ),
            "method_log_loss": float(
                (heuristic_row["method_log_loss"] - best["method_log_loss"])
                / heuristic_row["method_log_loss"]
            ),
            "goes_distance_brier": float(
                (
                    heuristic_row["goes_distance_brier"]
                    - best["goes_distance_brier"]
                )
                / heuristic_row["goes_distance_brier"]
            ),
            "fight_time_mae": float(
                (heuristic_row["fight_time_mae"] - best["fight_time_mae"])
                / heuristic_row["fight_time_mae"]
            ),
            "strike_attempt_mae": float(
                (
                    heuristic_row["strike_attempt_mae"]
                    - best["strike_attempt_mae"]
                )
                / heuristic_row["strike_attempt_mae"]
            ),
        },
        "promotion_gate": {
            "winner_beats_baseline": bool(best["winner_brier"] < baseline_winner),
            "method_beats_baseline": bool(
                best["method_log_loss"] < baseline_method
            ),
            "distance_beats_baseline": bool(
                best["goes_distance_brier"] < baseline_distance
            ),
            "status": "blocked",
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("=" * 100)
    print("HISTORICAL TRAINED-COMPONENT SIMULATOR REPLAY")
    print("=" * 100)
    print(comparison.to_string(index=False))
    print("\nPromotion remains blocked pending explicit review of all gates.")
    print(f"Summary: {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
