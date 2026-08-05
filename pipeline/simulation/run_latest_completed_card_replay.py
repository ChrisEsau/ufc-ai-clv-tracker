"""Run the current trained-component simulator on the latest completed UFC card.

The card is selected from the authoritative local master/training data rather than
from an upcoming-events feed. Fighter states and model calibration use only fights
completed before the selected event. Actual results are attached only for grading.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH, MODEL_LAB_DIR
from pipeline.simulation.artifacts import SIMULATION_TRAINING_DATASET_PATH
from pipeline.simulation.finish_hazard_holdout import (
    build_counterfactual_finish_predictions,
)
from pipeline.simulation.historical_simulator_replay import (
    build_fighter_fight_history,
    build_holdout_matchups,
    population_priors,
)
from pipeline.simulation.historical_trained_component_replay import (
    run_historical_trained_component_replay,
)
from pipeline.simulation.run_historical_simulator_replay import (
    _attach_scoring_labels,
)


OUTPUT_DIR = MODEL_LAB_DIR / "simulation" / "latest_completed_card_replay_v0"
DEFAULT_FINISH_DIR = (
    MODEL_LAB_DIR / "simulation" / "models" / "finish_hazard_prefight_v0"
)
METHOD_PROBABILITY_COLUMNS = {
    "decision": "sim_decision_probability",
    "ko_tko": "sim_ko_tko_probability",
    "submission": "sim_submission_probability",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay the current trained simulator on the latest completed card"
    )
    parser.add_argument("--input", type=Path, default=SIMULATION_TRAINING_DATASET_PATH)
    parser.add_argument("--master", type=Path, default=MASTER_PATH)
    parser.add_argument(
        "--finish-calibration-schedule",
        type=Path,
        default=DEFAULT_FINISH_DIR / "calibration_schedule.csv",
    )
    parser.add_argument("--finish-model", default="xgb_prefight_context")
    parser.add_argument("--test-year", type=int, default=2026)
    parser.add_argument("--simulations-per-fight", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=91)
    return parser


def select_latest_card_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return every matchup from the newest completed event represented in records."""
    if not records:
        raise ValueError("No scoreable completed fight records were supplied")
    latest_date = max(pd.Timestamp(record["date"]) for record in records)
    date_records = [
        record for record in records if pd.Timestamp(record["date"]) == latest_date
    ]
    event_counts: dict[str, int] = {}
    for record in date_records:
        event_id = str(record["matchup"].event_id)
        event_counts[event_id] = event_counts.get(event_id, 0) + 1
    selected_event = sorted(
        event_counts,
        key=lambda event_id: (-event_counts[event_id], event_id),
    )[0]
    selected = [
        record
        for record in date_records
        if str(record["matchup"].event_id) == selected_event
    ]
    selected.sort(key=lambda record: str(record["matchup"].fight_id))
    if not selected:
        raise ValueError("Latest completed event contained no scoreable fights")
    return selected


def _grade_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    graded = predictions.copy()
    graded["predicted_winner_corner"] = np.where(
        graded["sim_red_win_probability"].ge(0.5), "red", "blue"
    )
    graded["predicted_winner_probability"] = np.where(
        graded["predicted_winner_corner"].eq("red"),
        graded["sim_red_win_probability"],
        1.0 - graded["sim_red_win_probability"],
    )
    graded["predicted_winner_name"] = np.where(
        graded["predicted_winner_corner"].eq("red"),
        graded["red_fighter_name"],
        graded["blue_fighter_name"],
    )
    graded["actual_winner_name"] = np.where(
        graded["actual_winner_corner"].eq("red"),
        graded["red_fighter_name"],
        graded["blue_fighter_name"],
    )
    graded["winner_correct"] = graded["predicted_winner_corner"].eq(
        graded["actual_winner_corner"]
    )

    method_columns = list(METHOD_PROBABILITY_COLUMNS.values())
    method_labels = list(METHOD_PROBABILITY_COLUMNS)
    method_index = np.argmax(graded[method_columns].to_numpy(dtype=float), axis=1)
    graded["predicted_method"] = [method_labels[index] for index in method_index]
    graded["predicted_method_probability"] = graded[method_columns].max(axis=1)
    graded["method_correct"] = graded["predicted_method"].eq(
        graded["actual_method"]
    )
    graded["actual_goes_distance"] = graded["actual_method"].eq("decision")
    graded["predicted_goes_distance"] = graded["sim_decision_probability"].ge(0.5)
    graded["distance_correct"] = graded["predicted_goes_distance"].eq(
        graded["actual_goes_distance"]
    )
    graded["fight_time_absolute_error"] = (
        graded["sim_fight_time_seconds"] - graded["actual_fight_time_seconds"]
    ).abs()
    return graded


def _event_name(labeled: pd.DataFrame, fight_ids: set[str]) -> str:
    if "event_name" not in labeled.columns:
        return "Latest completed UFC event"
    names = (
        labeled.loc[labeled["fight_id"].astype(str).isin(fight_ids), "event_name"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    names = names.loc[names.ne("")].unique().tolist()
    return names[0] if len(names) == 1 else "Latest completed UFC event"


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

    history = build_fighter_fight_history(labeled)
    priors = population_priors(history, test_year=args.test_year)
    all_holdout_records = build_holdout_matchups(
        history,
        test_year=args.test_year,
        priors=priors,
    )
    latest_records = select_latest_card_records(all_holdout_records)
    latest_date = pd.Timestamp(latest_records[0]["date"])
    latest_event_id = str(latest_records[0]["matchup"].event_id)
    latest_fight_ids = {
        str(record["matchup"].fight_id) for record in latest_records
    }
    if any(pd.Timestamp(record["date"]) != latest_date for record in latest_records):
        raise RuntimeError("Latest-card selection crossed event dates")
    if any(
        str(record["matchup"].event_id) != latest_event_id
        for record in latest_records
    ):
        raise RuntimeError("Latest-card selection crossed event IDs")

    counterfactual = build_counterfactual_finish_predictions(
        training_df=labeled,
        calibration_schedule=schedule,
        test_year=args.test_year,
        model_name=args.finish_model,
        seed=7,
    )
    result = run_historical_trained_component_replay(
        labeled,
        finish_predictions=counterfactual.predictions,
        finish_model_name=args.finish_model,
        test_year=args.test_year,
        simulations_per_fight=args.simulations_per_fight,
        seed=args.seed,
        max_fights=len(latest_records),
    )
    predictions = result.fight_predictions.copy()
    resolved_ids = set(predictions["fight_id"].astype(str))
    if resolved_ids != latest_fight_ids:
        raise RuntimeError(
            "Latest-card replay did not resolve the exact selected fight set"
        )
    if predictions["event_id"].astype(str).nunique() != 1:
        raise RuntimeError("Latest-card predictions contain multiple events")
    predictions = _grade_predictions(predictions)

    event_name = _event_name(labeled, latest_fight_ids)
    card_summary = {
        "status": "shadow_only",
        "event_name": event_name,
        "event_id": latest_event_id,
        "event_date": str(latest_date.date()),
        "fights": int(len(predictions)),
        "simulations_per_fight": int(args.simulations_per_fight),
        "total_simulated_paths": int(
            len(predictions) * args.simulations_per_fight
        ),
        "winner_accuracy": float(predictions["winner_correct"].mean()),
        "method_accuracy": float(predictions["method_correct"].mean()),
        "distance_accuracy": float(predictions["distance_correct"].mean()),
        "fight_time_mae_seconds": float(
            predictions["fight_time_absolute_error"].mean()
        ),
        "mean_predicted_winner_confidence": float(
            predictions["predicted_winner_probability"].mean()
        ),
        "finish_model": args.finish_model,
        "simulator_version": str(predictions["simulator_version"].iloc[0]),
        "promotion_status": "blocked",
    }

    output_columns = [
        "fight_id",
        "red_fighter_name",
        "blue_fighter_name",
        "predicted_winner_name",
        "predicted_winner_probability",
        "actual_winner_name",
        "winner_correct",
        "predicted_method",
        "predicted_method_probability",
        "actual_method",
        "method_correct",
        "sim_decision_probability",
        "sim_ko_tko_probability",
        "sim_submission_probability",
        "distance_correct",
        "sim_fight_time_seconds",
        "actual_fight_time_seconds",
        "fight_time_absolute_error",
        "sim_red_sig_attempted",
        "actual_red_sig_attempted",
        "sim_blue_sig_attempted",
        "actual_blue_sig_attempted",
        "red_prior_fights",
        "blue_prior_fights",
    ]
    report = predictions[output_columns].sort_values(
        "predicted_winner_probability", ascending=False
    )
    report.to_csv(OUTPUT_DIR / "fight_report.csv", index=False)
    predictions.to_parquet(OUTPUT_DIR / "fight_predictions.parquet", index=False)
    result.metrics.to_csv(OUTPUT_DIR / "card_metrics.csv", index=False)
    result.calibration.to_csv(OUTPUT_DIR / "card_calibration.csv", index=False)
    result.aggregate_comparison.to_csv(
        OUTPUT_DIR / "card_aggregate_comparison.csv", index=False
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(card_summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("=" * 100)
    print("LATEST COMPLETED CARD — CURRENT TRAINED-COMPONENT SIMULATOR")
    print("=" * 100)
    print(f"Event: {event_name}")
    print(f"Date: {latest_date.date()} | Event ID: {latest_event_id}")
    print(
        f"Fights: {len(report)} | Paths per fight: {args.simulations_per_fight:,}"
    )
    print(report.to_string(index=False))
    print("\nCard summary:")
    print(json.dumps(card_summary, indent=2, sort_keys=True))
    print("Shadow-only. No production or wagering artifact was promoted.")


if __name__ == "__main__":
    main()
