"""Run the heuristic fight simulator against completed historical holdouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.common.fight_time import repair_elapsed_match_time
from pipeline.common.paths import MASTER_PATH, MODEL_LAB_DIR
from pipeline.simulation.artifacts import SIMULATION_TRAINING_DATASET_PATH
from pipeline.simulation.historical_simulator_replay import (
    metric_lookup,
    run_historical_simulator_replay,
)


OUTPUT_DIR = MODEL_LAB_DIR / "simulation" / "historical_replay_v0"
SCOREABLE_METHODS = frozenset({"decision", "ko_tko", "submission"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay the full heuristic simulator")
    parser.add_argument("--input", type=Path, default=SIMULATION_TRAINING_DATASET_PATH)
    parser.add_argument("--master", type=Path, default=MASTER_PATH)
    parser.add_argument("--test-year", type=int, default=2026)
    parser.add_argument("--simulations-per-fight", type=int, default=750)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--max-fights", type=int, default=None)
    return parser


def _method_family(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if "ko" in text or "tko" in text:
        return "ko_tko"
    if "sub" in text:
        return "submission"
    if "dec" in text:
        return "decision"
    return "other"


def _attach_scoring_labels(training: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Attach scoreable labels with one validated elapsed-time definition.

    The authoritative master may store ``match_time_sec`` as either total elapsed
    fight time or the clock inside the final round. The leakage-safe simulator
    training builder already repairs that ambiguity before creating
    ``target_elapsed_fight_seconds``. Historical replay therefore repairs master
    labels, verifies exact agreement with the training target at fight grain, and
    scores against that validated elapsed value.

    Draws, no-contests, overturned bouts, and rows without a complete winner/time
    cannot be scored by the current two-corner simulator contract. They are
    excluded from replay evaluation without changing the training artifact or any
    fighter's pre-fight historical state.
    """
    master_required = [
        "fight_id",
        "winner_id",
        "method",
        "finish_round",
        "match_time_sec",
    ]
    training_required = ["fight_id", "target_elapsed_fight_seconds"]
    missing_master = [column for column in master_required if column not in master]
    missing_training = [
        column for column in training_required if column not in training
    ]
    if missing_master:
        raise ValueError(f"Master fight table is missing replay labels: {missing_master}")
    if missing_training:
        raise ValueError(
            "Simulator training table is missing repaired elapsed-time targets: "
            f"{missing_training}"
        )

    elapsed = training[["fight_id", "target_elapsed_fight_seconds"]].copy()
    elapsed["target_elapsed_fight_seconds"] = pd.to_numeric(
        elapsed["target_elapsed_fight_seconds"], errors="coerce"
    )
    consistency = elapsed.groupby("fight_id", dropna=False)[
        "target_elapsed_fight_seconds"
    ].agg(["min", "max", "count"])
    inconsistent = (
        consistency["min"].isna()
        | consistency["max"].isna()
        | consistency["min"].ne(consistency["max"])
    )
    if inconsistent.any():
        sample = consistency.loc[inconsistent].head(10).reset_index().to_dict(
            orient="records"
        )
        raise ValueError(
            "Training elapsed-time targets are missing or inconsistent within fights: "
            f"{sample}"
        )
    elapsed = consistency[["min"]].rename(
        columns={"min": "target_elapsed_fight_seconds"}
    ).reset_index()

    repaired_master = repair_elapsed_match_time(master)
    labels = repaired_master.loc[
        repaired_master["fight_id"].isin(training["fight_id"].unique()),
        master_required,
    ].copy()
    labels["method_family"] = labels["method"].map(_method_family)
    labels["match_time_sec"] = pd.to_numeric(
        labels["match_time_sec"], errors="coerce"
    )
    labels["winner_id"] = labels["winner_id"].astype("string").str.strip()
    if labels.duplicated(["fight_id"]).any():
        raise ValueError("Master fight table has duplicate fight_id labels")

    labels = labels.merge(
        elapsed,
        on="fight_id",
        how="inner",
        validate="one_to_one",
    )
    comparable = labels["match_time_sec"].notna() & labels[
        "target_elapsed_fight_seconds"
    ].notna()
    mismatch = comparable & (
        labels["match_time_sec"] - labels["target_elapsed_fight_seconds"]
    ).abs().gt(1e-6)
    if mismatch.any():
        sample = labels.loc[
            mismatch,
            ["fight_id", "finish_round", "match_time_sec", "target_elapsed_fight_seconds"],
        ].head(10).to_dict(orient="records")
        raise ValueError(
            "Repaired master elapsed time disagrees with simulator training targets: "
            f"{sample}"
        )
    labels["match_time_sec"] = labels["target_elapsed_fight_seconds"]

    scoreable = (
        labels["winner_id"].notna()
        & labels["winner_id"].ne("")
        & labels["match_time_sec"].notna()
        & labels["match_time_sec"].ge(0)
        & labels["method_family"].isin(SCOREABLE_METHODS)
    )
    labels = labels.loc[scoreable].copy()
    if labels.empty:
        raise ValueError("No scoreable master fight labels were available for replay")

    labeled = training.merge(
        labels[["fight_id", "winner_id", "method_family", "match_time_sec"]],
        on="fight_id",
        how="inner",
        validate="many_to_one",
    )
    if labeled.empty:
        raise ValueError("No training rows matched scoreable master fight labels")
    return labeled


def main() -> None:
    args = build_parser().parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Training table not found: {args.input}. Run the simulator training builder first."
        )
    if not args.master.exists():
        raise FileNotFoundError(f"Master fight table not found: {args.master}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training = pd.read_parquet(args.input)
    master = pd.read_parquet(args.master)
    labeled_training = _attach_scoring_labels(training, master)

    result = run_historical_simulator_replay(
        labeled_training,
        test_year=args.test_year,
        simulations_per_fight=args.simulations_per_fight,
        seed=args.seed,
        max_fights=args.max_fights,
    )

    predictions_path = OUTPUT_DIR / "fight_predictions.parquet"
    metrics_path = OUTPUT_DIR / "metrics.csv"
    calibration_path = OUTPUT_DIR / "calibration.csv"
    aggregate_path = OUTPUT_DIR / "aggregate_comparison.csv"
    summary_path = OUTPUT_DIR / "summary.json"

    result.fight_predictions.to_parquet(predictions_path, index=False)
    result.metrics.to_csv(metrics_path, index=False)
    result.calibration.to_csv(calibration_path, index=False)
    result.aggregate_comparison.to_csv(aggregate_path, index=False)

    winner_sim = metric_lookup(result.metrics, "winner", "simulator", "brier")
    winner_base = metric_lookup(
        result.metrics, "winner", "historical_baseline", "brier"
    )
    method_sim = metric_lookup(result.metrics, "method", "simulator", "log_loss")
    method_base = metric_lookup(
        result.metrics, "method", "historical_baseline", "log_loss"
    )
    time_sim = metric_lookup(
        result.metrics, "fight_time_seconds", "simulator", "mae"
    )
    time_base = metric_lookup(
        result.metrics, "fight_time_seconds", "historical_baseline", "mae"
    )
    strikes_sim = metric_lookup(
        result.metrics, "fighter_sig_attempted", "simulator", "mae"
    )
    strikes_base = metric_lookup(
        result.metrics, "fighter_sig_attempted", "historical_baseline", "mae"
    )

    summary = {
        "status": "shadow_only",
        "test_year": int(args.test_year),
        "fights": int(len(result.fight_predictions)),
        "simulations_per_fight": int(args.simulations_per_fight),
        "winner_brier": {
            "simulator": winner_sim,
            "baseline": winner_base,
            "relative_improvement": (winner_base - winner_sim) / winner_base,
        },
        "method_log_loss": {
            "simulator": method_sim,
            "baseline": method_base,
            "relative_improvement": (method_base - method_sim) / method_base,
        },
        "fight_time_mae_seconds": {
            "simulator": time_sim,
            "baseline": time_base,
            "relative_improvement": (time_base - time_sim) / time_base,
        },
        "fighter_sig_attempt_mae": {
            "simulator": strikes_sim,
            "baseline": strikes_base,
            "relative_improvement": (strikes_base - strikes_sim) / strikes_base,
        },
        "aggregate_comparison": result.aggregate_comparison.to_dict(orient="records"),
        "population_priors": dict(result.population_priors),
        "artifacts": {
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "calibration": str(calibration_path),
            "aggregate_comparison": str(aggregate_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 80)
    print("HISTORICAL FULL SIMULATOR REPLAY")
    print("=" * 80)
    print(result.metrics.to_string(index=False))
    print("\nAggregate comparison:")
    print(result.aggregate_comparison.to_string(index=False))
    print(f"\nSummary: {summary_path}")
    print("Shadow-only replay. No production artifact was changed.")


if __name__ == "__main__":
    main()
