"""Walk-forward model backtest runner.

This runner retrains a fresh model for each test year using only data available
before that year, calibrates on the immediately prior year, predicts the target
year, joins historical market outcomes, and scores betting performance.

Example:

    python -m pipeline.backtesting.run_walk_forward_backtest_v2 \
        --model-config-path configs/models/moneyline_xgboost_v8.yaml \
        --feature-view-path data/features/moneyline_feature_view.parquet \
        --historical-market-path data/market/historical_market_outcomes.parquet \
        --train-start-date 2014-01-01 \
        --first-test-year 2022 \
        --last-test-year 2025
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pipeline.backtesting.model_outcomes import build_backtest_model_outcomes
from pipeline.backtesting.run_backtest_v2 import (
    apply_filters,
    bucket_summary,
    score_bets,
    standardize_market,
    summarize,
)
from pipeline.common.paths import MARKET_DIR, MODEL_LAB_DIR, MONEYLINE_FEATURE_VIEW_PATH, ensure_data_dirs
from pipeline.features.registry_feature_builder import apply_registry_feature_definitions
from pipeline.training.calibration import calibrate_model, predict_positive_class_probability
from pipeline.training.feature_selection import resolve_features_from_model_config
from pipeline.training.metrics import evaluate_binary_probabilities
from pipeline.training.model_training import train_model
from pipeline.training.symmetry import apply_symmetry_augmentation

DEFAULT_CONFIG_PATH = Path("configs/models/moneyline_xgboost_v8.yaml")
DEFAULT_FEATURE_VIEW_PATH = MONEYLINE_FEATURE_VIEW_PATH
DEFAULT_MARKET_PATH = MARKET_DIR / "historical_market_outcomes.parquet"
DEFAULT_OUTPUT_ROOT = MODEL_LAB_DIR / "walk_forward_backtests"


@dataclass(frozen=True)
class WalkForwardWindow:
    """One walk-forward train/calibration/test window."""

    test_year: int
    train_start_date: str
    train_end_date: str
    calibration_start_date: str
    calibration_end_date: str
    test_start_date: str
    test_end_date: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run yearly walk-forward UFC model backtest.")
    parser.add_argument("--model-config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--feature-view-path", default=str(DEFAULT_FEATURE_VIEW_PATH))
    parser.add_argument("--historical-market-path", default=str(DEFAULT_MARKET_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--market-key", default="moneyline")
    parser.add_argument("--train-start-date", default="2014-01-01")
    parser.add_argument("--first-test-year", type=int, default=2022)
    parser.add_argument("--last-test-year", type=int, default=2025)
    parser.add_argument("--starting-bankroll", type=float, default=10000.0)
    parser.add_argument("--flat-stake", type=float, default=100.0)
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--max-stake-pct", type=float, default=0.05)
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-odds", type=float, default=-1000.0)
    parser.add_argument("--max-odds", type=float, default=3000.0)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load model config YAML."""

    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Model config must be a dict: {path}")
    return config


def load_feature_view(path: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Load feature view and materialize selected registry-defined features."""

    if not path.exists():
        raise FileNotFoundError(f"Feature view not found: {path}")

    df = pd.read_parquet(path).copy()
    if "date" not in df.columns:
        raise ValueError("Feature view must contain date column for walk-forward splitting.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()

    selected_features = (config.get("features") or {}).get("feature_columns") or []
    build_result = apply_registry_feature_definitions(
        df,
        selected_features=selected_features,
        allowed_statuses={"active", "draft"},
        overwrite_existing=True,
    )
    if build_result.generated_columns:
        print(
            "Registry features materialized: "
            f"{len(build_result.generated_columns)} ({build_result.generated_columns})"
        )

    feature_df = build_result.dataframe
    feature_columns = resolve_features_from_model_config(
        df=feature_df,
        model_config=config,
    )
    return feature_df, feature_columns


def resolve_prebuilt_flipped_feature_view_path(
    *,
    normal_feature_view_path: Path,
    config: dict[str, Any],
) -> Path:
    """Resolve the training-only prebuilt flipped feature-view path."""
    data_config = config.get("data", {}) or {}
    explicit = (
        data_config.get("flipped_rolling_features_path")
        or data_config.get("flipped_feature_view_path")
    )
    if explicit:
        return Path(explicit)

    return normal_feature_view_path.with_name(
        f"{normal_feature_view_path.stem}_flipped{normal_feature_view_path.suffix}"
    )


def should_use_prebuilt_flipped_training_view(config: dict[str, Any]) -> bool:
    """Return True when symmetry should come from a prebuilt flipped feature view."""
    symmetry_config = config.get("symmetry", {}) or {}
    if not symmetry_config.get("enabled", False):
        return False

    source = str(symmetry_config.get("source", "")).strip().lower()
    return source in {"feature_view_flipped", "flipped_feature_view", "prebuilt_feature_view"}


def load_prebuilt_flipped_training_view(
    *,
    normal_feature_view_path: Path,
    config: dict[str, Any],
    expected_feature_columns: list[str],
) -> pd.DataFrame | None:
    """Load a symmetric training-only feature view when configured."""
    if not should_use_prebuilt_flipped_training_view(config):
        return None

    flipped_path = resolve_prebuilt_flipped_feature_view_path(
        normal_feature_view_path=normal_feature_view_path,
        config=config,
    )
    if not flipped_path.exists():
        raise FileNotFoundError(f"Configured flipped feature view not found: {flipped_path}")

    flipped_df, flipped_feature_columns = load_feature_view(flipped_path, config)

    if flipped_feature_columns != expected_feature_columns:
        raise ValueError(
            "Flipped feature view resolved a different feature contract. "
            f"normal={len(expected_feature_columns)} flipped={len(flipped_feature_columns)}"
        )

    print(f"Using prebuilt flipped training view: {flipped_path}")
    print(f"Prebuilt flipped training shape    : {flipped_df.shape}")
    return flipped_df


def build_windows(args: argparse.Namespace) -> list[WalkForwardWindow]:
    """Build annual walk-forward windows.

    For target year Y:
    - train from train_start_date through Y-2
    - calibrate on Y-1
    - test on Y
    """

    if args.last_test_year < args.first_test_year:
        raise ValueError("last-test-year must be greater than or equal to first-test-year")

    windows = []
    for test_year in range(args.first_test_year, args.last_test_year + 1):
        calibration_year = test_year - 1
        train_end_year = test_year - 2
        windows.append(
            WalkForwardWindow(
                test_year=test_year,
                train_start_date=str(args.train_start_date),
                train_end_date=f"{train_end_year}-12-31",
                calibration_start_date=f"{calibration_year}-01-01",
                calibration_end_date=f"{calibration_year}-12-31",
                test_start_date=f"{test_year}-01-01",
                test_end_date=f"{test_year}-12-31",
            )
        )
    return windows


def slice_window(df: pd.DataFrame, window: WalkForwardWindow) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train/calibration/test rows for one window."""

    train_start = pd.to_datetime(window.train_start_date)
    train_end = pd.to_datetime(window.train_end_date)
    cal_start = pd.to_datetime(window.calibration_start_date)
    cal_end = pd.to_datetime(window.calibration_end_date)
    test_start = pd.to_datetime(window.test_start_date)
    test_end = pd.to_datetime(window.test_end_date)

    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
    calibration = df[(df["date"] >= cal_start) & (df["date"] <= cal_end)].copy()
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    return train, calibration, test


def prepare_matrix(df: pd.DataFrame, feature_columns: list[str], target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Build numeric X/y matrices."""

    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Feature view missing model features: {missing[:20]} total_missing={len(missing)}")

    X = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(df[target_column], errors="coerce").astype(int)
    return X, y


def maybe_apply_symmetry_to_training(
    train: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Apply configured symmetry augmentation to training rows only."""

    symmetry_config = config.get("symmetry", {}) or {}
    if not symmetry_config.get("enabled", False):
        return train

    # When training rows already come from a prebuilt flipped feature view,
    # do not apply runtime sign-flipping again.
    source = str(symmetry_config.get("source", "")).strip().lower()
    if source in {"feature_view_flipped", "flipped_feature_view", "prebuilt_feature_view"}:
        return train

    mode = str(symmetry_config.get("mode", "flip_all")).strip().lower()
    target_col = config.get("data", {}).get("target_column", "target")
    date_col = config.get("data", {}).get("date_column", "date")

    if mode == "flip_all":
        return apply_symmetry_augmentation(
            df=train,
            feature_columns=feature_columns,
            target_col=target_col,
            date_col=date_col,
        )

    if mode == "explicit":
        return apply_symmetry_augmentation(
            df=train,
            feature_columns=feature_columns,
            target_col=target_col,
            date_col=date_col,
            flip_feature_columns=symmetry_config.get("flip_features", []),
            preserve_feature_columns=symmetry_config.get("preserve_features", []),
        )

    raise ValueError(f"Unsupported symmetry mode: {mode}")


def clip_probabilities(probs: pd.Series, config: dict[str, Any]) -> pd.Series:
    """Clip probabilities using model config prediction settings."""

    probability_config = (config.get("prediction") or {}).get("probability", {}) or {}
    low = float(probability_config.get("clip_low", 0.0))
    high = float(probability_config.get("clip_high", 1.0))
    return probs.clip(lower=low, upper=high)


def run_one_window(
    *,
    window: WalkForwardWindow,
    feature_df: pd.DataFrame,
    training_feature_df: pd.DataFrame | None,
    market: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
    args: argparse.Namespace,
    run_id: str,
    timestamp: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train, predict, and score one walk-forward test year."""

    data_config = config.get("data", {}) or {}
    target_col = data_config.get("target_column", "target")

    train_source_df = training_feature_df if training_feature_df is not None else feature_df
    train_raw, _, _ = slice_window(train_source_df, window)
    _, calibration_raw, test_raw = slice_window(feature_df, window)
    if train_raw.empty or calibration_raw.empty or test_raw.empty:
        raise ValueError(
            f"Window {window.test_year} has empty split: "
            f"train={len(train_raw)}, calibration={len(calibration_raw)}, test={len(test_raw)}"
        )

    train_augmented = maybe_apply_symmetry_to_training(train_raw, feature_columns, config)
    X_train, y_train = prepare_matrix(train_augmented, feature_columns, target_col)
    X_calibration, y_calibration = prepare_matrix(calibration_raw, feature_columns, target_col)
    X_test, y_test = prepare_matrix(test_raw, feature_columns, target_col)

    training_result = train_model(
        algorithm=config["algorithm"],
        X_train=X_train,
        y_train=y_train,
        params=config.get("params", {}),
    )

    calibration_config = config.get("calibration", {}) or {}
    calibration_result = calibrate_model(
        model=training_result.model,
        X_calibration=X_calibration,
        y_calibration=y_calibration,
        method=calibration_config.get("method", "isotonic") if calibration_config.get("enabled", False) else "none",
    )
    final_model = calibration_result.calibrator or training_result.model
    probabilities = pd.Series(
        predict_positive_class_probability(final_model, X_test),
        index=test_raw.index,
        dtype="float64",
    )
    probabilities = clip_probabilities(probabilities, config)

    metric_config = config.get("metrics", {}) or {}
    evaluation = evaluate_binary_probabilities(
        y_true=y_test,
        probabilities=probabilities,
        threshold_min=float(metric_config.get("threshold_min", 0.40)),
        threshold_max=float(metric_config.get("threshold_max", 0.60)),
        threshold_step=float(metric_config.get("threshold_step", 0.01)),
        bucket_edges=metric_config.get("confidence_bucket_edges"),
        probability_label="walk_forward_probability",
    )

    window_run_id = f"{run_id}_{window.test_year}"
    model_outcomes = build_backtest_model_outcomes(
        feature_df=test_raw,
        probabilities=probabilities,
        model_config=config,
        prediction_run_id=window_run_id,
        prediction_timestamp=timestamp,
    )
    model_outcomes["walk_forward_test_year"] = window.test_year

    joined = model_outcomes.merge(
        market,
        on=["fight_id", "market_key", "outcome_join_key"],
        how="inner",
        suffixes=("_model", "_market"),
    )
    joined["walk_forward_test_year"] = window.test_year

    candidates = apply_filters(joined, args)
    scored = score_bets(candidates, args)
    scored["walk_forward_test_year"] = window.test_year

    backtest_id = f"{run_id}_{window.test_year}"
    summary = summarize(scored, args, backtest_id)
    summary.update(
        {
            "mode": "walk_forward",
            "test_year": window.test_year,
            "train_start_date": window.train_start_date,
            "train_end_date": window.train_end_date,
            "calibration_start_date": window.calibration_start_date,
            "calibration_end_date": window.calibration_end_date,
            "test_start_date": window.test_start_date,
            "test_end_date": window.test_end_date,
            "train_rows": int(len(train_raw)),
            "train_rows_after_symmetry": int(len(train_augmented)),
            "calibration_rows": int(len(calibration_raw)),
            "test_rows": int(len(test_raw)),
            "historical_model_outcome_rows": int(len(model_outcomes)),
            "joined_rows": int(len(joined)),
            "accuracy": float(evaluation.metrics["accuracy"]),
            "log_loss": float(evaluation.metrics["log_loss"]),
            "roc_auc": float(evaluation.metrics["roc_auc"]),
            "brier_score": float(evaluation.metrics["brier_score"]),
            "best_threshold": float(evaluation.best_threshold),
            "calibration_method": calibration_result.method,
            "calibration_rows_used": int(calibration_result.n_calibration_rows),
        }
    )

    metrics_row = {
        key: summary[key]
        for key in [
            "test_year",
            "train_start_date",
            "train_end_date",
            "calibration_start_date",
            "calibration_end_date",
            "test_start_date",
            "test_end_date",
            "train_rows",
            "train_rows_after_symmetry",
            "calibration_rows",
            "test_rows",
            "accuracy",
            "log_loss",
            "roc_auc",
            "brier_score",
            "best_threshold",
            "calibration_method",
        ]
    }

    return summary, scored, model_outcomes, pd.DataFrame([metrics_row])


def combined_summary(scored: pd.DataFrame, args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    """Build combined summary across all walk-forward test years."""

    summary = summarize(scored, args, run_id)
    summary.update(
        {
            "mode": "walk_forward",
            "first_test_year": int(args.first_test_year),
            "last_test_year": int(args.last_test_year),
            "train_start_date": str(args.train_start_date),
            "test_years": sorted(pd.to_numeric(scored.get("walk_forward_test_year"), errors="coerce").dropna().astype(int).unique().tolist()) if not scored.empty else [],
        }
    )
    return summary


def main() -> None:
    args = parse_args()
    ensure_data_dirs()

    run_time = datetime.now(timezone.utc)
    timestamp = run_time.isoformat()
    config_path = Path(args.model_config_path)
    config = load_config(config_path)
    args.model_id = str(config["model_id"])

    run_id = f"{args.model_id}_{args.market_key}_walk_forward_{run_time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    normal_feature_view_path = Path(args.feature_view_path)
    feature_df, feature_columns = load_feature_view(normal_feature_view_path, config)
    training_feature_df = load_prebuilt_flipped_training_view(
        normal_feature_view_path=normal_feature_view_path,
        config=config,
        expected_feature_columns=feature_columns,
    )
    market_raw = pd.read_parquet(args.historical_market_path)
    market_raw = market_raw[market_raw["market_key"].astype(str).str.lower() == args.market_key.lower()].copy()
    market = standardize_market(market_raw)

    windows = build_windows(args)
    yearly_summaries: list[dict[str, Any]] = []
    scored_frames: list[pd.DataFrame] = []
    outcome_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []

    for window in windows:
        print("=" * 80)
        print(f"WALK-FORWARD TEST YEAR {window.test_year}")
        print(json.dumps(asdict(window), indent=2))
        summary, scored, model_outcomes, metrics = run_one_window(
            window=window,
            feature_df=feature_df,
            training_feature_df=training_feature_df,
            market=market,
            feature_columns=feature_columns,
            config=config,
            args=args,
            run_id=run_id,
            timestamp=timestamp,
        )
        yearly_summaries.append(summary)
        scored_frames.append(scored)
        outcome_frames.append(model_outcomes)
        metric_frames.append(metrics)
        print(json.dumps(summary, indent=2, default=str))

    all_scored = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
    all_model_outcomes = pd.concat(outcome_frames, ignore_index=True) if outcome_frames else pd.DataFrame()
    yearly_results = pd.DataFrame(yearly_summaries)
    model_metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    buckets = bucket_summary(all_scored)
    summary = combined_summary(all_scored, args, run_id)
    summary.update(
        {
            "model_config_path": str(config_path),
            "feature_view_path": str(args.feature_view_path),
            "historical_market_path": str(args.historical_market_path),
            "feature_count": int(len(feature_columns)),
            "created_at_utc": timestamp,
        }
    )

    config_payload = vars(args) | {
        "run_id": run_id,
        "created_at_utc": timestamp,
        "mode": "walk_forward",
        "windows": [asdict(window) for window in windows],
    }

    (output_dir / "walk_forward_config.json").write_text(json.dumps(config_payload, indent=2, default=str), encoding="utf-8")
    (output_dir / "walk_forward_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    yearly_results.to_parquet(output_dir / "walk_forward_yearly_results.parquet", index=False)
    model_metrics.to_parquet(output_dir / "walk_forward_model_metrics.parquet", index=False)
    all_model_outcomes.to_parquet(output_dir / "walk_forward_model_outcomes.parquet", index=False)
    all_scored.to_parquet(output_dir / "walk_forward_bets.parquet", index=False)
    buckets.to_parquet(output_dir / "walk_forward_bucket_summary.parquet", index=False)

    registry_path = Path(args.output_root) / "walk_forward_registry.parquet"
    registry_row = pd.DataFrame([summary | {"output_dir": str(output_dir)}])
    registry = pd.concat([pd.read_parquet(registry_path), registry_row], ignore_index=True) if registry_path.exists() else registry_row
    registry.to_parquet(registry_path, index=False)

    print("=" * 80)
    print("WALK-FORWARD BACKTEST V2 SUMMARY")
    print("=" * 80)
    print("Run ID:", run_id)
    print("Output dir:", output_dir)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
