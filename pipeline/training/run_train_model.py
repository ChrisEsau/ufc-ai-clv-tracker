"""Generic UFC model training runner.

Run from repo root:

    python -m pipeline.training.run_train_model \
        --config configs/models/moneyline_xgb_base.yaml

The model config is the single source of truth for:
- feature sources
- explicit feature columns
- algorithm and parameters
- symmetry behavior
- temporal split dates
- calibration method
- metric settings
- artifact output paths
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from pipeline.features.registry_feature_builder import apply_registry_feature_definitions
from pipeline.training.calibration import calibrate_model, predict_positive_class_probability
from pipeline.training.feature_selection import (
    load_model_config,
    resolve_features_from_model_config,
)
from pipeline.training.metrics import evaluate_binary_probabilities
from pipeline.training.model_training import train_model
from pipeline.training.symmetry import apply_symmetry_augmentation
from pipeline.training.temporal_split import (
    build_temporal_train_calibration_test_split,
    build_temporal_train_test_split,
)

DEFAULT_CONFIG_PATH = "configs/models/moneyline_xgb_base.yaml"


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_model_config(config_path)

    print("=" * 80)
    print("TRAIN UFC MODEL")
    print("=" * 80)
    print(f"Config path : {config_path}")
    print(f"Model ID    : {config['model_id']}")
    print(f"Algorithm   : {config['algorithm']}")

    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_df = load_training_feature_dataframe(config)
    print(f"Feature dataframe shape: {feature_df.shape}")

    feature_columns = resolve_features_from_model_config(
        df=feature_df,
        model_config=config,
    )
    print(f"Resolved feature count: {len(feature_columns)}")

    model_df = maybe_apply_symmetry(
        df=feature_df,
        feature_columns=feature_columns,
        config=config,
    )
    print(f"Model dataframe shape: {model_df.shape}")

    split = build_split(
        df=model_df,
        feature_columns=feature_columns,
        config=config,
    )

    training_result = train_model(
        algorithm=config["algorithm"],
        X_train=split.X_train,
        y_train=split.y_train,
        params=config.get("params", {}),
    )
    print("Model training complete.")

    raw_test_probabilities = predict_positive_class_probability(
        training_result.model,
        split.X_test,
    )

    calibration_result = maybe_calibrate(
        model=training_result.model,
        split=split,
        config=config,
    )

    final_model = calibration_result.calibrator or training_result.model
    final_test_probabilities = predict_positive_class_probability(final_model, split.X_test)

    metric_config = config.get("metrics", {})
    evaluation = evaluate_binary_probabilities(
        y_true=split.y_test,
        probabilities=final_test_probabilities,
        threshold_min=float(metric_config.get("threshold_min", 0.40)),
        threshold_max=float(metric_config.get("threshold_max", 0.60)),
        threshold_step=float(metric_config.get("threshold_step", 0.01)),
        bucket_edges=metric_config.get("confidence_bucket_edges"),
        probability_label="calibrated_probability",
    )

    raw_evaluation = evaluate_binary_probabilities(
        y_true=split.y_test,
        probabilities=raw_test_probabilities,
        threshold_min=float(metric_config.get("threshold_min", 0.40)),
        threshold_max=float(metric_config.get("threshold_max", 0.60)),
        threshold_step=float(metric_config.get("threshold_step", 0.01)),
        bucket_edges=metric_config.get("confidence_bucket_edges"),
        probability_label="raw_probability",
    )

    save_artifacts(
        output_dir=output_dir,
        config_path=config_path,
        config=config,
        feature_columns=feature_columns,
        training_result=training_result,
        calibration_result=calibration_result,
        evaluation=evaluation,
        raw_evaluation=raw_evaluation,
        split=split,
    )

    print("=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Output dir        : {output_dir}")
    print(f"Train start date  : {getattr(split, 'train_start_date', None) or 'all'}")
    print(f"Train end date    : {getattr(split, 'train_end_date', '')}")
    print(f"Calibration end   : {getattr(split, 'calibration_end_date', 'n/a')}")
    print(f"Train rows        : {len(split.y_train)}")
    print(f"Calibration rows  : {getattr(split, 'y_calibration', pd.Series(dtype=int)).shape[0]}")
    print(f"Test rows         : {len(split.y_test)}")
    print(f"Feature count     : {len(feature_columns)}")
    print(f"Best threshold    : {evaluation.best_threshold:.2f}")
    print(f"Accuracy          : {evaluation.metrics['accuracy']:.4f}")
    print(f"ROC-AUC           : {evaluation.metrics['roc_auc']:.4f}")
    print(f"Log loss          : {evaluation.metrics['log_loss']:.4f}")
    print(f"Brier score       : {evaluation.metrics['brier_score']:.4f}")
    print("DONE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a UFC model from a YAML config.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to model config YAML.",
    )
    return parser.parse_args()


def load_training_feature_dataframe(config: dict[str, Any]) -> pd.DataFrame:
    """Load feature dataframe and materialize selected registry-defined features.

    Training may be launched before a feature-view artifact has been rebuilt.
    To keep Model Lab experiments runnable, selected feature columns that are
    defined in ``feature_registry.yaml`` are materialized in-memory before the
    feature contract is validated.
    """

    data_config = config.get("data", {})
    path = data_config.get("rolling_features_path")
    if not path:
        raise ValueError("Model config data section must define rolling_features_path")

    feature_path = Path(path)
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature warehouse not found: {feature_path}")

    feature_df = pd.read_parquet(feature_path)
    selected_features = (config.get("features") or {}).get("feature_columns") or []
    build_result = apply_registry_feature_definitions(
        feature_df,
        selected_features=selected_features,
        allowed_statuses={"active", "draft"},
        overwrite_existing=True,
    )

    if build_result.generated_columns:
        print(
            "Registry features materialized: "
            f"{len(build_result.generated_columns)} ({build_result.generated_columns})"
        )
    selected_missing = {
        feature_id: missing
        for feature_id, missing in build_result.missing_inputs.items()
        if feature_id in set(str(item) for item in selected_features)
    }
    if selected_missing:
        print(f"Registry features with missing inputs: {selected_missing}")

    return build_result.dataframe


def maybe_apply_symmetry(
    df: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Apply symmetry augmentation if enabled in the model config."""
    symmetry_config = config.get("symmetry", {})
    if not symmetry_config.get("enabled", False):
        return df

    mode = str(symmetry_config.get("mode", "flip_all")).strip().lower()
    target_col = config.get("data", {}).get("target_column", "target")
    date_col = config.get("data", {}).get("date_column", "date")

    if mode == "flip_all":
        return apply_symmetry_augmentation(
            df=df,
            feature_columns=feature_columns,
            target_col=target_col,
            date_col=date_col,
        )

    if mode == "explicit":
        return apply_symmetry_augmentation(
            df=df,
            feature_columns=feature_columns,
            target_col=target_col,
            date_col=date_col,
            flip_feature_columns=symmetry_config.get("flip_features", []),
            preserve_feature_columns=symmetry_config.get("preserve_features", []),
        )

    raise ValueError(f"Unsupported symmetry mode: {mode}")


def _optional_config_string(value: Any) -> str | None:
    """Return a stripped string value, treating blank config fields as None."""

    normalized = str(value or "").strip()
    return normalized or None


def build_split(
    df: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
) -> Any:
    """Build temporal split based on model config."""
    split_config = config.get("split", {})
    data_config = config.get("data", {})
    mode = str(split_config.get("mode", "train_calibration_test")).strip().lower()
    train_start_date = _optional_config_string(split_config.get("train_start_date"))

    if mode == "train_calibration_test":
        return build_temporal_train_calibration_test_split(
            df=df,
            feature_columns=feature_columns,
            train_start_date=train_start_date,
            train_end_date=split_config["train_end_date"],
            calibration_end_date=split_config["calibration_end_date"],
            target_col=data_config.get("target_column", "target"),
            date_col=data_config.get("date_column", "date"),
        )

    if mode == "train_test":
        return build_temporal_train_test_split(
            df=df,
            feature_columns=feature_columns,
            train_start_date=train_start_date,
            train_end_date=split_config["train_end_date"],
            target_col=data_config.get("target_column", "target"),
            date_col=data_config.get("date_column", "date"),
        )

    raise ValueError(f"Unsupported split mode: {mode}")


def maybe_calibrate(model: Any, split: Any, config: dict[str, Any]) -> Any:
    """Fit calibration layer if enabled, otherwise pass raw probabilities through."""
    calibration_config = config.get("calibration", {})
    if not calibration_config.get("enabled", False):
        return calibrate_model(
            model=model,
            X_calibration=split.X_test,
            y_calibration=split.y_test,
            method="none",
        )

    if hasattr(split, "X_calibration") and hasattr(split, "y_calibration"):
        return calibrate_model(
            model=model,
            X_calibration=split.X_calibration,
            y_calibration=split.y_calibration,
            method=calibration_config.get("method", "isotonic"),
        )

    # Backward-compatible train/test mode. Prefer train_calibration_test for production.
    return calibrate_model(
        model=model,
        X_calibration=split.X_test,
        y_calibration=split.y_test,
        method=calibration_config.get("method", "isotonic"),
    )


def save_artifacts(
    output_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    feature_columns: list[str],
    training_result: Any,
    calibration_result: Any,
    evaluation: Any,
    raw_evaluation: Any,
    split: Any,
) -> None:
    """Save model artifacts and evaluation outputs."""
    artifact_config = config.get("artifacts", {})

    if artifact_config.get("save_raw_model", True):
        joblib.dump(training_result.model, output_dir / "raw_model.joblib")

    if artifact_config.get("save_calibrated_model", True) and calibration_result.calibrator is not None:
        joblib.dump(calibration_result.calibrator, output_dir / "calibrated_model.joblib")

    if artifact_config.get("save_feature_columns", True):
        joblib.dump(feature_columns, output_dir / "feature_columns.joblib")
        (output_dir / "feature_columns.json").write_text(
            json.dumps(feature_columns, indent=2),
            encoding="utf-8",
        )

    if artifact_config.get("save_metrics", True):
        metrics_payload = {
            "model_id": config["model_id"],
            "raw_metrics": raw_evaluation.metrics,
            "final_metrics": evaluation.metrics,
            "best_threshold": evaluation.best_threshold,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_start_date": getattr(split, "train_start_date", None),
            "train_end_date": getattr(split, "train_end_date", None),
            "calibration_end_date": getattr(split, "calibration_end_date", None),
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics_payload, indent=2),
            encoding="utf-8",
        )

    if artifact_config.get("save_threshold_sweep", True):
        evaluation.threshold_sweep.to_parquet(output_dir / "threshold_sweep.parquet", index=False)
        raw_evaluation.threshold_sweep.to_parquet(output_dir / "raw_threshold_sweep.parquet", index=False)

    if artifact_config.get("save_confidence_buckets", True):
        evaluation.confidence_buckets.to_parquet(output_dir / "confidence_buckets.parquet", index=False)
        raw_evaluation.confidence_buckets.to_parquet(output_dir / "raw_confidence_buckets.parquet", index=False)

    if artifact_config.get("save_model_card", True):
        model_card = build_model_card(
            config_path=config_path,
            config=config,
            feature_columns=feature_columns,
            training_result=training_result,
            calibration_result=calibration_result,
            evaluation=evaluation,
            split=split,
        )
        (output_dir / "model_card.yaml").write_text(
            yaml.safe_dump(model_card, sort_keys=False),
            encoding="utf-8",
        )


def build_model_card(
    config_path: Path,
    config: dict[str, Any],
    feature_columns: list[str],
    training_result: Any,
    calibration_result: Any,
    evaluation: Any,
    split: Any,
) -> dict[str, Any]:
    """Build a compact model card for tracking trained artifacts."""
    return {
        "model_id": config["model_id"],
        "model_family": config.get("model_family"),
        "artifact_name": config.get("artifact_name"),
        "algorithm": training_result.algorithm,
        "config_path": str(config_path),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": config.get("data", {}),
        "split": {
            "mode": config.get("split", {}).get("mode"),
            "train_start_date": getattr(split, "train_start_date", None),
            "train_end_date": getattr(split, "train_end_date", None),
            "calibration_end_date": getattr(split, "calibration_end_date", None),
            "train_rows": int(len(split.y_train)),
            "calibration_rows": int(getattr(split, "y_calibration", pd.Series(dtype=int)).shape[0]),
            "test_rows": int(len(split.y_test)),
        },
        "features": {
            "feature_count": len(feature_columns),
            "expected_feature_count": config.get("features", {}).get("expected_feature_count"),
        },
        "symmetry": config.get("symmetry", {}),
        "calibration": {
            "enabled": config.get("calibration", {}).get("enabled", False),
            "method": calibration_result.method,
            "calibration_rows": calibration_result.n_calibration_rows,
        },
        "params": training_result.params,
        "metrics": evaluation.metrics,
        "best_threshold": evaluation.best_threshold,
    }


if __name__ == "__main__":
    main()
