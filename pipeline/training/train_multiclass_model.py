from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from pipeline.modeling.algorithms.xgboost_predictor import predict_class_probabilities
from pipeline.training.model_training import train_model
from pipeline.training.multiclass_metrics import evaluate_multiclass_probabilities
from pipeline.training.run_train_model import (
    build_shap_importance,
    build_split,
    load_training_feature_dataframe,
    maybe_apply_symmetry,
)
from pipeline.training.feature_selection import resolve_features_from_model_config


class MulticlassTrainingError(RuntimeError):
    """Raised when multiclass training configuration is invalid."""


def train_multiclass_from_config(*, config: dict[str, Any], config_path: Path) -> None:
    """Train a multiclass model from the standard model YAML contract."""

    prediction_config = config.get("prediction", {}) or {}
    class_labels = prediction_config.get("class_labels") or prediction_config.get("classes")
    if not isinstance(class_labels, list) or len(class_labels) < 3:
        raise MulticlassTrainingError("Multiclass training requires prediction.class_labels with at least 3 classes.")
    class_labels = [str(label) for label in class_labels]

    calibration_config = config.get("calibration", {}) or {}
    if calibration_config.get("enabled", False):
        raise MulticlassTrainingError(
            "Multiclass calibration is not supported yet. Set calibration.enabled: false."
        )

    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRAIN UFC MULTICLASS MODEL")
    print("=" * 80)
    print(f"Config path : {config_path}")
    print(f"Model ID    : {config['model_id']}")
    print(f"Algorithm   : {config['algorithm']}")
    print(f"Classes     : {class_labels}")

    feature_df = load_training_feature_dataframe(config)
    print(f"Feature dataframe shape: {feature_df.shape}")

    feature_columns = resolve_features_from_model_config(df=feature_df, model_config=config)
    print(f"Resolved feature count: {len(feature_columns)}")

    model_df = maybe_apply_symmetry(df=feature_df, feature_columns=feature_columns, config=config)
    print(f"Model dataframe shape: {model_df.shape}")

    split = build_split(df=model_df, feature_columns=feature_columns, config=config)
    _validate_multiclass_target(split=split, class_labels=class_labels)

    params = dict(config.get("params", {}) or {})
    params.setdefault("objective", "multi:softprob")
    params.setdefault("num_class", len(class_labels))
    params.setdefault("eval_metric", "mlogloss")

    training_result = train_model(
        algorithm=config["algorithm"],
        X_train=split.X_train,
        y_train=pd.to_numeric(split.y_train, errors="raise").astype(int),
        params=params,
    )
    print("Multiclass model training complete.")

    test_probabilities = predict_class_probabilities(training_result.model, split.X_test)
    evaluation = evaluate_multiclass_probabilities(
        y_true=split.y_test,
        probabilities=test_probabilities,
        class_labels=class_labels,
    )

    save_multiclass_artifacts(
        output_dir=output_dir,
        config_path=config_path,
        config=config,
        class_labels=class_labels,
        feature_columns=feature_columns,
        training_result=training_result,
        evaluation=evaluation,
        split=split,
    )

    print("=" * 80)
    print("MULTICLASS TRAINING SUMMARY")
    print("=" * 80)
    print(f"Output dir        : {output_dir}")
    print(f"Train start date  : {getattr(split, 'train_start_date', None) or 'all'}")
    print(f"Train end date    : {getattr(split, 'train_end_date', '')}")
    print(f"Calibration end   : {getattr(split, 'calibration_end_date', 'n/a')}")
    print(f"Train rows        : {len(split.y_train)}")
    print(f"Test rows         : {len(split.y_test)}")
    print(f"Feature count     : {len(feature_columns)}")
    print(f"Accuracy          : {evaluation.metrics['accuracy']:.4f}")
    print(f"Log loss          : {evaluation.metrics['log_loss']:.4f}")
    print(f"Macro F1          : {evaluation.metrics['macro_f1']:.4f}")
    print(f"Weighted F1       : {evaluation.metrics['weighted_f1']:.4f}")
    print("DONE")


def _validate_multiclass_target(*, split: Any, class_labels: list[str]) -> None:
    expected_classes = set(range(len(class_labels)))
    for name, y in [
        ("train", split.y_train),
        ("test", split.y_test),
        ("calibration", getattr(split, "y_calibration", pd.Series(dtype=int))),
    ]:
        if y is None or len(y) == 0:
            continue
        y_numeric = pd.to_numeric(y, errors="raise").astype(int)
        observed = set(y_numeric.unique().tolist())
        unexpected = sorted(observed - expected_classes)
        if unexpected:
            raise MulticlassTrainingError(f"{name} target contains unexpected classes: {unexpected}")
    train_classes = set(pd.to_numeric(split.y_train, errors="raise").astype(int).unique().tolist())
    if len(train_classes) < len(class_labels):
        missing = sorted(expected_classes - train_classes)
        raise MulticlassTrainingError(f"Training target is missing classes required by class_labels: {missing}")


def save_multiclass_artifacts(
    *,
    output_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    class_labels: list[str],
    feature_columns: list[str],
    training_result: Any,
    evaluation: Any,
    split: Any,
) -> None:
    artifact_config = config.get("artifacts", {}) or {}

    if artifact_config.get("save_raw_model", True):
        joblib.dump(training_result.model, output_dir / "raw_model.joblib")

    if artifact_config.get("save_feature_columns", True):
        joblib.dump(feature_columns, output_dir / "feature_columns.joblib")
        (output_dir / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")

    (output_dir / "class_labels.json").write_text(json.dumps(class_labels, indent=2), encoding="utf-8")

    if artifact_config.get("save_metrics", True):
        metrics_payload = {
            "model_id": config["model_id"],
            "metrics": evaluation.metrics,
            "class_labels": class_labels,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_start_date": getattr(split, "train_start_date", None),
            "train_end_date": getattr(split, "train_end_date", None),
            "calibration_end_date": getattr(split, "calibration_end_date", None),
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    evaluation.class_metrics.to_parquet(output_dir / "class_metrics.parquet", index=False)
    evaluation.confusion_matrix.to_parquet(output_dir / "confusion_matrix.parquet", index=False)

    if artifact_config.get("save_shap_importance", True):
        try:
            shap_df = build_shap_importance(
                model=training_result.model,
                X=split.X_test,
                feature_columns=feature_columns,
                max_rows=int(artifact_config.get("shap_max_rows", 1000)),
                random_state=int(config.get("params", {}).get("random_state", 42)),
            )
            shap_df.to_csv(output_dir / "shap_importance.csv", index=False)
            print(f"Saved SHAP importance: {output_dir / 'shap_importance.csv'}")
        except Exception as exc:  # noqa: BLE001 - SHAP should not fail the training run.
            print(f"WARNING: SHAP importance generation failed: {exc}")

    if artifact_config.get("save_model_card", True):
        model_card = {
            "model_id": config["model_id"],
            "model_family": config.get("model_family"),
            "market_key": config.get("market_key"),
            "artifact_name": config.get("artifact_name"),
            "algorithm": config.get("algorithm"),
            "config_path": str(config_path),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "prediction": {
                "format": "multiclass",
                "class_labels": class_labels,
            },
            "data": config.get("data", {}),
            "split": {
                **(config.get("split", {}) or {}),
                "train_rows": int(len(split.y_train)),
                "test_rows": int(len(split.y_test)),
            },
            "features": {
                "feature_count": int(len(feature_columns)),
                "expected_feature_count": (config.get("features", {}) or {}).get("expected_feature_count"),
            },
            "calibration": {"enabled": False, "reason": "multiclass calibration not supported yet"},
            "params": training_result.params,
            "metrics": evaluation.metrics,
        }
        (output_dir / "model_card.yaml").write_text(yaml.safe_dump(model_card, sort_keys=False), encoding="utf-8")
