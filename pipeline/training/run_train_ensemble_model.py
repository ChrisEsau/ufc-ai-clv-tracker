"""Train configurable market-aware ensemble models.

Run from repo root:

    python -m pipeline.training.run_train_ensemble_model \
        --config configs/models/moneyline_xgboost_v13_ensemble_top60.yaml

This runner is intentionally separate from pipeline.training.run_train_model so
the current single-model production trainer remains untouched.

The trainer supports:
- any number of child members
- separate feature sets per child
- favorite and dog perspective feature prefixes
- optional market features per child
- weighted group-level probability averaging
- member artifacts saved under members/<member_id>/
- parent ensemble manifest and group evaluation artifacts
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from pipeline.training.calibration import (
    calibrate_model,
    predict_positive_class_probability,
)
from pipeline.training.metrics import evaluate_binary_probabilities
from pipeline.training.model_training import train_model
from pipeline.training.temporal_split import (
    build_temporal_train_calibration_test_split,
    build_temporal_train_test_split,
)

DEFAULT_CONFIG_PATH = "configs/models/moneyline_xgboost_v13_ensemble_top60.yaml"


@dataclass(frozen=True)
class MemberTrainingResult:
    member_id: str
    probability_group: str
    target_column: str
    artifact_dir: str
    feature_count: int
    train_rows: int
    calibration_rows: int
    test_rows: int
    best_threshold: float
    accuracy: float
    roc_auc: float
    log_loss: float
    brier_score: float
    weight: float
    uses_calibration: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a UFC ensemble model from YAML config.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to ensemble model config YAML.")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f) or {}

    if not isinstance(value, dict):
        raise ValueError(f"YAML file did not load to a dictionary: {path}")

    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def normalize_string_list(values: Any, *, label: str) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list.")

    normalized = [str(value) for value in values if value is not None]
    normalized = list(dict.fromkeys(normalized))

    if not normalized:
        raise ValueError(f"{label} resolved to an empty list.")

    return normalized


def nested_get(payload: dict[str, Any], dotted_key: str) -> Any:
    value: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Could not resolve nested config key: {dotted_key}")
        value = value[part]
    return value


def resolve_clean_fight_features(config: dict[str, Any]) -> list[str]:
    """Resolve clean fight features used before perspective prefixing."""

    feature_config = config.get("features") or {}

    if feature_config.get("clean_fight_features"):
        return normalize_string_list(
            feature_config["clean_fight_features"],
            label="features.clean_fight_features",
        )

    source_config_path = feature_config.get("clean_feature_source_config")
    source_key = feature_config.get("clean_feature_source_key", "features.feature_columns")

    if source_config_path:
        source_config = load_yaml(source_config_path)
        return normalize_string_list(
            nested_get(source_config, str(source_key)),
            label=f"{source_config_path}:{source_key}",
        )

    raise ValueError(
        "Ensemble config must define either features.clean_fight_features "
        "or features.clean_feature_source_config."
    )


def resolve_market_features(config: dict[str, Any]) -> list[str]:
    feature_config = config.get("features") or {}
    return normalize_string_list(
        feature_config.get("market_features", []),
        label="features.market_features",
    )


def resolve_member_features(
    *,
    member_config: dict[str, Any],
    clean_fight_features: list[str],
    market_features: list[str],
) -> list[str]:
    """Resolve concrete dataframe columns for one ensemble member."""

    if member_config.get("feature_columns"):
        return normalize_string_list(
            member_config["feature_columns"],
            label=f"{member_config.get('member_id')}.feature_columns",
        )

    perspective = str(member_config.get("perspective", "")).strip().lower()
    if perspective in {"favorite", "fav", "regular"}:
        prefix = "favpersp__"
    elif perspective in {"dog", "underdog"}:
        prefix = "dogpersp__"
    else:
        raise ValueError(
            f"Unsupported member perspective for {member_config.get('member_id')}: {perspective}"
        )

    top_n = member_config.get("top_n")
    selected_clean_features = clean_fight_features
    if top_n is not None:
        selected_clean_features = clean_fight_features[: int(top_n)]

    columns = [f"{prefix}{feature}" for feature in selected_clean_features]

    if bool(member_config.get("include_market_features", False)):
        columns.extend(market_features)

    return list(dict.fromkeys(columns))


def validate_required_columns(
    *,
    df: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def build_split(
    *,
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    config: dict[str, Any],
) -> Any:
    split_config = config.get("split", {}) or {}
    mode = str(split_config.get("mode", "train_calibration_test")).strip().lower()
    train_start_date = str(split_config.get("train_start_date") or "").strip() or None
    date_column = str((config.get("data") or {}).get("date_column", "date"))

    if mode == "train_calibration_test":
        return build_temporal_train_calibration_test_split(
            df=df,
            feature_columns=feature_columns,
            train_start_date=train_start_date,
            train_end_date=split_config["train_end_date"],
            calibration_end_date=split_config["calibration_end_date"],
            target_col=target_column,
            date_col=date_column,
        )

    if mode == "train_test":
        return build_temporal_train_test_split(
            df=df,
            feature_columns=feature_columns,
            train_start_date=train_start_date,
            train_end_date=split_config["train_end_date"],
            target_col=target_column,
            date_col=date_column,
        )

    raise ValueError(f"Unsupported ensemble split mode: {mode}")


def fit_calibrator_if_enabled(
    *,
    model: Any,
    split: Any,
    config: dict[str, Any],
) -> Any:
    calibration_config = config.get("calibration", {}) or {}

    if not bool(calibration_config.get("enabled", False)):
        return calibrate_model(
            model=model,
            X_calibration=split.X_test,
            y_calibration=split.y_test,
            method="none",
            config=None,
        )

    method = str(calibration_config.get("method", "isotonic"))

    if hasattr(split, "X_calibration") and hasattr(split, "y_calibration"):
        return calibrate_model(
            model=model,
            X_calibration=split.X_calibration,
            y_calibration=split.y_calibration,
            method=method,
            config=calibration_config,
        )

    return calibrate_model(
        model=model,
        X_calibration=split.X_test,
        y_calibration=split.y_test,
        method=method,
        config=calibration_config,
    )


def metric_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("metrics", {}) or {}


def evaluate_probabilities(
    *,
    y_true: pd.Series,
    probabilities: np.ndarray,
    config: dict[str, Any],
    probability_label: str,
) -> Any:
    metrics_config = metric_config(config)
    return evaluate_binary_probabilities(
        y_true=y_true,
        probabilities=probabilities,
        threshold_min=float(metrics_config.get("threshold_min", 0.40)),
        threshold_max=float(metrics_config.get("threshold_max", 0.60)),
        threshold_step=float(metrics_config.get("threshold_step", 0.01)),
        bucket_edges=metrics_config.get("confidence_bucket_edges"),
        probability_label=probability_label,
    )


def save_member_artifacts(
    *,
    member_dir: Path,
    parent_config_path: Path,
    parent_config: dict[str, Any],
    member_config: dict[str, Any],
    feature_columns: list[str],
    training_result: Any,
    calibration_result: Any,
    evaluation: Any,
    raw_evaluation: Any,
    split: Any,
) -> None:
    member_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(training_result.model, member_dir / "raw_model.joblib")
    if calibration_result.calibrator is not None:
        joblib.dump(calibration_result.calibrator, member_dir / "calibrated_model.joblib")

    joblib.dump(feature_columns, member_dir / "feature_columns.joblib")
    write_json(member_dir / "feature_columns.json", feature_columns)

    training_metadata = {
        "early_stopping_enabled": bool(getattr(training_result, "early_stopping_enabled", False)),
        "early_stopping_rounds": getattr(training_result, "early_stopping_rounds", None),
        "early_stopping_metric": getattr(training_result, "early_stopping_metric", None),
        "best_iteration": getattr(training_result, "best_iteration", None),
        "best_score": getattr(training_result, "best_score", None),
    }

    metrics_payload = {
        "model_id": parent_config.get("model_id"),
        "member_id": member_config.get("member_id"),
        "probability_group": member_config.get("probability_group"),
        "target_column": member_config.get("target_column"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(parent_config_path),
        "feature_count": len(feature_columns),
        "best_threshold": evaluation.best_threshold,
        "metrics": evaluation.metrics,
        "raw_metrics": raw_evaluation.metrics,
        "training": training_metadata,
        "train_rows": int(len(split.y_train)),
        "calibration_rows": int(getattr(split, "y_calibration", pd.Series(dtype=int)).shape[0]),
        "test_rows": int(len(split.y_test)),
        "train_start_date": getattr(split, "train_start_date", None),
        "train_end_date": getattr(split, "train_end_date", None),
        "calibration_end_date": getattr(split, "calibration_end_date", None),
    }
    write_json(member_dir / "metrics.json", metrics_payload)

    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in evaluation.metrics.items()]
    ).to_csv(member_dir / "metrics.csv", index=False)

    evaluation.threshold_sweep.to_csv(member_dir / "threshold_sweep.csv", index=False)
    evaluation.threshold_sweep.to_parquet(member_dir / "threshold_sweep.parquet", index=False)
    raw_evaluation.threshold_sweep.to_csv(member_dir / "raw_threshold_sweep.csv", index=False)
    raw_evaluation.threshold_sweep.to_parquet(member_dir / "raw_threshold_sweep.parquet", index=False)

    evaluation.confidence_buckets.to_csv(member_dir / "confidence_buckets.csv", index=False)
    evaluation.confidence_buckets.to_parquet(member_dir / "confidence_buckets.parquet", index=False)
    raw_evaluation.confidence_buckets.to_csv(member_dir / "raw_confidence_buckets.csv", index=False)
    raw_evaluation.confidence_buckets.to_parquet(member_dir / "raw_confidence_buckets.parquet", index=False)

    model_card = {
        "model_id": parent_config.get("model_id"),
        "model_family": parent_config.get("model_family"),
        "artifact_name": parent_config.get("artifact_name"),
        "algorithm": parent_config.get("algorithm"),
        "architecture": "ensemble_member",
        "member": member_config,
        "status": parent_config.get("status"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(parent_config_path),
        "artifacts_output_dir": str(member_dir),
        "feature_count": len(feature_columns),
        "data": parent_config.get("data"),
        "split": parent_config.get("split"),
        "calibration": parent_config.get("calibration"),
        "early_stopping": parent_config.get("early_stopping"),
        "training": training_metadata,
        "params": training_result.params,
        "metrics": evaluation.metrics,
        "raw_metrics": raw_evaluation.metrics,
        "best_threshold": evaluation.best_threshold,
    }
    write_json(member_dir / "model_card.json", model_card)
    (member_dir / "model_card.yaml").write_text(
        yaml.safe_dump(model_card, sort_keys=False),
        encoding="utf-8",
    )


def train_one_member(
    *,
    df: pd.DataFrame,
    config_path: Path,
    config: dict[str, Any],
    member_config: dict[str, Any],
    clean_fight_features: list[str],
    market_features: list[str],
    output_dir: Path,
) -> tuple[MemberTrainingResult, pd.DataFrame]:
    member_id = str(member_config["member_id"])
    probability_group = str(member_config["probability_group"])
    target_column = str(member_config["target_column"])
    weight = float(member_config.get("weight", 1.0))

    feature_columns = resolve_member_features(
        member_config=member_config,
        clean_fight_features=clean_fight_features,
        market_features=market_features,
    )

    validate_required_columns(
        df=df,
        columns=["_ensemble_row_id", "fight_id", "date", target_column, *feature_columns],
        label=member_id,
    )

    split = build_split(
        df=df,
        feature_columns=feature_columns,
        target_column=target_column,
        config=config,
    )

    early_stopping_config = config.get("early_stopping", {}) or {}
    early_stopping_enabled = bool(early_stopping_config.get("enabled", False))
    X_validation = getattr(split, "X_calibration", None) if early_stopping_enabled else None
    y_validation = getattr(split, "y_calibration", None) if early_stopping_enabled else None

    params = (config.get("params") or {}).copy()
    params.update(member_config.get("params") or {})
    if member_config.get("random_state") is not None:
        params["random_state"] = int(member_config["random_state"])

    training_result = train_model(
        algorithm=str(config["algorithm"]),
        X_train=split.X_train,
        y_train=split.y_train,
        params=params,
        X_validation=X_validation,
        y_validation=y_validation,
        early_stopping_config=early_stopping_config,
    )

    raw_test_probabilities = predict_positive_class_probability(training_result.model, split.X_test)
    calibration_result = fit_calibrator_if_enabled(
        model=training_result.model,
        split=split,
        config=config,
    )
    final_model = calibration_result.calibrator or training_result.model
    final_test_probabilities = predict_positive_class_probability(final_model, split.X_test)

    evaluation = evaluate_probabilities(
        y_true=split.y_test,
        probabilities=final_test_probabilities,
        config=config,
        probability_label="calibrated_probability",
    )
    raw_evaluation = evaluate_probabilities(
        y_true=split.y_test,
        probabilities=raw_test_probabilities,
        config=config,
        probability_label="raw_probability",
    )

    member_dir = output_dir / "members" / member_id
    save_member_artifacts(
        member_dir=member_dir,
        parent_config_path=config_path,
        parent_config=config,
        member_config=member_config,
        feature_columns=feature_columns,
        training_result=training_result,
        calibration_result=calibration_result,
        evaluation=evaluation,
        raw_evaluation=raw_evaluation,
        split=split,
    )

    # Avoid selecting target_column twice when it is already favorite_won or dog_won.
    # Duplicate dataframe column names break concat/reindexing later.
    prediction_columns = [
        "_ensemble_row_id",
        "fight_id",
        "date",
        "event_name",
        "favorite_fighter_name",
        "dog_fighter_name",
        "favorite_won",
        "dog_won",
    ]
    prediction_df = split.test_df[prediction_columns].copy()
    prediction_df["y_true"] = split.test_df[target_column].astype(int).to_numpy()
    prediction_df["model_id"] = config.get("model_id")
    prediction_df["member_id"] = member_id
    prediction_df["probability_group"] = probability_group
    prediction_df["target_column"] = target_column
    prediction_df["raw_probability"] = raw_test_probabilities
    prediction_df["calibrated_probability"] = final_test_probabilities
    prediction_df["member_probability"] = final_test_probabilities
    prediction_df["weight"] = weight
    prediction_df["feature_count"] = len(feature_columns)

    result = MemberTrainingResult(
        member_id=member_id,
        probability_group=probability_group,
        target_column=target_column,
        artifact_dir=str(member_dir),
        feature_count=len(feature_columns),
        train_rows=int(len(split.y_train)),
        calibration_rows=int(getattr(split, "y_calibration", pd.Series(dtype=int)).shape[0]),
        test_rows=int(len(split.y_test)),
        best_threshold=float(evaluation.best_threshold),
        accuracy=float(evaluation.metrics["accuracy"]),
        roc_auc=float(evaluation.metrics["roc_auc"]),
        log_loss=float(evaluation.metrics["log_loss"]),
        brier_score=float(evaluation.metrics["brier_score"]),
        weight=weight,
        uses_calibration=calibration_result.calibrator is not None,
    )

    return result, prediction_df


def combine_group_predictions(
    *,
    member_predictions: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build weighted group probability averages and evaluate each group."""

    combined_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    for group_name, group_df in member_predictions.groupby("probability_group"):
        tmp = group_df.copy()
        tmp["weighted_probability"] = tmp["member_probability"] * tmp["weight"]

        grouped = (
            tmp.groupby("_ensemble_row_id", as_index=False)
            .agg(
                fight_id=("fight_id", "first"),
                date=("date", "first"),
                event_name=("event_name", "first"),
                favorite_fighter_name=("favorite_fighter_name", "first"),
                dog_fighter_name=("dog_fighter_name", "first"),
                favorite_won=("favorite_won", "first"),
                dog_won=("dog_won", "first"),
                target_column=("target_column", "first"),
                y_true=("y_true", "first"),
                weighted_probability=("weighted_probability", "sum"),
                weight_sum=("weight", "sum"),
                member_count=("member_id", "nunique"),
            )
            .copy()
        )

        grouped["probability_group"] = group_name
        grouped["ensemble_probability"] = grouped["weighted_probability"] / grouped["weight_sum"]

        evaluation = evaluate_probabilities(
            y_true=grouped["y_true"],
            probabilities=grouped["ensemble_probability"].to_numpy(dtype=float),
            config=config,
            probability_label=f"{group_name}_ensemble_probability",
        )

        row = {
            "model_id": config.get("model_id"),
            "probability_group": group_name,
            "target_column": grouped["target_column"].iloc[0],
            "member_count": int(grouped["member_count"].max()),
            "row_count": int(len(grouped)),
            "best_threshold": float(evaluation.best_threshold),
            **evaluation.metrics,
        }
        metric_rows.append(row)
        combined_frames.append(grouped)

    combined_predictions = pd.concat(combined_frames, ignore_index=True)
    group_metrics = pd.DataFrame(metric_rows)

    return combined_predictions, group_metrics


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml(config_path)

    print("=" * 80)
    print("TRAIN UFC ENSEMBLE MODEL")
    print("=" * 80)
    print(f"Config path : {config_path}")
    print(f"Model ID    : {config['model_id']}")
    print(f"Algorithm   : {config['algorithm']}")

    data_config = config.get("data") or {}
    feature_view_path = Path(data_config["feature_view_path"])
    if not feature_view_path.exists():
        raise FileNotFoundError(f"Feature view not found: {feature_view_path}")

    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(feature_view_path).reset_index(drop=True)
    df["_ensemble_row_id"] = np.arange(len(df), dtype=int)

    print(f"Feature view path : {feature_view_path}")
    print(f"Feature view shape: {df.shape}")
    print(f"Output dir        : {output_dir}")

    clean_fight_features = resolve_clean_fight_features(config)
    market_features = resolve_market_features(config)

    print(f"Clean fight features: {len(clean_fight_features)}")
    print(f"Market features     : {len(market_features)}")

    members = (config.get("ensemble") or {}).get("members") or []
    if not members:
        raise ValueError("Ensemble config must define ensemble.members.")

    member_results: list[MemberTrainingResult] = []
    member_prediction_frames: list[pd.DataFrame] = []

    for index, member_config in enumerate(members, start=1):
        print()
        print("-" * 80)
        print(f"TRAIN MEMBER {index}/{len(members)}: {member_config['member_id']}")
        print("-" * 80)

        result, prediction_df = train_one_member(
            df=df,
            config_path=config_path,
            config=config,
            member_config=member_config,
            clean_fight_features=clean_fight_features,
            market_features=market_features,
            output_dir=output_dir,
        )

        member_results.append(result)
        member_prediction_frames.append(prediction_df)

        print(f"Member dir       : {result.artifact_dir}")
        print(f"Target column    : {result.target_column}")
        print(f"Probability group: {result.probability_group}")
        print(f"Feature count    : {result.feature_count}")
        print(f"Train rows       : {result.train_rows}")
        print(f"Calibration rows : {result.calibration_rows}")
        print(f"Test rows        : {result.test_rows}")
        print(f"Accuracy         : {result.accuracy:.4f}")
        print(f"ROC-AUC          : {result.roc_auc:.4f}")
        print(f"Log loss         : {result.log_loss:.4f}")
        print(f"Brier score      : {result.brier_score:.4f}")

    member_summary = pd.DataFrame([asdict(result) for result in member_results])
    member_predictions = pd.concat(member_prediction_frames, ignore_index=True)

    combined_predictions, group_metrics = combine_group_predictions(
        member_predictions=member_predictions,
        config=config,
    )

    member_summary.to_csv(output_dir / "member_summary.csv", index=False)
    member_predictions.to_csv(output_dir / "member_test_predictions.csv", index=False)
    member_predictions.to_parquet(output_dir / "member_test_predictions.parquet", index=False)

    combined_predictions.to_csv(output_dir / "ensemble_group_test_predictions.csv", index=False)
    combined_predictions.to_parquet(output_dir / "ensemble_group_test_predictions.parquet", index=False)

    group_metrics.to_csv(output_dir / "ensemble_group_metrics.csv", index=False)
    write_json(output_dir / "ensemble_group_metrics.json", group_metrics.to_dict(orient="records"))

    parent_feature_union = sorted(
        set(
            column
            for member_config in members
            for column in resolve_member_features(
                member_config=member_config,
                clean_fight_features=clean_fight_features,
                market_features=market_features,
            )
        )
    )
    write_json(output_dir / "feature_columns.json", parent_feature_union)
    joblib.dump(parent_feature_union, output_dir / "feature_columns.joblib")

    manifest = {
        "model_id": config.get("model_id"),
        "model_family": config.get("model_family"),
        "artifact_name": config.get("artifact_name"),
        "algorithm": config.get("algorithm"),
        "architecture": "market_aware_ensemble",
        "status": config.get("status"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "feature_view_path": str(feature_view_path),
        "output_dir": str(output_dir),
        "clean_fight_feature_count": len(clean_fight_features),
        "market_feature_count": len(market_features),
        "member_count": len(members),
        "members": [asdict(result) for result in member_results],
        "group_metrics": group_metrics.to_dict(orient="records"),
        "ensemble": config.get("ensemble"),
        "decision_rule": config.get("decision_rule"),
        "split": config.get("split"),
        "calibration": config.get("calibration"),
        "params": config.get("params"),
    }
    write_json(output_dir / "ensemble_manifest.json", manifest)
    (output_dir / "training_config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("ENSEMBLE TRAINING SUMMARY")
    print("=" * 80)
    print(f"Output dir     : {output_dir}")
    print(f"Member count   : {len(member_results)}")
    print(f"Member summary : {output_dir / 'member_summary.csv'}")
    print(f"Manifest       : {output_dir / 'ensemble_manifest.json'}")
    print()
    print(group_metrics.to_string(index=False))
    print("DONE")


if __name__ == "__main__":
    main()
