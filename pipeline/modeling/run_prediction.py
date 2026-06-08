from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from pipeline.common.paths import PREDICTIONS_DIR
from pipeline.modeling.model_config import get_model_id, get_prediction_config, load_model_config
from pipeline.modeling.model_loader import load_model_bundle
from pipeline.modeling.model_registry import (
    get_model_entry,
    load_model_registry,
    resolve_selected_model_id,
)
from pipeline.modeling.prediction_adapter import run_prediction_adapter
from pipeline.prediction.live_feature_builder import (
    build_live_model_features,
    write_live_feature_outputs,
)


DEFAULT_MODEL_FAMILY = "moneyline"
DEFAULT_LIVE_FEATURE_OUTPUT_PATH = PREDICTIONS_DIR / "live_model_features.parquet"
DEFAULT_MODEL_OUTCOMES_PATH = PREDICTIONS_DIR / "model_outcomes.parquet"


class PredictionRunnerError(RuntimeError):
    """Raised when the Prediction V2 runner fails."""



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UFC Prediction V2 and write outcome-level predictions.",
    )
    parser.add_argument(
        "--model-family",
        default=DEFAULT_MODEL_FAMILY,
        help="Model family to resolve from the registry when --model-id is not supplied.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Explicit model ID. Overrides UFC_MODEL_ID and registry active model.",
    )
    parser.add_argument(
        "--registry-path",
        default="configs/models/model_registry.yaml",
        help="Path to model registry YAML.",
    )
    parser.add_argument(
        "--live-feature-output-path",
        default=str(DEFAULT_LIVE_FEATURE_OUTPUT_PATH),
        help="Optional path to persist live model-ready features.",
    )
    parser.add_argument(
        "--model-outcomes-path",
        default=str(DEFAULT_MODEL_OUTCOMES_PATH),
        help="Path to write canonical model outcome predictions.",
    )
    parser.add_argument(
        "--prefer-raw-model",
        action="store_true",
        help="Use raw_model.joblib when available instead of preferring calibrated_model.joblib.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    prediction_run_id = _make_prediction_run_id()
    prediction_timestamp = datetime.now(timezone.utc).isoformat()

    print("=" * 80)
    print("UFC PREDICTION V2")
    print("=" * 80)
    print(f"Prediction run ID: {prediction_run_id}")

    registry = load_model_registry(args.registry_path)
    model_id = resolve_selected_model_id(
        model_family=args.model_family,
        registry=registry,
        model_id=args.model_id,
    )
    model_entry = get_model_entry(model_id, registry)
    config_path = Path(model_entry["config_path"])

    print(f"Model family: {args.model_family}")
    print(f"Model ID: {model_id}")
    print(f"Config path: {config_path}")

    model_config = load_model_config(config_path, require_prediction=True)
    _validate_model_id_match(model_id=model_id, model_config=model_config)

    model_bundle = load_model_bundle(
        model_config,
        prefer_calibrated=not args.prefer_raw_model,
    )

    print(f"Artifact dir: {model_bundle.artifact_dir}")
    print(f"Model artifact: {model_bundle.model_artifact_path}")
    print(f"Uses calibrated model: {model_bundle.uses_calibrated_model}")
    print(f"Feature count: {len(model_bundle.feature_columns)}")

    live_result = build_live_model_features(
        feature_columns=model_bundle.feature_columns,
    )
    write_live_feature_outputs(
        live_result,
        live_feature_output_path=args.live_feature_output_path,
    )

    print(f"Live feature rows: {len(live_result.live_feature_df)}")
    print(f"Live feature output: {args.live_feature_output_path}")

    adapter_result = run_prediction_adapter(
        model_bundle=model_bundle,
        model_config=model_config,
        live_feature_df=live_result.live_feature_df,
        prediction_run_id=prediction_run_id,
        prediction_timestamp=prediction_timestamp,
    )

    model_outcomes_path = Path(args.model_outcomes_path)
    model_outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_result.outcome_df.to_parquet(model_outcomes_path, index=False)

    _write_model_scoped_outputs(
        outcome_df=adapter_result.outcome_df,
        model_config=model_config,
    )

    print(f"Outcome rows: {len(adapter_result.outcome_df)}")
    print(f"Model outcomes output: {model_outcomes_path}")
    print("Prediction V2 complete.")



def _write_model_scoped_outputs(*, outcome_df, model_config: dict) -> None:
    prediction_config = get_prediction_config(model_config)
    output_config = prediction_config.get("output", {}) or {}
    path_template = output_config.get("model_scoped_path_template")

    if not path_template:
        return

    model_id = get_model_id(model_config)
    path = Path(str(path_template).format(model_id=model_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    outcome_df.to_parquet(path, index=False)
    print(f"Model-scoped outcomes output: {path}")



def _validate_model_id_match(*, model_id: str, model_config: dict) -> None:
    config_model_id = get_model_id(model_config)
    if config_model_id != model_id:
        raise PredictionRunnerError(
            f"Registry selected model_id '{model_id}' but config has model_id '{config_model_id}'."
        )



def _make_prediction_run_id() -> str:
    return datetime.now(timezone.utc).strftime("pred_%Y%m%d_%H%M%S")



if __name__ == "__main__":
    main()
