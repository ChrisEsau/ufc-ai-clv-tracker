from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ModelConfigError(RuntimeError):
    """Raised when a model config is missing required contract fields."""


REQUIRED_TOP_LEVEL_FIELDS = [
    "model_id",
    "model_family",
    "algorithm",
    "features",
    "artifacts",
]

REQUIRED_FEATURE_FIELDS = [
    "feature_columns",
]

REQUIRED_ARTIFACT_FIELDS = [
    "output_dir",
]

# The current moneyline training config predates Prediction V2, so prediction
# fields are optional at load time. V2 runners can request strict prediction
# validation once the configs are migrated.
REQUIRED_PREDICTION_FIELDS = [
    "format",
    "market_key",
]



def load_model_config(
    config_path: str | Path,
    *,
    require_prediction: bool = False,
) -> dict[str, Any]:
    """Load and validate a model config YAML.

    Parameters
    ----------
    config_path:
        Path to the model config YAML.
    require_prediction:
        When True, require the V2 prediction section. This should be enabled by
        V2 prediction runners after model configs are migrated.

    Returns
    -------
    dict
        Parsed model configuration.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise ModelConfigError(
            f"Model config not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ModelConfigError(
            f"Model config must deserialize into a dictionary: {config_path}"
        )

    validate_model_config(
        config,
        config_path=config_path,
        require_prediction=require_prediction,
    )

    return config



def validate_model_config(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    require_prediction: bool = False,
) -> None:
    """Validate required model config contract fields."""

    context = f" in {config_path}" if config_path else ""

    missing_top_level = [
        field for field in REQUIRED_TOP_LEVEL_FIELDS
        if field not in config
    ]

    if missing_top_level:
        raise ModelConfigError(
            f"Model config missing required top-level fields{context}: "
            f"{missing_top_level}"
        )

    features = config.get("features")
    if not isinstance(features, dict):
        raise ModelConfigError(
            f"Model config 'features' section must be a dictionary{context}."
        )

    missing_feature_fields = [
        field for field in REQUIRED_FEATURE_FIELDS
        if field not in features
    ]

    if missing_feature_fields:
        raise ModelConfigError(
            f"Model config missing required feature fields{context}: "
            f"{missing_feature_fields}"
        )

    feature_columns = features.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ModelConfigError(
            f"Model config features.feature_columns must be a non-empty list{context}."
        )

    artifacts = config.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ModelConfigError(
            f"Model config 'artifacts' section must be a dictionary{context}."
        )

    missing_artifact_fields = [
        field for field in REQUIRED_ARTIFACT_FIELDS
        if field not in artifacts
    ]

    if missing_artifact_fields:
        raise ModelConfigError(
            f"Model config missing required artifact fields{context}: "
            f"{missing_artifact_fields}"
        )

    if require_prediction:
        prediction = config.get("prediction")
        if not isinstance(prediction, dict):
            raise ModelConfigError(
                f"Model config requires a V2 'prediction' section{context}."
            )

        missing_prediction_fields = [
            field for field in REQUIRED_PREDICTION_FIELDS
            if field not in prediction
        ]

        if missing_prediction_fields:
            raise ModelConfigError(
                f"Model config missing required prediction fields{context}: "
                f"{missing_prediction_fields}"
            )



def get_model_id(config: dict[str, Any]) -> str:
    """Return model_id from a validated config."""

    return str(config["model_id"])



def get_model_family(config: dict[str, Any]) -> str:
    """Return model_family from a validated config."""

    return str(config["model_family"])



def get_algorithm(config: dict[str, Any]) -> str:
    """Return algorithm from a validated config."""

    return str(config["algorithm"]).lower().strip()



def get_artifact_dir(config: dict[str, Any]) -> Path:
    """Return model artifact directory from config."""

    return Path(config["artifacts"]["output_dir"])



def get_feature_columns(config: dict[str, Any]) -> list[str]:
    """Return the explicit feature column contract from config."""

    return [str(column) for column in config["features"]["feature_columns"]]



def get_prediction_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return prediction config section, or an empty dict if not migrated yet."""

    prediction = config.get("prediction", {})
    if prediction is None:
        return {}
    if not isinstance(prediction, dict):
        raise ModelConfigError(
            "Model config 'prediction' section must be a dictionary when provided."
        )
    return prediction
