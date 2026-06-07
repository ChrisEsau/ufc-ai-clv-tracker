from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import yaml

from pipeline.modeling.model_config import (
    get_algorithm,
    get_artifact_dir,
    get_feature_columns,
    get_model_family,
    get_model_id,
)


class ModelLoaderError(RuntimeError):
    """Raised when a trained model bundle cannot be loaded."""


@dataclass(frozen=True)
class ModelBundle:
    """Loaded model artifacts needed by prediction V2."""

    model: Any
    model_id: str
    model_family: str
    algorithm: str
    artifact_dir: Path
    feature_columns: list[str]
    metrics: dict[str, Any] | None
    model_card: dict[str, Any] | None
    uses_calibrated_model: bool
    model_artifact_path: Path



def load_model_bundle(
    model_config: dict[str, Any],
    *,
    prefer_calibrated: bool = True,
) -> ModelBundle:
    """Load trained model artifacts from a validated model config.

    Parameters
    ----------
    model_config:
        Parsed model YAML config.
    prefer_calibrated:
        Prefer ``calibrated_model.joblib`` over ``raw_model.joblib`` when both
        are present. Production prediction should usually prefer calibrated.

    Returns
    -------
    ModelBundle
        Loaded model and metadata needed by prediction code.
    """

    artifact_dir = get_artifact_dir(model_config)

    if not artifact_dir.exists():
        raise ModelLoaderError(
            f"Model artifact directory not found: {artifact_dir}"
        )

    model_artifact_path, uses_calibrated = _resolve_model_artifact_path(
        artifact_dir=artifact_dir,
        prefer_calibrated=prefer_calibrated,
    )

    model = joblib.load(model_artifact_path)

    feature_columns = _load_feature_columns(
        artifact_dir=artifact_dir,
        fallback_feature_columns=get_feature_columns(model_config),
    )

    return ModelBundle(
        model=model,
        model_id=get_model_id(model_config),
        model_family=get_model_family(model_config),
        algorithm=get_algorithm(model_config),
        artifact_dir=artifact_dir,
        feature_columns=feature_columns,
        metrics=_load_optional_json(artifact_dir / "metrics.json"),
        model_card=_load_optional_yaml(artifact_dir / "model_card.yaml"),
        uses_calibrated_model=uses_calibrated,
        model_artifact_path=model_artifact_path,
    )



def _resolve_model_artifact_path(
    artifact_dir: Path,
    prefer_calibrated: bool,
) -> tuple[Path, bool]:
    """Return model artifact path and whether it is calibrated."""

    calibrated_path = artifact_dir / "calibrated_model.joblib"
    raw_path = artifact_dir / "raw_model.joblib"

    if prefer_calibrated and calibrated_path.exists():
        return calibrated_path, True

    if raw_path.exists():
        return raw_path, False

    if calibrated_path.exists():
        return calibrated_path, True

    raise ModelLoaderError(
        "No model artifact found. Expected one of: "
        f"{calibrated_path}, {raw_path}"
    )



def _load_feature_columns(
    artifact_dir: Path,
    fallback_feature_columns: list[str],
) -> list[str]:
    """Load feature columns from artifact files, falling back to config."""

    json_path = artifact_dir / "feature_columns.json"
    joblib_path = artifact_dir / "feature_columns.joblib"

    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            values = json.load(f)
        return _normalize_feature_columns(values, source=json_path)

    if joblib_path.exists():
        values = joblib.load(joblib_path)
        return _normalize_feature_columns(values, source=joblib_path)

    if fallback_feature_columns:
        return _normalize_feature_columns(
            fallback_feature_columns,
            source="model config fallback",
        )

    raise ModelLoaderError(
        "No feature columns found. Expected feature_columns.json, "
        "feature_columns.joblib, or config features.feature_columns."
    )



def _normalize_feature_columns(values: Any, source: str | Path) -> list[str]:
    """Validate and normalize loaded feature columns."""

    if not isinstance(values, list):
        raise ModelLoaderError(
            f"Feature columns from {source} must be a list."
        )

    feature_columns = [str(value) for value in values if value is not None]
    feature_columns = list(dict.fromkeys(feature_columns))

    if not feature_columns:
        raise ModelLoaderError(
            f"Feature columns from {source} resolved to an empty list."
        )

    return feature_columns



def _load_optional_json(path: Path) -> dict[str, Any] | None:
    """Load optional JSON artifact."""

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)

    if not isinstance(value, dict):
        raise ModelLoaderError(
            f"Optional JSON artifact must deserialize into a dictionary: {path}"
        )

    return value



def _load_optional_yaml(path: Path) -> dict[str, Any] | None:
    """Load optional YAML artifact."""

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f)

    if value is None:
        return None

    if not isinstance(value, dict):
        raise ModelLoaderError(
            f"Optional YAML artifact must deserialize into a dictionary: {path}"
        )

    return value
