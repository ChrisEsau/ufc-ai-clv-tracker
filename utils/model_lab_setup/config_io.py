from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from utils.model_lab_setup.versioning import artifact_dir_for_market, safe_model_id


DEFAULT_SECTIONS: dict[str, Any] = {
    "split": {},
    "calibration": {},
    "params": {},
    "prediction": {"probability": {}, "threshold": {}},
    "features": {},
    "artifacts": {},
    "data": {},
    "metrics": {},
    "symmetry": {},
}


def load_model_config(config_path: str | Path) -> dict[str, Any]:
    """Load a model config YAML file."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Model config must be a mapping: {path}")
    return payload


def dump_model_config(config: dict[str, Any]) -> str:
    """Return a readable YAML string for a model config."""

    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def normalize_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized config copy with required default sections."""

    normalized = deepcopy(config)
    for key, default_value in DEFAULT_SECTIONS.items():
        existing = normalized.get(key)
        if not isinstance(existing, dict):
            normalized[key] = deepcopy(default_value)
        else:
            merged = deepcopy(default_value)
            merged.update(existing)
            normalized[key] = merged

    prediction = normalized.setdefault("prediction", {})
    if not isinstance(prediction.get("probability"), dict):
        prediction["probability"] = {}
    if not isinstance(prediction.get("threshold"), dict):
        prediction["threshold"] = {}
    if not isinstance(normalized.get("symmetry"), dict):
        normalized["symmetry"] = {}
    return normalized


def build_config_path(model_id: str) -> str:
    """Return canonical model config path."""

    return f"configs/models/{safe_model_id(model_id)}.yaml"


def build_config_payload_from_form(context: dict[str, Any], form_payload: dict[str, Any]) -> dict[str, Any]:
    """Build a model config payload from context and nested Model Setup form payload."""

    config = normalize_model_config(context.get("config") or {})
    identity = form_payload.get("identity") or {}
    training = form_payload.get("training") or {}
    behavior = form_payload.get("behavior") or {}
    hyperparameters = form_payload.get("hyperparameters") or {}
    features = form_payload.get("features") or {}

    model_id = safe_model_id(context.get("model_id") or identity.get("model_id") or "")
    model_family = str(identity.get("model_family") or context.get("model_family") or "moneyline").strip().lower()
    market_key = str(identity.get("market_key") or context.get("market_key") or "moneyline").strip().lower()

    config["model_id"] = model_id
    config["model_family"] = model_family
    config["market_key"] = market_key
    config["artifact_name"] = model_id
    config["status"] = str(context.get("status") or identity.get("status") or "draft")

    split = config.setdefault("split", {})
    split["train_start_date"] = training.get("train_start_date", split.get("train_start_date"))
    split["train_end_date"] = training.get("train_end_date", split.get("train_end_date"))
    split["calibration_end_date"] = training.get("calibration_end_date", split.get("calibration_end_date"))

    calibration = config.setdefault("calibration", {})
    if "calibration_enabled" in behavior:
        calibration["enabled"] = bool(behavior["calibration_enabled"])
    if behavior.get("calibration_method") is not None:
        calibration["method"] = str(behavior["calibration_method"])

    probability = config.setdefault("prediction", {}).setdefault("probability", {})
    if behavior.get("clip_low") is not None:
        probability["clip_low"] = float(behavior["clip_low"])
    if behavior.get("clip_high") is not None:
        probability["clip_high"] = float(behavior["clip_high"])

    threshold = config.setdefault("prediction", {}).setdefault("threshold", {})
    if behavior.get("threshold_source") is not None:
        threshold["source"] = str(behavior["threshold_source"])
    if behavior.get("threshold_value") is not None:
        threshold["value"] = float(behavior["threshold_value"])

    symmetry = config.setdefault("symmetry", {})
    if "symmetry_enabled" in behavior:
        symmetry["enabled"] = bool(behavior["symmetry_enabled"])
    if behavior.get("symmetry_mode") is not None:
        symmetry["mode"] = str(behavior["symmetry_mode"])

    params = hyperparameters.get("params") or {}
    if params:
        config["params"] = deepcopy(params)

    feature_config = config.setdefault("features", {})
    for key in ["selected_bundles", "include_features", "exclude_features", "resolved_features", "expected_feature_count"]:
        if key in features:
            feature_config[key] = deepcopy(features[key])
    if "resolved_features" in features:
        feature_config["feature_columns"] = list(features.get("resolved_features") or [])

    config.setdefault("prediction", {})["market_key"] = market_key
    config.setdefault("artifacts", {})["output_dir"] = artifact_dir_for_market(model_id, market_key)
    return config
