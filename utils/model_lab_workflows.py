from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

from utils.github_actions import trigger_workflow


MODEL_REGISTRY_PATH = Path("configs/models/model_registry.yaml")

FEATURE_VIEW_CONFIG_BY_SOURCE = {
    "moneyline_feature_view": "configs/feature_views/moneyline_base.yaml",
    # Add future prop feature-view configs here once they are committed.
    "prop_goes_distance_feature_view": "configs/feature_views/prop_goes_distance.yaml",
}

WORKFLOWS = {
    "feature_view": "run-build-feature-view-v2.yml",
    "training": "run-train-model-v2.yml",
    "prediction": "run-prediction-v2.yml",
}


class ModelLabWorkflowError(RuntimeError):
    """Raised when Model Lab cannot resolve registry-driven workflow inputs."""


@st.cache_data(show_spinner=False)
def load_yaml_file(path_text: str) -> dict[str, Any]:
    """Load a local YAML file as a dictionary."""

    path = Path(path_text)
    if not path.exists():
        raise ModelLabWorkflowError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    if not isinstance(payload, dict):
        raise ModelLabWorkflowError(f"YAML file must contain a mapping: {path}")

    return payload


def load_model_registry(path: Path = MODEL_REGISTRY_PATH) -> dict[str, Any]:
    return load_yaml_file(str(path))


def get_registered_model_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model_id, entry in (registry.get("models", {}) or {}).items():
        if not isinstance(entry, dict):
            continue
        rows.append({
            "model_id": str(model_id),
            "display_name": entry.get("display_name") or str(model_id),
            "model_family": entry.get("model_family", ""),
            "market_key": entry.get("market_key", ""),
            "algorithm": entry.get("algorithm", ""),
            "status": entry.get("status", ""),
            "config_path": entry.get("config_path", ""),
            "artifact_dir": entry.get("artifact_dir", ""),
        })
    return rows


def _model_label(row):
    status = row.get("status") or "unknown"
    family = row.get("model_family") or "unknown"
    return f"{row['model_id']} — {row.get('display_name', row['model_id'])} ({family}, {status})"


def _first_feature_source(model_config):
    sources = model_config.get("feature_sources") or []
    return str(sources[0]) if sources else None


def _feature_view_config_path(model_config):
    explicit_path = model_config.get("feature_view_config_path") or (model_config.get("feature_view") or {}).get("config_path")
    if explicit_path:
        path = Path(str(explicit_path))
        return path if path.exists() else None
    feature_source = _first_feature_source(model_config)
    mapped_path = FEATURE_VIEW_CONFIG_BY_SOURCE.get(feature_source or "")
    if not mapped_path:
        return None
    path = Path(mapped_path)
    return path if path.exists() else None


def _feature_view_output_path(*, model_config, feature_view_config_path):
    if feature_view_config_path is not None:
        feature_config = load_yaml_file(str(feature_view_config_path))
        output_path = (feature_config.get("output") or {}).get("feature_view_path")
        if output_path:
            return Path(str(output_path))
    model_data_path = (model_config.get("data") or {}).get("rolling_features_path")
    return Path(str(model_data_path)) if model_data_path else None


def resolve_model_workflow_context(*, registry, model_id):
    models = registry.get("models", {}) or {}
    entry = models[model_id]
    config_path = Path(str(entry.get("config_path") or ""))
    model_config = load_yaml_file(str(config_path))
    feature_view_config_path = _feature_view_config_path(model_config)
    feature_view_output_path = _feature_view_output_path(model_config=model_config, feature_view_config_path=feature_view_config_path)
    return {
        "model_id": model_id,
        "display_name": entry.get("display_name") or model_id,
        "status": entry.get("status", ""),
        "model_family": str(entry.get("model_family") or model_config.get("model_family") or ""),
        "market_key": entry.get("market_key") or model_config.get("market_key") or "",
        "algorithm": entry.get("algorithm") or model_config.get("algorithm") or "",
        "config_path": str(config_path),
        "artifact_dir": str(entry.get("artifact_dir") or (model_config.get("artifacts") or {}).get("output_dir") or ""),
        "feature_source": _first_feature_source(model_config) or "",
        "feature_view_config_path": str(feature_view_config_path) if feature_view_config_path else "",
        "feature_view_output_path": str(feature_view_output_path) if feature_view_output_path else "",
    }


def _dispatch_button(*, label, workflow_file, inputs, disabled=False, help_text=None, key=""):
    if st.button(label, disabled=disabled, help=help_text, key=key):
        ok, message = trigger_workflow(workflow_file, inputs=inputs)
        st.success(message) if ok else st.error(message)


def render_model_workflow_launcher():
    st.write('')