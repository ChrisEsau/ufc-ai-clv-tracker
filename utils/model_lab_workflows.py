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
    "prop_goes_distance_feature_view": "configs/feature_views/prop_goes_distance.yaml",
}

WORKFLOWS = {
    "feature_view": "run-build-feature-view-v2.yml",
    "training": "run-train-model-v2.yml",
    "prediction": "run-prediction-v2.yml",
    "betting_outcomes": "run-betting-outcomes-v2.yml",
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
    """Load the model registry used by V2 workflow selection."""

    return load_yaml_file(str(path))


def get_registered_model_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all registered model rows for the Model Lab selector."""

    rows: list[dict[str, Any]] = []
    for model_id, entry in (registry.get("models", {}) or {}).items():
        if not isinstance(entry, dict):
            continue

        rows.append(
            {
                "model_id": str(model_id),
                "display_name": entry.get("display_name") or str(model_id),
                "model_family": entry.get("model_family", ""),
                "market_key": entry.get("market_key", ""),
                "algorithm": entry.get("algorithm", ""),
                "status": entry.get("status", ""),
                "config_path": entry.get("config_path", ""),
                "artifact_dir": entry.get("artifact_dir", ""),
            }
        )

    return rows


def _model_label(row: dict[str, Any]) -> str:
    """Build the dropdown label for a registered model."""

    status = row.get("status") or "unknown"
    family = row.get("model_family") or "unknown"
    return f"{row['model_id']} — {row.get('display_name', row['model_id'])} ({family}, {status})"


def _first_feature_source(model_config: dict[str, Any]) -> str | None:
    sources = model_config.get("feature_sources") or []
    if not sources:
        return None
    return str(sources[0])


def _feature_view_config_path(model_config: dict[str, Any]) -> Path | None:
    """Resolve feature-view config path from model config metadata."""

    explicit_path = (
        model_config.get("feature_view_config_path")
        or (model_config.get("feature_view") or {}).get("config_path")
    )
    if explicit_path:
        path = Path(str(explicit_path))
        return path if path.exists() else None

    feature_source = _first_feature_source(model_config)
    mapped_path = FEATURE_VIEW_CONFIG_BY_SOURCE.get(feature_source or "")
    if not mapped_path:
        return None

    path = Path(mapped_path)
    return path if path.exists() else None


def _feature_view_output_path(
    *,
    model_config: dict[str, Any],
    feature_view_config_path: Path | None,
) -> Path | None:
    """Resolve the feature-view parquet path to pass to the workflow."""

    if feature_view_config_path is not None:
        feature_config = load_yaml_file(str(feature_view_config_path))
        output_path = (feature_config.get("output") or {}).get("feature_view_path")
        if output_path:
            return Path(str(output_path))

    model_data_path = (model_config.get("data") or {}).get("rolling_features_path")
    return Path(str(model_data_path)) if model_data_path else None


def resolve_model_workflow_context(
    *,
    registry: dict[str, Any],
    model_id: str,
) -> dict[str, Any]:
    """Resolve all workflow inputs for the selected registered model."""

    models = registry.get("models", {}) or {}
    if model_id not in models:
        raise ModelLabWorkflowError(f"Model is not registered: {model_id}")

    entry = models[model_id]
    config_path = Path(str(entry.get("config_path") or ""))
    if not config_path.exists():
        raise ModelLabWorkflowError(f"Model config path does not exist: {config_path}")

    model_config = load_yaml_file(str(config_path))
    feature_view_config_path = _feature_view_config_path(model_config)
    feature_view_output_path = _feature_view_output_path(
        model_config=model_config,
        feature_view_config_path=feature_view_config_path,
    )

    model_family = str(entry.get("model_family") or model_config.get("model_family") or "")
    artifact_dir = str(
        entry.get("artifact_dir")
        or (model_config.get("artifacts") or {}).get("output_dir")
        or ""
    )

    return {
        "model_id": model_id,
        "display_name": entry.get("display_name") or model_id,
        "status": entry.get("status", ""),
        "model_family": model_family,
        "market_key": entry.get("market_key") or model_config.get("market_key") or "",
        "algorithm": entry.get("algorithm") or model_config.get("algorithm") or "",
        "config_path": str(config_path),
        "artifact_dir": artifact_dir,
        "feature_source": _first_feature_source(model_config) or "",
        "feature_view_config_path": str(feature_view_config_path) if feature_view_config_path else "",
        "feature_view_output_path": str(feature_view_output_path) if feature_view_output_path else "",
    }


def _dispatch_button(
    *,
    label: str,
    workflow_file: str,
    inputs: dict[str, str],
    disabled: bool = False,
    help_text: str | None = None,
    key: str,
) -> None:
    """Render a workflow dispatch button."""

    if st.button(label, disabled=disabled, help=help_text, key=key):
        ok, message = trigger_workflow(workflow_file, inputs=inputs)
        if ok:
            st.success(message)
        else:
            st.error(message)


def _render_betting_outcomes_launcher() -> None:
    """Render a simple workflow launcher for Betting Outcomes V2."""

    st.divider()
    st.subheader("Betting Board / Outcomes")
    st.caption(
        "Run DraftKings discovery, normalize provider markets, match markets to the live card, "
        "and rebuild betting_outcomes.parquet for the Betting Board tab."
    )

    col_event, col_score = st.columns([2, 1])
    with col_event:
        draftkings_event_id = st.text_input(
            "DraftKings Event ID",
            value=st.session_state.get("model_lab_draftkings_event_id", "33525834"),
            key="model_lab_draftkings_event_id",
            help="DraftKings event ID passed to run-betting-outcomes-v2.yml.",
        )
    with col_score:
        min_match_score = st.number_input(
            "Min Match Score",
            min_value=0,
            max_value=100,
            value=int(st.session_state.get("model_lab_min_match_score", 65)),
            step=1,
            key="model_lab_min_match_score",
            help="Minimum market matching score passed to the DraftKings market matcher.",
        )

    inputs = {
        "draftkings_event_id": str(draftkings_event_id).strip(),
        "min_match_score": str(int(min_match_score)),
    }
    disabled = not bool(inputs["draftkings_event_id"])

    _dispatch_button(
        label="Run Betting Outcomes / Update Betting Board",
        workflow_file=WORKFLOWS["betting_outcomes"],
        inputs=inputs,
        disabled=disabled,
        help_text=(
            "Enter a DraftKings event ID before launching."
            if disabled
            else "Launch run-betting-outcomes-v2.yml."
        ),
        key="run_betting_outcomes_v2",
    )


def render_model_workflow_launcher() -> None:
    """Render a simple registry-driven Model Lab workflow launcher."""

    with st.expander("V2 Workflow Launcher", expanded=True):
        st.caption(
            "Select a registered model, then launch V2 feature-view, training, or prediction workflows. "
            "Workflow inputs are resolved from the model registry and model config."
        )

        try:
            registry = load_model_registry()
            model_rows = get_registered_model_rows(registry)
        except Exception as exc:
            st.error(f"Unable to load model registry: {exc}")
            return

        if not model_rows:
            st.info("No models are registered in configs/models/model_registry.yaml.")
            return

        model_ids = [row["model_id"] for row in model_rows]
        row_by_id = {row["model_id"]: row for row in model_rows}

        selected_model_id = st.selectbox(
            "Registered model",
            model_ids,
            format_func=lambda model_id: _model_label(row_by_id[model_id]),
            key="model_lab_registered_model_id",
        )

        try:
            context = resolve_model_workflow_context(
                registry=registry,
                model_id=selected_model_id,
            )
        except Exception as exc:
            st.error(f"Unable to resolve workflow inputs: {exc}")
            return

        summary_df = pd.DataFrame(
            [
                {"field": "model_id", "value": context["model_id"]},
                {"field": "model_family", "value": context["model_family"]},
                {"field": "market_key", "value": context["market_key"] or "—"},
                {"field": "status", "value": context["status"] or "—"},
                {"field": "model_config", "value": context["config_path"]},
                {"field": "artifact_dir", "value": context["artifact_dir"] or "—"},
                {"field": "feature_source", "value": context["feature_source"] or "—"},
                {"field": "feature_view_config", "value": context["feature_view_config_path"] or "not configured"},
                {"field": "feature_view_output", "value": context["feature_view_output_path"] or "—"},
            ]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        feature_inputs = {
            "config_path": context["feature_view_config_path"],
            "output_path": context["feature_view_output_path"],
        }
        training_inputs = {
            "config_path": context["config_path"],
            "artifact_dir": context["artifact_dir"],
        }
        prediction_inputs = {
            "model_family": context["model_family"],
            "model_id": context["model_id"],
        }

        feature_disabled = not bool(context["feature_view_config_path"] and context["feature_view_output_path"])
        train_disabled = not bool(context["config_path"] and context["artifact_dir"])
        prediction_disabled = not bool(context["model_family"] and context["model_id"])

        cols = st.columns(3)
        with cols[0]:
            _dispatch_button(
                label="Build Feature View",
                workflow_file=WORKFLOWS["feature_view"],
                inputs=feature_inputs,
                disabled=feature_disabled,
                help_text=(
                    "Feature-view config is not configured for this model."
                    if feature_disabled
                    else "Launch run-build-feature-view-v2.yml."
                ),
                key=f"build_feature_view_{selected_model_id}",
            )
        with cols[1]:
            _dispatch_button(
                label="Train Model",
                workflow_file=WORKFLOWS["training"],
                inputs=training_inputs,
                disabled=train_disabled,
                help_text="Launch run-train-model-v2.yml.",
                key=f"train_model_{selected_model_id}",
            )
        with cols[2]:
            _dispatch_button(
                label="Run Prediction",
                workflow_file=WORKFLOWS["prediction"],
                inputs=prediction_inputs,
                disabled=prediction_disabled,
                help_text="Launch run-prediction-v2.yml.",
                key=f"run_prediction_{selected_model_id}",
            )

        _render_betting_outcomes_launcher()
