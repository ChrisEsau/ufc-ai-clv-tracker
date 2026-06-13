from __future__ import annotations

from copy import deepcopy
from typing import Any

import requests
import streamlit as st

import utils.model_lab_workflows as mlw
from utils.model_lab_feature_selection import render_feature_checklist


NEW_MODEL_SENTINEL = "__new_model__"
WORKSPACES = ["Overview", "Configuration", "Features", "Performance", "Comparison", "Actions"]
MARKET_OPTIONS = {
    "moneyline": ["moneyline"],
    "prop": [
        "goes_distance",
        "inside_distance",
        "ko_tko",
        "submission",
        "decision",
        "over_1_5",
        "over_2_5",
        "over_3_5",
    ],
}


def _safe_model_id(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _safe_path_key(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unknown_market"


def _artifact_dir_for_market(model_id: str, market_key: str) -> str:
    return f"models/{_safe_path_key(market_key)}/{_safe_model_id(model_id)}"


def _default_market_family(context: dict[str, Any]) -> str:
    family = str(context.get("model_family") or "moneyline").strip().lower()
    return family if family in MARKET_OPTIONS else "moneyline"


def _default_market_key(context: dict[str, Any], market_family: str) -> str:
    config = context.get("config") or {}
    prediction = config.get("prediction") or {}
    key = str(
        context.get("market_key")
        or config.get("market_key")
        or prediction.get("market_key")
        or "moneyline"
    ).strip().lower()
    options = MARKET_OPTIONS.get(market_family, MARKET_OPTIONS["moneyline"])
    return key if key in options else options[0]


def _load_registry_rows() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    registry = mlw.load_model_registry()
    rows = mlw.get_registered_model_rows(registry)
    return registry, rows, {row["model_id"]: row for row in rows}


def _active_primary(registry: dict[str, Any], context: dict[str, Any]) -> bool:
    family = context.get("model_family", "")
    model_id = context.get("model_id", "")
    return registry.get("active_models", {}).get(family, {}).get("primary") == model_id


def _existing_model_selector(
    registry: dict[str, Any],
    rows: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    *,
    key: str,
) -> dict[str, Any]:
    model_ids = [row["model_id"] for row in rows]
    current = st.session_state.get("mlab_active_model_id", model_ids[0])
    index = model_ids.index(current) if current in model_ids else 0
    selected = st.selectbox(
        "Active Model",
        model_ids,
        index=index,
        format_func=lambda mid: mlw._model_label(row_by_id[mid]),
        key=key,
    )
    st.session_state["mlab_active_model_id"] = selected
    return mlw.resolve_model_workflow_context(registry=registry, model_id=selected)


def _build_new_context(
    template_context: dict[str, Any],
    *,
    model_id: str,
    market_family: str,
    market_key: str,
) -> dict[str, Any]:
    model_id = _safe_model_id(model_id)
    market_family = str(market_family or "moneyline").strip().lower()
    market_key = str(market_key or "moneyline").strip().lower()
    context = deepcopy(template_context)
    config = deepcopy(context["config"])
    artifact_dir = _artifact_dir_for_market(model_id, market_key)
    config["model_id"] = model_id
    config["model_family"] = market_family
    config["market_key"] = market_key
    config["artifact_name"] = model_id
    config["status"] = "draft"
    config.setdefault("prediction", {})["market_key"] = market_key
    config.setdefault("artifacts", {})["output_dir"] = artifact_dir
    context.update(
        {
            "model_id": model_id,
            "display_name": f"{template_context.get('display_name', template_context['model_id'])} Experiment",
            "description": f"Draft experiment created from {template_context['model_id']}.",
            "status": "draft",
            "dashboard_selectable": False,
            "config_path": f"configs/models/{model_id}.yaml" if model_id else "",
            "artifact_dir": artifact_dir,
            "model_family": market_family,
            "market_key": market_key,
            "config": config,
            "is_new_model": True,
            "template_model_id": template_context["model_id"],
        }
    )
    return context


def _save_new_or_existing_model(
    *,
    context: dict[str, Any],
    registry: dict[str, Any],
    form_values: dict[str, Any],
    feature_values: dict[str, Any],
) -> tuple[bool, str]:
    model_id = _safe_model_id(context.get("model_id", ""))
    if not model_id:
        return False, "Model ID is required."
    if context.get("is_new_model") and model_id in (registry.get("models") or {}):
        return False, f"Model already exists: {model_id}"

    market_family = str(context.get("model_family") or "moneyline").strip().lower()
    market_key = str(context.get("market_key") or "moneyline").strip().lower()
    artifact_dir = _artifact_dir_for_market(model_id, market_key)
    updated_config = mlw._apply_config_updates(context, form_values, feature_values)
    updated_config["model_id"] = model_id
    updated_config["model_family"] = market_family
    updated_config["market_key"] = market_key
    updated_config["artifact_name"] = model_id
    updated_config["status"] = "draft"
    updated_config.setdefault("prediction", {})["market_key"] = market_key
    updated_config.setdefault("artifacts", {})["output_dir"] = artifact_dir

    updated_registry = deepcopy(registry)
    updated_registry.setdefault("models", {})[model_id] = {
        "display_name": form_values["display_name"],
        "description": form_values["description"],
        "model_family": market_family,
        "market_key": market_key,
        "algorithm": context["algorithm"] or updated_config.get("algorithm", "xgboost"),
        "config_path": context["config_path"],
        "artifact_dir": artifact_dir,
        "status": "draft",
        "dashboard_selectable": bool(form_values["dashboard_selectable"]),
        "outcome_architecture": True,
    }

    ok, msg = mlw._github_write_file(
        context["config_path"],
        mlw._yaml_dump(updated_config),
        f"Save draft model config {model_id}",
    )
    if not ok:
        return ok, msg
    ok, msg = mlw._save_registry(updated_registry)
    return (ok, msg if not ok else f"Saved draft model {model_id}.")


def _github_delete_file(path: str, message: str) -> tuple[bool, str]:
    owner, repo, token, branch = mlw.get_github_config()
    if not owner or not repo or not token:
        return False, "Missing GitHub Streamlit secrets."
    ok, _, sha = mlw._github_read_file(path)
    if not ok:
        return False, "Could not inspect existing GitHub file before delete."
    if not sha:
        return True, f"No config file found at {path}; registry entry can still be removed."
    response = requests.delete(
        f"{mlw.GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}",
        headers=mlw.github_headers(token),
        json={"message": message, "sha": sha, "branch": branch},
        timeout=20,
    )
    if response.status_code in {200, 201}:
        return True, f"Deleted {path}."
    return False, f"GitHub API error {response.status_code}: {response.text}"


def _delete_model(context: dict[str, Any], registry: dict[str, Any]) -> tuple[bool, str]:
    model_id = context["model_id"]
    status = str(context.get("status") or "draft").lower()
    if status == "production":
        return False, "Production models cannot be deleted."
    if _active_primary(registry, context):
        return False, "Active primary models cannot be deleted. Change active model first."
    config_path = context.get("config_path", "")
    if config_path:
        ok, msg = _github_delete_file(config_path, f"Delete model config {model_id}")
        if not ok:
            return ok, msg
    updated_registry = deepcopy(registry)
    updated_registry.get("models", {}).pop(model_id, None)
    ok, msg = mlw._save_registry(updated_registry)
    return (ok, msg if not ok else f"Deleted model {model_id}. Artifacts were not deleted.")


def _render_delete_dialog(context: dict[str, Any], registry: dict[str, Any]) -> None:
    model_id = context["model_id"]

    def body() -> None:
        st.warning("This removes the registry entry and config YAML. Model artifacts are not deleted.")
        st.write(f"Model: `{model_id}`")
        confirmation = st.text_input("Type the model ID to confirm", key=f"delete_confirm_{model_id}")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Cancel", use_container_width=True, key=f"delete_cancel_{model_id}"):
                st.session_state.pop("mlab_delete_candidate", None)
                st.rerun()
        with c2:
            if st.button(
                "Delete Model",
                type="primary",
                disabled=confirmation != model_id,
                use_container_width=True,
                key=f"delete_execute_{model_id}",
            ):
                ok, msg = _delete_model(context, registry)
                st.success(msg) if ok else st.error(msg)
                if ok:
                    st.cache_data.clear()
                    st.session_state.pop("mlab_delete_candidate", None)
                    st.rerun()

    if hasattr(st, "dialog"):
        @st.dialog("Delete Model")
        def confirm_dialog():
            body()
        confirm_dialog()
    else:
        with st.expander("Confirm Delete Model", expanded=True):
            body()


def _inject_model_lab_control_css() -> None:
    """Keep Model Lab number-input stepper buttons aligned with input cells."""

    st.markdown(
        """
        <style>
        div[data-testid="stNumberInput"] button,
        div[data-testid="stNumberInput"] button:disabled {
            background: rgba(7, 17, 31, 0.9) !important;
            background-color: rgba(7, 17, 31, 0.9) !important;
            border-color: rgba(38, 54, 74, 0.95) !important;
            color: #ffffff !important;
            opacity: 1 !important;
        }

        div[data-testid="stNumberInput"] button:hover,
        div[data-testid="stNumberInput"] button:focus,
        div[data-testid="stNumberInput"] button:active {
            background: rgba(7, 17, 31, 0.98) !important;
            background-color: rgba(7, 17, 31, 0.98) !important;
            border-color: rgba(59, 130, 246, 0.65) !important;
            color: #ffffff !important;
        }

        div[data-testid="stNumberInput"] button svg,
        div[data-testid="stNumberInput"] button svg path {
            fill: #ffffff !important;
            color: #ffffff !important;
            opacity: 1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_advanced_configuration(context: dict[str, Any], *, can_delete: bool) -> None:
    """Render rarely used model metadata and destructive controls above the save action."""

    model_id = str(context.get("model_id") or "new_model")
    with st.expander("Advanced", expanded=False):
        st.caption("Technical metadata and destructive actions.")

        meta_left, meta_right = st.columns(2)
        with meta_left:
            st.text_input(
                "Config Path",
                value=str(context.get("config_path") or ""),
                disabled=True,
                key=f"model_lab_adv_config_path_{model_id}",
            )
            st.text_input(
                "Artifact Directory",
                value=str(context.get("artifact_dir") or ""),
                disabled=True,
                key=f"model_lab_adv_artifact_dir_{model_id}",
            )
        with meta_right:
            st.text_input(
                "Model Family",
                value=str(context.get("model_family") or ""),
                disabled=True,
                key=f"model_lab_adv_model_family_{model_id}",
            )
            st.text_input(
                "Market Key",
                value=str(context.get("market_key") or ""),
                disabled=True,
                key=f"model_lab_adv_market_key_{model_id}",
            )

        st.divider()
        st.caption("Danger zone")
        if st.button(
            "Delete Model",
            disabled=not can_delete,
            use_container_width=True,
            key=f"model_lab_adv_delete_{model_id}",
        ):
            st.session_state["mlab_delete_candidate"] = context["model_id"]


def _render_configuration(registry: dict[str, Any], rows: list[dict[str, Any]], row_by_id: dict[str, dict[str, Any]]) -> None:
    st.markdown("## Configuration")
    options = [NEW_MODEL_SENTINEL] + [row["model_id"] for row in rows]
    selected = st.selectbox(
        "Selected Model",
        options,
        format_func=lambda mid: "New Model" if mid == NEW_MODEL_SENTINEL else mlw._model_label(row_by_id[mid]),
        key="model_lab_config_selected_model",
    )
    if selected == NEW_MODEL_SENTINEL:
        template_id = st.selectbox(
            "Template Model",
            [row["model_id"] for row in rows],
            format_func=lambda mid: mlw._model_label(row_by_id[mid]),
            key="model_lab_template_model_id",
        )
        template_context = mlw.resolve_model_workflow_context(registry=registry, model_id=template_id)
        new_model_id = _safe_model_id(
            st.text_input("New Model ID", value=f"{template_id}_exp01", key="model_lab_new_model_id")
        )
        default_family = _default_market_family(template_context)
        family_options = list(MARKET_OPTIONS.keys())
        family_index = family_options.index(default_family) if default_family in family_options else 0
        c_family, c_key = st.columns(2)
        with c_family:
            market_family = st.selectbox(
                "Market Family",
                family_options,
                index=family_index,
                key="model_lab_new_market_family",
            )
        key_options = MARKET_OPTIONS.get(market_family, MARKET_OPTIONS["moneyline"])
        default_key = _default_market_key(template_context, market_family)
        key_index = key_options.index(default_key) if default_key in key_options else 0
        with c_key:
            market_key = st.selectbox(
                "Market Key",
                key_options,
                index=key_index,
                key="model_lab_new_market_key",
            )
        context = _build_new_context(
            template_context,
            model_id=new_model_id,
            market_family=market_family,
            market_key=market_key,
        )
    else:
        st.session_state["mlab_active_model_id"] = selected
        context = mlw.resolve_model_workflow_context(registry=registry, model_id=selected)

    mlw._render_model_bar(context, registry)
    c1, c2 = st.columns([1.05, 1.45], gap="medium")
    with c1:
        if context.get("is_new_model"):
            st.info("New model draft. Press Save Draft Configuration to create config YAML and registry entry.")
        elif context["status"] == "production":
            st.info("Production models are read-only. Select New Model and use this model as a template to tune an experiment.")
        form_values = mlw._render_config_editor(context, registry)
    with c2:
        feature_values = render_feature_checklist(context)

    can_save = context.get("is_new_model") or context.get("status") == "draft"
    can_delete = (
        not context.get("is_new_model")
        and str(context.get("status") or "").lower() in {"draft", "archived"}
        and not _active_primary(registry, context)
    )

    _render_advanced_configuration(context, can_delete=can_delete)

    if st.button("Save Draft Configuration", type="primary", disabled=not can_save, use_container_width=True):
        ok, msg = _save_new_or_existing_model(
            context=context,
            registry=registry,
            form_values=form_values,
            feature_values=feature_values,
        )
        st.success(msg) if ok else st.error(msg)
        if ok:
            st.session_state["mlab_active_model_id"] = context["model_id"]
            st.cache_data.clear()
            st.rerun()

    if st.session_state.get("mlab_delete_candidate") == context.get("model_id"):
        _render_delete_dialog(context, registry)


def _render_overview(registry: dict[str, Any], rows: list[dict[str, Any]], row_by_id: dict[str, dict[str, Any]]) -> None:
    st.markdown("## Overview")
    context = _existing_model_selector(registry, rows, row_by_id, key="mlab_overview_model")
    mlw._render_kpis(context)
    mlw._render_model_bar(context, registry)
    mlw._render_registry_table(rows)


def _render_features() -> None:
    st.markdown("## Features")
    st.info("Feature and bundle creation workspace coming soon. Model-level bundle selection stays in Configuration.")


def _render_performance(registry: dict[str, Any], rows: list[dict[str, Any]], row_by_id: dict[str, dict[str, Any]]) -> None:
    st.markdown("## Performance")
    context = _existing_model_selector(registry, rows, row_by_id, key="mlab_performance_model")
    mlw._render_model_bar(context, registry)
    mlw._render_performance(context)


def _render_comparison(registry: dict[str, Any], rows: list[dict[str, Any]], row_by_id: dict[str, dict[str, Any]]) -> None:
    st.markdown("## Comparison")
    context = _existing_model_selector(registry, rows, row_by_id, key="mlab_comparison_model")
    mlw._render_comparison(context, registry, context["model_id"])


def _render_actions(registry: dict[str, Any], rows: list[dict[str, Any]], row_by_id: dict[str, dict[str, Any]]) -> None:
    st.markdown("## Actions")
    context = _existing_model_selector(registry, rows, row_by_id, key="mlab_actions_model")
    mlw._render_model_bar(context, registry)
    mlw._render_actions(context)


def render_model_lab():
    mlw._inject_css()
    _inject_model_lab_control_css()
    mlw._render_header()
    try:
        registry, rows, row_by_id = _load_registry_rows()
        if not rows:
            st.info("No models are registered in configs/models/model_registry.yaml.")
            return
        workspace = st.session_state.get("sidebar_model_lab_workspace", "Configuration")
        if workspace not in WORKSPACES:
            workspace = "Configuration"
        if workspace == "Overview":
            _render_overview(registry, rows, row_by_id)
        elif workspace == "Configuration":
            _render_configuration(registry, rows, row_by_id)
        elif workspace == "Features":
            _render_features()
        elif workspace == "Performance":
            _render_performance(registry, rows, row_by_id)
        elif workspace == "Comparison":
            _render_comparison(registry, rows, row_by_id)
        elif workspace == "Actions":
            _render_actions(registry, rows, row_by_id)
    except Exception as exc:
        st.error(f"Unable to render Model Lab: {exc}")
