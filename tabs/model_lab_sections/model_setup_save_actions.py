from __future__ import annotations

from copy import deepcopy
from typing import Any

import requests
import streamlit as st

import utils.model_lab_workflows as mlw


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


def _active_primary(registry: dict[str, Any], context: dict[str, Any]) -> bool:
    family = str(context.get("model_family") or "").strip().lower()
    market_key = str(context.get("market_key") or "").strip().lower()
    model_id = str(context.get("model_id") or "")
    active_family = (registry.get("active_models", {}).get(family) or {})
    if market_key:
        market_model = active_family.get(market_key)
        if market_model:
            return str(market_model) == model_id
        if family == "moneyline" and market_key == "moneyline":
            return str(active_family.get("primary") or "") == model_id
        return False
    return str(active_family.get("primary") or "") == model_id


def apply_advanced_config_updates(
    updated_config: dict[str, Any],
    advanced_values: dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply advanced Model Setup form values to the config payload."""

    if not advanced_values:
        return updated_config

    updated_config.setdefault("data", {})["target_column"] = str(advanced_values["target_column"])

    symmetry = updated_config.setdefault("symmetry", {})
    symmetry["enabled"] = bool(advanced_values["symmetry_enabled"])
    symmetry["mode"] = str(advanced_values["symmetry_mode"])

    metrics = updated_config.setdefault("metrics", {})
    metrics["threshold_min"] = float(advanced_values["threshold_min"])
    metrics["threshold_max"] = float(advanced_values["threshold_max"])
    metrics["threshold_step"] = float(advanced_values["threshold_step"])

    threshold = updated_config.setdefault("prediction", {}).setdefault("threshold", {})
    threshold["source"] = str(advanced_values["prediction_threshold_source"])
    threshold["value"] = float(advanced_values["prediction_threshold_value"])
    return updated_config


def save_new_or_existing_model(
    *,
    context: dict[str, Any],
    registry: dict[str, Any],
    form_values: dict[str, Any],
    feature_values: dict[str, Any],
    advanced_values: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Save a draft/new Model Setup config and registry entry."""

    model_id = _safe_model_id(context.get("model_id", ""))
    if not model_id:
        return False, "Model ID is required."
    if context.get("is_new_model") and model_id in (registry.get("models") or {}):
        return False, f"Model already exists: {model_id}"

    market_family = str(context.get("model_family") or "moneyline").strip().lower()
    market_key = str(context.get("market_key") or "moneyline").strip().lower()
    artifact_dir = _artifact_dir_for_market(model_id, market_key)
    updated_config = mlw._apply_config_updates(context, form_values, feature_values)
    updated_config = apply_advanced_config_updates(updated_config, advanced_values)
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
    return (ok, msg if not ok else f"Saved draft model {model_id} to GitHub.")


def github_delete_file(path: str, message: str) -> tuple[bool, str]:
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


def delete_model(context: dict[str, Any], registry: dict[str, Any]) -> tuple[bool, str]:
    model_id = context["model_id"]
    status = str(context.get("status") or "draft").lower()
    if status == "production":
        return False, "Production models cannot be deleted."
    if _active_primary(registry, context):
        return False, "Active production models cannot be deleted. Change active model first."
    config_path = context.get("config_path", "")
    if config_path:
        ok, msg = github_delete_file(config_path, f"Delete model config {model_id}")
        if not ok:
            return ok, msg
    updated_registry = deepcopy(registry)
    updated_registry.get("models", {}).pop(model_id, None)
    ok, msg = mlw._save_registry(updated_registry)
    return (ok, msg if not ok else f"Deleted model {model_id}. Artifacts were not deleted.")


def render_delete_dialog(context: dict[str, Any], registry: dict[str, Any]) -> None:
    """Render the Model Setup delete confirmation dialog."""

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
                ok, msg = delete_model(context, registry)
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
