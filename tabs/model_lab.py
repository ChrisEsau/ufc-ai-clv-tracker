from __future__ import annotations

from copy import deepcopy
from typing import Any

import requests
import streamlit as st

import utils.model_lab_workflows as mlw


NEW_MODEL_SENTINEL = "__new_model__"

MODEL_LAB_VISUAL_REFINEMENT_CSS = """
<style>
section.main > div.block-container { padding-top: 1.05rem; }
.mlab-hero {
    background: radial-gradient(circle at 14% 12%, rgba(59,130,246,.23), transparent 34%), radial-gradient(circle at 88% 18%, rgba(14,165,233,.18), transparent 30%), linear-gradient(135deg, rgba(15,31,52,.98), rgba(7,15,26,.98));
    border: 1px solid rgba(64,93,132,.92); border-radius: 14px; padding: 1.05rem 1.15rem; box-shadow: 0 26px 58px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.04);
}
.mlab-title { font-size: 2.22rem !important; letter-spacing: -.055em !important; }
.mlab-subtitle { color: #aebdd2 !important; }
div[data-testid="stSelectbox"] label, div[data-testid="stTextInput"] label, div[data-testid="stTextArea"] label, div[data-testid="stNumberInput"] label, div[data-testid="stMultiSelect"] label {
    color: #dbe7f5 !important; font-size: .68rem !important; font-weight: 900 !important; letter-spacing: .035em !important; text-transform: uppercase !important;
}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div, div[data-baseweb="base-input"], textarea, input {
    background-color: rgba(9,19,32,.96) !important; border-color: rgba(56,79,111,.96) !important; color: #f5f7fb !important;
}
.mlab-card { border-radius: 13px !important; border: 1px solid rgba(50,75,108,.96) !important; background: linear-gradient(180deg, rgba(18,34,55,.96), rgba(8,17,30,.99)) !important; box-shadow: 0 20px 46px rgba(0,0,0,.31), inset 0 1px 0 rgba(255,255,255,.035) !important; overflow: hidden; }
.mlab-section { padding: 1rem 1.05rem 1.05rem !important; }
.mlab-section-title { display: flex; align-items: center; gap: .45rem; color: #f8fafc !important; font-size: .76rem !important; letter-spacing: .08em !important; padding-bottom: .55rem; margin-bottom: .85rem !important; border-bottom: 1px solid rgba(53,76,110,.82); }
.mlab-section-title::before { content: ""; width: 7px; height: 7px; border-radius: 999px; background: #3b82f6; box-shadow: 0 0 13px rgba(59,130,246,.9); }
.mlab-kpis { gap: .72rem !important; margin: .82rem 0 .9rem !important; }
.mlab-kpi { min-height: 94px !important; padding: .95rem .7rem .78rem !important; position: relative; }
.mlab-kpi::after { content: ""; position: absolute; left: 13%; right: 13%; bottom: 0; height: 2px; background: linear-gradient(90deg, transparent, rgba(59,130,246,.8), transparent); }
.mlab-label { color: #9fb0c4 !important; font-size: .62rem !important; letter-spacing: .075em !important; }
.mlab-value { font-size: 1.62rem !important; line-height: 1.05 !important; }
.mlab-caption { color: #91a3ba !important; }
.mlab-model-bar { margin-top: .15rem; padding: 1rem 1.1rem !important; background: linear-gradient(90deg, rgba(15,38,68,.98), rgba(8,18,31,.98)) !important; }
.mlab-model-name { font-size: 1.42rem !important; letter-spacing: -.025em; }
.mlab-pill { margin-left: .42rem; border: 1px solid rgba(255,255,255,.14); box-shadow: inset 0 1px 0 rgba(255,255,255,.06); }
div[data-testid="stButton"] > button { border-radius: 9px !important; border: 1px solid rgba(59,130,246,.46) !important; background: linear-gradient(180deg, rgba(37,99,235,.95), rgba(29,78,216,.92)) !important; color: #f8fafc !important; font-weight: 900 !important; letter-spacing: .01em !important; box-shadow: 0 10px 24px rgba(0,0,0,.25) !important; }
div[data-testid="stButton"] > button:hover { border-color: rgba(147,197,253,.78) !important; filter: brightness(1.08); }
div[data-testid="stButton"] > button:disabled { background: linear-gradient(180deg, rgba(43,57,78,.78), rgba(24,35,51,.9)) !important; border-color: rgba(71,85,105,.5) !important; color: #94a3b8 !important; }
div[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden; border: 1px solid rgba(43,60,82,.78); }
details[data-testid="stExpander"] { background: rgba(9,19,32,.72) !important; border: 1px solid rgba(43,60,82,.85) !important; border-radius: 12px !important; }
details[data-testid="stExpander"] summary { color: #f5f7fb !important; font-weight: 900 !important; }
div[data-testid="stTabs"] button { color: #dbe7f5 !important; font-weight: 850 !important; }
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: #3b82f6 !important; }
div[data-testid="stAlert"] { border-radius: 10px !important; border: 1px solid rgba(59,130,246,.24) !important; background: rgba(15,35,60,.72) !important; }
div[data-testid="stVerticalBlock"] > div:has(.mlab-card) { gap: .55rem !important; }
hr { border-color: rgba(43,60,82,.7) !important; }
@media (max-width: 1200px) { .mlab-title { font-size: 1.75rem !important; } .mlab-model-name { font-size: 1.12rem !important; } }
</style>
"""


def _safe_model_id(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _default_artifact_dir(model_family: str, market_key: str, model_id: str) -> str:
    if model_family == "prop":
        return f"models/props/{market_key or 'unknown_market'}/{model_id}"
    return f"models/{model_family or 'models'}/{model_id}"


def _registry_is_active_primary(registry: dict[str, Any], context: dict[str, Any]) -> bool:
    family = context.get("model_family", "")
    model_id = context.get("model_id", "")
    return registry.get("active_models", {}).get(family, {}).get("primary") == model_id


def _build_new_context(template_context: dict[str, Any], *, model_id: str, artifact_dir: str) -> dict[str, Any]:
    context = deepcopy(template_context)
    model_id = _safe_model_id(model_id)
    config = deepcopy(context["config"])
    config["model_id"] = model_id
    config["artifact_name"] = model_id
    config["status"] = "draft"
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

    is_new = bool(context.get("is_new_model"))
    if is_new and model_id in (registry.get("models") or {}):
        return False, f"Model already exists: {model_id}"

    updated_config = mlw._apply_config_updates(context, form_values, feature_values)
    updated_config["model_id"] = model_id
    updated_config["artifact_name"] = model_id
    updated_config["status"] = "draft"
    updated_config.setdefault("artifacts", {})["output_dir"] = context["artifact_dir"]

    updated_registry = deepcopy(registry)
    updated_registry.setdefault("models", {})[model_id] = {
        "display_name": form_values["display_name"],
        "description": form_values["description"],
        "model_family": context["model_family"],
        "market_key": context["market_key"],
        "algorithm": context["algorithm"] or updated_config.get("algorithm", "xgboost"),
        "config_path": context["config_path"],
        "artifact_dir": context["artifact_dir"],
        "status": "draft",
        "dashboard_selectable": bool(form_values["dashboard_selectable"]),
        "outcome_architecture": True,
    }

    ok, msg = mlw._github_write_file(context["config_path"], mlw._yaml_dump(updated_config), f"Save draft model config {model_id}")
    if not ok:
        return ok, msg
    ok, msg = mlw._save_registry(updated_registry)
    if not ok:
        return ok, msg
    return True, f"Saved draft model {model_id}."


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
    if _registry_is_active_primary(registry, context):
        return False, "Active primary models cannot be deleted. Change active model first."

    config_path = context.get("config_path", "")
    if config_path:
        ok, msg = _github_delete_file(config_path, f"Delete model config {model_id}")
        if not ok:
            return ok, msg

    updated_registry = deepcopy(registry)
    updated_registry.get("models", {}).pop(model_id, None)
    ok, msg = mlw._save_registry(updated_registry)
    if not ok:
        return ok, msg
    return True, f"Deleted model {model_id} from registry and removed config YAML. Artifacts were not deleted."


def _render_delete_dialog(context: dict[str, Any], registry: dict[str, Any]) -> None:
    model_id = context["model_id"]

    def _dialog_body() -> None:
        st.warning("This removes the registry entry and config YAML. Model artifacts are not deleted.")
        st.write(f"Model: `{model_id}`")
        confirmation = st.text_input("Type the model ID to confirm", key=f"delete_confirm_{model_id}")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Cancel", use_container_width=True, key=f"delete_cancel_{model_id}"):
                st.session_state.pop("mlab_delete_candidate", None)
                st.rerun()
        with c2:
            if st.button("Delete Model", type="primary", disabled=confirmation != model_id, use_container_width=True, key=f"delete_execute_{model_id}"):
                ok, msg = _delete_model(context, registry)
                st.success(msg) if ok else st.error(msg)
                if ok:
                    st.cache_data.clear()
                    st.session_state.pop("mlab_delete_candidate", None)
                    st.rerun()

    if hasattr(st, "dialog"):
        @st.dialog("Delete Model")
        def confirm_dialog():
            _dialog_body()
        confirm_dialog()
    else:
        with st.expander("Confirm Delete Model", expanded=True):
            _dialog_body()


def _patched_lifecycle_controls(context: dict[str, Any], registry: dict[str, Any]) -> None:
    status = str(context.get("status") or "draft").lower()
    st.markdown("#### Lifecycle")
    if context.get("is_new_model"):
        st.info("New model draft. Press Save Draft Configuration to create config YAML and registry entry.")
    elif status == "production":
        st.info("Production models are read-only. Select New Model and use this model as a template to tune an experiment.")
    elif status == "draft":
        st.success("Draft model is editable.")
    else:
        st.warning("Archived models are read-only for editing, but can be deleted if not active.")


def _render_single_editor() -> None:
    registry = mlw.load_model_registry()
    rows = mlw.get_registered_model_rows(registry)
    if not rows:
        st.info("No models are registered in configs/models/model_registry.yaml.")
        return

    row_by_id = {row["model_id"]: row for row in rows}
    model_options = [NEW_MODEL_SENTINEL] + [row["model_id"] for row in rows]
    selected = st.selectbox(
        "Selected Model",
        model_options,
        format_func=lambda mid: "New Model" if mid == NEW_MODEL_SENTINEL else mlw._model_label(row_by_id[mid]),
        key="model_lab_registered_model_id_single_editor",
    )

    if selected == NEW_MODEL_SENTINEL:
        template_id = st.selectbox(
            "Template Model",
            [row["model_id"] for row in rows],
            format_func=lambda mid: mlw._model_label(row_by_id[mid]),
            key="model_lab_template_model_id",
        )
        template_context = mlw.resolve_model_workflow_context(registry=registry, model_id=template_id)
        new_model_id = _safe_model_id(st.text_input("New Model ID", value=f"{template_id}_exp01", key="model_lab_new_model_id"))
        artifact_dir = st.text_input(
            "Artifact Directory",
            value=_default_artifact_dir(template_context["model_family"], template_context["market_key"], new_model_id or "new_model"),
            key="model_lab_new_artifact_dir",
        )
        context = _build_new_context(template_context, model_id=new_model_id, artifact_dir=artifact_dir)
    else:
        context = mlw.resolve_model_workflow_context(registry=registry, model_id=selected)

    mlw._render_kpis(context)
    mlw._render_model_bar(context, registry)
    mlw._render_registry_table(rows)

    top_left, top_right = st.columns([1.05, 1.45], gap="medium")
    with top_left:
        _patched_lifecycle_controls(context, registry)
        form_values = mlw._render_config_editor(context, registry)
    with top_right:
        feature_values = mlw._render_feature_bundle_editor(context)
        mlw._render_performance(context)

    can_save = context.get("is_new_model") or context.get("status") == "draft"
    can_delete = (not context.get("is_new_model")) and str(context.get("status") or "").lower() in {"draft", "archived"} and not _registry_is_active_primary(registry, context)
    save_col, delete_col = st.columns([3, 1])
    with save_col:
        if st.button("Save Draft Configuration", type="primary", disabled=not can_save, use_container_width=True, key="mlab_save_draft_config_single"):
            ok, msg = _save_new_or_existing_model(context=context, registry=registry, form_values=form_values, feature_values=feature_values)
            st.success(msg) if ok else st.error(msg)
            if ok:
                st.cache_data.clear()
                st.rerun()
    with delete_col:
        if st.button("Delete Model", disabled=not can_delete, use_container_width=True, key="mlab_delete_model_button"):
            st.session_state["mlab_delete_candidate"] = context["model_id"]

    if st.session_state.get("mlab_delete_candidate") == context.get("model_id"):
        _render_delete_dialog(context, registry)

    bottom_left, bottom_right = st.columns([1.05, 1.35], gap="medium")
    with bottom_left:
        if not context.get("is_new_model"):
            mlw._render_comparison(context, registry, context["model_id"])
        else:
            st.info("Save the new draft before comparing it to existing models.")
    with bottom_right:
        if not context.get("is_new_model"):
            mlw._render_actions(context)
        else:
            st.info("Save the new draft before running workflows.")


def render_model_lab():
    st.markdown(MODEL_LAB_VISUAL_REFINEMENT_CSS, unsafe_allow_html=True)
    mlw._inject_css()
    mlw._render_header()
    try:
        _render_single_editor()
    except Exception as exc:
        st.error(f"Unable to render Model Lab: {exc}")
