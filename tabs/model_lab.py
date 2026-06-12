from __future__ import annotations

from copy import deepcopy
from typing import Any

import streamlit as st

import utils.model_lab_workflows as mlw


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


def _parameter_controls(params: dict[str, Any], *, key_prefix: str) -> dict[str, Any]:
    st.markdown("##### XGBoost Parameters")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        n_estimators = st.number_input("N Estimators", value=int(params.get("n_estimators", 500)), min_value=50, step=50, key=f"{key_prefix}_n_estimators")
    with c2:
        max_depth = st.number_input("Max Depth", value=int(params.get("max_depth", 4)), min_value=1, max_value=12, step=1, key=f"{key_prefix}_max_depth")
    with c3:
        learning_rate = st.number_input("Learning Rate", value=float(params.get("learning_rate", 0.03)), min_value=0.001, max_value=1.0, step=0.01, format="%.3f", key=f"{key_prefix}_learning_rate")
    with c4:
        subsample = st.number_input("Subsample", value=float(params.get("subsample", 0.8)), min_value=0.1, max_value=1.0, step=0.05, format="%.2f", key=f"{key_prefix}_subsample")
    with c5:
        colsample = st.number_input("Colsample", value=float(params.get("colsample_bytree", 0.8)), min_value=0.1, max_value=1.0, step=0.05, format="%.2f", key=f"{key_prefix}_colsample")
    return {
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "learning_rate": float(learning_rate),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample),
        "random_state": int(params.get("random_state", 42)),
        "eval_metric": params.get("eval_metric", "logloss"),
    }


def _feature_controls(context: dict[str, Any], *, key_prefix: str) -> dict[str, Any]:
    available = mlw._available_feature_columns(context)
    bundle_map = mlw._bundle_map(available)
    default_bundles = mlw._infer_selected_bundles(context["config"], available)
    selected = st.multiselect(
        "Feature Bundles",
        list(bundle_map.keys()),
        default=[bundle for bundle in default_bundles if bundle in bundle_map],
        format_func=lambda key: f"{mlw.BUNDLE_LABELS.get(key, key)} ({len(bundle_map.get(key, []))})",
        key=f"{key_prefix}_bundles",
    )
    include_text = st.text_area("Include Feature Overrides", value="", height=60, key=f"{key_prefix}_include")
    exclude_text = st.text_area("Exclude Feature Overrides", value="", height=60, key=f"{key_prefix}_exclude")
    include = mlw._csv_to_list(include_text)
    exclude = mlw._csv_to_list(exclude_text)
    resolved = mlw._resolve_features_from_bundles(
        available_features=available,
        selected_bundles=selected,
        include_overrides=include,
        exclude_overrides=exclude,
    )
    if not resolved:
        resolved = list((context["config"].get("features") or {}).get("feature_columns") or [])
    st.caption(f"Resolved feature count: {len(resolved):,}")
    return {"selected_bundles": selected, "include_features": include, "exclude_features": exclude, "resolved_features": resolved}


def _apply_manager_updates(config: dict[str, Any], *, model_id: str, display_name: str, model_family: str, market_key: str, artifact_dir: str, params: dict[str, Any], calibration_enabled: bool, calibration_method: str, clip_low: float, clip_high: float, feature_values: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(config)
    updated["model_id"] = model_id
    updated["artifact_name"] = model_id
    updated["display_name"] = display_name
    updated["model_family"] = model_family
    updated["market_key"] = market_key
    updated["status"] = "draft"
    updated.setdefault("artifacts", {})["output_dir"] = artifact_dir
    updated["params"] = params
    updated.setdefault("calibration", {})["enabled"] = bool(calibration_enabled)
    updated.setdefault("calibration", {})["method"] = calibration_method
    updated.setdefault("prediction", {}).setdefault("probability", {})["clip_low"] = float(clip_low)
    updated.setdefault("prediction", {}).setdefault("probability", {})["clip_high"] = float(clip_high)
    features = updated.setdefault("features", {})
    features["selection_mode"] = "explicit"
    features["selected_bundles"] = feature_values["selected_bundles"]
    features["include_features"] = feature_values["include_features"]
    features["exclude_features"] = feature_values["exclude_features"]
    features["feature_columns"] = feature_values["resolved_features"]
    features["expected_feature_count"] = len(feature_values["resolved_features"])
    return updated


def _save_model_config_and_registry(*, registry: dict[str, Any], model_id: str, config: dict[str, Any], display_name: str, description: str, model_family: str, market_key: str, artifact_dir: str, config_path: str) -> tuple[bool, str]:
    if not model_id:
        return False, "Model ID is required."
    if not (config.get("features") or {}).get("feature_columns"):
        return False, "Feature selection resolved to zero features."

    updated_registry = deepcopy(registry)
    entry = updated_registry.setdefault("models", {}).setdefault(model_id, {})
    entry.update(
        {
            "display_name": display_name,
            "description": description,
            "model_family": model_family,
            "market_key": market_key,
            "algorithm": config.get("algorithm", "xgboost"),
            "config_path": config_path,
            "artifact_dir": artifact_dir,
            "status": "draft",
            "dashboard_selectable": False,
            "outcome_architecture": True,
        }
    )

    ok, msg = mlw._github_write_file(config_path, mlw._yaml_dump(config), f"Save model config {model_id} from Model Lab")
    if not ok:
        return ok, msg
    ok, msg = mlw._save_registry(updated_registry)
    if not ok:
        return ok, msg
    return True, f"Saved draft model {model_id}."


def _delete_model_from_registry(*, registry: dict[str, Any], model_id: str) -> tuple[bool, str]:
    models = registry.get("models") or {}
    if model_id not in models:
        return False, f"Model is not registered: {model_id}"

    entry = models[model_id]
    status = str(entry.get("status") or "draft").lower()
    if status == "production":
        return False, "Production models cannot be deleted. Move them out of production or archive them first."

    updated = deepcopy(registry)
    updated.get("models", {}).pop(model_id, None)

    # Remove any active-model pointer that references the deleted model.
    for family_entry in (updated.get("active_models") or {}).values():
        if isinstance(family_entry, dict) and family_entry.get("primary") == model_id:
            family_entry.pop("primary", None)

    ok, msg = mlw._save_registry(updated)
    if not ok:
        return ok, msg
    return True, f"Deleted registry entry for {model_id}. Config YAML and trained artifacts were left in the repo for audit/history."


def _show_delete_confirmation(registry: dict[str, Any], model_id: str) -> None:
    entry = (registry.get("models") or {}).get(model_id, {})
    status = str(entry.get("status") or "draft").lower()

    def _dialog_body() -> None:
        st.warning(f"You are about to delete model `{model_id}` from the model registry.")
        st.caption("This removes the model from dashboard/runtime selection. It does not delete the config YAML or trained artifacts from the repository.")
        typed = st.text_input("Type the model ID to confirm", key=f"mlab_delete_confirm_text_{model_id}")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Delete Model", disabled=(typed != model_id or status == "production"), type="primary", use_container_width=True, key=f"mlab_delete_confirm_button_{model_id}"):
                ok, msg = _delete_model_from_registry(registry=registry, model_id=model_id)
                st.success(msg) if ok else st.error(msg)
                if ok:
                    st.cache_data.clear()
                    st.rerun()
        with c2:
            if st.button("Cancel", use_container_width=True, key=f"mlab_delete_cancel_{model_id}"):
                st.rerun()

    if hasattr(st, "dialog"):
        @st.dialog("Confirm Delete Existing Model")
        def _confirm_dialog() -> None:
            _dialog_body()

        _confirm_dialog()
    else:
        st.error("Confirm delete")
        _dialog_body()


def _render_model_configuration_manager() -> None:
    try:
        registry = mlw.load_model_registry()
        rows = mlw.get_registered_model_rows(registry)
    except Exception as exc:
        st.warning(f"Model Configuration Manager unavailable: {exc}")
        return

    if not rows:
        return

    model_ids = [row["model_id"] for row in rows]
    row_by_id = {row["model_id"]: row for row in rows}
    with st.expander("Model Configuration Manager", expanded=True):
        st.caption("Create new draft experiments, edit existing drafts, or delete non-production registry entries. Production models are protected.")
        mode = st.radio("Mode", ["Create New Model", "Edit Existing Model", "Delete Existing Model"], horizontal=True, key="mlab_manager_mode")

        selected_model_id = st.selectbox(
            "Model / Template",
            model_ids,
            format_func=lambda mid: f"{mid} ({row_by_id[mid].get('status', 'unknown')})",
            key="mlab_manager_model_select",
        )
        context = mlw.resolve_model_workflow_context(registry=registry, model_id=selected_model_id)
        source_status = str(context.get("status") or "draft").lower()

        if mode == "Delete Existing Model":
            st.markdown("##### Delete Existing Model")
            st.caption("Deletes the registry entry only. Config YAML and trained artifacts remain available in the repository for audit/history.")
            if source_status == "production":
                st.error("Production models cannot be deleted from Model Lab. Archive or demote them first.")
            else:
                st.warning(f"Selected model: `{selected_model_id}` ({source_status})")
                if st.button("Delete Existing Model", type="primary", use_container_width=True, key="mlab_delete_open_dialog"):
                    _show_delete_confirmation(registry, selected_model_id)
            return

        source_config = context["config"]
        source_params = source_config.get("params") or {}
        source_probability = (source_config.get("prediction") or {}).get("probability") or {}
        source_calibration = source_config.get("calibration") or {}

        is_create = mode == "Create New Model"
        if not is_create and source_status != "draft":
            st.info("This model is production/archived and is read-only. Switch to Create New Model to create an editable draft experiment from it.")
            return

        default_model_id = f"{selected_model_id}_exp01" if is_create else selected_model_id
        raw_model_id = st.text_input("Model ID", value=default_model_id, disabled=not is_create, key=f"mlab_manager_model_id_{mode}")
        model_id = _safe_model_id(raw_model_id)
        display_name = st.text_input("Display Name", value=(f"{context.get('display_name', selected_model_id)} Experiment" if is_create else context.get("display_name", selected_model_id)), key=f"mlab_manager_display_{mode}")
        description = st.text_area("Description", value=(f"Draft experiment created from {selected_model_id}." if is_create else context.get("description", "")), height=68, key=f"mlab_manager_desc_{mode}")

        c1, c2, c3 = st.columns(3)
        with c1:
            model_family = st.selectbox("Model Family", ["moneyline", "prop"], index=1 if context.get("model_family") == "prop" else 0, key=f"mlab_manager_family_{mode}")
        with c2:
            market_defaults = list(dict.fromkeys([context.get("market_key") or "moneyline", "moneyline", "goes_distance", "win_by_ko_tko_dq", "win_by_submission", "win_by_decision"]))
            market_key = st.selectbox("Market Key", market_defaults, key=f"mlab_manager_market_{mode}")
        with c3:
            artifact_default = _default_artifact_dir(model_family, market_key, model_id or "new_model") if is_create else context.get("artifact_dir", "")
            artifact_dir = st.text_input("Artifact Dir", value=artifact_default, key=f"mlab_manager_artifact_{mode}")

        params = _parameter_controls(source_params, key_prefix=f"mlab_manager_{mode}")
        c4, c5, c6, c7 = st.columns(4)
        with c4:
            calibration_enabled = st.toggle("Calibration Enabled", value=bool(source_calibration.get("enabled", True)), key=f"mlab_manager_cal_enabled_{mode}")
        with c5:
            method_value = str(source_calibration.get("method", "isotonic"))
            method_options = ["isotonic", "sigmoid", "none"]
            calibration_method = st.selectbox("Calibration Method", method_options, index=method_options.index(method_value) if method_value in method_options else 0, key=f"mlab_manager_cal_method_{mode}")
        with c6:
            clip_low = st.number_input("Clip Low", value=float(source_probability.get("clip_low", 0.02)), min_value=0.0, max_value=0.49, step=0.01, format="%.2f", key=f"mlab_manager_clip_low_{mode}")
        with c7:
            clip_high = st.number_input("Clip High", value=float(source_probability.get("clip_high", 0.98)), min_value=0.51, max_value=1.0, step=0.01, format="%.2f", key=f"mlab_manager_clip_high_{mode}")

        feature_values = _feature_controls(context, key_prefix=f"mlab_manager_{mode}")
        config_path = f"configs/models/{model_id}.yaml" if is_create else context["config_path"]
        updated_config = _apply_manager_updates(
            source_config,
            model_id=model_id,
            display_name=display_name,
            model_family=model_family,
            market_key=market_key,
            artifact_dir=artifact_dir,
            params=params,
            calibration_enabled=calibration_enabled,
            calibration_method=calibration_method,
            clip_low=clip_low,
            clip_high=clip_high,
            feature_values=feature_values,
        )

        button_label = "Create New Draft Model" if is_create else "Save Existing Draft Model"
        if st.button(button_label, type="primary", use_container_width=True, disabled=not bool(model_id and display_name), key=f"mlab_manager_save_{mode}"):
            if is_create and model_id in (registry.get("models") or {}):
                st.error(f"Model already exists: {model_id}")
            else:
                ok, msg = _save_model_config_and_registry(
                    registry=registry,
                    model_id=model_id,
                    config=updated_config,
                    display_name=display_name,
                    description=description,
                    model_family=model_family,
                    market_key=market_key,
                    artifact_dir=artifact_dir,
                    config_path=config_path,
                )
                st.success(msg) if ok else st.error(msg)
                if ok:
                    st.cache_data.clear()
                    st.rerun()


def _patched_lifecycle_controls(context: dict[str, Any], registry: dict[str, Any]) -> None:
    status = str(context.get("status") or "draft").lower()
    st.markdown("#### Lifecycle")
    if status == "production":
        st.info("Production configs are read-only. Use Model Configuration Manager → Create New Model to create an editable draft experiment from this model.")
    elif status == "draft":
        st.success("Draft model is editable. Use Model Configuration Manager or the draft editor to update config values.")
        new_status = st.selectbox("Status", mlw.STATUS_OPTIONS, index=mlw.STATUS_OPTIONS.index(status), key="mlab_status_select")
        set_active = st.toggle("Make active primary for this family on save", value=False, key="mlab_set_active_primary")
        if st.button("Update Registry Status", use_container_width=True, key="mlab_update_status"):
            updated = deepcopy(registry)
            updated["models"][context["model_id"]]["status"] = new_status
            if set_active and new_status == "production":
                updated.setdefault("active_models", {}).setdefault(context["model_family"], {})["primary"] = context["model_id"]
            ok, msg = mlw._save_registry(updated)
            st.success(msg) if ok else st.error(msg)
            if ok:
                st.cache_data.clear()
                st.rerun()
    else:
        st.warning("Archived models are read-only. Use Model Configuration Manager → Create New Model to start a new draft experiment.")


def render_model_lab():
    st.markdown(MODEL_LAB_VISUAL_REFINEMENT_CSS, unsafe_allow_html=True)
    mlw._render_lifecycle_controls = _patched_lifecycle_controls
    _render_model_configuration_manager()
    mlw.render_model_workflow_launcher()
