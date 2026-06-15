from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

import utils.feature_registry as fr
import utils.model_lab_workflows as mlw


def _feature_options(registry: dict[str, Any]) -> list[str]:
    return sorted(fr.feature_map(registry).keys())


def _save_registry_to_github(registry: dict[str, Any]) -> tuple[bool, str]:
    return mlw._github_write_file(
        str(fr.FEATURE_REGISTRY_PATH),
        fr.dump_yaml(registry),
        "Update Model Lab feature registry",
    )


def _render_summary(registry: dict[str, Any]) -> None:
    features = fr.feature_map(registry)
    bundles = fr.bundle_map(registry)
    findings = fr.validate_registry(registry)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Features", len(features))
    c2.metric("Bundles", len(bundles))
    c3.metric("Formula", sum(1 for item in features.values() if item.get("type") == "formula"))
    c4.metric("Findings", len(findings))

    st.caption("Feature Studio reads canonical feature definitions from feature_registry.yaml and bundles from feature_bundles.yaml.")


def _render_validation(registry: dict[str, Any]) -> None:
    findings = fr.validate_registry(registry)
    with st.expander("Validation Results", expanded=bool(findings)):
        if not findings:
            st.success("Feature registry validation passed.")
            return
        st.dataframe(pd.DataFrame(findings), use_container_width=True, hide_index=True)


def _render_feature_table(registry: dict[str, Any]) -> None:
    st.markdown("#### Feature Library")
    rows = fr.feature_rows(registry)
    if not rows:
        st.info("No canonical feature definitions found yet.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_bundle_table(registry: dict[str, Any]) -> None:
    st.markdown("#### Bundle Library")
    rows = fr.bundle_rows(registry)
    if not rows:
        st.info("No bundle definitions found in configs/features/feature_bundles.yaml.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_feature_editor(registry: dict[str, Any]) -> dict[str, Any] | None:
    features = fr.feature_map(registry)
    choices = ["+ Create new feature"] + _feature_options(registry)
    selected = st.selectbox("Feature", choices, key="mlab_feature_studio_feature_select")
    is_new = selected == "+ Create new feature"
    current = {} if is_new else features.get(selected, {})

    default_id = "" if is_new else selected
    feature_id = fr.safe_feature_id(st.text_input("Feature ID", value=default_id, key="mlab_feature_id"))
    label = st.text_input("Label", value=current.get("label", feature_id), key="mlab_feature_label")
    feature_type = st.selectbox(
        "Build Type",
        fr.FEATURE_TYPES,
        index=fr.FEATURE_TYPES.index(current.get("type", "formula")) if current.get("type", "formula") in fr.FEATURE_TYPES else 1,
        key="mlab_feature_type",
    )
    status = st.selectbox(
        "Status",
        fr.FEATURE_STATUSES,
        index=fr.FEATURE_STATUSES.index(current.get("status", "draft")) if current.get("status", "draft") in fr.FEATURE_STATUSES else 0,
        key="mlab_feature_status",
    )
    family = st.text_input("Family", value=current.get("family", "custom"), key="mlab_feature_family")
    description = st.text_area("Description", value=current.get("description", ""), key="mlab_feature_description")
    inputs = fr.csv_to_list(st.text_area("Inputs comma/newline separated", value="\n".join(current.get("inputs", []) or []), key="mlab_feature_inputs"))

    payload: dict[str, Any] = {
        "label": label or feature_id,
        "type": feature_type,
        "status": status,
        "family": family,
        "description": description,
        "inputs": inputs,
        "leakage_safe": st.checkbox("Leakage safe", value=bool(current.get("leakage_safe", True)), key="mlab_feature_leakage_safe"),
    }

    if feature_type == "formula":
        payload["formula"] = st.text_area("Formula", value=current.get("formula", ""), key="mlab_feature_formula")
        issues = fr.validate_formula_syntax(payload["formula"])
        if issues:
            st.warning("Formula validation: " + "; ".join(issues))
        else:
            st.success("Formula syntax looks valid.")
    elif feature_type == "pipeline":
        payload["builder"] = st.text_input("Builder path", value=current.get("builder", ""), key="mlab_feature_builder")
    elif feature_type == "transform":
        payload["transform"] = st.text_input("Transform ID", value=current.get("transform", "red_minus_blue"), key="mlab_feature_transform")
        payload["source_columns"] = fr.csv_to_list(st.text_area("Source columns", value="\n".join(current.get("source_columns", []) or []), key="mlab_feature_source_columns"))
    else:
        payload["source_column"] = st.text_input("Source column", value=current.get("source_column", ""), key="mlab_feature_source_column")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Stage Feature Change", use_container_width=True, key="mlab_stage_feature"):
            if not feature_id:
                st.error("Feature ID is required.")
                return None
            updated = fr.upsert_feature(registry, feature_id, payload)
            st.session_state["mlab_feature_registry_staged"] = updated
            st.success(f"Staged feature: {feature_id}")
            return updated
    with c2:
        if not is_new and st.button("Archive Feature", use_container_width=True, key="mlab_archive_feature"):
            updated = fr.archive_feature(registry, selected)
            st.session_state["mlab_feature_registry_staged"] = updated
            st.warning(f"Staged archive: {selected}")
            return updated
    with c3:
        delete_enabled = not is_new and st.checkbox("Confirm delete feature", key="mlab_confirm_delete_feature")
        if not is_new and st.button("Delete Feature", disabled=not delete_enabled, use_container_width=True, key="mlab_delete_feature"):
            updated = fr.delete_feature(registry, selected)
            st.session_state["mlab_feature_registry_staged"] = updated
            st.error(f"Staged delete: {selected}")
            return updated
    return None


def _render_bundle_editor(registry: dict[str, Any]) -> None:
    st.markdown("#### Bundle Registry")
    st.caption("Bundles are separate from feature families and currently read from configs/features/feature_bundles.yaml. Editing remains disabled during the 10-feature registry test slice.")
    _render_bundle_table(registry)


def _render_save_controls(registry: dict[str, Any]) -> None:
    staged = st.session_state.get("mlab_feature_registry_staged")
    active_registry = staged or registry
    st.markdown("#### Save Registry")
    if staged:
        st.info("You have staged feature registry changes. Save to GitHub to persist them.")
    else:
        st.caption("No staged feature-definition changes.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save Feature Registry to GitHub", use_container_width=True, key="mlab_save_feature_registry"):
            ok, msg = _save_registry_to_github(active_registry)
            if ok:
                st.success(msg)
                st.session_state.pop("mlab_feature_registry_staged", None)
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(msg)
    with c2:
        if st.button("Discard Staged Changes", use_container_width=True, key="mlab_discard_feature_registry"):
            st.session_state.pop("mlab_feature_registry_staged", None)
            st.info("Discarded staged feature registry changes.")
            st.rerun()


def render_features(
    registry: dict[str, Any],
    rows: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    *,
    existing_model_selector,
) -> None:
    st.markdown("## Features")
    st.caption("Feature Studio is the UI over canonical feature definitions and separate bundle definitions.")

    loaded = fr.load_feature_registry()
    active_registry = st.session_state.get("mlab_feature_registry_staged") or loaded

    _render_summary(active_registry)
    _render_validation(active_registry)

    tab_library, tab_feature, tab_bundle, tab_yaml = st.tabs([
        "Library",
        "Feature Editor",
        "Bundle Registry",
        "Registry YAML",
    ])

    with tab_library:
        _render_feature_table(active_registry)
        _render_bundle_table(active_registry)

    with tab_feature:
        updated = _render_feature_editor(active_registry)
        if updated is not None:
            active_registry = updated

    with tab_bundle:
        _render_bundle_editor(active_registry)

    with tab_yaml:
        st.code(fr.dump_yaml(active_registry), language="yaml")

    _render_save_controls(active_registry)
