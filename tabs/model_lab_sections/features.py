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


def _save_bundle_registry_to_github(registry: dict[str, Any]) -> tuple[bool, str]:
    return mlw._github_write_file(
        str(fr.FEATURE_BUNDLE_REGISTRY_PATH),
        fr.dump_yaml(registry),
        "Update Model Lab feature bundle registry",
    )


def _features_by_family(registry: dict[str, Any]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for feature_id, feature in fr.feature_map(registry).items():
        family = str((feature or {}).get("family") or "unassigned")
        grouped.setdefault(family, []).append(str(feature_id))
    return {family: sorted(features) for family, features in sorted(grouped.items())}


def _sync_feature_editor_state(selected: str, current: dict[str, Any], is_new: bool) -> None:
    if st.session_state.get("mlab_feature_editor_loaded") == selected:
        return
    st.session_state["mlab_feature_editor_loaded"] = selected
    st.session_state["mlab_feature_id"] = "" if is_new else selected
    st.session_state["mlab_feature_label"] = current.get("label", "" if is_new else selected)
    st.session_state["mlab_feature_type"] = current.get("type", "formula") if current.get("type", "formula") in fr.FEATURE_TYPES else "formula"
    st.session_state["mlab_feature_status"] = current.get("status", "draft") if current.get("status", "draft") in fr.FEATURE_STATUSES else "draft"
    st.session_state["mlab_feature_family"] = current.get("family", "custom")
    st.session_state["mlab_feature_description"] = current.get("description", "")
    st.session_state["mlab_feature_inputs"] = "\n".join(str(item) for item in current.get("inputs", []) or [])
    st.session_state["mlab_feature_leakage_safe"] = bool(current.get("leakage_safe", True))
    st.session_state["mlab_feature_formula"] = current.get("formula", "")
    st.session_state["mlab_feature_builder"] = current.get("builder", "")
    st.session_state["mlab_feature_transform"] = current.get("transform", "red_minus_blue")
    st.session_state["mlab_feature_source_columns"] = "\n".join(str(item) for item in current.get("source_columns", []) or [])
    st.session_state["mlab_feature_source_column"] = current.get("source_column", "")


def _sync_bundle_editor_state(selected: str, current: dict[str, Any], is_new: bool) -> None:
    if st.session_state.get("mlab_bundle_editor_loaded") == selected:
        return
    st.session_state["mlab_bundle_editor_loaded"] = selected
    st.session_state["mlab_bundle_id"] = "" if is_new else selected
    st.session_state["mlab_bundle_description"] = current.get("description", "")
    st.session_state["mlab_bundle_source_layer"] = current.get("source_layer", "fighter_state")
    st.session_state["mlab_bundle_source_prefix"] = current.get("source_prefix", "")
    st.session_state["mlab_bundle_candidate_columns"] = "\n".join(str(item) for item in current.get("candidate_columns", []) or [])
    st.session_state["mlab_bundle_markets"] = current.get("markets", []) or []
    current_transforms = [str(item) for item in current.get("recommended_transforms", []) or []]
    transform_options = fr.transform_options(include_planned=True)
    st.session_state["mlab_bundle_recommended_transforms"] = [item for item in current_transforms if item in transform_options]
    st.session_state["mlab_bundle_unknown_transforms"] = "\n".join(item for item in current_transforms if item not in transform_options)


def _render_summary(registry: dict[str, Any]) -> None:
    features = fr.feature_map(registry)
    bundles = fr.bundle_map()
    findings = fr.validate_registry(registry)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Features", len(features))
    c2.metric("Bundles", len(bundles))
    c3.metric("Formula", sum(1 for item in features.values() if item.get("type") == "formula"))
    c4.metric("Findings", len(findings))
    st.caption("Feature Studio reads canonical feature definitions from feature_registry.yaml and bundles from feature_bundles.yaml.")


def _render_validation(registry: dict[str, Any]) -> None:
    findings = fr.validate_registry(registry)
    bundle_findings = fr.validate_bundle_registry(fr.load_feature_bundle_registry())
    with st.expander("Validation Results", expanded=bool(findings or bundle_findings)):
        if not findings and not bundle_findings:
            st.success("Feature and bundle registry validation passed.")
            return
        if findings:
            st.markdown("**Feature findings**")
            st.dataframe(pd.DataFrame(findings), use_container_width=True, hide_index=True)
        if bundle_findings:
            st.markdown("**Bundle findings**")
            st.dataframe(pd.DataFrame(bundle_findings), use_container_width=True, hide_index=True)


def _render_feature_table(registry: dict[str, Any]) -> None:
    st.markdown("#### Feature Library")
    rows = fr.feature_rows(registry)
    if not rows:
        st.info("No canonical feature definitions found yet.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_bundle_table(bundle_registry: dict[str, Any] | None = None) -> None:
    st.markdown("#### Bundle Library")
    rows = fr.bundle_rows(bundle_registry)
    if not rows:
        st.info("No bundle definitions found in configs/features/feature_bundles.yaml.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_transform_selector() -> str:
    transform_ids = fr.transform_options(include_planned=True)
    current_transform = str(st.session_state.get("mlab_feature_transform") or "")
    if current_transform and current_transform not in transform_ids:
        transform_ids = [current_transform] + transform_ids
    if not transform_ids:
        st.error("No transforms found in configs/features/transform_registry.yaml.")
        return current_transform
    if not current_transform:
        st.session_state["mlab_feature_transform"] = transform_ids[0]
    st.selectbox("Transform ID", transform_ids, key="mlab_feature_transform", help="Loaded from configs/features/transform_registry.yaml")
    return str(st.session_state.get("mlab_feature_transform") or "")


def _render_feature_editor(registry: dict[str, Any]) -> dict[str, Any] | None:
    features = fr.feature_map(registry)
    choices = ["+ Create new feature"] + _feature_options(registry)
    selected = st.selectbox("Feature", choices, key="mlab_feature_studio_feature_select")
    is_new = selected == "+ Create new feature"
    current = {} if is_new else features.get(selected, {})
    _sync_feature_editor_state(selected, current, is_new)

    st.text_input("Feature ID", key="mlab_feature_id")
    feature_id = fr.safe_feature_id(st.session_state.get("mlab_feature_id", ""))
    st.text_input("Label", key="mlab_feature_label")
    feature_type = st.selectbox("Build Type", fr.FEATURE_TYPES, key="mlab_feature_type")
    st.selectbox("Status", fr.FEATURE_STATUSES, key="mlab_feature_status")
    st.text_input("Family", key="mlab_feature_family")
    st.text_area("Description", key="mlab_feature_description")
    st.text_area("Inputs comma/newline separated", key="mlab_feature_inputs")

    payload: dict[str, Any] = {
        "label": st.session_state.get("mlab_feature_label") or feature_id,
        "type": st.session_state.get("mlab_feature_type"),
        "status": st.session_state.get("mlab_feature_status"),
        "family": st.session_state.get("mlab_feature_family"),
        "description": st.session_state.get("mlab_feature_description", ""),
        "inputs": fr.csv_to_list(st.session_state.get("mlab_feature_inputs", "")),
        "leakage_safe": st.checkbox("Leakage safe", key="mlab_feature_leakage_safe"),
    }

    if feature_type == "formula":
        st.text_area("Formula", key="mlab_feature_formula")
        payload["formula"] = st.session_state.get("mlab_feature_formula", "")
        issues = fr.validate_formula_syntax(payload["formula"])
        st.warning("Formula validation: " + "; ".join(issues)) if issues else st.success("Formula syntax looks valid.")
    elif feature_type == "pipeline":
        st.text_input("Builder path", key="mlab_feature_builder")
        payload["builder"] = st.session_state.get("mlab_feature_builder", "")
    elif feature_type == "transform":
        payload["transform"] = _render_transform_selector()
        st.text_area("Source columns", key="mlab_feature_source_columns")
        payload["source_columns"] = fr.csv_to_list(st.session_state.get("mlab_feature_source_columns", ""))
    else:
        st.text_input("Source column", key="mlab_feature_source_column")
        payload["source_column"] = st.session_state.get("mlab_feature_source_column", "")

    for optional_key in ["output_column", "model_input_allowed", "current_moneyline_v5"]:
        if optional_key in current:
            payload[optional_key] = current[optional_key]

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Save Feature to Registry", use_container_width=True, key="mlab_stage_feature"):
            if not feature_id:
                st.error("Feature ID is required.")
                return None
            updated = fr.upsert_feature(registry, feature_id, payload)
            ok, msg = _save_registry_to_github(updated)
            if ok:
                st.cache_data.clear()
                st.success(f"Saved feature: {feature_id}")
                st.rerun()
            else:
                st.error(msg)
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


def _render_family_feature_selector(feature_registry: dict[str, Any], current_candidates: list[str]) -> list[str]:
    grouped = _features_by_family(feature_registry)
    if not grouped:
        st.info("No registered feature definitions available for family selection.")
        return []

    current_set = set(current_candidates)
    selected_families = st.multiselect(
        "Feature Families",
        list(grouped.keys()),
        default=[family for family, features in grouped.items() if current_set.intersection(features)],
        help="Pick families, then check the specific features to include in the bundle.",
        key="mlab_bundle_selected_families",
    )

    selected_features: list[str] = []
    for family in selected_families:
        features = grouped.get(family, [])
        with st.expander(f"{family} ({len(features)} features)", expanded=True):
            cols = st.columns(3)
            for i, feature_id in enumerate(features):
                default = feature_id in current_set
                key = f"mlab_bundle_family_feature_{fr.safe_feature_id(family)}_{fr.safe_feature_id(feature_id)}"
                label = feature_registry.get("feature_definitions", {}).get(feature_id, {}).get("label", feature_id)
                with cols[i % 3]:
                    checked = st.checkbox(str(label), value=default, key=key, help=feature_id)
                if checked:
                    selected_features.append(feature_id)

    return list(dict.fromkeys(selected_features))


def _render_bundle_editor() -> None:
    st.markdown("#### Bundle Editor")
    bundle_registry = fr.load_feature_bundle_registry()
    feature_registry = fr.load_feature_registry()
    bundles = fr.bundle_map(bundle_registry)
    choices = ["+ Create new bundle"] + sorted(bundles.keys())
    selected = st.selectbox("Bundle", choices, key="mlab_bundle_select")
    is_new = selected == "+ Create new bundle"
    current = {} if is_new else bundles.get(selected, {})
    _sync_bundle_editor_state(selected, current, is_new)

    st.text_input("Bundle ID", key="mlab_bundle_id")
    bundle_id = fr.safe_feature_id(st.session_state.get("mlab_bundle_id", ""))
    st.text_area("Description", key="mlab_bundle_description")
    st.text_input("Source layer", key="mlab_bundle_source_layer")
    st.text_input("Source prefix optional", key="mlab_bundle_source_prefix")

    current_candidates = fr.csv_to_list(st.session_state.get("mlab_bundle_candidate_columns", ""))
    selected_family_features = _render_family_feature_selector(feature_registry, current_candidates)
    registered_features = set(fr.feature_map(feature_registry).keys())
    manual_defaults = [item for item in current_candidates if item not in registered_features]
    manual_candidates = st.text_area(
        "Manual extra candidate columns",
        value="\n".join(manual_defaults),
        key="mlab_bundle_manual_candidate_columns",
        help="Optional: raw/generated columns that are not registered as feature definitions yet.",
    )
    candidate_columns = list(dict.fromkeys(selected_family_features + fr.csv_to_list(manual_candidates)))
    st.caption(f"Bundle candidate count: {len(candidate_columns)}")

    transform_ids = fr.transform_options(include_planned=True)
    st.multiselect("Recommended transforms", transform_ids, key="mlab_bundle_recommended_transforms")
    unknown_transforms = fr.csv_to_list(st.text_area(
        "Unknown/custom transforms optional",
        key="mlab_bundle_unknown_transforms",
        help="Normally leave empty. Entries here will trigger validation warnings until registered.",
    ))

    market_options = ["moneyline", "props", "ko_tko", "submission", "decision", "goes_distance", "rounds"]
    st.multiselect("Markets", market_options, key="mlab_bundle_markets")

    payload: dict[str, Any] = {
        "description": st.session_state.get("mlab_bundle_description", ""),
        "source_layer": st.session_state.get("mlab_bundle_source_layer", ""),
        "candidate_columns": candidate_columns,
        "recommended_transforms": list(dict.fromkeys(list(st.session_state.get("mlab_bundle_recommended_transforms", []) or []) + unknown_transforms)),
        "markets": st.session_state.get("mlab_bundle_markets", []) or [],
    }
    source_prefix = str(st.session_state.get("mlab_bundle_source_prefix", "")).strip()
    if source_prefix:
        payload["source_prefix"] = source_prefix

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save Bundle to Registry", use_container_width=True, key="mlab_save_bundle"):
            if not bundle_id:
                st.error("Bundle ID is required.")
                return
            updated = fr.upsert_bundle(bundle_registry, bundle_id, payload)
            findings = fr.validate_bundle_registry(updated)
            blocking = [item for item in findings if item.get("level") == "error"]
            if blocking:
                st.error("Bundle has validation errors. Fix errors before saving.")
                st.dataframe(pd.DataFrame(blocking), use_container_width=True, hide_index=True)
                return
            ok, msg = _save_bundle_registry_to_github(updated)
            if ok:
                st.cache_data.clear()
                st.success(f"Saved bundle: {bundle_id}")
                st.rerun()
            else:
                st.error(msg)
    with c2:
        delete_enabled = not is_new and st.checkbox("Confirm delete bundle", key="mlab_confirm_delete_bundle")
        if not is_new and st.button("Delete Bundle", disabled=not delete_enabled, use_container_width=True, key="mlab_delete_bundle"):
            updated = fr.delete_bundle(bundle_registry, selected)
            ok, msg = _save_bundle_registry_to_github(updated)
            if ok:
                st.cache_data.clear()
                st.warning(f"Deleted bundle: {selected}")
                st.rerun()
            else:
                st.error(msg)

    findings = fr.validate_bundle_registry(bundle_registry)
    with st.expander("Bundle Validation", expanded=bool(findings)):
        if findings:
            st.dataframe(pd.DataFrame(findings), use_container_width=True, hide_index=True)
        else:
            st.success("Bundle registry validation passed.")
    _render_bundle_table(bundle_registry)


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
        if st.button("Save Staged Registry to GitHub", use_container_width=True, key="mlab_save_feature_registry"):
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

    tab_library, tab_feature, tab_bundle, tab_yaml = st.tabs(["Library", "Feature Editor", "Bundle Editor", "Registry YAML"])

    with tab_library:
        _render_feature_table(active_registry)
        _render_bundle_table(fr.load_feature_bundle_registry())
    with tab_feature:
        updated = _render_feature_editor(active_registry)
        if updated is not None:
            active_registry = updated
    with tab_bundle:
        _render_bundle_editor()
    with tab_yaml:
        st.markdown("**Feature Registry**")
        st.code(fr.dump_yaml(active_registry), language="yaml")
        st.markdown("**Bundle Registry**")
        st.code(fr.dump_yaml(fr.load_feature_bundle_registry()), language="yaml")

    _render_save_controls(active_registry)
