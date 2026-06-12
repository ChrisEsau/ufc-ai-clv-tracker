from __future__ import annotations

import re

import streamlit as st
import utils.model_lab_workflows as mlw


def _safe_key(value) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", str(value or ""))
    return cleaned.strip("_") or "feature"


def _inject_feature_selector_css() -> None:
    st.markdown(
        """
        <style>
        div[data-baseweb="tag"] {
            background: linear-gradient(180deg, rgba(37,99,235,.98), rgba(29,78,216,.96)) !important;
            border: 1px solid rgba(96,165,250,.85) !important;
            color: #ffffff !important;
        }
        div[data-baseweb="tag"] span,
        div[data-baseweb="tag"] svg {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        div[data-testid="stCheckbox"] label,
        div[data-testid="stCheckbox"] label span,
        div[data-testid="stCheckbox"] p,
        details[data-testid="stExpander"] summary,
        details[data-testid="stExpander"] summary p,
        details[data-testid="stExpander"] summary span {
            color: #f8fafc !important;
            opacity: 1 !important;
        }
        div[data-testid="stCheckbox"] input:checked + div {
            background-color: #2563eb !important;
            border-color: #60a5fa !important;
        }
        div[data-testid="stCheckbox"] input[type="checkbox"] {
            accent-color: #2563eb !important;
        }
        details[data-testid="stExpander"] {
            background: linear-gradient(180deg, rgba(9,19,32,.95), rgba(7,16,28,.98)) !important;
            border: 1px solid rgba(51,75,108,.95) !important;
            border-radius: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_feature_checklist(context):
    _inject_feature_selector_css()

    config = context["config"]
    editable = context.get("is_new_model") or context["status"] in mlw.EDITABLE_STATUSES
    model_key = _safe_key(context.get("model_id") or "new_model")
    feature_config = config.get("features") or {}
    current = set(feature_config.get("feature_columns") or [])
    available = mlw._available_feature_columns(context)
    bundle_map = mlw._bundle_map(available)
    saved = list(feature_config.get("selected_bundles") or [])
    if not saved:
        saved = mlw._infer_selected_bundles(config, available)
    saved_set = set(saved)

    st.markdown("#### Feature Selection")
    selected = st.multiselect(
        "Selected Bundles",
        list(bundle_map.keys()),
        default=[b for b in saved if b in bundle_map],
        format_func=lambda b: f"{mlw.BUNDLE_LABELS.get(b, b)} ({len(bundle_map.get(b, []))})",
        disabled=not editable,
        key=f"mlab_selected_bundles_{model_key}",
    )

    included = []
    removed = []
    universe = []
    for bundle in selected:
        features = list(bundle_map.get(bundle, []))
        universe.extend(features)
        label = mlw.BUNDLE_LABELS.get(bundle, bundle)
        with st.expander(f"{label} ({len(features)} features)", expanded=False):
            cols = st.columns(3)
            for i, feature in enumerate(features):
                default = feature in current if bundle in saved_set else True
                key = f"mlab_feature_{model_key}_{_safe_key(bundle)}_{_safe_key(feature)}"
                with cols[i % 3]:
                    value = st.checkbox(feature, value=default, disabled=not editable, key=key)
                if value:
                    included.append(feature)
                else:
                    removed.append(feature)

    resolved = list(dict.fromkeys(included))
    removed = list(dict.fromkeys(removed))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bundles", len(selected))
    c2.metric("Available", len(list(dict.fromkeys(universe))))
    c3.metric("Included", len(resolved))
    c4.metric("Unchecked", len(removed))
    return {"selected_bundles": selected, "include_features": [], "exclude_features": removed, "resolved_features": resolved}
