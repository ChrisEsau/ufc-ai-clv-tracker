from __future__ import annotations

from typing import Any

import streamlit as st


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def render_feature_selection_section(context: dict[str, Any]) -> dict[str, Any]:
    """Render the Feature Selection card and return feature payload.

    This is model-specific feature selection only. It does not edit the global
    feature registry.
    """

    editable = bool(context.get("is_editable"))
    feature_config = (context.get("config") or {}).get("features") or {}

    selected_bundles = _as_list(feature_config.get("selected_bundles") or feature_config.get("bundles"))
    include_features = _as_list(feature_config.get("include_features"))
    exclude_features = _as_list(feature_config.get("exclude_features"))
    resolved_features = _as_list(
        feature_config.get("resolved_features")
        or feature_config.get("feature_columns")
        or feature_config.get("features")
    )
    expected_feature_count = feature_config.get("expected_feature_count")
    if expected_feature_count is None and resolved_features:
        expected_feature_count = len(resolved_features)

    st.markdown("#### 5. Feature Selection")
    st.caption("Model-specific feature configuration. Global feature registry editing belongs in the Features workspace.")

    c1, c2 = st.columns(2)
    with c1:
        bundles_text = st.text_area(
            "Selected Bundles",
            value="\n".join(selected_bundles),
            disabled=not editable,
            height=96,
            key="model_setup_features_selected_bundles",
            help="One bundle per line.",
        )
        include_text = st.text_area(
            "Included Features",
            value="\n".join(include_features),
            disabled=not editable,
            height=132,
            key="model_setup_features_include_features",
            help="Manual include list. One feature per line.",
        )
    with c2:
        exclude_text = st.text_area(
            "Excluded Features",
            value="\n".join(exclude_features),
            disabled=not editable,
            height=96,
            key="model_setup_features_exclude_features",
            help="Manual exclude list. One feature per line.",
        )
        resolved_preview = st.text_area(
            "Resolved Features Preview",
            value="\n".join(resolved_features[:50]),
            disabled=True,
            height=132,
            key="model_setup_features_resolved_preview",
            help="Preview only. Backend resolver will own final resolution.",
        )

    c3, c4 = st.columns(2)
    with c3:
        expected_count = st.number_input(
            "Expected Feature Count",
            value=int(expected_feature_count or 0),
            min_value=0,
            step=1,
            disabled=not editable,
            key="model_setup_features_expected_count",
        )
    with c4:
        resolved_count = st.number_input(
            "Resolved Feature Count",
            value=len(resolved_features),
            min_value=0,
            step=1,
            disabled=True,
            key="model_setup_features_resolved_count",
        )

    selected_bundles_payload = [line.strip() for line in bundles_text.splitlines() if line.strip()]
    include_features_payload = [line.strip() for line in include_text.splitlines() if line.strip()]
    exclude_features_payload = [line.strip() for line in exclude_text.splitlines() if line.strip()]

    return {
        "selected_bundles": selected_bundles_payload,
        "include_features": include_features_payload,
        "exclude_features": exclude_features_payload,
        "resolved_features": resolved_features,
        "expected_feature_count": int(expected_count) if expected_count else None,
        "resolved_feature_count": int(resolved_count),
    }
