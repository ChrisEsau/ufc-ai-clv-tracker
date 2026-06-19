from __future__ import annotations

from typing import Any

import streamlit as st

from utils.model_lab_feature_selection import render_feature_checklist


def render_feature_selection_section(context: dict[str, Any]) -> dict[str, Any]:
    """Render the model-specific Feature Selection card."""

    st.caption("Model-specific feature selection. Global feature registry editing belongs in the Features workspace.")
    feature_payload = render_feature_checklist(context)
    resolved_features = list(feature_payload.get("resolved_features") or [])
    feature_payload["expected_feature_count"] = len(resolved_features)
    feature_payload["resolved_feature_count"] = len(resolved_features)
    return feature_payload
