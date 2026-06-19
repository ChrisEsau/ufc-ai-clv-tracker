from __future__ import annotations

from typing import Any

import streamlit as st


MODEL_SETUP_STATE_PREFIXES_TO_CLEAR = (
    "model_setup_identity_",
    "model_setup_training_",
    "model_setup_behavior_",
    "model_setup_hyperparameters_",
    "model_setup_features_",
    "mlab_selected_bundles_",
    "mlab_feature_",
)

MODEL_SETUP_STATE_KEYS_TO_CLEAR = {
    "model_setup_draft_context",
    "model_setup_draft_template_model_id",
    "model_setup_identity_payload",
    "model_setup_training_payload",
    "model_setup_behavior_payload",
    "model_setup_hyperparameters_payload",
    "model_setup_feature_payload",
    "model_setup_validation_result",
    "model_setup_last_context_id",
}


def _clear_model_setup_form_state() -> None:
    """Clear cached widget state when switching selected models.

    Streamlit widgets keep values by key across reruns. Model Setup fields use
    stable keys so edited values do not vanish while working on one model, but
    those stable keys must be cleared when the selected model changes; otherwise
    the old model's field values can remain visible for the newly selected model.
    """

    keys_to_clear = []
    for key in st.session_state.keys():
        key_text = str(key)
        if key_text in MODEL_SETUP_STATE_KEYS_TO_CLEAR:
            keys_to_clear.append(key_text)
            continue
        if any(key_text.startswith(prefix) for prefix in MODEL_SETUP_STATE_PREFIXES_TO_CLEAR):
            keys_to_clear.append(key_text)

    for key in keys_to_clear:
        st.session_state.pop(key, None)


def _model_label(row: dict[str, Any]) -> str:
    model_id = str(row.get("model_id") or "")
    status = str(row.get("status") or "draft").upper()
    market = str(row.get("market_key") or "moneyline")
    return f"{model_id}  ·  {status}  ·  {market}"


def render_model_selector(rows: list[dict[str, Any]]) -> str | None:
    """Render the existing-model selector.

    The selector only loads existing models. New model creation is prepared by
    the page-level New button and persisted by Save when the generated ID does
    not yet exist in the registry.
    """

    if not rows:
        st.info("No registered models found.")
        return None

    model_ids = [str(row["model_id"]) for row in rows]
    row_by_id = {str(row["model_id"]): row for row in rows}
    current = st.session_state.get("model_setup_selected_model_id", model_ids[0])
    index = model_ids.index(current) if current in model_ids else 0

    selected = st.selectbox(
        "Model",
        model_ids,
        index=index,
        format_func=lambda model_id: _model_label(row_by_id[model_id]),
        key="model_setup_selected_model_id",
        on_change=_clear_model_setup_form_state,
        help="Select an existing model to view/edit or use as the template for New.",
    )
    return str(selected)
