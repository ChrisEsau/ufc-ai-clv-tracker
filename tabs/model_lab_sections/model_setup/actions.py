from __future__ import annotations

from typing import Any

import streamlit as st

from utils.model_lab_setup import model_context, persistence
from utils.model_lab_setup.validators import validate_delete_allowed


def _new_template_model_id(context: dict[str, Any]) -> str:
    if context.get("is_new_model"):
        return str(context.get("template_model_id") or st.session_state.get("model_setup_draft_template_model_id") or "")
    return str(context.get("model_id") or "")


def _inject_action_styles() -> None:
    st.markdown(
        """
        <style>
        .st-key-model_setup_action_delete button,
        .st-key-model_setup_delete_confirm button {
            background: linear-gradient(180deg, rgba(220, 38, 38, 0.98), rgba(153, 27, 27, 0.98)) !important;
            border-color: rgba(248, 113, 113, 0.85) !important;
            color: #ffffff !important;
        }
        .st-key-model_setup_action_delete button:hover,
        .st-key-model_setup_delete_confirm button:hover {
            background: linear-gradient(180deg, rgba(239, 68, 68, 1), rgba(185, 28, 28, 1)) !important;
            border-color: rgba(252, 165, 165, 0.95) !important;
            color: #ffffff !important;
        }
        .st-key-model_setup_action_delete button:disabled,
        .st-key-model_setup_delete_confirm button:disabled {
            background: rgba(69, 26, 26, 0.55) !important;
            border-color: rgba(127, 29, 29, 0.8) !important;
            color: rgba(255, 255, 255, 0.48) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_action_bar(
    context: dict[str, Any],
    registry: dict[str, Any],
    payload: dict[str, Any],
    validation_result: dict[str, Any] | None = None,
) -> None:
    """Render New, Save, and Delete actions for Model Setup."""

    _inject_action_styles()

    full_validation = validation_result or {"ok": True, "errors": [], "warnings": []}
    delete_validation = validate_delete_allowed(context, registry)
    save_disabled = not bool(full_validation.get("ok"))
    delete_disabled = not delete_validation["ok"] or bool(context.get("is_new_model"))

    st.markdown("#### Actions")
    c1, c2, c3, c4 = st.columns([1.0, 1.0, 1.0, 2.0])

    with c1:
        if st.button("New", use_container_width=True, key="model_setup_action_new"):
            template_model_id = _new_template_model_id(context)
            if not template_model_id:
                st.error("Unable to determine template model for new draft.")
            else:
                draft_context = model_context.build_new_model_context(
                    registry,
                    template_model_id,
                    str(context.get("model_family") or "moneyline"),
                    str(context.get("market_key") or "moneyline"),
                )
                st.session_state["model_setup_draft_context"] = draft_context
                st.session_state["model_setup_draft_template_model_id"] = template_model_id
                st.rerun()

    with c2:
        if st.button("Save", type="primary", use_container_width=True, disabled=save_disabled, key="model_setup_action_save"):
            result = persistence.save_model_setup(context, registry, payload)
            if result.get("ok"):
                saved_model_id = str(result.get("model_id") or context.get("model_id") or "")
                st.success(result.get("message") or "Saved model.")
                st.session_state["model_setup_selected_model_id"] = saved_model_id
                st.session_state.pop("model_setup_draft_context", None)
                st.session_state.pop("model_setup_draft_template_model_id", None)
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(result.get("message") or "Save failed.")

    with c3:
        if st.button("Delete", use_container_width=True, disabled=delete_disabled, key="model_setup_action_delete"):
            st.session_state["model_setup_delete_requested"] = True

    with c4:
        if save_disabled:
            st.caption("Save disabled: " + "; ".join(full_validation.get("errors") or []))
        elif delete_disabled and delete_validation.get("errors"):
            st.caption("Delete disabled: " + "; ".join(delete_validation.get("errors") or []))
        else:
            st.caption("New prepares a fresh unsaved next version. Save creates or updates the model.")

    if st.session_state.get("model_setup_delete_requested"):
        st.warning("Delete removes the config YAML and registry entry. Artifacts are not deleted.")
        confirmation = st.text_input(
            "Type the model ID to confirm delete",
            key="model_setup_delete_confirmation",
        )
        dc1, dc2 = st.columns(2)
        with dc1:
            if st.button("Cancel Delete", use_container_width=True, key="model_setup_delete_cancel"):
                st.session_state.pop("model_setup_delete_requested", None)
                st.session_state.pop("model_setup_delete_confirmation", None)
                st.rerun()
        with dc2:
            if st.button(
                "Confirm Delete",
                use_container_width=True,
                disabled=confirmation != str(context.get("model_id") or ""),
                key="model_setup_delete_confirm",
            ):
                result = persistence.delete_model_setup(context, registry)
                if result.get("ok"):
                    st.success(result.get("message") or "Deleted model.")
                    st.session_state.pop("model_setup_delete_requested", None)
                    st.session_state.pop("model_setup_delete_confirmation", None)
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(result.get("message") or "Delete failed.")