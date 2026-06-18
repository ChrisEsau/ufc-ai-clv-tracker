from __future__ import annotations

from typing import Any

import streamlit as st

from utils.model_lab_setup import validators


def render_validation_summary(
    context: dict[str, Any],
    registry: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Render live Model Setup validation and return the validation result."""

    result = validators.validate_model_setup_form(context, registry, payload)
    errors = result.get("errors") or []
    warnings = result.get("warnings") or []

    st.markdown("#### Validation Summary")

    if result.get("ok") and not warnings:
        st.success("Model setup is ready to save.")
    elif result.get("ok"):
        st.warning("Model setup can be saved, but warnings should be reviewed.")
    else:
        st.error("Model setup has errors that must be fixed before save.")

    if errors:
        st.markdown("**Errors**")
        for error in errors:
            st.write(f"- {error}")

    if warnings:
        st.markdown("**Warnings**")
        for warning in warnings:
            st.write(f"- {warning}")

    if not errors and not warnings:
        st.caption("No validation issues detected.")

    return result
