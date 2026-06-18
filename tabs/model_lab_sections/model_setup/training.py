from __future__ import annotations

from typing import Any

import streamlit as st


def render_training_section(context: dict[str, Any]) -> dict[str, Any]:
    """Render the Training Setup card and return training payload."""

    editable = bool(context.get("is_editable"))
    split = (context.get("config") or {}).get("split") or {}

    st.markdown("#### 2. Training Setup")
    c1, c2, c3 = st.columns(3)
    with c1:
        train_start_date = st.text_input(
            "Train Start Date",
            value=str(split.get("train_start_date") or ""),
            disabled=not editable,
            key="model_setup_training_train_start_date",
            help="Required for new configs. Existing configs may warn if missing.",
        )
    with c2:
        train_end_date = st.text_input(
            "Train End Date",
            value=str(split.get("train_end_date") or ""),
            disabled=not editable,
            key="model_setup_training_train_end_date",
        )
    with c3:
        calibration_end_date = st.text_input(
            "Calibration End Date",
            value=str(split.get("calibration_end_date") or ""),
            disabled=not editable,
            key="model_setup_training_calibration_end_date",
        )

    return {
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "calibration_end_date": calibration_end_date,
    }
