from __future__ import annotations

from typing import Any

import streamlit as st


def render_training(split: dict[str, Any], *, editable: bool) -> dict[str, Any]:
    """Render training/calibration date controls."""

    train_end = st.text_input(
        "Train End Date",
        value=str(split.get("train_end_date", "2022-12-31")),
        disabled=not editable,
        key="mlab_train_end",
    )
    cal_end = st.text_input(
        "Calibration End Date",
        value=str(split.get("calibration_end_date", "2023-12-31")),
        disabled=not editable,
        key="mlab_cal_end",
    )
    return {
        "train_end_date": train_end,
        "calibration_end_date": cal_end,
    }
