from __future__ import annotations

import re
from typing import Any

import streamlit as st


def _safe_widget_key(value: Any) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", str(value or ""))
    return cleaned.strip("_") or "model"


def _training_key(context: dict[str, Any], suffix: str) -> str:
    model_key = _safe_widget_key(context.get("model_id") or context.get("config_path") or "model")
    return f"model_setup_training_{suffix}_{model_key}"


def render_training_section(context: dict[str, Any]) -> dict[str, Any]:
    editable = bool(context.get("is_editable"))
    split = (context.get("config") or {}).get("split") or {}

    st.markdown("#### 2. Training Setup")
    train_start_date = st.text_input("Train Start Date", value=str(split.get("train_start_date") or ""), disabled=not editable, key=_training_key(context, "train_start_date"))
    train_end_date = st.text_input("Train End Date", value=str(split.get("train_end_date") or ""), disabled=not editable, key=_training_key(context, "train_end_date"))
    calibration_end_date = st.text_input("Calibration End Date", value=str(split.get("calibration_end_date") or ""), disabled=not editable, key=_training_key(context, "calibration_end_date"))

    return {"train_start_date": train_start_date, "train_end_date": train_end_date, "calibration_end_date": calibration_end_date}
