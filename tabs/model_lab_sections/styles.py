from __future__ import annotations

import streamlit as st


def inject_model_lab_control_css() -> None:
    """Keep Model Lab number-input stepper buttons aligned with input cells."""

    st.markdown(
        """
        <style>
        div[data-testid="stNumberInput"] button,
        div[data-testid="stNumberInput"] button:disabled {
            background: rgba(7, 17, 31, 0.9) !important;
            background-color: rgba(7, 17, 31, 0.9) !important;
            border-color: rgba(38, 54, 74, 0.95) !important;
            color: #ffffff !important;
            opacity: 1 !important;
        }

        div[data-testid="stNumberInput"] button:hover,
        div[data-testid="stNumberInput"] button:focus,
        div[data-testid="stNumberInput"] button:active {
            background: rgba(7, 17, 31, 0.98) !important;
            background-color: rgba(7, 17, 31, 0.98) !important;
            border-color: rgba(59, 130, 246, 0.65) !important;
            color: #ffffff !important;
        }

        div[data-testid="stNumberInput"] button svg,
        div[data-testid="stNumberInput"] button svg path {
            fill: #ffffff !important;
            color: #ffffff !important;
            opacity: 1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
