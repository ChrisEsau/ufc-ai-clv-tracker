from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    """Inject styles scoped to the new clean Model Setup workspace."""

    st.markdown(
        """
        <style>
        .model-setup-page-title {
            color: #f8fbff;
            font-size: 1.65rem;
            font-weight: 950;
            letter-spacing: -0.04em;
            line-height: 1.05;
            margin: 0 0 0.15rem 0;
        }
        .model-setup-page-caption {
            color: #b6c4d7;
            font-size: 0.92rem;
            margin-bottom: 1rem;
        }
        .model-setup-shell {
            border: 1px solid rgba(43, 60, 82, 0.95);
            border-radius: 12px;
            padding: 1.15rem 1.35rem;
            margin: 0.65rem 0 1rem 0;
            background:
                radial-gradient(circle at 10% 0%, rgba(37, 99, 235, 0.16), transparent 38%),
                linear-gradient(180deg, rgba(17, 31, 49, 0.97), rgba(8, 18, 31, 0.98));
            box-shadow: 0 22px 46px rgba(0, 0, 0, 0.28);
        }
        .model-setup-title {
            color: #f5f7fb;
            font-size: 1.38rem;
            font-weight: 950;
            letter-spacing: -0.035em;
            margin-bottom: 0.35rem;
        }
        .model-setup-status {
            display: inline-block;
            vertical-align: middle;
            margin-left: 0.45rem;
            padding: 0.16rem 0.48rem;
            border-radius: 7px;
            background: rgba(37, 99, 235, 0.36);
            color: #dcecff;
            border: 1px solid rgba(59, 130, 246, 0.42);
            font-size: 0.72rem;
            line-height: 1;
            letter-spacing: 0.01em;
        }
        .model-setup-subtitle {
            color: #dbe7f5;
            font-size: 0.92rem;
            margin-bottom: 0.7rem;
        }
        .model-setup-note {
            color: #93a6bd;
            font-size: 0.76rem;
            line-height: 1.45;
        }
        .model-setup-footer-spacer {
            height: 0.35rem;
        }

        /* Card polish for the Model Setup layout. These rules are intentionally
           visual-only and do not alter widget keys, values, payloads, or callbacks. */
        section.main div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: rgba(43, 60, 82, 0.95) !important;
            border-radius: 12px !important;
            background:
                linear-gradient(180deg, rgba(13, 27, 45, 0.96), rgba(7, 17, 31, 0.98)) !important;
            box-shadow: 0 18px 38px rgba(0, 0, 0, 0.22) !important;
        }
        section.main div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stMarkdownContainer"] h4 {
            color: #2f9bff !important;
            font-size: 1.02rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em !important;
            margin-bottom: 0.75rem !important;
        }
        section.main label,
        section.main div[data-testid="stWidgetLabel"] p {
            color: #d7e2f0 !important;
            font-size: 0.78rem !important;
            font-weight: 650 !important;
        }
        section.main input,
        section.main textarea,
        section.main div[data-baseweb="select"] > div {
            border-color: rgba(61, 84, 112, 0.9) !important;
            background-color: rgba(4, 14, 26, 0.82) !important;
            color: #f8fbff !important;
            border-radius: 7px !important;
        }
        section.main div[data-testid="stNumberInput"] button,
        section.main div[data-testid="stNumberInput"] button:disabled {
            background: rgba(7, 17, 31, 0.95) !important;
            border-color: rgba(61, 84, 112, 0.9) !important;
            color: #ffffff !important;
            opacity: 1 !important;
        }
        section.main div[data-testid="stNumberInput"] button svg,
        section.main div[data-testid="stNumberInput"] button svg path {
            fill: #ffffff !important;
            color: #ffffff !important;
        }
        section.main div[data-testid="stButton"] button {
            border-radius: 8px !important;
            border-color: rgba(59, 130, 246, 0.72) !important;
            background: linear-gradient(180deg, rgba(17, 77, 160, 0.98), rgba(14, 55, 116, 0.98)) !important;
            color: #f8fbff !important;
            font-weight: 760 !important;
            min-height: 2.35rem !important;
        }
        section.main div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(180deg, rgba(30, 96, 210, 1), rgba(21, 68, 154, 1)) !important;
        }
        section.main div[data-testid="stMetric"] {
            border: 1px solid rgba(43, 60, 82, 0.95);
            border-radius: 10px;
            padding: 0.6rem 0.75rem;
            background: rgba(10, 28, 48, 0.86);
        }
        section.main div[data-testid="stExpander"] {
            border-color: rgba(43, 60, 82, 0.95) !important;
            border-radius: 9px !important;
            background: rgba(4, 14, 26, 0.42) !important;
        }
        section.main div[data-testid="stAlert"] {
            border-radius: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
