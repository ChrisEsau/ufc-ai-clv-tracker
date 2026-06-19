from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    """Inject styles scoped to the new clean Model Setup workspace."""

    st.markdown(
        """
        <style>
        .model-setup-shell {
            border: 1px solid rgba(43, 60, 82, 0.95);
            border-radius: 12px;
            padding: 1rem 1.15rem 1rem;
            margin: 0.72rem 0 0.9rem 0;
            background:
                radial-gradient(circle at 8% 0%, rgba(37, 99, 235, 0.15), transparent 42%),
                linear-gradient(180deg, rgba(15, 32, 54, 0.98), rgba(7, 18, 32, 0.99));
            box-shadow: 0 18px 38px rgba(0, 0, 0, 0.24);
        }
        .model-setup-summary-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(320px, .95fr);
            gap: 1.4rem;
            align-items: start;
        }
        .model-setup-summary-left,
        .model-setup-summary-right {
            min-width: 0;
        }
        .model-setup-summary-right {
            display: grid;
            gap: .58rem;
            padding-top: .15rem;
        }
        .model-setup-kicker {
            color: #9fb0c4;
            text-transform: uppercase;
            letter-spacing: .055em;
            font-size: .66rem;
            font-weight: 720;
            margin-bottom: .24rem;
        }
        .model-setup-title {
            color: #f5f7fb;
            font-size: 1.34rem;
            font-weight: 760;
            letter-spacing: -0.025em;
            line-height: 1.08;
            margin-bottom: 0.7rem;
        }
        .model-setup-status {
            display: inline-block;
            vertical-align: middle;
            margin-left: 0.42rem;
            padding: 0.16rem 0.48rem;
            border-radius: 7px;
            background: rgba(37, 99, 235, 0.34);
            color: #dcecff;
            border: 1px solid rgba(59, 130, 246, 0.48);
            font-size: 0.66rem;
            line-height: 1;
            letter-spacing: 0.01em;
            font-weight: 660;
        }
        .model-setup-path-item {
            color: #f8fbff;
            font-size: .78rem;
            line-height: 1.38;
            font-weight: 520;
            overflow-wrap: anywhere;
        }
        .model-setup-path-item span,
        .model-setup-meta-grid span {
            display: block;
            color: #9fb0c4;
            font-size: .64rem;
            text-transform: uppercase;
            letter-spacing: .045em;
            font-weight: 680;
            margin-bottom: .12rem;
        }
        .model-setup-meta-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1.2rem;
            margin-top: .3rem;
            color: #f8fbff;
            font-size: .8rem;
            font-weight: 560;
        }
        .model-setup-footer-spacer {
            height: 0.22rem;
        }

        section.main div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: rgba(43, 60, 82, 0.95) !important;
            border-radius: 12px !important;
            background:
                linear-gradient(180deg, rgba(12, 27, 47, 0.97), rgba(6, 17, 31, 0.99)) !important;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.18) !important;
        }
        section.main div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stMarkdownContainer"] h4 {
            color: #2f9bff !important;
            font-size: .98rem !important;
            font-weight: 680 !important;
            letter-spacing: -0.015em !important;
            margin-bottom: 0.62rem !important;
        }
        section.main label,
        section.main div[data-testid="stWidgetLabel"] p {
            color: #d7e2f0 !important;
            font-size: 0.72rem !important;
            font-weight: 560 !important;
        }
        section.main input,
        section.main textarea,
        section.main div[data-baseweb="select"] > div {
            border-color: rgba(61, 84, 112, 0.92) !important;
            background-color: rgba(4, 14, 26, 0.84) !important;
            color: #f8fbff !important;
            border-radius: 7px !important;
            font-size: .8rem !important;
            font-weight: 500 !important;
        }
        section.main textarea {
            min-height: 70px !important;
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
            border-color: rgba(59, 130, 246, 0.7) !important;
            background: linear-gradient(180deg, rgba(17, 77, 160, 0.96), rgba(14, 55, 116, 0.98)) !important;
            color: #f8fbff !important;
            font-weight: 620 !important;
            min-height: 2.25rem !important;
            font-size: .82rem !important;
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
        section.main div[data-testid="stAlert"] {
            border-radius: 10px !important;
        }
        @media (max-width: 1200px) {
            .model-setup-summary-grid,
            .model-setup-meta-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )