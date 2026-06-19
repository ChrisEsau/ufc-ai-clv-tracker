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
            padding: 1.05rem 1.25rem;
            margin: 0.75rem 0 1rem 0;
            background:
                radial-gradient(circle at 8% 0%, rgba(37, 99, 235, 0.18), transparent 42%),
                linear-gradient(180deg, rgba(15, 32, 54, 0.98), rgba(7, 18, 32, 0.99));
            box-shadow: 0 22px 46px rgba(0, 0, 0, 0.28);
        }
        .model-setup-banner-main {
            display: grid;
            grid-template-columns: minmax(0, 1.15fr) minmax(260px, .85fr);
            gap: 1.2rem;
            align-items: start;
        }
        .model-setup-kicker {
            color: #9fb0c4;
            text-transform: uppercase;
            letter-spacing: .055em;
            font-size: .68rem;
            font-weight: 900;
            margin-bottom: .22rem;
        }
        .model-setup-title {
            color: #f5f7fb;
            font-size: 1.42rem;
            font-weight: 950;
            letter-spacing: -0.035em;
            margin-bottom: 0.2rem;
        }
        .model-setup-status {
            display: inline-block;
            vertical-align: middle;
            margin-left: 0.45rem;
            padding: 0.18rem 0.5rem;
            border-radius: 7px;
            background: rgba(37, 99, 235, 0.36);
            color: #dcecff;
            border: 1px solid rgba(59, 130, 246, 0.48);
            font-size: 0.72rem;
            line-height: 1;
            letter-spacing: 0.01em;
        }
        .model-setup-banner-paths {
            display: grid;
            gap: .45rem;
            color: #dbe7f5;
            font-size: .8rem;
            line-height: 1.35;
        }
        .model-setup-banner-paths span,
        .model-setup-meta-grid span {
            display: block;
            color: #93a6bd;
            font-size: .68rem;
            text-transform: uppercase;
            letter-spacing: .04em;
            font-weight: 850;
            margin-bottom: .1rem;
        }
        .model-setup-meta-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .9rem;
            margin-top: .9rem;
            color: #f8fbff;
            font-size: .84rem;
            font-weight: 780;
        }
        .model-setup-footer-spacer {
            height: 0.25rem;
        }

        section.main div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: rgba(43, 60, 82, 0.95) !important;
            border-radius: 12px !important;
            background:
                linear-gradient(180deg, rgba(12, 27, 47, 0.97), rgba(6, 17, 31, 0.99)) !important;
            box-shadow: 0 18px 38px rgba(0, 0, 0, 0.20) !important;
        }
        section.main div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stMarkdownContainer"] h4 {
            color: #2f9bff !important;
            font-size: 1.02rem !important;
            font-weight: 860 !important;
            letter-spacing: -0.02em !important;
            margin-bottom: 0.72rem !important;
        }
        section.main label,
        section.main div[data-testid="stWidgetLabel"] p {
            color: #d7e2f0 !important;
            font-size: 0.75rem !important;
            font-weight: 650 !important;
        }
        section.main input,
        section.main textarea,
        section.main div[data-baseweb="select"] > div {
            border-color: rgba(61, 84, 112, 0.92) !important;
            background-color: rgba(4, 14, 26, 0.84) !important;
            color: #f8fbff !important;
            border-radius: 7px !important;
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
        section.main div[data-testid="stAlert"] {
            border-radius: 10px !important;
        }
        @media (max-width: 1200px) {
            .model-setup-banner-main,
            .model-setup-meta-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
