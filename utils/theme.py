import streamlit as st


def apply_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: linear-gradient(135deg, #0B1220 0%, #111827 45%, #0F172A 100%);
            color: #F9FAFB;
        }

        section[data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid #334155;
        }

        h1 {
            font-size: 34px !important;
            font-weight: 800 !important;
            color: #F9FAFB !important;
            letter-spacing: -0.03em;
        }

        h2, h3 {
            color: #F9FAFB !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 1600px;
        }

        .sidebar-section {
            color: #94A3B8;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.08em;
            margin: 14px 0 10px 0;
        }

        .sidebar-version {
            color: #64748B;
            font-size: 12px;
            margin-top: 20px;
            text-align: center;
        }

        .metric-card {
            background: linear-gradient(180deg, #1E293B 0%, #172033 100%);
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 10px 28px rgba(0,0,0,0.32);
            min-height: 118px;
        }

        .metric-label {
            color: #93A4BA;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 10px;
        }

        .metric-value {
            color: #FFFFFF;
            font-size: 30px;
            font-weight: 800;
            line-height: 1.1;
        }

        .metric-subtext {
            color: #94A3B8;
            font-size: 13px;
            margin-top: 8px;
        }

        .section-header {
            font-size: 24px;
            font-weight: 800;
            margin-top: 28px;
            margin-bottom: 12px;
            color: #F8FAFC;
            letter-spacing: -0.03em;
        }

        .panel {
            background: linear-gradient(180deg, #1E293B 0%, #172033 100%);
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 10px 28px rgba(0,0,0,0.28);
        }

        .small-muted {
            color: #94A3B8;
            font-size: 13px;
        }

        div[data-testid="stDataFrame"] {
            background: #1E293B;
            border-radius: 14px;
            border: 1px solid #334155;
            overflow: hidden;
        }

        .positive {
            color: #22C55E;
            font-weight: 800;
        }

        .negative {
            color: #EF4444;
            font-weight: 800;
        }

        .neutral {
            color: #CBD5E1;
            font-weight: 700;
        }

        .status-pill {
            padding: 4px 9px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 800;
            display: inline-block;
        }

        .pill-official {
            background: rgba(16,185,129,0.20);
            color: #34D399;
            border: 1px solid rgba(16,185,129,0.45);
        }

        .pill-watchlist {
            background: rgba(245,158,11,0.18);
            color: #FBBF24;
            border: 1px solid rgba(245,158,11,0.45);
        }

        .pill-danger {
            background: rgba(239,68,68,0.18);
            color: #F87171;
            border: 1px solid rgba(239,68,68,0.45);
        }

        .pill-info {
            background: rgba(59,130,246,0.18);
            color: #60A5FA;
            border: 1px solid rgba(59,130,246,0.45);
        }

        /* Sidebar buttons */
        section[data-testid="stSidebar"] .stButton button {
            background: #1E293B;
            color: #F8FAFC;
            border: 1px solid #334155;
            border-radius: 12px;
            font-weight: 700;
            font-size: 15px;
            padding: 0.7rem 1rem;
            text-align: left;
            transition: all 0.18s ease;
        }

        section[data-testid="stSidebar"] .stButton button:hover {
            background: #334155;
            border-color: #475569;
            color: #FFFFFF;
        }

        /* Selectbox container */
        div[data-baseweb="select"] > div {
            background-color: #0F172A !important;
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
            color: #F8FAFC !important;
            min-height: 42px !important;
        }

        div[data-baseweb="select"] span {
            color: #F8FAFC !important;
            font-weight: 600 !important;
        }

        div[data-baseweb="select"] svg {
            color: #CBD5E1 !important;
            fill: #CBD5E1 !important;
        }

        div[data-baseweb="select"] input {
            color: #F8FAFC !important;
        }

        label[data-testid="stWidgetLabel"] {
            color: #CBD5E1 !important;
            font-weight: 700 !important;
            font-size: 13px !important;
        }

        ul[role="listbox"] {
            background-color: #111827 !important;
            border: 1px solid #334155 !important;
        }

        ul[role="listbox"] li {
            background-color: #111827 !important;
            color: #F8FAFC !important;
        }

        ul[role="listbox"] li:hover {
            background-color: #1E293B !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )