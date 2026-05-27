import streamlit as st


def apply_theme():
    st.markdown(
        """
        <style>
        /* paste your main CSS block here */
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        /* paste your selectbox CSS block here */
        </style>
        """,
        unsafe_allow_html=True,
    )
    # ============================================================
    # THEME / CSS
    # ============================================================
    
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
    
        .stSelectbox label,
        .stMultiSelect label,
        .stRadio label {
            color: #CBD5E1 !important;
            font-weight: 600 !important;
        }
    
        .stButton button {
            background: #1E293B;
            color: #F8FAFC;
            border: 1px solid #475569;
            border-radius: 10px;
            font-weight: 700;
        }
    
        .stButton button:hover {
            background: #334155;
            color: #FFFFFF;
            border-color: #60A5FA;
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
        
        section[data-testid="stSidebar"] .stButton button {
            background: transparent;
            border: 1px solid transparent;
            color: #CBD5E1;
            font-weight: 700;
            font-size: 15px;
            text-align: left;
            border-radius: 12px;
            padding: 10px 12px;
            transition: all 0.18s ease;
        }
        
        section[data-testid="stSidebar"] .stButton button:hover {
            background: rgba(148, 163, 184, 0.08);
            border-color: rgba(148, 163, 184, 0.15);
            color: #FFFFFF;
        }
        .workflow-row {
            margin-bottom: 8px;
        }
        
        .workflow-row-active {
            background: linear-gradient(
                135deg,
                rgba(239,68,68,0.92),
                rgba(127,29,29,0.82)
            );
            border: 1px solid rgba(248,113,113,0.45);
            border-radius: 12px;
            padding: 2px 4px;
            margin-bottom: 8px;
            box-shadow: 0 8px 22px rgba(239,68,68,0.20);
        }
        
        section[data-testid="stSidebar"] .stButton button {
            background: transparent;
            border: none;
            color: #E2E8F0;
            font-weight: 700;
            font-size: 15px;
            text-align: left;
            box-shadow: none;
            padding-top: 12px;
            padding-bottom: 12px;
        }
        
        section[data-testid="stSidebar"] .stButton button:hover {
            background: rgba(148,163,184,0.08);
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        /* Selectbox container */
        div[data-baseweb="select"] > div {
            background-color: #0F172A !important;
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
            color: #F8FAFC !important;
            min-height: 42px !important;
        }
    
        /* Selected text */
        div[data-baseweb="select"] span {
            color: #F8FAFC !important;
            font-weight: 600 !important;
        }
    
        /* Dropdown arrow */
        div[data-baseweb="select"] svg {
            color: #CBD5E1 !important;
            fill: #CBD5E1 !important;
        }
    
        /* Input text area */
        div[data-baseweb="select"] input {
            color: #F8FAFC !important;
        }
    
        /* Selectbox label */
        label[data-testid="stWidgetLabel"] {
            color: #CBD5E1 !important;
            font-weight: 700 !important;
            font-size: 13px !important;
        }
    
        /* Dropdown menu */
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
