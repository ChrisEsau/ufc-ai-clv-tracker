"""Global Streamlit theme for the dark UFC SaaS-style dashboard."""

from __future__ import annotations

import streamlit as st


THEME = {
    "background": "#05080d",
    "panel": "#0d1727",
    "panel_alt": "#101c2d",
    "border": "#26364a",
    "text": "#f5f7fb",
    "muted": "#cbd5e1",
    "green": "#35d96b",
    "blue": "#3b82f6",
    "yellow": "#facc15",
    "red": "#ef4444",
    "purple": "#a855f7",
}


def apply_theme() -> None:
    """Apply CSS used by all Streamlit workspaces."""

    st.set_page_config(
        page_title="UFC Betting Intelligence",
        page_icon="🥊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        :root {{
            --ufc-bg: {THEME["background"]};
            --ufc-panel: {THEME["panel"]};
            --ufc-panel-alt: {THEME["panel_alt"]};
            --ufc-border: {THEME["border"]};
            --ufc-text: {THEME["text"]};
            --ufc-muted: {THEME["muted"]};
            --ufc-green: {THEME["green"]};
            --ufc-blue: {THEME["blue"]};
            --ufc-yellow: {THEME["yellow"]};
            --ufc-red: {THEME["red"]};
            --ufc-purple: {THEME["purple"]};
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .stApp {{
            color: var(--ufc-text);
            background: #05080d;
        }}

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        .main {{
            background: #05080d !important;
        }}

        [data-testid="stHeader"] {{
            background: #05080d !important;
            height: 0 !important;
            min-height: 0 !important;
            visibility: hidden !important;
        }}

        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        #MainMenu {{
            display: none !important;
        }}

        .block-container {{
            max-width: 1580px;
            padding: 0.45rem 1.65rem 4rem;
        }}

        section[data-testid="stSidebar"] {{
            background: #05080d !important;
            border-right: 1px solid rgba(38, 54, 74, 0.95);
        }}

        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
            padding-top: 0.15rem;
        }}

        section[data-testid="stSidebar"] img {{
            margin-top: -0.35rem;
            margin-bottom: -0.2rem;
        }}

        h1, h2, h3 {{
            color: var(--ufc-text) !important;
            letter-spacing: -0.035em;
        }}

        p, span, label {{
            color: inherit;
        }}

        .ufc-page-header {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            margin: 0 0 0.75rem;
        }}

        .ufc-kicker {{
            color: var(--ufc-blue);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }}

        .ufc-title {{
            color: var(--ufc-text);
            font-size: clamp(1.75rem, 3vw, 2.35rem);
            font-weight: 900;
            letter-spacing: -0.045em;
            line-height: 1;
            margin: 0;
        }}

        .ufc-subtitle {{
            color: var(--ufc-muted);
            font-size: 0.98rem;
            margin-top: 0.18rem;
        }}

        .ufc-updated {{
            color: var(--ufc-muted);
            font-size: 0.78rem;
            white-space: nowrap;
            padding-top: 0.35rem;
        }}

        .ufc-card, .metric-card, .panel {{
            background: linear-gradient(180deg, rgba(16, 28, 45, 0.92), rgba(13, 23, 39, 0.94));
            border: 1px solid rgba(38, 54, 74, 0.96);
            border-radius: 12px;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.20);
        }}

        .metric-card {{
            min-height: 104px;
            padding: 1rem 1.1rem;
            text-align: center;
        }}

        .metric-label {{
            color: var(--ufc-muted);
            font-size: 0.70rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }}

        .metric-value {{
            color: var(--ufc-text);
            font-size: 1.8rem;
            font-weight: 900;
            line-height: 1.05;
        }}

        .metric-delta {{
            font-size: 0.82rem;
            font-weight: 800;
            margin-top: 0.25rem;
        }}

        .metric-subtext {{
            color: var(--ufc-muted);
            font-size: 0.78rem;
            margin-top: 0.35rem;
        }}

        .section-header {{
            color: var(--ufc-text);
            font-size: 1.05rem;
            font-weight: 900;
            letter-spacing: -0.025em;
            margin: 1.2rem 0 0.25rem;
        }}

        .section-caption {{
            color: var(--ufc-muted);
            font-size: 0.84rem;
            margin-bottom: 0.7rem;
        }}

        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border-radius: 999px;
            padding: 0.22rem 0.58rem;
            font-size: 0.74rem;
            font-weight: 800;
            border: 1px solid transparent;
            white-space: nowrap;
        }}
        .status-success {{ color: var(--ufc-green); background: rgba(53, 217, 107, .12); border-color: rgba(53,217,107,.34); }}
        .status-warning {{ color: var(--ufc-yellow); background: rgba(250, 204, 21, .12); border-color: rgba(250,204,21,.34); }}
        .status-danger {{ color: var(--ufc-red); background: rgba(239, 68, 68, .12); border-color: rgba(239,68,68,.34); }}
        .status-info {{ color: var(--ufc-blue); background: rgba(59, 130, 246, .12); border-color: rgba(59,130,246,.34); }}
        .status-neutral {{ color: #cbd5e1; background: rgba(148, 163, 184, .11); border-color: rgba(148,163,184,.24); }}

        .sidebar-section {{
            color: #dbeafe;
            font-size: 0.72rem;
            font-weight: 900;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            margin: 0.85rem 0 0.35rem;
        }}
        .sidebar-section.compact {{ margin-top: 0.30rem; }}
        .sidebar-divider {{ border-top: 1px solid rgba(245, 247, 251, 0.14); height: 1px; margin: 0.40rem 0 0.15rem; }}
        .sidebar-divider.compact {{ margin-top: 0.35rem; margin-bottom: 0.10rem; }}
        .sidebar-stat-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: .75rem;
            color: #dbe7f5;
            font-size: .80rem;
            padding: .34rem 0;
            border-bottom: 1px solid rgba(245,247,251,.08);
        }}
        .sidebar-stat-row strong {{ color: #31df63; font-size: .92rem; }}
        .sidebar-version, .sidebar-note {{ color: #dbe7f5; font-size: 0.76rem; }}

        section.main .mlab-nav-note,
        section.main div[data-testid="stRadio"] {{
            display: none !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
