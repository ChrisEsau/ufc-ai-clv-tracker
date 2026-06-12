import streamlit as st

from utils.model_lab_workflows import render_model_workflow_launcher


MODEL_LAB_VISUAL_REFINEMENT_CSS = """
<style>
/* --------------------------------------------------------------------------
   Model Lab V2 visual refinement layer
   -------------------------------------------------------------------------- */

section.main > div.block-container {
    padding-top: 1.05rem;
}

/* Hero/header: closer to the supplied mockup with a blue glow and dashboard feel. */
.mlab-hero {
    background:
        radial-gradient(circle at 14% 12%, rgba(59,130,246,.23), transparent 34%),
        radial-gradient(circle at 88% 18%, rgba(14,165,233,.18), transparent 30%),
        linear-gradient(135deg, rgba(15,31,52,.98), rgba(7,15,26,.98));
    border: 1px solid rgba(64,93,132,.92);
    border-radius: 14px;
    padding: 1.05rem 1.15rem;
    box-shadow: 0 26px 58px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.04);
}

.mlab-title {
    font-size: 2.22rem !important;
    letter-spacing: -.055em !important;
}

.mlab-subtitle {
    color: #aebdd2 !important;
}

/* Top selector should feel like a toolbar rather than a normal form row. */
div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stMultiSelect"] label {
    color: #dbe7f5 !important;
    font-size: .68rem !important;
    font-weight: 900 !important;
    letter-spacing: .035em !important;
    text-transform: uppercase !important;
}

/* Dark inputs. */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div,
div[data-baseweb="base-input"],
textarea,
input {
    background-color: rgba(9,19,32,.96) !important;
    border-color: rgba(56,79,111,.96) !important;
    color: #f5f7fb !important;
}

/* Main card system. */
.mlab-card {
    border-radius: 13px !important;
    border: 1px solid rgba(50,75,108,.96) !important;
    background:
        linear-gradient(180deg, rgba(18,34,55,.96), rgba(8,17,30,.99)) !important;
    box-shadow: 0 20px 46px rgba(0,0,0,.31), inset 0 1px 0 rgba(255,255,255,.035) !important;
    overflow: hidden;
}

.mlab-section {
    padding: 1rem 1.05rem 1.05rem !important;
}

.mlab-section-title {
    display: flex;
    align-items: center;
    gap: .45rem;
    color: #f8fafc !important;
    font-size: .76rem !important;
    letter-spacing: .08em !important;
    padding-bottom: .55rem;
    margin-bottom: .85rem !important;
    border-bottom: 1px solid rgba(53,76,110,.82);
}

.mlab-section-title::before {
    content: "";
    width: 7px;
    height: 7px;
    border-radius: 999px;
    background: #3b82f6;
    box-shadow: 0 0 13px rgba(59,130,246,.9);
}

/* KPI row refinement. */
.mlab-kpis {
    gap: .72rem !important;
    margin: .82rem 0 .9rem !important;
}

.mlab-kpi {
    min-height: 94px !important;
    padding: .95rem .7rem .78rem !important;
    position: relative;
}

.mlab-kpi::after {
    content: "";
    position: absolute;
    left: 13%;
    right: 13%;
    bottom: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,.8), transparent);
}

.mlab-label {
    color: #9fb0c4 !important;
    font-size: .62rem !important;
    letter-spacing: .075em !important;
}

.mlab-value {
    font-size: 1.62rem !important;
    line-height: 1.05 !important;
}

.mlab-caption {
    color: #91a3ba !important;
}

/* Selected model bar: make it look like the mockup's model comparison header. */
.mlab-model-bar {
    margin-top: .15rem;
    padding: 1rem 1.1rem !important;
    background:
        linear-gradient(90deg, rgba(15,38,68,.98), rgba(8,18,31,.98)) !important;
}

.mlab-model-name {
    font-size: 1.42rem !important;
    letter-spacing: -.025em;
}

.mlab-pill {
    margin-left: .42rem;
    border: 1px solid rgba(255,255,255,.14);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.06);
}

/* Make Streamlit buttons more like action tiles. */
div[data-testid="stButton"] > button {
    border-radius: 9px !important;
    border: 1px solid rgba(59,130,246,.46) !important;
    background: linear-gradient(180deg, rgba(37,99,235,.95), rgba(29,78,216,.92)) !important;
    color: #f8fafc !important;
    font-weight: 900 !important;
    letter-spacing: .01em !important;
    box-shadow: 0 10px 24px rgba(0,0,0,.25) !important;
}

div[data-testid="stButton"] > button:hover {
    border-color: rgba(147,197,253,.78) !important;
    filter: brightness(1.08);
}

div[data-testid="stButton"] > button:disabled {
    background: linear-gradient(180deg, rgba(43,57,78,.78), rgba(24,35,51,.9)) !important;
    border-color: rgba(71,85,105,.5) !important;
    color: #94a3b8 !important;
}

/* Dataframes inside cards. */
div[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden;
    border: 1px solid rgba(43,60,82,.78);
}

/* Expander styling for registry. */
details[data-testid="stExpander"] {
    background: rgba(9,19,32,.72) !important;
    border: 1px solid rgba(43,60,82,.85) !important;
    border-radius: 12px !important;
}

details[data-testid="stExpander"] summary {
    color: #f5f7fb !important;
    font-weight: 900 !important;
}

/* Tabs need to look like dashboard pills. */
div[data-testid="stTabs"] button {
    color: #dbe7f5 !important;
    font-weight: 850 !important;
}

div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #3b82f6 !important;
}

/* Alerts less bulky. */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1px solid rgba(59,130,246,.24) !important;
    background: rgba(15,35,60,.72) !important;
}

/* Reduce vertical whitespace caused by Streamlit default blocks. */
div[data-testid="stVerticalBlock"] > div:has(.mlab-card) {
    gap: .55rem !important;
}

hr {
    border-color: rgba(43,60,82,.7) !important;
}

@media (max-width: 1200px) {
    .mlab-title { font-size: 1.75rem !important; }
    .mlab-model-name { font-size: 1.12rem !important; }
}
</style>
"""


def render_model_lab():
    st.markdown(MODEL_LAB_VISUAL_REFINEMENT_CSS, unsafe_allow_html=True)
    render_model_workflow_launcher()
