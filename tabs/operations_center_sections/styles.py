from __future__ import annotations

import streamlit as st


def inject_operations_css() -> None:
    st.html(
        """
        <style>
        .ops-title { color:#f5f7fb; font-size:2rem; line-height:1; font-weight:900; letter-spacing:-.04em; }
        .ops-subtitle { color:#dbe7f5; font-size:.98rem; margin-top:.35rem; }
        .ops-actions { display:flex; justify-content:flex-end; align-items:center; color:#dbe7f5; font-size:.78rem; white-space:nowrap; padding-top:.35rem; }
        .ops-card { background:linear-gradient(180deg, rgba(17,31,49,.94), rgba(12,24,39,.98)); border:1px solid rgba(43,60,82,.96); border-radius:8px; box-shadow:0 20px 40px rgba(0,0,0,.22); }
        .ops-kpis { display:grid; grid-template-columns:repeat(6, minmax(0,1fr)); gap:.7rem; margin:.9rem 0 .85rem; }
        .ops-kpi { min-height:106px; padding:1rem .8rem .85rem; display:flex; gap:.7rem; align-items:center; }
        .ops-kpi-icon { font-size:1.9rem; line-height:1; width:2.2rem; text-align:center; }
        .ops-kpi-label { color:#dbe7f5; text-transform:uppercase; font-size:.68rem; font-weight:850; }
        .ops-kpi-value { color:#31df63; font-size:1.45rem; line-height:1.15; font-weight:900; margin-top:.28rem; }
        .ops-kpi-value.warning { color:#facc15; } .ops-kpi-value.danger { color:#ff4949; }
        .ops-kpi-caption { color:#f5f7fb; font-size:.75rem; margin-top:.25rem; }
        .ops-flow { padding:.85rem 1rem; margin:0 0 .9rem; }
        .ops-flow-title { color:#dbe7f5; text-transform:uppercase; font-size:.72rem; font-weight:900; letter-spacing:.02em; margin-bottom:.65rem; }
        .ops-flow-grid { display:grid; grid-template-columns:1fr auto 1fr auto 1fr auto 1fr auto 1fr; gap:.55rem; align-items:center; }
        .ops-flow-stage { text-align:center; min-width:0; }
        .ops-flow-dot { width:.62rem; height:.62rem; border-radius:50%; background:#31df63; margin:0 auto .25rem; box-shadow:0 0 14px rgba(49,223,99,.35); }
        .ops-flow-dot.warning { background:#facc15; box-shadow:0 0 14px rgba(250,204,21,.35); } .ops-flow-dot.danger { background:#ff4949; box-shadow:0 0 14px rgba(255,73,73,.35); }
        .ops-flow-label { color:#f5f7fb; font-size:.82rem; font-weight:900; }
        .ops-flow-value { color:#31df63; font-size:.78rem; font-weight:850; margin-top:.1rem; }
        .ops-flow-caption { color:#dbe7f5; font-size:.65rem; margin-top:.05rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .ops-flow-arrow { color:#64748b; font-size:1.25rem; font-weight:900; }
        .ops-model-strip { display:grid; grid-template-columns:1.2fr 1.1fr 1.35fr .9fr; gap:.75rem; padding:1rem; margin-bottom:.9rem; }
        .ops-strip-label { color:#dbe7f5; text-transform:uppercase; font-size:.68rem; font-weight:850; }
        .ops-strip-value { color:#f5f7fb; font-size:1rem; margin-top:.35rem; }
        .ops-badge { display:inline-block; border-radius:5px; padding:.16rem .45rem; font-weight:850; font-size:.68rem; margin-left:.45rem; }
        .ops-badge.success { color:#31df63; background:rgba(49,223,99,.13); border:1px solid rgba(49,223,99,.28); }
        .ops-badge.warning { color:#facc15; background:rgba(250,204,21,.12); border:1px solid rgba(250,204,21,.28); }
        .ops-group-accent { height:3px; border-radius:999px; margin:.05rem .05rem .65rem; background:#334155; }
        .ops-group-accent.market { background:#2f8cff; box-shadow:0 0 16px rgba(47,140,255,.28); }
        .ops-group-accent.prediction { background:#31df63; box-shadow:0 0 16px rgba(49,223,99,.24); }
        .ops-group-accent.model { background:#b56dff; box-shadow:0 0 16px rgba(181,109,255,.25); }
        .ops-group-accent.data { background:#f59e0b; box-shadow:0 0 16px rgba(245,158,11,.23); }
        .ops-group-header { padding:.4rem .15rem .75rem; border-bottom:1px solid rgba(43,60,82,.7); min-height:86px; }
        .ops-group-title { color:#f5f7fb; font-size:.95rem; font-weight:900; text-transform:uppercase; }
        .ops-group-subtitle { color:#dbe7f5; font-size:.74rem; margin-top:.25rem; line-height:1.25; }
        .ops-action-copy { padding:.42rem .05rem .2rem; color:#f5f7fb; min-height:74px; }
        .ops-action-title { color:#f5f7fb; font-size:.78rem; font-weight:900; }
        .ops-action-desc { color:#dbe7f5; font-size:.66rem; margin-top:.16rem; line-height:1.25; }
        .ops-action-status { color:#dbe7f5; font-size:.68rem; margin-top:.38rem; }
        .ops-action-divider { border-bottom:1px solid rgba(43,60,82,.55); margin:.15rem 0 .25rem; }
        .ops-dot { width:.45rem; height:.45rem; border-radius:50%; display:inline-block; margin-right:.28rem; background:#31df63; vertical-align:middle; }
        .ops-dot.warning { background:#facc15; } .ops-dot.disabled { background:#64748b; }
        .ops-link { color:#2f8cff; text-align:center; padding:.65rem .25rem .25rem; font-size:.82rem; }
        .ops-bottom { display:grid; grid-template-columns:1fr 1.05fr 1.1fr; gap:.7rem; margin-top:.8rem; }
        .ops-card-title { color:#f5f7fb; text-transform:uppercase; font-size:.82rem; font-weight:900; padding:1rem 1rem .45rem; }
        .ops-table { width:100%; border-collapse:collapse; color:#f5f7fb; font-size:.78rem; }
        .ops-table th { color:#dbe7f5; text-align:left; text-transform:uppercase; font-size:.65rem; padding:.55rem .75rem; border-bottom:1px solid rgba(43,60,82,.9); }
        .ops-table td { padding:.58rem .75rem; border-bottom:1px solid rgba(43,60,82,.62); }
        .ops-green { color:#31df63; } .ops-yellow { color:#facc15; } .ops-blue { color:#2f8cff; }
        div[data-testid="stButton"] > button { white-space:nowrap; min-width:3.25rem; }
        @media (max-width:1300px) { .ops-kpis { grid-template-columns:repeat(3, minmax(0,1fr)); } .ops-bottom, .ops-model-strip { grid-template-columns:1fr; } .ops-flow-grid { grid-template-columns:1fr; } .ops-flow-arrow { display:none; } }
        @media (max-width:760px) { .ops-kpis { grid-template-columns:1fr; } }
        </style>
        """
    )
