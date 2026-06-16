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
        .ops-model-strip { display:grid; grid-template-columns:1.2fr 1.1fr 1.35fr .9fr; gap:.75rem; padding:1rem; margin-bottom:.9rem; }
        .ops-strip-label { color:#dbe7f5; text-transform:uppercase; font-size:.68rem; font-weight:850; }
        .ops-strip-value { color:#f5f7fb; font-size:1rem; margin-top:.35rem; }
        .ops-badge { display:inline-block; border-radius:5px; padding:.16rem .45rem; font-weight:850; font-size:.68rem; margin-left:.45rem; }
        .ops-badge.success { color:#31df63; background:rgba(49,223,99,.13); border:1px solid rgba(49,223,99,.28); }
        .ops-badge.warning { color:#facc15; background:rgba(250,204,21,.12); border:1px solid rgba(250,204,21,.28); }
        .ops-groups { display:grid; grid-template-columns:repeat(5, minmax(0,1fr)); gap:.7rem; margin:.75rem 0; }
        .ops-group-header { padding:1rem 1rem .7rem; border-bottom:1px solid rgba(43,60,82,.7); min-height:74px; }
        .ops-group-title { color:#f5f7fb; font-size:.95rem; font-weight:900; text-transform:uppercase; }
        .ops-group-subtitle { color:#dbe7f5; font-size:.74rem; margin-top:.25rem; line-height:1.25; }
        .ops-action-row { display:grid; grid-template-columns:1fr auto auto; align-items:center; gap:.45rem; padding:.55rem .8rem; border-bottom:1px solid rgba(43,60,82,.55); color:#f5f7fb; font-size:.76rem; }
        .ops-action-desc { color:#dbe7f5; font-size:.66rem; margin-top:.15rem; }
        .ops-dot { width:.45rem; height:.45rem; border-radius:50%; display:inline-block; margin-right:.25rem; background:#31df63; }
        .ops-dot.warning { background:#facc15; } .ops-dot.disabled { background:#64748b; }
        .ops-run { border:1px solid #2f7de1; color:#2f8cff; border-radius:5px; padding:.18rem .45rem; font-size:.7rem; }
        .ops-run.disabled { border-color:#334155; color:#64748b; }
        .ops-link { color:#2f8cff; text-align:center; padding:.8rem; font-size:.82rem; }
        .ops-bottom { display:grid; grid-template-columns:1fr 1.05fr 1.1fr; gap:.7rem; margin-top:.8rem; }
        .ops-card-title { color:#f5f7fb; text-transform:uppercase; font-size:.82rem; font-weight:900; padding:1rem 1rem .45rem; }
        .ops-table { width:100%; border-collapse:collapse; color:#f5f7fb; font-size:.78rem; }
        .ops-table th { color:#dbe7f5; text-align:left; text-transform:uppercase; font-size:.65rem; padding:.55rem .75rem; border-bottom:1px solid rgba(43,60,82,.9); }
        .ops-table td { padding:.58rem .75rem; border-bottom:1px solid rgba(43,60,82,.62); }
        .ops-green { color:#31df63; } .ops-yellow { color:#facc15; } .ops-blue { color:#2f8cff; }
        @media (max-width:1300px) { .ops-kpis { grid-template-columns:repeat(3, minmax(0,1fr)); } .ops-groups { grid-template-columns:repeat(2, minmax(0,1fr)); } .ops-bottom, .ops-model-strip { grid-template-columns:1fr; } }
        @media (max-width:760px) { .ops-kpis, .ops-groups { grid-template-columns:1fr; } }
        </style>
        """
    )
