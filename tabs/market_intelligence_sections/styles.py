from __future__ import annotations

import streamlit as st


def inject_market_intelligence_css() -> None:
    st.html(
        """
        <style>
        .mi-title { color:#f5f7fb; font-size:2rem; line-height:1; font-weight:900; letter-spacing:-.04em; }
        .mi-subtitle { color:#dbe7f5; font-size:.98rem; margin-top:.35rem; }
        .mi-actions { display:flex; justify-content:flex-end; align-items:center; color:#dbe7f5; font-size:.78rem; white-space:nowrap; padding-top:.35rem; }

        .mi-kpis { display:grid; grid-template-columns:repeat(5, minmax(0,1fr)); gap:.7rem; margin:.9rem 0 .85rem; }
        .mi-card { background:linear-gradient(180deg, rgba(17,31,49,.94), rgba(12,24,39,.98)); border:1px solid rgba(43,60,82,.96); border-radius:8px; box-shadow:0 20px 40px rgba(0,0,0,.22); }
        .mi-kpi { min-height:108px; padding:1rem .95rem; display:flex; align-items:center; gap:.9rem; text-align:left; }
        .mi-kpi-icon { width:3.2rem; height:3.2rem; border-radius:50%; display:flex; align-items:center; justify-content:center; color:#2f8cff; background:rgba(47,140,255,.13); border:1px solid rgba(47,140,255,.28); font-size:1.7rem; font-weight:900; }
        .mi-kpi-label { color:#dbe7f5; text-transform:uppercase; font-size:.68rem; font-weight:850; }
        .mi-kpi-value { color:#31df63; font-size:1.45rem; line-height:1.15; font-weight:900; margin-top:.28rem; }
        .mi-kpi-caption { color:#f5f7fb; font-size:.75rem; margin-top:.25rem; }

        .mi-section-title { color:#f5f7fb; text-transform:uppercase; font-size:.88rem; font-weight:900; padding:1rem 1rem .25rem; }
        .mi-section-subtitle { color:#dbe7f5; font-size:.75rem; padding:0 1rem .8rem; }
        .mi-body { padding:0 1rem 1rem; }
        .mi-empty { color:#dbe7f5; padding:1rem; }

        .mi-layout-main { display:grid; grid-template-columns:1.18fr .92fr; gap:.7rem; margin-top:.7rem; }
        .mi-layout-bottom { display:grid; grid-template-columns:1fr 1fr 1.05fr; gap:.7rem; margin-top:.7rem; }


        .mi-signal-shell { background:linear-gradient(180deg, rgba(17,31,49,.94), rgba(12,24,39,.98)); border:1px solid rgba(43,60,82,.96); border-radius:8px; box-shadow:0 20px 40px rgba(0,0,0,.22); overflow:hidden; }
        .mi-signal-header { border-bottom:1px solid rgba(43,60,82,.72); }
        .mi-signal-filter-row { padding:.8rem 1rem .75rem; border-bottom:1px solid rgba(43,60,82,.55); }
        .mi-signal-scroll { height:470px; overflow-y:auto; padding:.7rem .85rem .85rem; scrollbar-width:thin; scrollbar-color:#64748b rgba(10,22,36,.95); }
        .mi-signal-scroll::-webkit-scrollbar { width:8px; }
        .mi-signal-scroll::-webkit-scrollbar-track { background:rgba(10,22,36,.95); border-radius:999px; }
        .mi-signal-scroll::-webkit-scrollbar-thumb { background:#64748b; border-radius:999px; }
        .mi-signal-footer { border-top:1px solid rgba(43,60,82,.65); color:#2f8cff; text-align:center; padding:.55rem; font-size:.78rem; font-weight:850; }
        .mi-signal-list { display:flex; flex-direction:column; gap:.55rem; margin-top:.65rem; }
        .mi-signal-card { background:linear-gradient(180deg, rgba(17,31,49,.96), rgba(10,22,36,.98)); border:1px solid rgba(43,60,82,.96); border-left:4px solid #64748b; border-radius:8px; padding:.85rem .95rem; box-shadow:0 12px 28px rgba(0,0,0,.18); }
        .mi-signal-card.opportunity { border-left-color:#31df63; }
        .mi-signal-card.watch { border-left-color:#facc15; }
        .mi-signal-card.info { border-left-color:#2f8cff; }
        .mi-signal-top { display:flex; align-items:center; gap:.45rem; }
        .mi-signal-badge { border-radius:5px; padding:.16rem .45rem; font-size:.64rem; font-weight:900; letter-spacing:.02em; }
        .mi-signal-badge.opportunity { color:#31df63; background:rgba(49,223,99,.13); border:1px solid rgba(49,223,99,.28); }
        .mi-signal-badge.watch { color:#facc15; background:rgba(250,204,21,.12); border:1px solid rgba(250,204,21,.28); }
        .mi-signal-badge.info { color:#2f8cff; background:rgba(47,140,255,.13); border:1px solid rgba(47,140,255,.28); }
        .mi-signal-type { color:#dbe7f5; font-size:.72rem; font-weight:900; text-transform:uppercase; }
        .mi-signal-confidence { margin-left:auto; color:#31df63; font-size:.78rem; font-weight:900; }
        .mi-signal-title { color:#f5f7fb; font-size:.98rem; font-weight:900; margin-top:.55rem; }
        .mi-signal-subtitle { color:#8fb6df; font-size:.78rem; margin-top:.12rem; }
        .mi-signal-metrics { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:.55rem; }
        .mi-signal-metrics span { color:#dbe7f5; background:rgba(15,30,49,.9); border:1px solid rgba(43,60,82,.8); border-radius:999px; padding:.18rem .5rem; font-size:.68rem; }
        .mi-signal-metrics b { color:#f5f7fb; }
        .mi-signal-explanation { color:#dbe7f5; font-size:.75rem; line-height:1.35; margin-top:.55rem; }

        .mi-panel { overflow:hidden; }
        .mi-panel-head { display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; border-bottom:1px solid rgba(43,60,82,.68); }
        .mi-panel-link { color:#2f8cff; font-size:.78rem; font-weight:850; padding:1rem 1rem 0 0; white-space:nowrap; }
        .mi-small-panel { min-height:230px; }
        .mi-placeholder { margin:1rem; min-height:120px; border:1px dashed rgba(148,163,184,.34); border-radius:8px; color:#dbe7f5; display:flex; align-items:center; justify-content:center; gap:1rem; padding:1rem; }
        .mi-placeholder-icon { color:#94a3b8; font-size:2.2rem; }
        .mi-placeholder span { color:#8fb6df; font-size:.8rem; }
        .mi-table { width:100%; border-collapse:collapse; color:#f5f7fb; font-size:.76rem; }
        .mi-table th { color:#dbe7f5; text-align:left; text-transform:uppercase; font-size:.62rem; padding:.55rem .55rem; border-bottom:1px solid rgba(43,60,82,.9); }
        .mi-table td { padding:.52rem .55rem; border-bottom:1px solid rgba(43,60,82,.58); vertical-align:top; }
        .mi-table td span { color:#8fb6df; font-size:.68rem; }
        .mi-book { display:inline-block; color:#f5f7fb !important; background:rgba(47,140,255,.12); border:1px solid rgba(47,140,255,.28); border-radius:5px; padding:.05rem .28rem; margin-left:.25rem; font-size:.62rem !important; font-weight:900; }
        .mi-red { color:#ff4949; }

        .mi-steam-list { padding:.7rem 1rem 1rem; display:flex; flex-direction:column; gap:.5rem; }
        .mi-steam-row { display:grid; grid-template-columns:1.4fr .7fr .55fr; gap:.75rem; align-items:center; padding:.65rem .75rem; border:1px solid rgba(43,60,82,.72); border-radius:8px; background:rgba(10,22,36,.78); }
        .mi-steam-title { color:#f5f7fb; font-size:.82rem; font-weight:900; }
        .mi-steam-subtitle { color:#8fb6df; font-size:.68rem; margin-top:.12rem; }
        .mi-steam-value { color:#facc15; font-size:.9rem; font-weight:900; }
        .mi-steam-confidence { color:#31df63; font-size:.9rem; font-weight:900; }
        .mi-status-bar { display:flex; gap:1.4rem; align-items:center; color:#dbe7f5; font-size:.78rem; padding:.8rem 1rem; border-top:1px solid rgba(43,60,82,.75); margin-top:.9rem; background:rgba(5,11,20,.72); }
        .mi-gauge-label { color:#dbe7f5; text-transform:uppercase; font-size:.72rem; font-weight:850; text-align:center; }
        .mi-gauge { color:#f5f7fb; font-size:2rem; font-weight:900; text-align:center; padding:1rem 0; }
        .mi-green { color:#31df63; }
        .mi-model-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.4rem; padding:1rem; }
        @media (max-width:1300px) { .mi-kpis { grid-template-columns:repeat(3, minmax(0,1fr)); } .mi-layout-main, .mi-layout-bottom { grid-template-columns:1fr; } }
        @media (max-width:760px) { .mi-kpis { grid-template-columns:1fr; } }
        </style>
        """
    )
