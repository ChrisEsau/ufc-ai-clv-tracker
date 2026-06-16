from __future__ import annotations

import streamlit as st


def inject_autopilot_css() -> None:
    st.html(
        """
        <style>
        .ops-auto-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: .75rem;
            margin: 1rem 0;
        }
        .ops-auto-card {
            min-height: 104px;
            padding: 1rem;
            display: flex;
            align-items: center;
            gap: .85rem;
        }
        .ops-auto-icon {
            width: 3rem;
            height: 3rem;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            background: rgba(47,140,255,.14);
            color: #2f8cff;
            font-size: .78rem;
        }
        .ops-auto-icon.success { background: rgba(49,223,99,.14); color: #31df63; }
        .ops-auto-icon.warning { background: rgba(245,158,11,.14); color: #f59e0b; }
        .ops-auto-icon.purple { background: rgba(181,109,255,.14); color: #b56dff; }
        .ops-auto-label {
            color: #dbe7f5;
            text-transform: uppercase;
            font-size: .68rem;
            font-weight: 900;
        }
        .ops-auto-value {
            color: #31df63;
            font-size: 1.35rem;
            line-height: 1.1;
            font-weight: 900;
            margin-top: .28rem;
        }
        .ops-auto-value.info { color: #2f8cff; }
        .ops-auto-value.warning { color: #f59e0b; }
        .ops-auto-value.purple { color: #b56dff; }
        .ops-auto-caption {
            color: #dbe7f5;
            font-size: .76rem;
            margin-top: .28rem;
        }
        .ops-panel { padding: 1rem; }
        .ops-panel-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: .8rem;
        }
        .ops-panel-title {
            color: #f5f7fb;
            text-transform: uppercase;
            font-size: .95rem;
            font-weight: 900;
        }
        .ops-panel-subtitle { color: #dbe7f5; font-size: .85rem; margin-top: .2rem; }
        .ops-link-inline { color: #2f8cff; font-size: .82rem; white-space: nowrap; }
        .ops-legend { display: flex; gap: .9rem; color: #dbe7f5; font-size: .75rem; flex-wrap: wrap; }
        .ops-legend i { width: .55rem; height: .55rem; border-radius: 50%; display: inline-block; margin-right: .25rem; vertical-align: middle; }
        .ops-legend .complete { background: #31df63; }
        .ops-legend .progress { background: #2f8cff; }
        .ops-legend .waiting { background: #94a3b8; }
        .ops-legend .failed { background: #ff4949; }
        .ops-runbook-list { display: grid; gap: .45rem; }
        .ops-runbook-row {
            display: grid;
            grid-template-columns: auto 1fr 8.5rem 6.5rem;
            gap: .75rem;
            align-items: center;
            border: 1px solid rgba(43,60,82,.55);
            border-radius: 8px;
            padding: .72rem .8rem;
            background: rgba(8,18,30,.36);
        }
        .ops-runbook-row.progress { border-color: rgba(47,140,255,.45); background: rgba(47,140,255,.08); }
        .ops-runbook-num {
            width: 1.45rem;
            height: 1.45rem;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: .72rem;
            font-weight: 900;
            background: #64748b;
            color: #08121e;
        }
        .ops-runbook-num.complete { background: #31df63; }
        .ops-runbook-num.progress { background: #2f8cff; }
        .ops-runbook-num.waiting { background: #64748b; }
        .ops-runbook-title { color: #f5f7fb; font-size: .9rem; font-weight: 900; }
        .ops-runbook-desc { color: #dbe7f5; font-size: .74rem; margin-top: .12rem; }
        .ops-runbook-time { color: #dbe7f5; font-size: .78rem; }
        .ops-runbook-state { color: #94a3b8; font-size: .82rem; font-weight: 850; text-align: right; }
        .ops-runbook-state.complete { color: #31df63; }
        .ops-runbook-state.progress { color: #2f8cff; }
        .ops-runbook-state.failed { color: #ff4949; }
        .ops-panel-note { color: #94a3b8; font-size: .78rem; margin-top: .75rem; }
        .ops-upcoming-row, .ops-alert-row, .ops-activity-row, .ops-health-row {
            display: grid;
            align-items: center;
            gap: .75rem;
            border-bottom: 1px solid rgba(43,60,82,.55);
            padding: .72rem 0;
            color: #f5f7fb;
        }
        .ops-upcoming-row { grid-template-columns: 2rem 1fr 10rem 5.8rem; }
        .ops-upcoming-icon { color: #94a3b8; font-weight: 900; }
        .ops-upcoming-title, .ops-alert-title { font-weight: 900; }
        .ops-upcoming-desc, .ops-alert-detail { color: #dbe7f5; font-size: .74rem; margin-top: .12rem; }
        .ops-upcoming-time, .ops-alert-time { color: #dbe7f5; font-size: .78rem; }
        .ops-mini-badge { border-radius: 6px; padding: .28rem .45rem; text-align: center; font-size: .72rem; background: rgba(47,140,255,.14); color: #2f8cff; }
        .ops-mini-badge.purple { background: rgba(181,109,255,.14); color: #b56dff; }
        .ops-alert-row { grid-template-columns: 3.2rem 1fr 6.5rem 5.5rem; }
        .ops-alert-badge { background: rgba(245,158,11,.18); color: #f59e0b; border: 1px solid rgba(245,158,11,.3); border-radius: 6px; padding: .24rem .35rem; font-size: .7rem; font-weight: 900; text-align: center; }
        .ops-alert-edge { color: #31df63; text-align: right; font-size: .8rem; }
        .ops-alert-edge span { color: #dbe7f5; font-size: .68rem; }
        .ops-alert-footer, .ops-health-footer { color: #f59e0b; margin-top: .65rem; font-size: .85rem; }
        .ops-health-row { grid-template-columns: 1fr 5rem 3rem; font-size: .82rem; }
        .ops-activity-row { grid-template-columns: 2.8rem 1fr 4.8rem; font-size: .82rem; }
        .ops-footer {
            margin-top: .8rem;
            padding: .8rem 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            color: #dbe7f5;
        }
        .ops-settings-button { color: #f5f7fb; background: #165dbd; border: 1px solid #2f8cff; border-radius: 8px; padding: .55rem 1rem; }
        @media (max-width: 1300px) {
            .ops-auto-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .ops-runbook-row, .ops-upcoming-row, .ops-alert-row { grid-template-columns: 1fr; }
        }
        @media (max-width: 760px) {
            .ops-auto-grid { grid-template-columns: 1fr; }
            .ops-panel-header, .ops-footer { flex-direction: column; align-items: flex-start; }
        }
        </style>
        """
    )
