from __future__ import annotations

import html
from datetime import datetime, timezone

import streamlit as st

from utils.github_actions import trigger_workflow
from utils.operations_registry import OPERATION_GROUPS, OperationAction, OperationGroup


def _escape(value) -> str:
    return html.escape(str(value or ""))


def _status_label(action: OperationAction) -> tuple[str, str]:
    if not action.enabled or not action.workflow:
        return "Not wired", "disabled"
    return "Ready", "success"


def _run_action(action: OperationAction) -> None:
    if not action.enabled or not action.workflow:
        st.info(f"{action.label} is a placeholder and is not wired to a workflow yet.")
        return
    ok, msg = trigger_workflow(action.workflow)
    if ok:
        st.session_state.setdefault("ops_recent_dispatches", [])
        st.session_state["ops_recent_dispatches"].insert(
            0,
            {
                "label": action.label,
                "workflow": action.workflow,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": msg,
            },
        )
        st.success(msg)
    else:
        st.error(msg)


def _render_group(group: OperationGroup) -> None:
    st.html(
        '<div class="ops-card">'
        '<div class="ops-group-header">'
        f'<div class="ops-group-title">{_escape(group.icon)}&nbsp;&nbsp;{_escape(group.title)}</div>'
        f'<div class="ops-group-subtitle">{_escape(group.subtitle)}</div>'
        '</div>'
    )
    for idx, action in enumerate(group.actions):
        status_text, status_class = _status_label(action)
        cols = st.columns([1.8, .55, .42])
        with cols[0]:
            st.html(
                '<div class="ops-action-row" style="grid-template-columns:1fr; border-bottom:0; padding:.35rem .2rem;">'
                f'<div><strong>{_escape(action.label)}</strong><div class="ops-action-desc">{_escape(action.description)}</div></div>'
                '</div>'
            )
        with cols[1]:
            dot_class = "ops-dot" if status_class == "success" else f"ops-dot {status_class}"
            st.html(f'<div style="padding-top:.55rem;"><span class="{dot_class}"></span><span class="ops-kpi-caption">{_escape(status_text)}</span></div>')
        with cols[2]:
            st.button(
                "Run",
                key=f"ops_run_{group.title}_{idx}_{action.label}",
                disabled=not action.enabled or not action.workflow,
                use_container_width=True,
                on_click=_run_action,
                args=(action,),
            )
    st.html(f'<div class="ops-link">View {_escape(group.title.replace(" Operations", ""))} Logs →</div></div>')


def render_operation_cards() -> None:
    cols = st.columns(len(OPERATION_GROUPS))
    for col, group in zip(cols, OPERATION_GROUPS):
        with col:
            _render_group(group)
