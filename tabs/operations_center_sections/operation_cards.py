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


def _accent_class(group: OperationGroup) -> str:
    return {
        "market": "market",
        "predictions": "prediction",
        "model": "model",
        "data": "data",
    }.get(group.status_key, "neutral")


def _render_group_header(group: OperationGroup) -> None:
    accent = _accent_class(group)
    st.html(
        f'<div class="ops-group-accent {accent}"></div>'
        '<div class="ops-group-header">'
        f'<div class="ops-group-title">{_escape(group.icon)}&nbsp;&nbsp;{_escape(group.title)}</div>'
        f'<div class="ops-group-subtitle">{_escape(group.subtitle)}</div>'
        '</div>'
    )


def _render_action(group: OperationGroup, action: OperationAction, idx: int) -> None:
    status_text, status_class = _status_label(action)
    dot_class = "ops-dot" if status_class == "success" else f"ops-dot {status_class}"

    with st.container(border=False):
        cols = st.columns([3.2, 1.0], gap="small")
        with cols[0]:
            st.html(
                '<div class="ops-action-copy">'
                f'<div class="ops-action-title">{_escape(action.label)}</div>'
                f'<div class="ops-action-desc">{_escape(action.description)}</div>'
                f'<div class="ops-action-status"><span class="{dot_class}"></span>{_escape(status_text)}</div>'
                '</div>'
            )
        with cols[1]:
            st.button(
                "Run",
                key=f"ops_run_{group.title}_{idx}_{action.label}",
                disabled=not action.enabled or not action.workflow,
                use_container_width=True,
                on_click=_run_action,
                args=(action,),
            )
        st.html('<div class="ops-action-divider"></div>')


def _render_group(group: OperationGroup) -> None:
    with st.container(border=True):
        _render_group_header(group)
        for idx, action in enumerate(group.actions):
            _render_action(group, action, idx)
        st.html(f'<div class="ops-link">View {_escape(group.title.replace(" Operations", ""))} Logs →</div>')


def render_operation_cards() -> None:
    cols = st.columns(len(OPERATION_GROUPS), gap="medium")
    for col, group in zip(cols, OPERATION_GROUPS):
        with col:
            _render_group(group)
