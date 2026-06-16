from __future__ import annotations

import html

import streamlit as st

from utils.operations_status import system_rows


def _escape(value) -> str:
    return html.escape(str(value or ""))


def render_system_status() -> None:
    rows = []
    for name, status, detail in system_rows():
        rows.append(
            "<tr>"
            f"<td>✅ {_escape(name)}</td>"
            f"<td class='ops-green'>{_escape(status)}</td>"
            f"<td>{_escape(detail)}</td>"
            "</tr>"
        )
    st.html(
        '<div class="ops-card"><div class="ops-card-title">System Status</div>'
        '<table class="ops-table"><tbody>' + ''.join(rows) + '</tbody></table></div>'
    )
