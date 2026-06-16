from __future__ import annotations

import html

import streamlit as st

from utils.operations_status import StatusTile, operation_status_tiles


def _escape(value) -> str:
    return html.escape(str(value or ""))


def _tile(tile: StatusTile) -> str:
    value_class = "ops-kpi-value"
    if tile.status in {"warning", "danger"}:
        value_class += f" {tile.status}"
    return (
        '<div class="ops-card ops-kpi">'
        f'<div class="ops-kpi-icon">{_escape(tile.icon)}</div>'
        '<div>'
        f'<div class="ops-kpi-label">{_escape(tile.label)}</div>'
        f'<div class="{value_class}">{_escape(tile.value)}</div>'
        f'<div class="ops-kpi-caption">{_escape(tile.caption)}</div>'
        '</div></div>'
    )


def render_status_summary() -> None:
    st.html('<div class="ops-kpis">' + ''.join(_tile(tile) for tile in operation_status_tiles()) + '</div>')
