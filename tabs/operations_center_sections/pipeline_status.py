from __future__ import annotations

import html

import streamlit as st

from utils.operations_status import operation_status_tiles


def _escape(value) -> str:
    return html.escape(str(value or ""))


def render_pipeline_status() -> None:
    tile_by_label = {tile.label: tile for tile in operation_status_tiles()}
    stages = [
        ("Market", tile_by_label.get("Market Status")),
        ("Features", tile_by_label.get("Features Status")),
        ("Predictions", tile_by_label.get("Predictions Status")),
        ("Betting Board", tile_by_label.get("Predictions Status")),
        ("CLV", tile_by_label.get("CLV Status")),
    ]

    cells = []
    for idx, (label, tile) in enumerate(stages):
        status = tile.status if tile else "warning"
        value = tile.value if tile else "Unknown"
        caption = tile.caption if tile else "No status"
        dot_class = "ops-flow-dot"
        if status in {"warning", "danger"}:
            dot_class += f" {status}"
        cells.append(
            '<div class="ops-flow-stage">'
            f'<div class="{dot_class}"></div>'
            f'<div class="ops-flow-label">{_escape(label)}</div>'
            f'<div class="ops-flow-value">{_escape(value)}</div>'
            f'<div class="ops-flow-caption">{_escape(caption)}</div>'
            '</div>'
        )
        if idx < len(stages) - 1:
            cells.append('<div class="ops-flow-arrow">→</div>')

    st.html(
        '<div class="ops-card ops-flow">'
        '<div class="ops-flow-title">Pipeline Status</div>'
        '<div class="ops-flow-grid">'
        + ''.join(cells)
        + '</div></div>'
    )
