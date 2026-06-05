"""Status badge helpers."""

from __future__ import annotations

import html

_STATUS_CLASS = {
    "success": "status-success",
    "warning": "status-warning",
    "danger": "status-danger",
    "neutral": "status-neutral",
    "info": "status-info",
    "green": "status-success",
    "amber": "status-warning",
    "red": "status-danger",
    "blue": "status-info",
}


def status_badge(label: str, status: str = "neutral") -> str:
    """Return a styled HTML pill for safe use in markdown tables/cards."""

    css = _STATUS_CLASS.get(status, "status-neutral")
    return f'<span class="status-pill {css}">{html.escape(str(label))}</span>'
