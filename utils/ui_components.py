"""Backward-compatible formatting and UI helpers."""

import pandas as pd

from utils.ui.cards import metric_card


def money(x):
    if pd.isna(x):
        return ""
    return f"${x:,.0f}"


def american(x):
    if pd.isna(x):
        return ""
    x = int(round(x))
    return f"+{x}" if x > 0 else str(x)


def pct(x):
    if pd.isna(x):
        return ""
    return f"{x * 100:.1f}%"


def pct_already(x):
    if pd.isna(x):
        return ""
    return f"{x:.1f}%"


def render_metric(label, value, subtext="", accent="neutral"):
    status = {
        "green": "success",
        "red": "danger",
        "blue": "info",
        "amber": "warning",
        "purple": "purple",
        "neutral": "neutral",
    }.get(accent, accent)
    metric_card(label, value, status=status, caption=subtext or None)
