"""Plotly chart theming helpers."""

from __future__ import annotations

import plotly.graph_objects as go


def apply_plotly_theme(fig: go.Figure, height: int | None = None) -> go.Figure:
    """Apply the dashboard dark theme to an existing Plotly figure."""

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,17,31,0.35)",
        font={"color": "#f5f7fb", "family": "Inter, sans-serif"},
        margin={"l": 30, "r": 22, "t": 35, "b": 30},
        legend={"orientation": "h", "y": -0.18},
    )
    fig.update_xaxes(gridcolor="rgba(38,54,74,.55)", zerolinecolor="rgba(154,168,189,.45)")
    fig.update_yaxes(gridcolor="rgba(38,54,74,.55)", zerolinecolor="rgba(154,168,189,.45)")
    if height:
        fig.update_layout(height=height)
    return fig


def empty_chart_figure(message: str = "No data available yet") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font={"color": "#9aa8bd", "size": 16})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_plotly_theme(fig, height=260)
