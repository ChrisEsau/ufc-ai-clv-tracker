"""Reusable table display helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def styled_dataframe(df: pd.DataFrame, columns: list[str] | None = None, height: int | None = None) -> None:
    """Render a compact wide dataframe with consistent defaults."""

    if df is None or df.empty:
        st.info("No data available yet.")
        return
    display = df[columns].copy() if columns else df.copy()
    st.dataframe(display, use_container_width=True, hide_index=True, height=height)
