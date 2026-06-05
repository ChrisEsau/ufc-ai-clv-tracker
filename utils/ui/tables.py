"""Reusable table display helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def styled_dataframe(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    height: int | None = None,
) -> None:
    """Render a compact wide dataframe with consistent defaults.

    Missing optional columns are ignored so display helpers remain safe around
    partially populated artifacts.
    """

    if df is None or df.empty:
        st.info("No data available yet.")
        return

    if columns:
        available_columns = [column for column in columns if column in df.columns]
        if not available_columns:
            st.info("Requested columns are not available yet.")
            return
        display = df[available_columns].copy()
    else:
        display = df.copy()

    st.dataframe(display, use_container_width=True, hide_index=True, height=height)
