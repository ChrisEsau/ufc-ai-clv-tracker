from __future__ import annotations

import streamlit as st

from utils import sidebar as legacy_sidebar


BACKTEST_WORKSPACE = ("Backtest", "◫")


def _ensure_backtest_workspace() -> None:
    """Add Backtest to the Model Lab sidebar navigation without duplicating sidebar code."""

    workspaces = legacy_sidebar.MODEL_LAB_WORKSPACES
    names = [name for name, _ in workspaces]
    if BACKTEST_WORKSPACE[0] in names:
        return

    try:
        performance_index = names.index("Performance")
        workspaces.insert(performance_index + 1, BACKTEST_WORKSPACE)
    except ValueError:
        workspaces.append(BACKTEST_WORKSPACE)


def render_sidebar():
    """Render the existing sidebar with modular additions."""

    _ensure_backtest_workspace()
    page = legacy_sidebar.render_sidebar()

    if page == "Bankroll":
        st.sidebar.markdown("---")
        if st.sidebar.button("Edit Bet", use_container_width=True, key="sidebar_bankroll_edit_bet"):
            st.session_state["bankroll_dialog"] = "edit"
            st.rerun()

    return page
