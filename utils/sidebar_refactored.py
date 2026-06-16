from __future__ import annotations

from utils import sidebar as legacy_sidebar


BACKTEST_WORKSPACE = ("Backtest", "◫")
OPERATIONS_NAV_ITEM = ("Operations Center", "↻", "Run and monitor workflows")


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


def _ensure_operations_nav_item() -> None:
    """Add Operations Center to the top-level sidebar without editing the legacy sidebar module."""

    nav_items = legacy_sidebar.NAV_ITEMS
    names = [name for name, _, _ in nav_items]
    if OPERATIONS_NAV_ITEM[0] in names:
        return

    try:
        bankroll_index = names.index("Bankroll")
        nav_items.insert(bankroll_index + 1, OPERATIONS_NAV_ITEM)
    except ValueError:
        nav_items.append(OPERATIONS_NAV_ITEM)


def render_sidebar():
    """Render the existing sidebar with modular navigation additions."""

    _ensure_backtest_workspace()
    _ensure_operations_nav_item()
    return legacy_sidebar.render_sidebar()
