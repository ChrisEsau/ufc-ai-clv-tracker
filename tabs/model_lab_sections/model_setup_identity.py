from __future__ import annotations

from typing import Any


def render_identity(context: dict[str, Any]) -> None:
    """Placeholder extraction point for model identity fields.

    The legacy configuration renderer still owns the UI. This module exists so
    future refactors can move identity controls here without changing routing.
    """
    return None
