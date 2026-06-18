from __future__ import annotations

from typing import Any

from tabs import model_lab as legacy_model_lab


def render_model_setup(
    registry: dict[str, Any],
    rows: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
) -> None:
    """Render the Model Setup workspace using the existing configuration flow."""

    legacy_model_lab._render_configuration(registry, rows, row_by_id)
