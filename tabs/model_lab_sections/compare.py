from __future__ import annotations

from typing import Any, Callable

from tabs.model_lab_sections.comparison import render_comparison


ExistingModelSelector = Callable[[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]], dict[str, Any]]


def render_compare(
    registry: dict[str, Any],
    rows: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    *,
    existing_model_selector: ExistingModelSelector,
) -> None:
    """Render the Compare workspace using the existing comparison diagnostics."""

    render_comparison(
        registry,
        rows,
        row_by_id,
        existing_model_selector=existing_model_selector,
    )
