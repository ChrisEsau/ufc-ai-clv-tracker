from __future__ import annotations

from typing import Any

from tabs import model_lab as legacy_model_lab

# Future extraction targets. These modules intentionally exist before behavior
# moves out of the legacy configuration renderer.
from tabs.model_lab_sections import model_setup_identity  # noqa: F401
from tabs.model_lab_sections import model_setup_training  # noqa: F401
from tabs.model_lab_sections import model_setup_behavior  # noqa: F401
from tabs.model_lab_sections import model_setup_hyperparameters  # noqa: F401
from tabs.model_lab_sections import model_setup_feature_selection  # noqa: F401
from tabs.model_lab_sections import model_setup_advanced  # noqa: F401


def render_model_setup(
    registry: dict[str, Any],
    rows: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
) -> None:
    """Render the Model Setup workspace.

    Phase 1 of modularization keeps existing behavior unchanged while creating
    dedicated extraction points for Identity, Training, Behavior,
    Hyperparameters, Feature Selection, and Advanced configuration.
    """

    legacy_model_lab._render_configuration(registry, rows, row_by_id)
