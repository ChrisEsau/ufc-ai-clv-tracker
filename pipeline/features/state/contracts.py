"""Contracts for modular fighter-state feature modules.

The state package is scaffolding only until the refactor is wired into the
feature build pipeline behind parity checks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FeatureStateModule(ABC):
    """Interface implemented by fighter-state feature family modules.

    Modules may depend on outputs/state owned by earlier modules. Dependencies
    are declared by module name and validated by the registry before execution.
    """

    name: str
    depends_on: list[str] = []
    output_columns: list[str] = []

    @abstractmethod
    def initial_state(self) -> dict[str, Any]:
        """Return this module's initial per-fighter state payload."""

    @abstractmethod
    def prefight_features(
        self,
        fighter_state: dict[str, dict[str, Any]],
        fighter_id: str,
        fight_date: Any,
    ) -> dict[str, Any]:
        """Return point-in-time prefight features for one fighter."""

    @abstractmethod
    def update_after_fight(
        self,
        fighter_state: dict[str, dict[str, Any]],
        fighter_id: str,
        fight_date: Any,
        won: bool,
        method: Any,
        own: dict[str, float],
        opp: dict[str, float],
        fight_time_sec: float,
        opponent_elo: float,
    ) -> None:
        """Update one fighter's state after a completed fight."""
