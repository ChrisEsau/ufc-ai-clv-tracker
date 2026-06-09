"""Legacy V5 fighter-state module adapter.

This module intentionally delegates to the existing V5 fighter-state functions so
that the new modular state package can be introduced without formula drift.
"""

from __future__ import annotations

from typing import Any

from pipeline.features.base.fighter_state import default_state, update_fighter
from pipeline.features.base.prefight_features import get_prefight_features
from pipeline.features.state.contracts import FeatureStateModule


class LegacyV5Module(FeatureStateModule):
    """Adapter around the existing monolithic V5 fighter-state implementation."""

    name = "legacy_v5"
    depends_on: list[str] = []
    output_columns: list[str] = []

    def initial_state(self) -> dict[str, Any]:
        """Return the existing V5 default fighter state."""

        return default_state()

    def prefight_features(
        self,
        fighter_state: dict[str, dict[str, Any]],
        fighter_id: str,
        fight_date: Any,
    ) -> dict[str, Any]:
        """Return existing V5 point-in-time prefight features."""

        return get_prefight_features(
            fighter_state=fighter_state,
            fighter_id=fighter_id,
            fight_date=fight_date,
        )

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
        """Update existing V5 fighter state after a completed fight."""

        update_fighter(
            fighter_state=fighter_state,
            fighter_id=fighter_id,
            fight_date=fight_date,
            won=won,
            method=method,
            own=own,
            opp=opp,
            fight_time_sec=fight_time_sec,
            opponent_elo=opponent_elo,
        )
