"""Future fighter-state orchestration layer.

This builder is intentionally not wired into production. It provides the
execution surface that future fighter-state modules will use.
"""

from __future__ import annotations

from pipeline.features.state.registry import resolve_state_modules


class FighterStateBuilder:
    """Minimal orchestration skeleton for modular fighter-state features."""

    def __init__(self):
        self.modules = resolve_state_modules()

    def build_prefight_state(
        self,
        fighter_state,
        fighter_id,
        fight_date,
    ):
        features = {}

        for module in self.modules:
            features.update(
                module.prefight_features(
                    fighter_state=fighter_state,
                    fighter_id=fighter_id,
                    fight_date=fight_date,
                )
            )

        return features
