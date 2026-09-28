"""Dynamic fighter-state contracts for RFS Monte Carlo V2.

Dynamic state describes temporary fight-specific conditions that evolve during
a simulated path. It is kept separate from baseline fighter parameters.

The shared physical position remains authoritative in ``SharedFightState``.
This module therefore does not duplicate phase, ownership, or position quality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
)


@dataclass(frozen=True)
class FighterDynamicState:
    """One fighter's temporary state during a simulated fight.

    Attributes:
        fatigue:
            Accumulated effort load. Zero represents fresh and one represents
            the maximum modeled fatigue burden.

        damage:
            Persistent accumulated damage. It may increase during the fight
            but will not automatically disappear between segments.

        acute_stress:
            Short-lived impairment caused by recent adversity. This may decay
            through recovery logic independently of persistent damage.
    """

    fatigue: float
    damage: float
    acute_stress: float

    def __post_init__(self) -> None:
        """Validate normalized dynamic-state values."""

        values = {
            "fatigue": self.fatigue,
            "damage": self.damage,
            "acute_stress": self.acute_stress,
        }

        for name, value in values.items():
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name} must be numeric"
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    f"{name} must be finite"
                )

            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )

    @classmethod
    def opening_state(cls) -> FighterDynamicState:
        """Return a fresh fighter state at fight opening."""

        return cls(
            fatigue=0.0,
            damage=0.0,
            acute_stress=0.0,
        )


@dataclass(frozen=True)
class FightDynamicState:
    """Temporary dynamic state for both fighters."""

    red: FighterDynamicState
    blue: FighterDynamicState

    @classmethod
    def opening_state(cls) -> FightDynamicState:
        """Return fresh dynamic state for both fighters."""

        return cls(
            red=FighterDynamicState.opening_state(),
            blue=FighterDynamicState.opening_state(),
        )

    def for_side(
        self,
        side: FighterSide,
    ) -> FighterDynamicState:
        """Return the dynamic state belonging to one fighter."""

        if side is FighterSide.RED:
            return self.red

        if side is FighterSide.BLUE:
            return self.blue

        raise ValueError(
            f"unsupported fighter side: {side}"
        )

    def replace_side(
        self,
        side: FighterSide,
        state: FighterDynamicState,
    ) -> FightDynamicState:
        """Return a new fight state with one fighter replaced."""

        if not isinstance(
            state,
            FighterDynamicState,
        ):
            raise TypeError(
                "state must be FighterDynamicState"
            )

        if side is FighterSide.RED:
            return replace(
                self,
                red=state,
            )

        if side is FighterSide.BLUE:
            return replace(
                self,
                blue=state,
            )

        raise ValueError(
            f"unsupported fighter side: {side}"
        )
