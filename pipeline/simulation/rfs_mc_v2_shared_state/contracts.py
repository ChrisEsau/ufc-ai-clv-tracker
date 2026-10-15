"""Core contracts for the RFS Monte Carlo V2 shared-state engine.

V2 models one coherent fight timeline per Monte Carlo path.

Both fighters always share:

- one fight phase
- one phase owner when clinching or grappling
- one position-quality value
- one round and segment location

This module contains contracts only. It does not calculate transitions,
generate activity, update dynamic fighter state, or evaluate finishes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase


SEGMENTS_PER_ROUND = 10


class FighterSide(str, Enum):
    """Red or blue fighter within one simulated matchup."""

    RED = "red"
    BLUE = "blue"

    @property
    def opponent(self) -> "FighterSide":
        """Return the opposite fighter side."""

        if self is FighterSide.RED:
            return FighterSide.BLUE

        return FighterSide.RED


@dataclass(frozen=True)
class SharedFightState:
    """Authoritative physical state shared by both fighters.

    ``phase_owner`` represents:

    - ``None`` while the fight is at distance
    - the controlling fighter while the fight is in the clinch
    - the top or positionally dominant fighter on the ground

    ``position_quality`` is measured from the owner's perspective:

    - 0.0 means weak or newly established control
    - 1.0 means highly dominant positional control

    Both fighters must read this same object. A simulation path may never
    maintain separate red and blue phases.
    """

    phase: FightPhase
    phase_owner: FighterSide | None

    phase_age_segments: int
    position_quality: float

    round_number: int
    segment_number: int

    def __post_init__(self) -> None:
        """Validate shared-state invariants."""

        if not 1 <= self.round_number <= 5:
            raise ValueError(
                "round_number must be between 1 and 5"
            )

        if not 1 <= self.segment_number <= SEGMENTS_PER_ROUND:
            raise ValueError(
                "segment_number must be between 1 "
                f"and {SEGMENTS_PER_ROUND}"
            )

        if self.phase_age_segments < 0:
            raise ValueError(
                "phase_age_segments cannot be negative"
            )

        if not 0.0 <= self.position_quality <= 1.0:
            raise ValueError(
                "position_quality must be between 0 and 1"
            )

        if self.phase is FightPhase.DISTANCE:
            if self.phase_owner is not None:
                raise ValueError(
                    "distance phase cannot have a phase owner"
                )

            if self.position_quality != 0.0:
                raise ValueError(
                    "distance phase must have zero "
                    "position quality"
                )

        if self.phase in {
            FightPhase.CLINCH,
            FightPhase.GROUND,
        }:
            if self.phase_owner is None:
                raise ValueError(
                    "clinch and ground phases require "
                    "a phase owner"
                )

    @classmethod
    def opening_state(
        cls,
        *,
        round_number: int = 1,
    ) -> "SharedFightState":
        """Create the required opening state for a fight or round."""

        return cls(
            phase=FightPhase.DISTANCE,
            phase_owner=None,
            phase_age_segments=0,
            position_quality=0.0,
            round_number=round_number,
            segment_number=1,
        )

    def reset_for_round(
        self,
        *,
        round_number: int,
    ) -> "SharedFightState":
        """Reset the physical fight state at a new round."""

        if round_number <= self.round_number:
            raise ValueError(
                "round_number must advance when resetting"
            )

        return SharedFightState.opening_state(
            round_number=round_number,
        )
