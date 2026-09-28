"""Distance-phase activity generation for RFS Monte Carlo V2.

The shared fight state determines that both fighters are at distance.
Each fighter's distance-specific parameters then determine activity during
the current 30-second segment.

Takedowns are not generated here. They remain end-of-segment transition
events handled by the shared transition engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    DistanceRateParameters,
)


@dataclass(frozen=True)
class DistanceFighterActivity:
    """One fighter's legal activity during a distance segment."""

    sig_str_attempted: int
    sig_str_landed: int
    knockdowns: int

    def __post_init__(self) -> None:
        """Validate distance activity counts."""

        nonnegative = {
            "sig_str_attempted": self.sig_str_attempted,
            "sig_str_landed": self.sig_str_landed,
            "knockdowns": self.knockdowns,
        }

        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(
                    f"{name} cannot be negative"
                )

        if self.sig_str_landed > self.sig_str_attempted:
            raise ValueError(
                "sig_str_landed cannot exceed "
                "sig_str_attempted"
            )

        if self.knockdowns > self.sig_str_landed:
            raise ValueError(
                "knockdowns cannot exceed "
                "sig_str_landed"
            )


@dataclass(frozen=True)
class DistanceSegmentActivity:
    """Both fighters' activity in one shared distance segment."""

    state: SharedFightState
    red: DistanceFighterActivity
    blue: DistanceFighterActivity

    def __post_init__(self) -> None:
        """Require an authoritative distance-phase state."""

        if self.state.phase is not FightPhase.DISTANCE:
            raise ValueError(
                "distance activity requires a distance "
                "shared state"
            )

        if self.state.phase_owner is not None:
            raise ValueError(
                "distance activity cannot have a phase owner"
            )


def _generate_fighter_distance_activity(
    parameters: DistanceRateParameters,
    rng: np.random.Generator,
) -> DistanceFighterActivity:
    """Generate one fighter's distance activity."""

    attempted = int(
        rng.poisson(
            parameters.sig_strike_attempt_rate
        )
    )

    landed = int(
        rng.binomial(
            attempted,
            parameters.sig_strike_accuracy,
        )
    )

    knockdowns = int(
        rng.binomial(
            landed,
            parameters.knockdown_probability_per_landed,
        )
    )

    return DistanceFighterActivity(
        sig_str_attempted=attempted,
        sig_str_landed=landed,
        knockdowns=knockdowns,
    )


def generate_distance_segment_activity(
    state: SharedFightState,
    red: DistanceRateParameters,
    blue: DistanceRateParameters,
    rng: np.random.Generator,
) -> DistanceSegmentActivity:
    """Generate both fighters' activity in one distance segment.

    Red and blue share the physical phase but retain separate historical
    activity rates. Dynamic-state modifiers will later produce temporary
    effective rate parameters before this function is called.
    """

    if state.phase is not FightPhase.DISTANCE:
        raise ValueError(
            "distance activity requires a distance "
            "shared state"
        )

    red_activity = _generate_fighter_distance_activity(
        red,
        rng,
    )
    blue_activity = _generate_fighter_distance_activity(
        blue,
        rng,
    )

    return DistanceSegmentActivity(
        state=state,
        red=red_activity,
        blue=blue_activity,
    )
