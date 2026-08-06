"""Ground-phase activity generation for RFS Monte Carlo V2.

Both fighters share one ground phase and one authoritative ground owner.

The owner may generate:

- ground strikes
- control time
- submission attempts
- position advancements

The defender may generate:

- escape attempts
- reversal attempts
- scramble attempts

Successful escapes, reversals, and scrambles remain end-of-segment transition
events. Submission defense is consumed later by the finish engine rather than
sampled as a segment activity count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    GroundDefenderRateParameters,
    GroundOwnerRateParameters,
    SEGMENT_SECONDS,
)


@dataclass(frozen=True)
class GroundFighterActivity:
    """One fighter's legal activity during a ground segment."""

    ground_str_attempted: int
    ground_str_landed: int
    control_seconds: int
    submission_attempts: int
    position_advancements: int

    escape_attempts: int
    reversal_attempts: int
    scramble_attempts: int

    def __post_init__(self) -> None:
        """Validate all ground activity counts."""

        integer_fields = {
            "ground_str_attempted": self.ground_str_attempted,
            "ground_str_landed": self.ground_str_landed,
            "control_seconds": self.control_seconds,
            "submission_attempts": self.submission_attempts,
            "position_advancements": self.position_advancements,
            "escape_attempts": self.escape_attempts,
            "reversal_attempts": self.reversal_attempts,
            "scramble_attempts": self.scramble_attempts,
        }

        for name, value in integer_fields.items():
            if not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer"
                )

            if value < 0:
                raise ValueError(
                    f"{name} cannot be negative"
                )

        if self.ground_str_landed > self.ground_str_attempted:
            raise ValueError(
                "ground_str_landed cannot exceed "
                "ground_str_attempted"
            )

        if self.control_seconds > SEGMENT_SECONDS:
            raise ValueError(
                f"control_seconds cannot exceed "
                f"{SEGMENT_SECONDS}"
            )

        if self.position_advancements > 1:
            raise ValueError(
                "position_advancements cannot exceed 1 "
                "per segment"
            )


@dataclass(frozen=True)
class GroundSegmentActivity:
    """Both fighters' activity in one shared ground segment."""

    state: SharedFightState
    red: GroundFighterActivity
    blue: GroundFighterActivity

    def __post_init__(self) -> None:
        """Enforce ground owner and defender role legality."""

        if self.state.phase is not FightPhase.GROUND:
            raise ValueError(
                "ground activity requires a ground "
                "shared state"
            )

        if self.state.phase_owner is None:
            raise ValueError(
                "ground activity requires a phase owner"
            )

        if self.state.phase_owner is FighterSide.RED:
            owner = self.red
            defender = self.blue
        else:
            owner = self.blue
            defender = self.red

        owner_defensive_activity = (
            owner.escape_attempts
            + owner.reversal_attempts
            + owner.scramble_attempts
        )

        if owner_defensive_activity != 0:
            raise ValueError(
                "ground owner cannot generate defender "
                "transition attempts"
            )

        defender_offensive_activity = (
            defender.ground_str_attempted
            + defender.ground_str_landed
            + defender.control_seconds
            + defender.submission_attempts
            + defender.position_advancements
        )

        if defender_offensive_activity != 0:
            raise ValueError(
                "ground defender cannot generate owner activity"
            )


def _empty_ground_activity() -> GroundFighterActivity:
    """Return a zero-valued ground activity record."""

    return GroundFighterActivity(
        ground_str_attempted=0,
        ground_str_landed=0,
        control_seconds=0,
        submission_attempts=0,
        position_advancements=0,
        escape_attempts=0,
        reversal_attempts=0,
        scramble_attempts=0,
    )


def _generate_owner_control_seconds(
    parameters: GroundOwnerRateParameters,
    rng: np.random.Generator,
) -> int:
    """Generate bounded ground-control time for the owner."""

    control_probability = (
        parameters.control_seconds_mean
        / SEGMENT_SECONDS
    )

    return int(
        rng.binomial(
            SEGMENT_SECONDS,
            control_probability,
        )
    )


def _generate_owner_activity(
    parameters: GroundOwnerRateParameters,
    rng: np.random.Generator,
) -> GroundFighterActivity:
    """Generate activity for the current ground owner."""

    attempted = int(
        rng.poisson(
            parameters.ground_strike_attempt_rate
        )
    )

    landed = int(
        rng.binomial(
            attempted,
            parameters.ground_strike_accuracy,
        )
    )

    control_seconds = _generate_owner_control_seconds(
        parameters,
        rng,
    )

    submission_attempts = int(
        rng.poisson(
            parameters.submission_attempt_rate
        )
    )

    position_advancements = int(
        rng.binomial(
            1,
            parameters.position_advancement_probability,
        )
    )

    return GroundFighterActivity(
        ground_str_attempted=attempted,
        ground_str_landed=landed,
        control_seconds=control_seconds,
        submission_attempts=submission_attempts,
        position_advancements=position_advancements,
        escape_attempts=0,
        reversal_attempts=0,
        scramble_attempts=0,
    )


def _generate_defender_activity(
    parameters: GroundDefenderRateParameters,
    rng: np.random.Generator,
) -> GroundFighterActivity:
    """Generate unsuccessful or potentially successful defensive attempts."""

    return GroundFighterActivity(
        ground_str_attempted=0,
        ground_str_landed=0,
        control_seconds=0,
        submission_attempts=0,
        position_advancements=0,
        escape_attempts=int(
            rng.poisson(
                parameters.escape_attempt_rate
            )
        ),
        reversal_attempts=int(
            rng.poisson(
                parameters.reversal_attempt_rate
            )
        ),
        scramble_attempts=int(
            rng.poisson(
                parameters.scramble_attempt_rate
            )
        ),
    )


def generate_ground_segment_activity(
    state: SharedFightState,
    owner_parameters: GroundOwnerRateParameters,
    defender_parameters: GroundDefenderRateParameters,
    rng: np.random.Generator,
) -> GroundSegmentActivity:
    """Generate role-legal activity in one shared ground segment.

    The caller selects the correct fighter parameters according to the
    authoritative ``state.phase_owner``.
    """

    if state.phase is not FightPhase.GROUND:
        raise ValueError(
            "ground activity requires a ground "
            "shared state"
        )

    if state.phase_owner is None:
        raise ValueError(
            "ground activity requires a phase owner"
        )

    owner_activity = _generate_owner_activity(
        owner_parameters,
        rng,
    )
    defender_activity = _generate_defender_activity(
        defender_parameters,
        rng,
    )

    if state.phase_owner is FighterSide.RED:
        red_activity = owner_activity
        blue_activity = defender_activity
    else:
        red_activity = defender_activity
        blue_activity = owner_activity

    return GroundSegmentActivity(
        state=state,
        red=red_activity,
        blue=blue_activity,
    )
