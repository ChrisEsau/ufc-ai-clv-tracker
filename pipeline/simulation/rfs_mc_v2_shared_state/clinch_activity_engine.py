"""Clinch-phase activity generation for RFS Monte Carlo V2.

Both fighters share one clinch phase. Both may generate clinch strikes, but
only the authoritative phase owner may accumulate clinch control time.

Takedowns remain end-of-segment transition events and are not generated here.
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
    ClinchRateParameters,
    SEGMENT_SECONDS,
)


@dataclass(frozen=True)
class ClinchFighterActivity:
    """One fighter's legal activity during a clinch segment."""

    clinch_str_attempted: int
    clinch_str_landed: int
    damaging_clinch_strikes: int
    control_seconds: int

    def __post_init__(self) -> None:
        """Validate clinch activity counts."""

        integer_fields = {
            "clinch_str_attempted": self.clinch_str_attempted,
            "clinch_str_landed": self.clinch_str_landed,
            "damaging_clinch_strikes": self.damaging_clinch_strikes,
            "control_seconds": self.control_seconds,
        }

        for name, value in integer_fields.items():
            if value < 0:
                raise ValueError(
                    f"{name} cannot be negative"
                )

        if self.clinch_str_landed > self.clinch_str_attempted:
            raise ValueError(
                "clinch_str_landed cannot exceed "
                "clinch_str_attempted"
            )

        if (
            self.damaging_clinch_strikes
            > self.clinch_str_landed
        ):
            raise ValueError(
                "damaging_clinch_strikes cannot exceed "
                "clinch_str_landed"
            )

        if self.control_seconds > SEGMENT_SECONDS:
            raise ValueError(
                f"control_seconds cannot exceed "
                f"{SEGMENT_SECONDS}"
            )


@dataclass(frozen=True)
class ClinchSegmentActivity:
    """Both fighters' activity in one shared clinch segment."""

    state: SharedFightState
    red: ClinchFighterActivity
    blue: ClinchFighterActivity

    def __post_init__(self) -> None:
        """Enforce shared-phase ownership rules."""

        if self.state.phase is not FightPhase.CLINCH:
            raise ValueError(
                "clinch activity requires a clinch "
                "shared state"
            )

        if self.state.phase_owner is None:
            raise ValueError(
                "clinch activity requires a phase owner"
            )

        if self.state.phase_owner is FighterSide.RED:
            if self.blue.control_seconds != 0:
                raise ValueError(
                    "clinch defender cannot accumulate "
                    "control time"
                )

        elif self.state.phase_owner is FighterSide.BLUE:
            if self.red.control_seconds != 0:
                raise ValueError(
                    "clinch defender cannot accumulate "
                    "control time"
                )


def _generate_clinch_striking(
    parameters: ClinchRateParameters,
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    """Generate phase-legal clinch striking activity."""

    attempted = int(
        rng.poisson(
            parameters.clinch_strike_attempt_rate
        )
    )

    landed = int(
        rng.binomial(
            attempted,
            parameters.clinch_strike_accuracy,
        )
    )

    damaging = int(
        rng.binomial(
            landed,
            parameters.damaging_clinch_probability,
        )
    )

    return attempted, landed, damaging


def _generate_owner_control_seconds(
    parameters: ClinchRateParameters,
    rng: np.random.Generator,
) -> int:
    """Generate bounded control time for the clinch owner.

    A binomial draw over the 30 seconds preserves the configured expected
    control duration while guaranteeing a legal segment maximum.
    """

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


def _generate_fighter_activity(
    parameters: ClinchRateParameters,
    *,
    is_owner: bool,
    rng: np.random.Generator,
) -> ClinchFighterActivity:
    """Generate one fighter's clinch activity."""

    attempted, landed, damaging = (
        _generate_clinch_striking(
            parameters,
            rng,
        )
    )

    control_seconds = (
        _generate_owner_control_seconds(
            parameters,
            rng,
        )
        if is_owner
        else 0
    )

    return ClinchFighterActivity(
        clinch_str_attempted=attempted,
        clinch_str_landed=landed,
        damaging_clinch_strikes=damaging,
        control_seconds=control_seconds,
    )


def generate_clinch_segment_activity(
    state: SharedFightState,
    red: ClinchRateParameters,
    blue: ClinchRateParameters,
    rng: np.random.Generator,
) -> ClinchSegmentActivity:
    """Generate both fighters' activity in one shared clinch segment."""

    if state.phase is not FightPhase.CLINCH:
        raise ValueError(
            "clinch activity requires a clinch "
            "shared state"
        )

    if state.phase_owner is None:
        raise ValueError(
            "clinch activity requires a phase owner"
        )

    red_activity = _generate_fighter_activity(
        red,
        is_owner=state.phase_owner is FighterSide.RED,
        rng=rng,
    )

    blue_activity = _generate_fighter_activity(
        blue,
        is_owner=state.phase_owner is FighterSide.BLUE,
        rng=rng,
    )

    return ClinchSegmentActivity(
        state=state,
        red=red_activity,
        blue=blue_activity,
    )
