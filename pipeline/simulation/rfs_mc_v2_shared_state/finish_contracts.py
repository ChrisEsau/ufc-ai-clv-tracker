"""Finish-result contracts for RFS Monte Carlo V2.

These contracts describe a completed simulated fight finish. They do not
calculate finish probability or alter a fight path.

Initial supported methods:

- KO/TKO
- submission

Decisions will be handled later by the scoring layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    SEGMENT_SECONDS,
)


class FinishMethod(str, Enum):
    """Supported non-decision fight-ending methods."""

    KO_TKO = "ko_tko"
    SUBMISSION = "submission"


@dataclass(frozen=True)
class FinishResult:
    """One completed simulated fight finish.

    Attributes:
        state:
            Authoritative shared fight state during the finishing segment.

        winner:
            Fighter who produced the finish.

        method:
            KO/TKO or submission.

        elapsed_seconds_in_segment:
            Approximate second within the 30-second segment when the finish
            occurred. This permits later conversion to official round time.
    """

    state: SharedFightState
    winner: FighterSide
    method: FinishMethod
    elapsed_seconds_in_segment: int

    def __post_init__(self) -> None:
        """Validate finish timing and physical legality."""

        if not isinstance(
            self.state,
            SharedFightState,
        ):
            raise TypeError(
                "state must be SharedFightState"
            )

        if not isinstance(
            self.winner,
            FighterSide,
        ):
            raise TypeError(
                "winner must be FighterSide"
            )

        if not isinstance(
            self.method,
            FinishMethod,
        ):
            raise TypeError(
                "method must be FinishMethod"
            )

        if not isinstance(
            self.elapsed_seconds_in_segment,
            int,
        ):
            raise TypeError(
                "elapsed_seconds_in_segment must be an integer"
            )

        if not (
            1
            <= self.elapsed_seconds_in_segment
            <= SEGMENT_SECONDS
        ):
            raise ValueError(
                "elapsed_seconds_in_segment must be between "
                f"1 and {SEGMENT_SECONDS}"
            )

        if self.method is FinishMethod.SUBMISSION:
            if self.state.phase is not FightPhase.GROUND:
                raise ValueError(
                    "submission finish requires a ground state"
                )

            if self.state.phase_owner is not self.winner:
                raise ValueError(
                    "submission winner must own the ground phase"
                )

    @property
    def loser(self) -> FighterSide:
        """Return the opposing fighter."""

        return self.winner.opponent

    @property
    def round_number(self) -> int:
        """Return the finish round."""

        return self.state.round_number

    @property
    def segment_number(self) -> int:
        """Return the finishing segment."""

        return self.state.segment_number

    @property
    def elapsed_seconds_in_round(self) -> int:
        """Return approximate elapsed seconds within the round."""

        return (
            (
                self.segment_number - 1
            )
            * SEGMENT_SECONDS
            + self.elapsed_seconds_in_segment
        )
