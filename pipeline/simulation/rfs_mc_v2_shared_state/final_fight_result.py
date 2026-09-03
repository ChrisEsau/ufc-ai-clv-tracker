"""Unified final fight-result resolution for RFS Monte Carlo V2.

Every completed simulation path resolves through exactly one branch:

- finish:
  - KO/TKO
  - submission

- scheduled distance:
  - unanimous, split, or majority decision
  - unanimous, split, or majority draw

The finish and scheduled-distance payloads remain available for detailed
auditing, while this module provides one authoritative top-level result.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
)
from pipeline.simulation.rfs_mc_v2_shared_state.decision_contracts import (
    DecisionType,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_contracts import (
    FinishMethod,
    FinishResult,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_path_contracts import (
    FinishEnabledDynamicPath,
)
from pipeline.simulation.rfs_mc_v2_shared_state.judge_scorecard_generator import (
    JudgeVariabilityCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.round_scoring_contracts import (
    JudgeScorecard,
)
from pipeline.simulation.rfs_mc_v2_shared_state.round_scoring_engine import (
    RoundScoringCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.scheduled_distance_result import (
    ScheduledDistanceResult,
    resolve_scheduled_distance_path,
)


class FightResultBranch(str, Enum):
    """Terminal branch used to resolve the fight."""

    FINISH = "finish"
    SCHEDULED_DISTANCE = "scheduled_distance"


@dataclass(frozen=True)
class FinalFightResult:
    """Authoritative resolved result for one completed simulation path."""

    path: FinishEnabledDynamicPath
    branch: FightResultBranch
    winner: FighterSide | None

    finish: FinishResult | None
    scheduled_distance: ScheduledDistanceResult | None

    def __post_init__(self) -> None:
        """Validate branch exclusivity and result consistency."""

        if not isinstance(
            self.path,
            FinishEnabledDynamicPath,
        ):
            raise TypeError(
                "path must be FinishEnabledDynamicPath"
            )

        if not isinstance(
            self.branch,
            FightResultBranch,
        ):
            raise TypeError(
                "branch must be FightResultBranch"
            )

        if (
            self.winner is not None
            and not isinstance(
                self.winner,
                FighterSide,
            )
        ):
            raise TypeError(
                "winner must be FighterSide or None"
            )

        if (
            self.finish is not None
            and not isinstance(
                self.finish,
                FinishResult,
            )
        ):
            raise TypeError(
                "finish must be FinishResult or None"
            )

        if (
            self.scheduled_distance is not None
            and not isinstance(
                self.scheduled_distance,
                ScheduledDistanceResult,
            )
        ):
            raise TypeError(
                "scheduled_distance must be "
                "ScheduledDistanceResult or None"
            )

        if self.branch is FightResultBranch.FINISH:
            self._validate_finish_branch()
        else:
            self._validate_scheduled_distance_branch()

    def _validate_finish_branch(self) -> None:
        """Validate a finish-branch result."""

        if self.finish is None:
            raise ValueError(
                "finish branch requires a finish result"
            )

        if self.scheduled_distance is not None:
            raise ValueError(
                "finish branch cannot contain a "
                "scheduled-distance result"
            )

        if self.path.finish is None:
            raise ValueError(
                "finish branch requires a path that "
                "ended by finish"
            )

        if self.finish != self.path.finish:
            raise ValueError(
                "finish result must match the path finish"
            )

        if self.winner is not self.finish.winner:
            raise ValueError(
                "winner must match the finish winner"
            )

    def _validate_scheduled_distance_branch(self) -> None:
        """Validate a scheduled-distance branch result."""

        if self.finish is not None:
            raise ValueError(
                "scheduled-distance branch cannot "
                "contain a finish result"
            )

        if self.scheduled_distance is None:
            raise ValueError(
                "scheduled-distance branch requires a "
                "scheduled-distance result"
            )

        if self.path.finish is not None:
            raise ValueError(
                "scheduled-distance branch cannot use "
                "a path that ended by finish"
            )

        if not self.path.reached_scheduled_distance:
            raise ValueError(
                "scheduled-distance branch requires a "
                "full-length path"
            )

        if self.scheduled_distance.path != self.path:
            raise ValueError(
                "scheduled-distance result path must "
                "match the final-result path"
            )

        if self.winner is not self.scheduled_distance.winner:
            raise ValueError(
                "winner must match the scheduled-distance "
                "result winner"
            )

    @property
    def scheduled_rounds(self) -> int:
        """Return the scheduled fight length."""

        return self.path.scheduled_rounds

    @property
    def seed(self) -> int:
        """Return the simulation seed."""

        return self.path.seed

    @property
    def is_finish(self) -> bool:
        """Return whether the fight ended before scheduled distance."""

        return self.branch is FightResultBranch.FINISH

    @property
    def is_scheduled_distance(self) -> bool:
        """Return whether the fight reached scheduled distance."""

        return (
            self.branch
            is FightResultBranch.SCHEDULED_DISTANCE
        )

    @property
    def is_draw(self) -> bool:
        """Return whether the official result is a draw."""

        return (
            self.is_scheduled_distance
            and self.winner is None
        )

    @property
    def finish_method(self) -> FinishMethod | None:
        """Return KO/TKO or submission for a finish result."""

        if self.finish is None:
            return None

        return self.finish.method

    @property
    def decision_type(self) -> DecisionType | None:
        """Return the decision classification at scheduled distance."""

        if self.scheduled_distance is None:
            return None

        return self.scheduled_distance.decision_type

    @property
    def official_method(
        self,
    ) -> FinishMethod | DecisionType:
        """Return the authoritative terminal method."""

        if self.finish is not None:
            return self.finish.method

        if self.scheduled_distance is None:
            raise RuntimeError(
                "final result has no terminal payload"
            )

        return self.scheduled_distance.decision_type

    @property
    def finish_round(self) -> int | None:
        """Return the finishing round, or None for decisions."""

        if self.finish is None:
            return None

        return self.finish.round_number

    @property
    def finish_segment(self) -> int | None:
        """Return the finishing segment, or None for decisions."""

        if self.finish is None:
            return None

        return self.finish.segment_number

    @property
    def elapsed_seconds_in_round(self) -> int | None:
        """Return finish time within the round."""

        if self.finish is None:
            return None

        return self.finish.elapsed_seconds_in_round

    @property
    def scorecards(
        self,
    ) -> tuple[JudgeScorecard, ...] | None:
        """Return official scorecards for scheduled-distance results."""

        if self.scheduled_distance is None:
            return None

        return self.scheduled_distance.scorecards


def resolve_final_fight_result(
    path: FinishEnabledDynamicPath,
    *,
    scoring_calibration: RoundScoringCalibration | None = None,
    variability_calibration: JudgeVariabilityCalibration | None = None,
) -> FinalFightResult:
    """Resolve one completed path through its correct terminal branch."""

    if not isinstance(
        path,
        FinishEnabledDynamicPath,
    ):
        raise TypeError(
            "path must be FinishEnabledDynamicPath"
        )

    if path.finish is not None:
        return FinalFightResult(
            path=path,
            branch=FightResultBranch.FINISH,
            winner=path.finish.winner,
            finish=path.finish,
            scheduled_distance=None,
        )

    if not path.reached_scheduled_distance:
        raise ValueError(
            "unfinished path has no resolvable final result"
        )

    scheduled_distance = resolve_scheduled_distance_path(
        path,
        scoring_calibration=scoring_calibration,
        variability_calibration=variability_calibration,
    )

    return FinalFightResult(
        path=path,
        branch=FightResultBranch.SCHEDULED_DISTANCE,
        winner=scheduled_distance.winner,
        finish=None,
        scheduled_distance=scheduled_distance,
    )
