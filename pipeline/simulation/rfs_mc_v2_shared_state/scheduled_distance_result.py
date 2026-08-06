"""Scheduled-distance result pipeline for RFS Monte Carlo V2.

This module connects an unfinished full-length simulation path through:

1. completed-round segmentation
2. deterministic round-evidence aggregation
3. deterministic round-scoring assessment
4. independently seeded three-judge scorecards
5. official decision resolution

Finish paths are intentionally rejected. They belong to the finish-result
branch of the final fight-result resolver.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    SEGMENTS_PER_ROUND,
    FighterSide,
)
from pipeline.simulation.rfs_mc_v2_shared_state.decision_contracts import (
    DecisionResult,
    DecisionType,
    resolve_decision,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_path_contracts import (
    FinishEnabledDynamicPath,
    FinishEvaluatedPathSegment,
)
from pipeline.simulation.rfs_mc_v2_shared_state.judge_scorecard_generator import (
    JudgePanelScorecards,
    JudgeVariabilityCalibration,
    generate_judge_panel_scorecards,
)
from pipeline.simulation.rfs_mc_v2_shared_state.round_evidence import (
    RoundEvidence,
    calculate_round_evidence,
)
from pipeline.simulation.rfs_mc_v2_shared_state.round_scoring_contracts import (
    JudgeScorecard,
)
from pipeline.simulation.rfs_mc_v2_shared_state.round_scoring_engine import (
    RoundScoringAssessment,
    RoundScoringCalibration,
    calculate_round_scoring_assessment,
)


@dataclass(frozen=True)
class ScheduledDistanceRoundResult:
    """Evidence and scoring output for one completed round."""

    round_number: int
    segments: tuple[FinishEvaluatedPathSegment, ...]
    evidence: RoundEvidence
    assessment: RoundScoringAssessment

    def __post_init__(self) -> None:
        """Validate round sequence and nested result consistency."""

        if type(self.round_number) is not int:
            raise TypeError(
                "round_number must be an integer"
            )

        if not 1 <= self.round_number <= 5:
            raise ValueError(
                "round_number must be between 1 and 5"
            )

        if not isinstance(
            self.segments,
            tuple,
        ):
            raise TypeError(
                "segments must be a tuple"
            )

        if len(self.segments) != SEGMENTS_PER_ROUND:
            raise ValueError(
                "scheduled-distance round must contain "
                f"exactly {SEGMENTS_PER_ROUND} segments"
            )

        for expected_segment, record in enumerate(
            self.segments,
            start=1,
        ):
            if not isinstance(
                record,
                FinishEvaluatedPathSegment,
            ):
                raise TypeError(
                    "segments must contain "
                    "FinishEvaluatedPathSegment values"
                )

            if record.finish is not None:
                raise ValueError(
                    "scheduled-distance rounds cannot "
                    "contain a finish"
                )

            if record.state.round_number != self.round_number:
                raise ValueError(
                    "all segments must match round_number"
                )

            if record.state.segment_number != expected_segment:
                raise ValueError(
                    "round segments must be sequential "
                    "from one through ten"
                )

        if not isinstance(
            self.evidence,
            RoundEvidence,
        ):
            raise TypeError(
                "evidence must be RoundEvidence"
            )

        if self.evidence.round_number != self.round_number:
            raise ValueError(
                "evidence round_number must match result"
            )

        if not isinstance(
            self.assessment,
            RoundScoringAssessment,
        ):
            raise TypeError(
                "assessment must be RoundScoringAssessment"
            )

        if self.assessment.round_number != self.round_number:
            raise ValueError(
                "assessment round_number must match result"
            )


@dataclass(frozen=True)
class ScheduledDistanceResult:
    """Complete scheduled-distance fight result."""

    path: FinishEnabledDynamicPath
    rounds: tuple[ScheduledDistanceRoundResult, ...]
    judge_panel: JudgePanelScorecards
    decision: DecisionResult

    def __post_init__(self) -> None:
        """Validate path, rounds, panel, and decision consistency."""

        if not isinstance(
            self.path,
            FinishEnabledDynamicPath,
        ):
            raise TypeError(
                "path must be FinishEnabledDynamicPath"
            )

        if self.path.finish is not None:
            raise ValueError(
                "scheduled-distance result cannot contain "
                "a finish"
            )

        if not self.path.reached_scheduled_distance:
            raise ValueError(
                "path must reach scheduled distance"
            )

        if not isinstance(
            self.rounds,
            tuple,
        ):
            raise TypeError(
                "rounds must be a tuple"
            )

        if len(self.rounds) != self.path.scheduled_rounds:
            raise ValueError(
                "round results must contain exactly one "
                "entry per scheduled round"
            )

        for expected_round, round_result in enumerate(
            self.rounds,
            start=1,
        ):
            if not isinstance(
                round_result,
                ScheduledDistanceRoundResult,
            ):
                raise TypeError(
                    "rounds must contain "
                    "ScheduledDistanceRoundResult values"
                )

            if round_result.round_number != expected_round:
                raise ValueError(
                    "round results must be sequential "
                    "starting at round one"
                )

        flattened_segments = tuple(
            segment
            for round_result in self.rounds
            for segment in round_result.segments
        )

        if flattened_segments != self.path.segments:
            raise ValueError(
                "round-result segments must exactly match "
                "the simulation path"
            )

        if not isinstance(
            self.judge_panel,
            JudgePanelScorecards,
        ):
            raise TypeError(
                "judge_panel must be JudgePanelScorecards"
            )

        if (
            self.judge_panel.scheduled_rounds
            != self.path.scheduled_rounds
        ):
            raise ValueError(
                "judge panel scheduled_rounds must match path"
            )

        if self.judge_panel.seed != self.path.seed:
            raise ValueError(
                "judge panel seed must match path seed"
            )

        if not isinstance(
            self.decision,
            DecisionResult,
        ):
            raise TypeError(
                "decision must be DecisionResult"
            )

        if (
            self.decision.scheduled_rounds
            != self.path.scheduled_rounds
        ):
            raise ValueError(
                "decision scheduled_rounds must match path"
            )

        if (
            self.decision.scorecards
            != self.judge_panel.scorecards
        ):
            raise ValueError(
                "decision scorecards must match judge panel"
            )

    @property
    def scheduled_rounds(self) -> int:
        """Return the fight's scheduled round count."""

        return self.path.scheduled_rounds

    @property
    def seed(self) -> int:
        """Return the simulation seed."""

        return self.path.seed

    @property
    def winner(self) -> FighterSide | None:
        """Return the official decision winner."""

        return self.decision.winner

    @property
    def decision_type(self) -> DecisionType:
        """Return the official decision classification."""

        return self.decision.decision_type

    @property
    def scorecards(self) -> tuple[JudgeScorecard, ...]:
        """Return the three official judge scorecards."""

        return self.judge_panel.scorecards

    @property
    def is_draw(self) -> bool:
        """Return whether the official result is a draw."""

        return self.decision.is_draw


def resolve_scheduled_distance_path(
    path: FinishEnabledDynamicPath,
    *,
    scoring_calibration: RoundScoringCalibration | None = None,
    variability_calibration: JudgeVariabilityCalibration | None = None,
) -> ScheduledDistanceResult:
    """Resolve one completed full-length path into an official decision."""

    if not isinstance(
        path,
        FinishEnabledDynamicPath,
    ):
        raise TypeError(
            "path must be FinishEnabledDynamicPath"
        )

    if path.finish is not None:
        raise ValueError(
            "cannot score a path that ended by finish"
        )

    if not path.reached_scheduled_distance:
        raise ValueError(
            "path must reach scheduled distance"
        )

    selected_scoring = (
        scoring_calibration
        if scoring_calibration is not None
        else RoundScoringCalibration()
    )

    if not isinstance(
        selected_scoring,
        RoundScoringCalibration,
    ):
        raise TypeError(
            "scoring_calibration must be "
            "RoundScoringCalibration"
        )

    selected_variability = (
        variability_calibration
        if variability_calibration is not None
        else JudgeVariabilityCalibration()
    )

    if not isinstance(
        selected_variability,
        JudgeVariabilityCalibration,
    ):
        raise TypeError(
            "variability_calibration must be "
            "JudgeVariabilityCalibration"
        )

    round_results: list[
        ScheduledDistanceRoundResult
    ] = []
    assessments: list[
        RoundScoringAssessment
    ] = []

    for round_number in range(
        1,
        path.scheduled_rounds + 1,
    ):
        start_index = (
            round_number - 1
        ) * SEGMENTS_PER_ROUND
        end_index = (
            start_index
            + SEGMENTS_PER_ROUND
        )

        round_segments = tuple(
            path.segments[
                start_index:end_index
            ]
        )

        evidence = calculate_round_evidence(
            round_segments
        )

        assessment = calculate_round_scoring_assessment(
            evidence,
            selected_scoring,
        )

        round_results.append(
            ScheduledDistanceRoundResult(
                round_number=round_number,
                segments=round_segments,
                evidence=evidence,
                assessment=assessment,
            )
        )
        assessments.append(
            assessment
        )

    judge_panel = generate_judge_panel_scorecards(
        tuple(assessments),
        seed=path.seed,
        variability_calibration=selected_variability,
        scoring_calibration=selected_scoring,
    )

    decision = resolve_decision(
        judge_panel.scorecards
    )

    return ScheduledDistanceResult(
        path=path,
        rounds=tuple(round_results),
        judge_panel=judge_panel,
        decision=decision,
    )
