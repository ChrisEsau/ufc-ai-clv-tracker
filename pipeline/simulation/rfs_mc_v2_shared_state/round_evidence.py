"""Deterministic completed-round evidence for RFS Monte Carlo V2.

This module aggregates the ten simulated segments of one completed round into
auditable fighter evidence.

It does not decide who won the round. Judge weighting, score margins, and
judge-specific variability belong to later layers.

Evidence includes:

- phase-specific strike attempts and landed strikes
- knockdowns and damaging clinch strikes
- control time
- submission and positional grappling actions
- modeled persistent damage inflicted
- modeled acute stress inflicted

Damage and stress inflicted by one fighter are derived from the opponent's
segment exposure because exposure records adversity received.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pipeline.simulation.rfs_mc_v2_shared_state.clinch_activity_engine import (
    ClinchSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    SEGMENTS_PER_ROUND,
    FighterSide,
)
from pipeline.simulation.rfs_mc_v2_shared_state.distance_activity_engine import (
    DistanceSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_path_contracts import (
    FinishEvaluatedPathSegment,
)
from pipeline.simulation.rfs_mc_v2_shared_state.ground_activity_engine import (
    GroundSegmentActivity,
)


INTEGER_EVIDENCE_FIELDS = (
    "distance_strikes_attempted",
    "distance_strikes_landed",
    "clinch_strikes_attempted",
    "clinch_strikes_landed",
    "ground_strikes_attempted",
    "ground_strikes_landed",
    "knockdowns",
    "damaging_clinch_strikes",
    "control_seconds",
    "submission_attempts",
    "position_advancements",
    "escape_attempts",
    "reversal_attempts",
    "scramble_attempts",
)

FLOAT_EVIDENCE_FIELDS = (
    "persistent_damage_inflicted",
    "acute_stress_inflicted",
)


@dataclass(frozen=True)
class FighterRoundEvidence:
    """Aggregated evidence produced by one fighter during one round."""

    distance_strikes_attempted: int
    distance_strikes_landed: int

    clinch_strikes_attempted: int
    clinch_strikes_landed: int

    ground_strikes_attempted: int
    ground_strikes_landed: int

    knockdowns: int
    damaging_clinch_strikes: int

    control_seconds: int

    submission_attempts: int
    position_advancements: int
    escape_attempts: int
    reversal_attempts: int
    scramble_attempts: int

    persistent_damage_inflicted: float
    acute_stress_inflicted: float

    def __post_init__(self) -> None:
        """Validate nonnegative round-evidence values."""

        for name in INTEGER_EVIDENCE_FIELDS:
            value = getattr(self, name)

            if type(value) is not int:
                raise TypeError(
                    f"{name} must be an integer"
                )

            if value < 0:
                raise ValueError(
                    f"{name} cannot be negative"
                )

        for name in FLOAT_EVIDENCE_FIELDS:
            value = getattr(self, name)

            if type(value) not in {
                int,
                float,
            }:
                raise TypeError(
                    f"{name} must be numeric"
                )

            selected = float(value)

            if not math.isfinite(selected):
                raise ValueError(
                    f"{name} must be finite"
                )

            if selected < 0.0:
                raise ValueError(
                    f"{name} cannot be negative"
                )

    @property
    def total_strikes_attempted(self) -> int:
        """Return attempts across distance, clinch, and ground."""

        return (
            self.distance_strikes_attempted
            + self.clinch_strikes_attempted
            + self.ground_strikes_attempted
        )

    @property
    def total_strikes_landed(self) -> int:
        """Return landed strikes across every phase."""

        return (
            self.distance_strikes_landed
            + self.clinch_strikes_landed
            + self.ground_strikes_landed
        )

    @property
    def striking_accuracy(self) -> float:
        """Return aggregate striking accuracy for the round."""

        if self.total_strikes_attempted == 0:
            return 0.0

        return (
            self.total_strikes_landed
            / self.total_strikes_attempted
        )

    @property
    def offensive_grappling_actions(self) -> int:
        """Return submission, advancement, and reversal actions."""

        return (
            self.submission_attempts
            + self.position_advancements
            + self.reversal_attempts
        )


@dataclass(frozen=True)
class RoundEvidence:
    """Complete deterministic evidence for one completed round."""

    round_number: int
    red: FighterRoundEvidence
    blue: FighterRoundEvidence

    def __post_init__(self) -> None:
        """Validate round identity and nested evidence contracts."""

        if type(self.round_number) is not int:
            raise TypeError(
                "round_number must be an integer"
            )

        if not 1 <= self.round_number <= 5:
            raise ValueError(
                "round_number must be between 1 and 5"
            )

        if not isinstance(
            self.red,
            FighterRoundEvidence,
        ):
            raise TypeError(
                "red must be FighterRoundEvidence"
            )

        if not isinstance(
            self.blue,
            FighterRoundEvidence,
        ):
            raise TypeError(
                "blue must be FighterRoundEvidence"
            )

    def for_side(
        self,
        side: FighterSide,
    ) -> FighterRoundEvidence:
        """Return evidence for the selected fighter."""

        if not isinstance(
            side,
            FighterSide,
        ):
            raise TypeError(
                "side must be FighterSide"
            )

        return (
            self.red
            if side is FighterSide.RED
            else self.blue
        )


def _empty_evidence_values() -> dict[str, int | float]:
    """Return mutable zero-valued aggregation fields."""

    values: dict[str, int | float] = {
        name: 0
        for name in INTEGER_EVIDENCE_FIELDS
    }

    values.update(
        {
            name: 0.0
            for name in FLOAT_EVIDENCE_FIELDS
        }
    )

    return values


def calculate_round_evidence(
    segments: tuple[FinishEvaluatedPathSegment, ...],
) -> RoundEvidence:
    """Aggregate exactly one completed ten-segment round."""

    if not isinstance(
        segments,
        tuple,
    ):
        raise TypeError(
            "segments must be a tuple"
        )

    if len(segments) != SEGMENTS_PER_ROUND:
        raise ValueError(
            "completed round must contain exactly "
            f"{SEGMENTS_PER_ROUND} segments"
        )

    for record in segments:
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
                "round evidence requires a completed "
                "non-finishing round"
            )

    round_number = segments[0].state.round_number

    for expected_segment, record in enumerate(
        segments,
        start=1,
    ):
        if record.state.round_number != round_number:
            raise ValueError(
                "all segments must belong to the same round"
            )

        if record.state.segment_number != expected_segment:
            raise ValueError(
                "round segments must be sequential "
                "from one through ten"
            )

    red_values = _empty_evidence_values()
    blue_values = _empty_evidence_values()

    for record in segments:
        activity = record.activity

        if isinstance(
            activity,
            DistanceSegmentActivity,
        ):
            red_values["distance_strikes_attempted"] += (
                activity.red.sig_str_attempted
            )
            red_values["distance_strikes_landed"] += (
                activity.red.sig_str_landed
            )
            red_values["knockdowns"] += (
                activity.red.knockdowns
            )

            blue_values["distance_strikes_attempted"] += (
                activity.blue.sig_str_attempted
            )
            blue_values["distance_strikes_landed"] += (
                activity.blue.sig_str_landed
            )
            blue_values["knockdowns"] += (
                activity.blue.knockdowns
            )

        elif isinstance(
            activity,
            ClinchSegmentActivity,
        ):
            red_values["clinch_strikes_attempted"] += (
                activity.red.clinch_str_attempted
            )
            red_values["clinch_strikes_landed"] += (
                activity.red.clinch_str_landed
            )
            red_values["damaging_clinch_strikes"] += (
                activity.red.damaging_clinch_strikes
            )
            red_values["control_seconds"] += (
                activity.red.control_seconds
            )

            blue_values["clinch_strikes_attempted"] += (
                activity.blue.clinch_str_attempted
            )
            blue_values["clinch_strikes_landed"] += (
                activity.blue.clinch_str_landed
            )
            blue_values["damaging_clinch_strikes"] += (
                activity.blue.damaging_clinch_strikes
            )
            blue_values["control_seconds"] += (
                activity.blue.control_seconds
            )

        elif isinstance(
            activity,
            GroundSegmentActivity,
        ):
            red_values["ground_strikes_attempted"] += (
                activity.red.ground_str_attempted
            )
            red_values["ground_strikes_landed"] += (
                activity.red.ground_str_landed
            )
            red_values["control_seconds"] += (
                activity.red.control_seconds
            )
            red_values["submission_attempts"] += (
                activity.red.submission_attempts
            )
            red_values["position_advancements"] += (
                activity.red.position_advancements
            )
            red_values["escape_attempts"] += (
                activity.red.escape_attempts
            )
            red_values["reversal_attempts"] += (
                activity.red.reversal_attempts
            )
            red_values["scramble_attempts"] += (
                activity.red.scramble_attempts
            )

            blue_values["ground_strikes_attempted"] += (
                activity.blue.ground_str_attempted
            )
            blue_values["ground_strikes_landed"] += (
                activity.blue.ground_str_landed
            )
            blue_values["control_seconds"] += (
                activity.blue.control_seconds
            )
            blue_values["submission_attempts"] += (
                activity.blue.submission_attempts
            )
            blue_values["position_advancements"] += (
                activity.blue.position_advancements
            )
            blue_values["escape_attempts"] += (
                activity.blue.escape_attempts
            )
            blue_values["reversal_attempts"] += (
                activity.blue.reversal_attempts
            )
            blue_values["scramble_attempts"] += (
                activity.blue.scramble_attempts
            )

        else:
            raise TypeError(
                "segment activity must be a supported "
                "phase activity contract"
            )

        # Exposure is adversity received, so the opponent inflicted it.
        red_values["persistent_damage_inflicted"] += (
            record.exposure.blue.persistent_damage_exposure
        )
        red_values["acute_stress_inflicted"] += (
            record.exposure.blue.acute_stress_exposure
        )

        blue_values["persistent_damage_inflicted"] += (
            record.exposure.red.persistent_damage_exposure
        )
        blue_values["acute_stress_inflicted"] += (
            record.exposure.red.acute_stress_exposure
        )

    return RoundEvidence(
        round_number=round_number,
        red=FighterRoundEvidence(
            **red_values
        ),
        blue=FighterRoundEvidence(
            **blue_values
        ),
    )
