"""Probabilistic finish hazards for RFS Monte Carlo V1.

KO/TKO and submission outcomes are competing hazards evaluated after each
30-second segment. No state value creates an automatic finish; even highly
damaged fighters retain a bounded chance of surviving the segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from pipeline.simulation.rfs_mc_v1.contracts import (
    DynamicFighterState,
    FighterSimulationProfile,
)
from pipeline.simulation.rfs_mc_v1.segment_engine import SegmentActivity


class FinishMethod(str, Enum):
    """Supported fight-ending methods."""

    KO_TKO = "ko_tko"
    SUBMISSION = "submission"


@dataclass(frozen=True)
class FinishHazardParameters:
    """Coefficients for per-segment competing finish hazards."""

    ko_intercept: float = -7.25
    ko_head_damage_weight: float = 3.00
    ko_chin_loss_weight: float = 2.25
    ko_defense_loss_weight: float = 1.25
    ko_energy_loss_weight: float = 0.75
    ko_landed_strike_weight: float = 0.10
    ko_ground_strike_weight: float = 0.15
    ko_knockdown_weight: float = 2.20

    submission_intercept: float = -7.50
    submission_danger_weight: float = 2.75
    submission_attempt_weight: float = 1.10
    submission_control_weight: float = 0.035
    submission_energy_loss_weight: float = 0.65
    submission_defense_loss_weight: float = 0.55

    minimum_hazard: float = 0.0
    maximum_hazard: float = 0.85

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_hazard < 1.0:
            raise ValueError("minimum_hazard must be in [0, 1)")

        if not 0.0 < self.maximum_hazard < 1.0:
            raise ValueError("maximum_hazard must be in (0, 1)")

        if self.minimum_hazard >= self.maximum_hazard:
            raise ValueError(
                "minimum_hazard must be below maximum_hazard"
            )


DEFAULT_FINISH_HAZARD_PARAMETERS = FinishHazardParameters()


@dataclass(frozen=True)
class FighterFinishHazards:
    """KO/TKO and submission hazards against one defending fighter."""

    ko_tko: float
    submission: float

    def __post_init__(self) -> None:
        for name, value in (
            ("ko_tko", self.ko_tko),
            ("submission", self.submission),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} hazard must be in [0, 1)")


@dataclass(frozen=True)
class SegmentFinishResult:
    """Outcome of one competing-hazard segment evaluation."""

    finished: bool
    winner: str | None = None
    loser: str | None = None
    method: FinishMethod | None = None

    red_hazards: FighterFinishHazards | None = None
    blue_hazards: FighterFinishHazards | None = None

    def __post_init__(self) -> None:
        if self.finished:
            if self.winner not in {"red", "blue"}:
                raise ValueError(
                    "Finished result requires red or blue winner"
                )
            if self.loser not in {"red", "blue"}:
                raise ValueError(
                    "Finished result requires red or blue loser"
                )
            if self.winner == self.loser:
                raise ValueError("winner and loser must differ")
            if self.method is None:
                raise ValueError(
                    "Finished result requires a finish method"
                )
        else:
            if any(
                value is not None
                for value in (
                    self.winner,
                    self.loser,
                    self.method,
                )
            ):
                raise ValueError(
                    "Unfinished result cannot have winner, loser, or method"
                )


def _sigmoid(value: float) -> float:
    """Numerically stable logistic transform."""

    if value >= 0:
        exp_value = np.exp(-value)
        return float(1.0 / (1.0 + exp_value))

    exp_value = np.exp(value)
    return float(exp_value / (1.0 + exp_value))


def _bounded_hazard(
    linear_predictor: float,
    parameters: FinishHazardParameters,
) -> float:
    """Convert a linear predictor to a bounded segment hazard."""

    probability = _sigmoid(linear_predictor)

    return float(
        np.clip(
            probability,
            parameters.minimum_hazard,
            parameters.maximum_hazard,
        )
    )


def calculate_finish_hazards(
    *,
    defender_state: DynamicFighterState,
    attacker_activity: SegmentActivity,
    defender_profile: FighterSimulationProfile,
    parameters: FinishHazardParameters = (
        DEFAULT_FINISH_HAZARD_PARAMETERS
    ),
) -> FighterFinishHazards:
    """Calculate finish hazards against one defender."""

    defender_state.validate()

    head_damage = max(defender_state.head_damage, 0.0)
    chin_loss = 1.0 - defender_state.chin_integrity
    defense_loss = 1.0 - defender_state.defensive_stability
    energy_loss = 1.0 - defender_state.energy

    ko_linear = (
        parameters.ko_intercept
        + parameters.ko_head_damage_weight * head_damage
        + parameters.ko_chin_loss_weight * chin_loss
        + parameters.ko_defense_loss_weight * defense_loss
        + parameters.ko_energy_loss_weight * energy_loss
        + parameters.ko_landed_strike_weight
        * attacker_activity.sig_str_landed
        + parameters.ko_ground_strike_weight
        * attacker_activity.ground_str_landed
        + parameters.ko_knockdown_weight
        * attacker_activity.knockdowns
    )

    submission_linear = (
        parameters.submission_intercept
        + parameters.submission_danger_weight
        * defender_state.submission_danger
        + parameters.submission_attempt_weight
        * attacker_activity.submission_attempts
        + parameters.submission_control_weight
        * attacker_activity.control_seconds
        + parameters.submission_energy_loss_weight * energy_loss
        + parameters.submission_defense_loss_weight * defense_loss
    )

    # Reserved for later calibration and hierarchical profile modifiers.
    _ = defender_profile

    return FighterFinishHazards(
        ko_tko=_bounded_hazard(ko_linear, parameters),
        submission=_bounded_hazard(
            submission_linear,
            parameters,
        ),
    )


def sample_competing_finish(
    *,
    red_state: DynamicFighterState,
    blue_state: DynamicFighterState,
    red_activity: SegmentActivity,
    blue_activity: SegmentActivity,
    red_profile: FighterSimulationProfile,
    blue_profile: FighterSimulationProfile,
    rng: np.random.Generator,
    parameters: FinishHazardParameters = (
        DEFAULT_FINISH_HAZARD_PARAMETERS
    ),
) -> SegmentFinishResult:
    """Evaluate red and blue finish hazards for one shared segment.

    Hazards are converted into four mutually exclusive finish events:

    - red KO/TKO win
    - red submission win
    - blue KO/TKO win
    - blue submission win

    The combined finish probability is capped below one, preserving a
    non-zero probability that the fight continues.
    """

    red_hazards = calculate_finish_hazards(
        defender_state=blue_state,
        attacker_activity=red_activity,
        defender_profile=blue_profile,
        parameters=parameters,
    )
    blue_hazards = calculate_finish_hazards(
        defender_state=red_state,
        attacker_activity=blue_activity,
        defender_profile=red_profile,
        parameters=parameters,
    )

    event_weights = np.array(
        [
            red_hazards.ko_tko,
            red_hazards.submission,
            blue_hazards.ko_tko,
            blue_hazards.submission,
        ],
        dtype=float,
    )

    total_finish_hazard = float(event_weights.sum())

    # Preserve survival probability and avoid deterministic finishes.
    maximum_total = parameters.maximum_hazard
    if total_finish_hazard > maximum_total:
        event_weights *= maximum_total / total_finish_hazard
        total_finish_hazard = maximum_total

    draw = float(rng.random())

    if draw >= total_finish_hazard:
        return SegmentFinishResult(
            finished=False,
            red_hazards=red_hazards,
            blue_hazards=blue_hazards,
        )

    cumulative = np.cumsum(event_weights)
    event_index = int(np.searchsorted(cumulative, draw, side="right"))

    event_map = (
        ("red", "blue", FinishMethod.KO_TKO),
        ("red", "blue", FinishMethod.SUBMISSION),
        ("blue", "red", FinishMethod.KO_TKO),
        ("blue", "red", FinishMethod.SUBMISSION),
    )

    winner, loser, method = event_map[min(event_index, 3)]

    return SegmentFinishResult(
        finished=True,
        winner=winner,
        loser=loser,
        method=method,
        red_hazards=red_hazards,
        blue_hazards=blue_hazards,
    )
