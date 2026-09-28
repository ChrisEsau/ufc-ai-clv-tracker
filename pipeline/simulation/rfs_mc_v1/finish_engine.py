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
    FightPhase,
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

    # Attacker historical finishing ability.
    submission_attacker_skill_weight: float = 2.00

    # Quality of the current grappling position.
    submission_position_quality_weight: float = 1.75

    # Defender historical and current vulnerability.
    submission_vulnerability_weight: float = 1.25

    # A fresh attacker converts attempts more effectively.
    submission_attacker_energy_weight: float = 0.75

    # Additional attempts in one segment increase danger, but are capped.
    submission_extra_attempt_weight: float = 0.40

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


def _profile_value(
    profile: FighterSimulationProfile,
    name: str,
    *,
    default: float = 0.0,
) -> float:
    """Read one finite simulation-profile value."""

    estimate = profile.parameters.get(name)

    if estimate is None:
        return default

    value = float(estimate.value)

    if not np.isfinite(value):
        return default

    return value


def _submission_is_eligible(
    activity: SegmentActivity,
) -> bool:
    """Return whether a submission finish may be evaluated.

    A submission requires an actual generated submission attempt and
    either a ground phase or positive control time.
    """

    return bool(
        activity.submission_attempts > 0
        and (
            activity.phase is FightPhase.GROUND
            or activity.control_seconds > 0
        )
    )


def _attacker_submission_skill(
    profile: FighterSimulationProfile,
) -> float:
    """Combine historical submission finishing signals."""

    conversion_rate = float(
        np.clip(
            _profile_value(
                profile,
                "prior_submission_conversion_rate",
                default=0.10,
            ),
            0.0,
            1.0,
        )
    )

    win_rate = float(
        np.clip(
            _profile_value(
                profile,
                "prior_submission_win_rate",
            )
            / 0.30,
            0.0,
            1.0,
        )
    )

    prior_wins = max(
        _profile_value(
            profile,
            "prior_submission_wins",
        ),
        0.0,
    )

    win_count_signal = float(
        1.0 - np.exp(-prior_wins / 2.0)
    )

    submission_pressure = float(
        np.clip(
            _profile_value(
                profile,
                "submission_pressure",
            )
            / 0.80,
            0.0,
            1.0,
        )
    )

    skill = (
        0.45 * conversion_rate
        + 0.20 * win_rate
        + 0.15 * win_count_signal
        + 0.20 * submission_pressure
    )

    return float(np.clip(skill, 0.0, 1.0))


def _submission_position_quality(
    *,
    activity: SegmentActivity,
    attacker_profile: FighterSimulationProfile,
) -> float:
    """Estimate the quality of the current submission position."""

    control_fraction = float(
        np.clip(
            activity.control_seconds / 30.0,
            0.0,
            1.0,
        )
    )

    control_stability = float(
        np.clip(
            _profile_value(
                attacker_profile,
                "control_stability",
            ),
            0.0,
            1.0,
        )
    )

    control_per_td_attempt = float(
        np.clip(
            _profile_value(
                attacker_profile,
                "control_per_td_attempt",
            )
            / 30.0,
            0.0,
            1.0,
        )
    )

    td_to_control_conversion = float(
        np.clip(
            _profile_value(
                attacker_profile,
                "td_to_control_conversion",
            ),
            0.0,
            1.0,
        )
    )

    position_quality = (
        0.35 * control_fraction
        + 0.30 * control_stability
        + 0.20 * control_per_td_attempt
        + 0.15 * td_to_control_conversion
    )

    return float(
        np.clip(position_quality, 0.0, 1.0)
    )


def _defender_submission_vulnerability(
    *,
    defender_state: DynamicFighterState,
    defender_profile: FighterSimulationProfile,
) -> float:
    """Combine prior submission losses and current deterioration."""

    historical_loss_rate = float(
        np.clip(
            _profile_value(
                defender_profile,
                "prior_submission_loss_rate",
            )
            / 0.25,
            0.0,
            1.0,
        )
    )

    current_danger = float(
        np.clip(
            defender_state.submission_danger,
            0.0,
            1.0,
        )
    )

    energy_loss = float(
        np.clip(
            1.0 - defender_state.energy,
            0.0,
            1.0,
        )
    )

    defense_loss = float(
        np.clip(
            1.0 - defender_state.defensive_stability,
            0.0,
            1.0,
        )
    )

    vulnerability = (
        0.35 * historical_loss_rate
        + 0.35 * current_danger
        + 0.15 * energy_loss
        + 0.15 * defense_loss
    )

    return float(
        np.clip(vulnerability, 0.0, 1.0)
    )


def calculate_finish_hazards(
    *,
    defender_state: DynamicFighterState,
    attacker_activity: SegmentActivity,
    defender_profile: FighterSimulationProfile,
    attacker_state: DynamicFighterState | None = None,
    attacker_profile: FighterSimulationProfile | None = None,
    parameters: FinishHazardParameters = (
        DEFAULT_FINISH_HAZARD_PARAMETERS
    ),
) -> FighterFinishHazards:
    """Calculate KO/TKO and submission hazards against one defender."""

    defender_state.validate()

    # Optional defaults preserve compatibility with older direct tests.
    if attacker_state is not None:
        attacker_state.validate()

    if attacker_profile is None:
        attacker_profile = defender_profile

    head_damage = max(
        defender_state.head_damage,
        0.0,
    )
    chin_loss = float(
        np.clip(
            1.0 - defender_state.chin_integrity,
            0.0,
            1.0,
        )
    )
    defense_loss = float(
        np.clip(
            1.0 - defender_state.defensive_stability,
            0.0,
            1.0,
        )
    )
    energy_loss = float(
        np.clip(
            1.0 - defender_state.energy,
            0.0,
            1.0,
        )
    )

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

    ko_hazard = _bounded_hazard(
        ko_linear,
        parameters,
    )

    # No attempt means no submission finish.
    if not _submission_is_eligible(
        attacker_activity
    ):
        return FighterFinishHazards(
            ko_tko=ko_hazard,
            submission=0.0,
        )

    attacker_skill = _attacker_submission_skill(
        attacker_profile
    )

    position_quality = _submission_position_quality(
        activity=attacker_activity,
        attacker_profile=attacker_profile,
    )

    defender_vulnerability = (
        _defender_submission_vulnerability(
            defender_state=defender_state,
            defender_profile=defender_profile,
        )
    )

    attacker_energy = (
        1.0
        if attacker_state is None
        else float(
            np.clip(
                attacker_state.energy,
                0.0,
                1.0,
            )
        )
    )

    extra_attempts = float(
        np.clip(
            attacker_activity.submission_attempts - 1,
            0,
            2,
        )
    )

    submission_linear = (
        parameters.submission_intercept
        + parameters.submission_danger_weight
        * defender_state.submission_danger
        + parameters.submission_attempt_weight
        * attacker_activity.submission_attempts
        + parameters.submission_control_weight
        * attacker_activity.control_seconds
        + parameters.submission_energy_loss_weight
        * energy_loss
        + parameters.submission_defense_loss_weight
        * defense_loss
        + parameters.submission_attacker_skill_weight
        * attacker_skill
        + parameters.submission_position_quality_weight
        * position_quality
        + parameters.submission_vulnerability_weight
        * defender_vulnerability
        + parameters.submission_attacker_energy_weight
        * attacker_energy
        + parameters.submission_extra_attempt_weight
        * extra_attempts
    )

    submission_hazard = _bounded_hazard(
        submission_linear,
        parameters,
    )

    return FighterFinishHazards(
        ko_tko=ko_hazard,
        submission=submission_hazard,
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
        attacker_state=red_state,
        attacker_activity=red_activity,
        defender_profile=blue_profile,
        attacker_profile=red_profile,
        parameters=parameters,
    )
    blue_hazards = calculate_finish_hazards(
        defender_state=red_state,
        attacker_state=blue_state,
        attacker_activity=blue_activity,
        defender_profile=red_profile,
        attacker_profile=blue_profile,
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
