"""Core contracts for the shadow-only RFS Monte Carlo V1 engine.

This module contains data structures only. It does not load files, generate
events, calculate finish hazards, score rounds, or write production artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class ProfileSource(str, Enum):
    """Provenance for an estimated simulation-profile parameter."""

    FIGHTER = "fighter"
    WEIGHT_CLASS_GENDER_ROUNDS = "weight_class_gender_rounds"
    WEIGHT_CLASS_GENDER = "weight_class_gender"
    GENDER = "gender"
    GLOBAL = "global"


class FightPhase(str, Enum):
    """Supported simulated fight phases."""

    DISTANCE = "distance"
    CLINCH = "clinch"
    GROUND = "ground"


@dataclass(frozen=True)
class ParameterEstimate:
    """One fixed pre-fight parameter with provenance and uncertainty."""

    value: float
    source: ProfileSource
    effective_sample_size: float
    uncertainty: float

    def __post_init__(self) -> None:
        if self.effective_sample_size < 0:
            raise ValueError("effective_sample_size cannot be negative")

        if self.uncertainty < 0:
            raise ValueError("uncertainty cannot be negative")


@dataclass(frozen=True)
class FighterSimulationProfile:
    """Leakage-safe fighter profile fixed before a target fight.

    All values must be derived exclusively from information available before
    ``target_date``. Dynamic simulation state must never overwrite this object.
    """

    fighter_id: str
    fighter_name: str
    target_date: str

    weight_class: str | None
    gender: str | None
    scheduled_rounds: int

    prior_fight_count: int
    valid_round_fight_count: int

    parameters: Mapping[str, ParameterEstimate]

    profile_version: str = "rfs_mc_v1_profile_v0"
    is_low_experience: bool = False

    def __post_init__(self) -> None:
        if not self.fighter_id:
            raise ValueError("fighter_id is required")

        if not self.fighter_name:
            raise ValueError("fighter_name is required")

        if self.scheduled_rounds not in {3, 5}:
            raise ValueError("scheduled_rounds must be 3 or 5")

        if self.prior_fight_count < 0:
            raise ValueError("prior_fight_count cannot be negative")

        if self.valid_round_fight_count < 0:
            raise ValueError("valid_round_fight_count cannot be negative")

        expected_low_experience = self.prior_fight_count < 3
        if self.is_low_experience != expected_low_experience:
            raise ValueError(
                "is_low_experience must equal prior_fight_count < 3"
            )

        if not self.parameters:
            raise ValueError("parameters cannot be empty")


@dataclass
class DynamicFighterState:
    """Mutable state belonging to one fighter in one simulation path."""

    energy: float = 1.0

    head_damage: float = 0.0
    body_damage: float = 0.0
    leg_damage: float = 0.0

    chin_integrity: float = 1.0
    defensive_stability: float = 1.0
    recovery_reserve: float = 1.0

    confidence: float = 0.5
    tactical_urgency: float = 0.0

    current_phase: FightPhase = FightPhase.DISTANCE
    control_position: float = 0.0
    submission_danger: float = 0.0

    cumulative_strike_activity: int = 0
    cumulative_wrestling_activity: int = 0
    knockdowns: int = 0
    rounds_won: int = 0

    score_state: dict[int, float] = field(default_factory=dict)

    def validate(self) -> None:
        bounded = {
            "energy": self.energy,
            "chin_integrity": self.chin_integrity,
            "defensive_stability": self.defensive_stability,
            "recovery_reserve": self.recovery_reserve,
            "confidence": self.confidence,
            "tactical_urgency": self.tactical_urgency,
            "control_position": self.control_position,
            "submission_danger": self.submission_danger,
        }

        for name, value in bounded.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        nonnegative = {
            "head_damage": self.head_damage,
            "body_damage": self.body_damage,
            "leg_damage": self.leg_damage,
            "cumulative_strike_activity": self.cumulative_strike_activity,
            "cumulative_wrestling_activity": self.cumulative_wrestling_activity,
            "knockdowns": self.knockdowns,
            "rounds_won": self.rounds_won,
        }

        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True)
class MatchupSimulationRequest:
    """Immutable inputs required to run one matchup simulation."""

    red_profile: FighterSimulationProfile
    blue_profile: FighterSimulationProfile

    path_count: int
    seed: int

    simulator_version: str = "rfs_mc_v1"
    calibration_version: str = "unselected"

    def __post_init__(self) -> None:
        if self.red_profile.fighter_id == self.blue_profile.fighter_id:
            raise ValueError("red and blue fighters must be different")

        if self.path_count <= 0:
            raise ValueError("path_count must be positive")

        if (
            self.red_profile.scheduled_rounds
            != self.blue_profile.scheduled_rounds
        ):
            raise ValueError("fighter profiles disagree on scheduled rounds")
