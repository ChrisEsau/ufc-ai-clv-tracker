"""Explicit matchup inputs and centralized structural MVP placeholders."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import math

from pipeline.simulation.event_clock_mc_v2.causal.state import Side


@dataclass(frozen=True)
class FighterMechanics:
    """Minimum matchup-resolved probabilities consumed by Stage 3 mechanics.

    Standing accuracy, takedown completion, and ground accuracy map directly to
    existing FSR V3 ``MatchupRuntimeInputs`` quantities. Submission, escape,
    and reversal probabilities remain explicit caller inputs until their frozen
    mechanics can be composed without importing a legacy state or engine.
    """

    standing_strike_landing_probability: float
    takedown_completion_probability: float
    ground_strike_landing_probability: float
    submission_success_probability: float
    ground_escape_probability: float
    ground_reversal_probability: float
    striking_power: float = 50.0
    damage_durability: float = 50.0
    knockdown_resistance: float = 50.0
    stamina_capacity: float = 100.0
    stamina_depletion_resistance: float = 50.0
    age_years: float = 30.0

    def __post_init__(self) -> None:
        for field in fields(self)[:6]:
            value = getattr(self, field.name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{field.name} must be a numeric probability")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field.name} must be between 0 and 1")
        # Power and KD resistance are synthetic, unbounded legacy-equivalent
        # coordinates.  The Stage-10 equations exponentiate centered versions
        # of them, so finite negative coordinates are mathematically valid.
        for field in fields(self)[6:9]:
            value = getattr(self, field.name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{field.name} must be finite")
            if field.name == "damage_durability" and value <= 0:
                raise ValueError("damage_durability must be finite and positive")
        for field in fields(self)[9:]:
            value = getattr(self, field.name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field.name} must be finite and positive")


class KOKDArchitecture(str, Enum):
    """Explicit strike-consequence architecture; coefficients are not overrides."""

    LEGACY_STAGE10 = "legacy_stage10"
    EMPIRICAL_EVENT2 = "empirical_event2"


@dataclass(frozen=True)
class MechanicsInputs:
    """Directional mechanics inputs for both fighters in one matchup."""

    red: FighterMechanics
    blue: FighterMechanics
    calibration: MechanicsCalibrationConfig = None  # type: ignore[name-defined]
    ko_kd_architecture: KOKDArchitecture | None = None

    def fighter(self, side: Side) -> FighterMechanics:
        if not isinstance(side, Side):
            raise ValueError("side must be a Side value")
        return self.red if side is Side.RED else self.blue


@dataclass(frozen=True)
class MechanicsCalibrationConfig:
    """Small canonical mechanics surface available to calibration research."""

    impact_scale: float = 0.50
    trauma_durability_divisor: float = 40.0
    kd_slope: float = 2.0
    kd_midpoint: float = 36.0
    finish_slope: float = 2.0
    finish_midpoint: float = 36.0
    post_kd_finish_logit_bonus: float = 1.0
    action_cost_scale: float = 1.0
    top_position_cost_per_second: float = 0.00025
    bottom_position_cost_per_second: float = 0.00035
    round_recovery_fraction: float = 0.40

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field.name} must be numeric")
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field.name} must be finite and positive")
        if self.round_recovery_fraction > 1.0:
            raise ValueError("round_recovery_fraction must not exceed 1")


DEFAULT_MECHANICS_CALIBRATION_CONFIG = MechanicsCalibrationConfig()


@dataclass(frozen=True)
class StructuralMVPPlaceholders:
    """Uncalibrated neutral clinch assumptions for structural validation only.

    These values must be revisited after the causal timeline is validated and
    empirical clinch capability is approved; they are not market-tuned inputs.
    """

    clinch_entry_success_probability: float = 0.50
    clinch_strike_landing_probability: float = 0.50
    break_clinch_success_probability: float = 0.50

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{field.name} must be a numeric probability")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field.name} must be between 0 and 1")
