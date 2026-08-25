"""Explicit matchup inputs and centralized structural MVP placeholders."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

from pipeline.simulation.event_clock_mc_v2.causal.state import Side


@dataclass(frozen=True)
class FighterMechanics:
    """Minimum matchup-resolved probabilities consumed by Stage 3 mechanics.

    Standing accuracy, takedown completion, and ground accuracy map directly to
    existing FSR V3 ``MatchupRuntimeInputs`` quantities. Submission, escape,
    and reversal probabilities remain explicit caller inputs until their frozen
    mechanics can be composed without importing a legacy state or engine.

    ``striking_power`` and ``knockdown_resistance`` retain the Stage-10 legacy
    rating coordinate for neutral/backward-compatible callers. Canonical FSR V3
    may instead provide its native log effects explicitly. The physiology
    resolver consumes the effective log effects, so native V3 latents do not
    need to be round-tripped through an artificial unbounded rating.
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
    striking_power_log_effect: float | None = None
    knockdown_resistance_log_effect: float | None = None

    def __post_init__(self) -> None:
        for field in fields(self)[:6]:
            value = getattr(self, field.name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{field.name} must be a numeric probability")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field.name} must be between 0 and 1")
        for field in fields(self)[6:11]:
            value = getattr(self, field.name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field.name} must be finite and positive")
        for field in fields(self)[11:]:
            value = getattr(self, field.name)
            if value is None:
                continue
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{field.name} must be finite when provided")

    @property
    def effective_power_log_effect(self) -> float:
        """Log multiplier used by the migrated Stage-10 impact formula."""
        if self.striking_power_log_effect is not None:
            return float(self.striking_power_log_effect)
        return (float(self.striking_power) - 50.0) / 55.0

    @property
    def effective_kd_resistance_log_effect(self) -> float:
        """Log resistance used by the migrated Stage-10 KD/finish formulas."""
        if self.knockdown_resistance_log_effect is not None:
            return float(self.knockdown_resistance_log_effect)
        return (float(self.knockdown_resistance) - 50.0) / 32.0


@dataclass(frozen=True)
class MechanicsInputs:
    """Directional mechanics inputs for both fighters in one matchup."""

    red: FighterMechanics
    blue: FighterMechanics

    def fighter(self, side: Side) -> FighterMechanics:
        if not isinstance(side, Side):
            raise ValueError("side must be a Side value")
        return self.red if side is Side.RED else self.blue


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
