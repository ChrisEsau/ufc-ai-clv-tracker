"""Immutable canonical FSR V2 boundary for EVENT MC."""

from dataclasses import dataclass, fields
from math import isfinite
from typing import Mapping

from pipeline.fsr_v2.physical import PHYSICAL_COLUMNS, STAMINA_CAPACITY
from .profiles import FighterProfile, MatchupProfiles, Side

FSR_V2_TRAIT_FIELDS = (
    "standing_striking_tendency", "standing_striking_suppression",
    "standing_striking_offense", "standing_striking_defense",
    "head_strike_tendency", "body_strike_tendency", "leg_strike_tendency",
    "takedown_tendency", "takedown_suppression", "takedown_offense", "takedown_defense",
    "escape_offense", "escape_defense",
    "ground_striking_tendency", "ground_striking_suppression",
    "ground_striking_offense", "ground_striking_defense",
    "submission_tendency", "submission_suppression", "submission_offense", "submission_defense",
)
FSR_V2_SIMULATOR_FIELDS = (*FSR_V2_TRAIT_FIELDS, *PHYSICAL_COLUMNS)
FSR_V2_POPULATION_FIELDS = (
    "standing_accuracy_baseline", "takedown_completion_baseline",
    "ground_accuracy_baseline", "submission_conversion_baseline",
    "escape_population_mean_seconds",
)


@dataclass(frozen=True)
class FSRV2FighterInput:
    fighter_id: str
    fighter_name: str
    standing_striking_tendency: float
    standing_striking_suppression: float
    standing_striking_offense: float
    standing_striking_defense: float
    head_strike_tendency: float
    body_strike_tendency: float
    leg_strike_tendency: float
    takedown_tendency: float
    takedown_suppression: float
    takedown_offense: float
    takedown_defense: float
    escape_offense: float
    escape_defense: float
    ground_striking_tendency: float
    ground_striking_suppression: float
    ground_striking_offense: float
    ground_striking_defense: float
    submission_tendency: float
    submission_suppression: float
    submission_offense: float
    submission_defense: float
    stamina_capacity: float
    stamina_depletion_resistance: float
    stamina_performance_resilience: float
    striking_power: float
    damage_durability: float
    knockdown_resistance: float
    standing_accuracy_baseline: float
    takedown_completion_baseline: float
    ground_accuracy_baseline: float
    submission_conversion_baseline: float
    escape_population_mean_seconds: float
    # Fight-context metadata, not an FSR V2 trait.
    # Age 30 is neutral for the TD-completion age adjustment.
    age_years: float = 30.0

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "FSRV2FighterInput":
        required = (*FSR_V2_SIMULATOR_FIELDS, *FSR_V2_POPULATION_FIELDS)
        missing = [name for name in required if name not in row or row[name] is None]
        if missing:
            raise ValueError(f"canonical FSR V2 row missing required fields: {missing}")
        fighter_id = str(row.get("fighter_id", "")).strip()
        if not fighter_id:
            raise ValueError("canonical FSR V2 row requires fighter_id")
        values = {name: float(row[name]) for name in required}
        bad = [name for name, value in values.items() if not isfinite(value)]
        if bad:
            raise ValueError(f"canonical FSR V2 row has non-finite fields: {bad}")

        age_years = float(row.get("age_years", 30.0))
        if not isfinite(age_years) or age_years <= 0.0:
            raise ValueError("age_years must be finite and positive")

        if values["stamina_capacity"] != STAMINA_CAPACITY:
            raise ValueError(f"stamina_capacity must be explicitly {STAMINA_CAPACITY}")
        for name in ("head_strike_tendency", "body_strike_tendency", "leg_strike_tendency"):
            if not 0.0 <= values[name] <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if abs(values["head_strike_tendency"] + values["body_strike_tendency"] - 1.0) > 1e-9:
            raise ValueError("conditional head/body target probabilities must sum to one")
        for name in FSR_V2_POPULATION_FIELDS[:-1]:
            if not 0.0 < values[name] < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        if values["escape_population_mean_seconds"] <= 0.0:
            raise ValueError("escape_population_mean_seconds must be positive")
        return cls(
            fighter_id,
            str(row.get("fighter_name", fighter_id)),
            **values,
            age_years=age_years,
        )

    def standing_target_probabilities(self) -> tuple[float, float, float]:
        non_leg = 1.0 - self.leg_strike_tendency
        return (non_leg * self.head_strike_tendency,
                non_leg * self.body_strike_tendency,
                self.leg_strike_tendency)

    def physical_profile(self) -> FighterProfile:
        # Legacy mechanics-only fields are inert in the FSR V2 clock path. The
        # six physical fields pass unchanged into the frozen consequence models.
        neutral = dict(distance_striking_pressure=50.0, distance_striking_precision=50.0,
                       distance_striking_defense=50.0, clinch_striking_pressure=50.0,
                       wrestling_entry=50.0, wrestling_conversion=50.0,
                       td_defense=50.0, control_imposition=50.0)
        return FighterProfile(self.fighter_id, self.fighter_name, **neutral,
            stamina_capacity=self.stamina_capacity,
            stamina_depletion_resistance=self.stamina_depletion_resistance,
            stamina_performance_resilience=self.stamina_performance_resilience,
            striking_power=self.striking_power, damage_durability=self.damage_durability,
            knockdown_resistance=self.knockdown_resistance,
            age_years=self.age_years)

    def audit_traits(self) -> dict[str, float]:
        return {field.name: getattr(self, field.name) for field in fields(self)
                if field.name in FSR_V2_SIMULATOR_FIELDS}


@dataclass(frozen=True)
class FSRV2Matchup:
    red: FSRV2FighterInput
    blue: FSRV2FighterInput

    def fighter(self, side: Side) -> FSRV2FighterInput:
        return self.red if side is Side.RED else self.blue

    def physical_profiles(self) -> MatchupProfiles:
        return MatchupProfiles(self.red.physical_profile(), self.blue.physical_profile())
