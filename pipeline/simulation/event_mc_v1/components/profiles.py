"""Minimal immutable FSR input adapter for DISTANCE parity mechanics."""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class Side(str, Enum):
    RED = "red"
    BLUE = "blue"

    @property
    def opponent(self) -> "Side":
        return Side.BLUE if self is Side.RED else Side.RED


@dataclass(frozen=True)
class FighterProfile:
    fighter_id: str
    fighter_name: str
    distance_striking_pressure: float
    distance_striking_precision: float
    distance_striking_defense: float
    clinch_striking_pressure: float
    wrestling_entry: float
    wrestling_conversion: float
    td_defense: float
    control_imposition: float
    clinch_striking_precision: float = 50.0
    clinch_striking_defense: float = 50.0
    ground_striking_pressure: float = 50.0
    ground_striking_precision: float = 50.0
    ground_striking_defense: float = 50.0
    control_resistance: float = 50.0
    submission_pressure: float = 50.0
    submission_conversion: float = 50.0
    submission_resistance: float = 50.0
    reversal_ability: float = 50.0
    stamina_capacity: float = 100.0
    stamina_depletion_resistance: float = 50.0
    stamina_performance_resilience: float = 50.0
    striking_power: float = 50.0
    damage_durability: float = 50.0
    knockdown_resistance: float = 50.0
    # Fight-context metadata, not a persisted fighter trait.
    age_years: float = 30.0

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "FighterProfile":
        def number(name: str, default: float = 50.0) -> float:
            value = row.get(name, default)
            return default if value is None else float(value)

        fighter_id = str(row["fighter_id"])
        return cls(
            fighter_id=fighter_id,
            fighter_name=str(row.get("fighter_name", fighter_id)),
            distance_striking_pressure=number("distance_striking_pressure"),
            distance_striking_precision=number("distance_striking_precision"),
            distance_striking_defense=number("distance_striking_defense"),
            clinch_striking_pressure=number("clinch_striking_pressure"),
            wrestling_entry=number("wrestling_entry"),
            wrestling_conversion=number("wrestling_conversion"),
            td_defense=number("td_defense"),
            control_imposition=number("control_imposition"),
            clinch_striking_precision=number("clinch_striking_precision"),
            clinch_striking_defense=number("clinch_striking_defense"),
            ground_striking_pressure=number("ground_striking_pressure"),
            ground_striking_precision=number("ground_striking_precision"),
            ground_striking_defense=number("ground_striking_defense"),
            control_resistance=number("control_resistance"),
            submission_pressure=number("submission_pressure"),
            submission_conversion=number("submission_conversion"),
            submission_resistance=number("submission_resistance"),
            reversal_ability=number("reversal_ability"),
            stamina_capacity=number("stamina_capacity"),
            stamina_depletion_resistance=number("stamina_depletion_resistance"),
            stamina_performance_resilience=number("stamina_performance_resilience"),
            striking_power=number("striking_power"),
            damage_durability=number("damage_durability"),
            knockdown_resistance=number("knockdown_resistance"),
            age_years=number("age_years", 30.0),
        )


@dataclass(frozen=True)
class MatchupProfiles:
    red: FighterProfile
    blue: FighterProfile

    def fighter(self, side: Side) -> FighterProfile:
        return self.red if side is Side.RED else self.blue
