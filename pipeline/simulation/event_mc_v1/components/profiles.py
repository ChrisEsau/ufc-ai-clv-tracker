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

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "FighterProfile":
        def number(name: str) -> float:
            value = row.get(name, 50.0)
            return 50.0 if value is None else float(value)

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
        )


@dataclass(frozen=True)
class MatchupProfiles:
    red: FighterProfile
    blue: FighterProfile

    def fighter(self, side: Side) -> FighterProfile:
        return self.red if side is Side.RED else self.blue
