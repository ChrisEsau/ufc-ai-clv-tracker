"""Typed contracts for the round-level UFC fight simulator.

The simulator intentionally separates parameter estimation from fight simulation.
Upstream models may later estimate these parameters from leakage-safe RFS and
fighter-state features. The simulation kernel only consumes validated values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


class SimulationContractError(ValueError):
    """Raised when simulator inputs violate the public parameter contract."""


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise SimulationContractError(f"{name} must be between 0 and 1; received {value!r}")


def _validate_nonnegative(name: str, value: float) -> None:
    if value < 0.0:
        raise SimulationContractError(f"{name} must be nonnegative; received {value!r}")


@dataclass(frozen=True)
class FighterSimulationState:
    """Pre-fight state for one fighter.

    Rate fields use interpretable units. State fields are normalized to [0, 1].
    No field is inferred inside the simulation engine; upstream adapters are
    responsible for translating trained model outputs into this contract.
    """

    fighter_id: str
    fighter_name: str
    sig_attempts_per_minute: float
    sig_accuracy: float
    sig_defense: float
    power: float
    durability: float
    td_attempts_per_15: float
    td_accuracy: float
    td_defense: float
    control_seconds_per_takedown: float
    submission_threat: float
    submission_defense: float
    cardio: float
    recovery: float
    pace_sustainability: float
    adaptability: float
    initiative: float = 0.5
    phase_imposition: float = 0.5
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.fighter_id).strip():
            raise SimulationContractError("fighter_id must be non-empty")
        if not str(self.fighter_name).strip():
            raise SimulationContractError("fighter_name must be non-empty")

        for name in (
            "sig_attempts_per_minute",
            "td_attempts_per_15",
            "control_seconds_per_takedown",
        ):
            _validate_nonnegative(name, float(getattr(self, name)))

        for name in (
            "sig_accuracy",
            "sig_defense",
            "power",
            "durability",
            "td_accuracy",
            "td_defense",
            "submission_threat",
            "submission_defense",
            "cardio",
            "recovery",
            "pace_sustainability",
            "adaptability",
            "initiative",
            "phase_imposition",
        ):
            _validate_probability(name, float(getattr(self, name)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FighterSimulationState":
        """Build a state from decoded JSON-compatible data."""
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MatchupSimulationInput:
    """Complete input contract for one simulated matchup."""

    fight_id: str
    event_id: str | None
    red: FighterSimulationState
    blue: FighterSimulationState
    scheduled_rounds: int = 3
    round_seconds: int = 300
    source_snapshot_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.fight_id).strip():
            raise SimulationContractError("fight_id must be non-empty")
        if self.red.fighter_id == self.blue.fighter_id:
            raise SimulationContractError("red and blue fighter_id values must differ")
        if self.scheduled_rounds not in (3, 5):
            raise SimulationContractError("scheduled_rounds must be 3 or 5")
        if self.round_seconds <= 0:
            raise SimulationContractError("round_seconds must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MatchupSimulationInput":
        payload = dict(value)
        payload["red"] = FighterSimulationState.from_mapping(payload["red"])
        payload["blue"] = FighterSimulationState.from_mapping(payload["blue"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimulatorConfig:
    """Runtime controls for Monte Carlo simulation."""

    simulations: int = 10_000
    seed: int = 7
    retain_outcomes: bool = False
    max_finish_probability_per_round: float = 0.72
    strike_overdispersion: float = 0.35
    takedown_overdispersion: float = 0.45

    def __post_init__(self) -> None:
        if self.simulations <= 0:
            raise SimulationContractError("simulations must be positive")
        if self.max_finish_probability_per_round <= 0 or self.max_finish_probability_per_round >= 1:
            raise SimulationContractError(
                "max_finish_probability_per_round must be strictly between 0 and 1"
            )
        _validate_nonnegative("strike_overdispersion", self.strike_overdispersion)
        _validate_nonnegative("takedown_overdispersion", self.takedown_overdispersion)


@dataclass(frozen=True)
class FighterFightTotals:
    sig_attempted: int = 0
    sig_landed: int = 0
    takedowns_attempted: int = 0
    takedowns_landed: int = 0
    control_seconds: float = 0.0
    knockdowns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FightSimulationOutcome:
    """One complete simulated fight path."""

    winner_corner: str
    method: str
    finish_round: int
    finish_time_seconds: float
    total_fight_seconds: float
    red_rounds_won: int
    blue_rounds_won: int
    red_totals: FighterFightTotals
    blue_totals: FighterFightTotals
    regime: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(frozen=True)
class SimulationSummary:
    """Aggregated Monte Carlo output suitable for a shadow prediction artifact."""

    fight_id: str
    event_id: str | None
    red_fighter_id: str
    red_fighter_name: str
    blue_fighter_id: str
    blue_fighter_name: str
    scheduled_rounds: int
    simulations: int
    seed: int
    probabilities: Mapping[str, float]
    expectations: Mapping[str, float]
    joint_probabilities: Mapping[str, float]
    regime_probabilities: Mapping[str, float]
    source_snapshot_id: str | None = None
    simulator_version: str = "round_simulator_v0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
