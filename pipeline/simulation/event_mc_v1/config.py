"""Immutable timing configuration for the generic event kernel."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FightConfig:
    scheduled_rounds: int = 3
    round_duration_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.scheduled_rounds <= 0:
            raise ValueError("scheduled_rounds must be positive")
        if self.round_duration_seconds <= 0:
            raise ValueError("round_duration_seconds must be positive")

    @property
    def fight_duration_seconds(self) -> float:
        return self.scheduled_rounds * self.round_duration_seconds

    @property
    def hard_boundaries_seconds(self) -> tuple[float, ...]:
        return tuple(
            round_number * self.round_duration_seconds
            for round_number in range(1, self.scheduled_rounds + 1)
        )

    def round_number_at(self, fight_time_seconds: float) -> int:
        if fight_time_seconds >= self.fight_duration_seconds:
            return self.scheduled_rounds
        return int(fight_time_seconds // self.round_duration_seconds) + 1

    def round_elapsed_seconds_at(self, fight_time_seconds: float) -> float:
        round_number = self.round_number_at(fight_time_seconds)
        round_start = (round_number - 1) * self.round_duration_seconds
        return min(self.round_duration_seconds, fight_time_seconds - round_start)

    def round_remaining_seconds_at(self, fight_time_seconds: float) -> float:
        return self.round_duration_seconds - self.round_elapsed_seconds_at(
            fight_time_seconds
        )

    def fight_remaining_seconds_at(self, fight_time_seconds: float) -> float:
        return max(0.0, self.fight_duration_seconds - fight_time_seconds)

    def next_boundary_after(self, fight_time_seconds: float) -> float:
        round_number = self.round_number_at(fight_time_seconds)
        return min(
            round_number * self.round_duration_seconds,
            self.fight_duration_seconds,
        )
