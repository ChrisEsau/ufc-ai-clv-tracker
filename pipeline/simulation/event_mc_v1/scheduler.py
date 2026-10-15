"""UFC-agnostic competing-risks scheduling in events per second."""

import math
from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class EventRate(Generic[T]):
    candidate: T
    rate_per_second: float


def probability_to_rate(p_interval: float, interval_seconds: float) -> float:
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("interval_seconds must be finite and positive")
    if not math.isfinite(p_interval) or not 0 <= p_interval < 1:
        raise ValueError("p_interval must be finite and in [0, 1)")
    return -math.log1p(-p_interval) / interval_seconds


class ExponentialScheduler:
    def sample(
        self,
        candidates: Sequence[EventRate[T]],
        rng: np.random.Generator,
    ) -> tuple[float, T | None]:
        positive: list[EventRate[T]] = []
        for candidate in candidates:
            rate = candidate.rate_per_second
            if not math.isfinite(rate) or rate < 0:
                raise ValueError("rates must be finite and non-negative")
            if rate > 0:
                positive.append(candidate)
        total_rate = math.fsum(item.rate_per_second for item in positive)
        if total_rate == 0:
            return math.inf, None
        dt = float(rng.exponential(1.0 / total_rate))
        threshold = float(rng.random()) * total_rate
        cumulative = 0.0
        for item in positive:
            cumulative += item.rate_per_second
            if threshold < cumulative:
                return dt, item.candidate
        return dt, positive[-1].candidate
