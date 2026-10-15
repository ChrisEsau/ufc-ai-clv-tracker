import math

import numpy as np
import pytest

from pipeline.simulation.event_mc_v1.scheduler import (
    EventRate,
    ExponentialScheduler,
    probability_to_rate,
)


def test_exponential_mean_and_competing_proportions() -> None:
    scheduler = ExponentialScheduler()
    rng = np.random.default_rng(20260813)
    candidates = [EventRate("a", 1.0), EventRate("b", 2.0), EventRate("c", 7.0)]
    samples = [scheduler.sample(candidates, rng) for _ in range(40_000)]
    assert np.mean([dt for dt, _ in samples]) == pytest.approx(0.1, rel=0.02)
    selections = [selected for _, selected in samples]
    assert selections.count("a") / len(samples) == pytest.approx(0.1, abs=0.01)
    assert selections.count("b") / len(samples) == pytest.approx(0.2, abs=0.01)
    assert selections.count("c") / len(samples) == pytest.approx(0.7, abs=0.01)


def test_single_positive_and_zero_total_rates() -> None:
    scheduler = ExponentialScheduler()
    rng = np.random.default_rng(7)
    samples = [scheduler.sample([EventRate("off", 0), EventRate("on", 2)], rng) for _ in range(10_000)]
    assert {selected for _, selected in samples} == {"on"}
    assert np.mean([dt for dt, _ in samples]) == pytest.approx(0.5, rel=0.03)
    assert scheduler.sample([EventRate("off", 0)], rng) == (math.inf, None)


@pytest.mark.parametrize("rate", [-1.0, math.nan, math.inf])
def test_invalid_rates_fail(rate: float) -> None:
    with pytest.raises(ValueError):
        ExponentialScheduler().sample([EventRate("bad", rate)], np.random.default_rng())


def test_probability_to_rate_exact_conversion_and_scaling() -> None:
    assert probability_to_rate(0, 10) == 0
    assert probability_to_rate(0.5, 2) == pytest.approx(math.log(2) / 2)
    assert probability_to_rate(0.5, 4) == pytest.approx(math.log(2) / 4)


@pytest.mark.parametrize("probability", [-0.1, 1, 1.1, math.nan])
def test_probability_to_rate_rejects_invalid_probability(probability: float) -> None:
    with pytest.raises(ValueError):
        probability_to_rate(probability, 1)


@pytest.mark.parametrize("interval", [0, -1, math.inf, math.nan])
def test_probability_to_rate_rejects_invalid_interval(interval: float) -> None:
    with pytest.raises(ValueError):
        probability_to_rate(0.5, interval)
