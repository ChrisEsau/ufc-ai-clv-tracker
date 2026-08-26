from dataclasses import dataclass, field

import numpy as np
import pytest

from pipeline.simulation.event_mc_v1.config import FightConfig
from pipeline.simulation.event_mc_v1.contracts import FightContext, Resolution
from pipeline.simulation.event_mc_v1.engine import SimulationEngine
from pipeline.simulation.event_mc_v1.events import PrimaryEvent, RoundEnded, RoundStarted
from pipeline.simulation.event_mc_v1.rng import RNGManager, RNGStream
from pipeline.simulation.event_mc_v1.scheduler import EventRate
from pipeline.simulation.event_mc_v1.sinks import FullTraceEventSink, NullEventSink, StatsEventSink
from pipeline.simulation.event_mc_v1.state import FightState, Phase, StateDelta


@dataclass
class SyntheticCandidate:
    candidate_id: str
    delta: StateDelta = field(default_factory=StateDelta)
    seen_times: list[float] = field(default_factory=list)
    rng_stream: RNGStream = RNGStream.STRIKE_RESOLUTION

    def resolve(self, state, context, rng: np.random.Generator) -> Resolution:
        self.seen_times.append(state.fight_time_seconds)
        return Resolution(delta=self.delta, payload=self.candidate_id)


@dataclass
class Provider:
    candidate: SyntheticCandidate
    rate: float
    observations: list[tuple[float, Phase, str | None, str | None]] = field(default_factory=list)

    def candidates(self, state, context):
        self.observations.append((state.fight_time_seconds, state.phase, state.ground_controller, state.clinch_controller))
        return [EventRate(self.candidate, self.rate)]


@dataclass
class Accumulator:
    elapsed: list[float] = field(default_factory=list)

    def advance(self, state, context, dt_seconds):
        self.elapsed.append(dt_seconds)
        return StateDelta()


class FixedScheduler:
    def __init__(self, waits):
        self.waits = iter(waits)

    def sample(self, candidates, rng):
        dt = next(self.waits)
        return dt, candidates[0].candidate


def test_boundaries_truncate_wait_reset_position_and_recompute_rates() -> None:
    candidate = SyntheticCandidate("move", StateDelta(phase=Phase.GROUND, ground_controller="red", set_ground_controller=True))
    provider = Provider(candidate, 1)
    advancer = Accumulator()
    trace = FullTraceEventSink()
    state = FightState(phase=Phase.CLINCH, clinch_controller="blue")
    result = SimulationEngine(
        FightConfig(2, 5), provider, advancer, RNGManager(1), trace,
        FixedScheduler([6, 1, 20]),
    ).run(state)
    assert advancer.elapsed == [5, 1, 4]
    assert candidate.seen_times == [6]
    assert provider.observations[1] == (5, Phase.DISTANCE, None, None)
    events = [entry.payload for entry in trace.entries if entry.kind == "event"]
    assert [event.timestamp_seconds for event in events] == sorted(event.timestamp_seconds for event in events)
    assert [(type(event), event.timestamp_seconds) for event in events if isinstance(event, (RoundEnded, RoundStarted))] == [
        (RoundStarted, 0), (RoundEnded, 5), (RoundStarted, 5), (RoundEnded, 10)
    ]
    assert result.state.fight_time_seconds == 10
    assert result.state.finish_reason == "scheduled_horizon"
    assert not any(isinstance(event, PrimaryEvent) and event.timestamp_seconds > 10 for event in events)


def test_continuous_advance_precedes_engine_owned_delta_and_explicit_finish() -> None:
    candidate = SyntheticCandidate("finish", StateDelta(phase=Phase.GROUND, finished=True, finish_reason="synthetic"))
    advancer = Accumulator()
    trace = FullTraceEventSink()
    result = SimulationEngine(
        FightConfig(1, 10), Provider(candidate, 1), advancer, RNGManager(2), trace,
        FixedScheduler([2]),
    ).run()
    assert advancer.elapsed == [2]
    assert candidate.seen_times == [2]
    assert result.state.phase == Phase.GROUND.value
    assert result.state.fight_time_seconds == 2
    assert result.state.finished
    timestamps = [entry.timestamp_seconds for entry in trace.entries]
    assert timestamps == sorted(timestamps)
    assert max(timestamps) == 2


def _run_with_sink(sink):
    candidate = SyntheticCandidate("finish", StateDelta(finished=True, finish_reason="done"))
    return SimulationEngine(
        FightConfig(1, 10), Provider(candidate, 1), Accumulator(), RNGManager(77), sink
    ).run()


def test_sink_modes_are_physics_and_rng_invariant() -> None:
    null = _run_with_sink(NullEventSink())
    stats = _run_with_sink(StatsEventSink())
    trace = _run_with_sink(FullTraceEventSink())
    assert null.state == stats.state == trace.state
    assert null.sink_result is None
    assert stats.sink_result["event_counts"]["PrimaryEvent"] == 1
    assert [entry.timestamp_seconds for entry in trace.sink_result] == sorted(entry.timestamp_seconds for entry in trace.sink_result)


def test_action_availability_is_inactive_by_default() -> None:
    availability = FightState().action_availability
    assert availability.busy_until_seconds is None
    assert dict(availability.cooldown_until_seconds) == {}
    assert availability.is_available("synthetic", 0)


def test_clock_views_are_derived_from_authoritative_fight_time() -> None:
    config = FightConfig(3, 90)
    assert config.round_number_at(115) == 2
    assert config.round_elapsed_seconds_at(115) == 25
    assert config.round_remaining_seconds_at(115) == 65
    assert config.fight_remaining_seconds_at(115) == 155
