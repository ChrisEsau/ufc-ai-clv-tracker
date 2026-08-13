import numpy as np

from pipeline.simulation.event_mc_v1.components.profiles import FighterProfile, MatchupProfiles, Side
from pipeline.simulation.event_mc_v1.finishes import KOTKOFinishModel
from pipeline.simulation.event_mc_v1.physiology import PhysiologyOutcome
from pipeline.simulation.event_mc_v1.state import FightState, StateDelta
from pipeline.simulation.event_mc_v1.config import FightConfig
from pipeline.simulation.event_mc_v1.contracts import NoOpTimeAdvanceModel, Resolution
from pipeline.simulation.event_mc_v1.engine import SimulationEngine
from pipeline.simulation.event_mc_v1.events import ConsequenceEvent, FightFinished, PrimaryEvent
from pipeline.simulation.event_mc_v1.rng import RNGManager, RNGStream
from pipeline.simulation.event_mc_v1.scheduler import EventRate
from pipeline.simulation.event_mc_v1.sinks import FullTraceEventSink
from pipeline.simulation.event_mc_v1.components.actions import ActionAttempt, ActionOutcome
from pipeline.simulation.event_mc_v1.modifiers import DynamicModifiers
from pipeline.simulation.event_mc_v1.calibration import load_event_mc_config
from pathlib import Path
import yaml


def fighter(name, **changes):
    values = dict(fighter_id=name, fighter_name=name, distance_striking_pressure=50,
                  distance_striking_precision=50, distance_striking_defense=50,
                  clinch_striking_pressure=50, wrestling_entry=50,
                  wrestling_conversion=50, td_defense=50, control_imposition=50)
    values.update(changes)
    return FighterProfile(**values)


def model(**blue):
    return KOTKOFinishModel(MatchupProfiles(fighter("red"), fighter("blue", **blue)))


def outcome(impact=2, kd=False):
    return PhysiologyOutcome(Side.RED, Side.BLUE, "distance", impact, 1, 1, 0.1, kd)


def test_impact_trauma_acute_and_kd_move_probability_monotonically():
    subject = model()
    low, _ = subject.probability(FightState(), outcome(1))
    high, _ = subject.probability(FightState(), outcome(4))
    trauma, _ = subject.probability(FightState(blue_cumulative_trauma=100), outcome(1))
    acute, _ = subject.probability(FightState(blue_acute_vulnerability=1), outcome(1))
    kd, _ = subject.probability(FightState(), outcome(1, True))
    assert high > low
    assert trauma > low
    assert acute > low
    assert kd >= low


def test_baseline_resistance_lowers_probability_and_no_trauma_threshold():
    low, _ = model(damage_durability=30, knockdown_resistance=30).probability(FightState(), outcome())
    high, _ = model(damage_durability=70, knockdown_resistance=70).probability(FightState(), outcome())
    extreme, _ = model().probability(FightState(blue_cumulative_trauma=1_000_000), outcome(0.001))
    assert high < low
    assert 0 < extreme < 1


def test_fresh_one_shot_finish_possible_and_terminal_delta_is_structured():
    class AlwaysFinish:
        def random(self): return 0.0
    delta, event = model().resolve(FightState(), outcome(20), 4.5, AlwaysFinish())
    assert delta.finished and delta.winner == "red" and delta.finish_method == "KO_TKO"
    assert event.payload.finished and event.timestamp_seconds == 4.5


def test_finish_sampling_is_deterministic_and_stochastic_across_seeds():
    subject = model()
    values = [subject.resolve(FightState(), outcome(30), 1, np.random.default_rng(seed))[1].payload.finished for seed in range(30)]
    repeat = [subject.resolve(FightState(), outcome(30), 1, np.random.default_rng(seed))[1].payload.finished for seed in range(30)]
    assert values == repeat
    assert len(set(values)) == 2


def test_engine_emits_one_lifecycle_finish_and_no_later_primary_events():
    class Candidate:
        candidate_id = "red_strike"
        rng_stream = RNGStream.STRIKE_RESOLUTION
        def resolve(self, state, context, rng):
            timestamp = state.fight_time_seconds
            return Resolution(
                payload=ActionAttempt(Side.RED, "strike", DynamicModifiers(1, 1), True),
                consequence_events=(
                    ConsequenceEvent(
                        timestamp,
                        "ActionOutcome",
                        ActionOutcome(Side.RED, "strike", "landed"),
                    ),
                ),
            )
    class Provider:
        def candidates(self, state, context): return (EventRate(Candidate(), 1000),)
    class Physiology:
        def resolve(self, state, payload, timestamp, damage_rng, kd_rng):
            from pipeline.simulation.event_mc_v1.events import ConsequenceEvent
            return StateDelta(blue_cumulative_trauma=1), (ConsequenceEvent(timestamp, "PhysiologyOutcome", outcome(100, True)),)
    trace = FullTraceEventSink()
    result = SimulationEngine(FightConfig(1, 10), Provider(), NoOpTimeAdvanceModel(), RNGManager(9), trace, physiology_model=Physiology(), finish_model=model()).run()
    events = [entry.payload for entry in result.sink_result if entry.kind == "event"]
    assert sum(isinstance(event, FightFinished) for event in events) == 1
    action_outcomes = [
        event.payload
        for event in events
        if isinstance(event, ConsequenceEvent)
        and isinstance(event.payload, ActionOutcome)
    ]
    assert action_outcomes == [ActionOutcome(Side.RED, "strike", "landed")]
    primary_indexes = [i for i, event in enumerate(events) if isinstance(event, PrimaryEvent)]
    finish_index = next(i for i, event in enumerate(events) if isinstance(event, FightFinished))
    assert all(i < finish_index for i in primary_indexes)
    assert finish_index == len(events) - 1
    assert result.state.finished and result.state.finish_method == "KO_TKO"


def test_synthetic_weight_class_override_reaches_finish_curve(tmp_path: Path):
    document = yaml.safe_load(Path("config/event_mc_v1.yaml").read_text())
    document["weight_classes"] = {"synthetic": {"finish": {"midpoint_impact_ratio": 72.0}}}
    path = tmp_path / "config.yaml"; path.write_text(yaml.safe_dump(document))
    resolver = load_event_mc_config(path)
    default = KOTKOFinishModel(model().profiles, resolver.for_weight_class())
    override = KOTKOFinishModel(model().profiles, resolver.for_weight_class("synthetic"))
    assert override.probability(FightState(), outcome(4))[0] < default.probability(FightState(), outcome(4))[0]
