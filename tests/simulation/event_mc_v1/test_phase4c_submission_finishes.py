from pathlib import Path

import numpy as np
import yaml

from pipeline.simulation.event_mc_v1.calibration import load_event_mc_config
from pipeline.simulation.event_mc_v1.components.actions import ActionAttempt, ActionOutcome
from pipeline.simulation.event_mc_v1.components.profiles import FighterProfile, MatchupProfiles, Side
from pipeline.simulation.event_mc_v1.config import FightConfig
from pipeline.simulation.event_mc_v1.contracts import NoOpTimeAdvanceModel, Resolution
from pipeline.simulation.event_mc_v1.engine import SimulationEngine
from pipeline.simulation.event_mc_v1.events import ConsequenceEvent, FightFinished, PrimaryEvent
from pipeline.simulation.event_mc_v1.modifiers import DynamicModifiers
from pipeline.simulation.event_mc_v1.rng import RNGManager, RNGStream
from pipeline.simulation.event_mc_v1.scheduler import EventRate
from pipeline.simulation.event_mc_v1.sinks import FullTraceEventSink
from pipeline.simulation.event_mc_v1.state import FightState, Phase
from pipeline.simulation.event_mc_v1.submission_finishes import SubmissionFinishModel, SubmissionFinishOutcome


def fighter(name, **changes):
    values = dict(fighter_id=name, fighter_name=name, distance_striking_pressure=50, distance_striking_precision=50, distance_striking_defense=50, clinch_striking_pressure=50, wrestling_entry=50, wrestling_conversion=50, td_defense=50, control_imposition=50)
    values.update(changes)
    return FighterProfile(**values)


def model(red=None, blue=None, calibration=None):
    return SubmissionFinishModel(MatchupProfiles(red or fighter("red"), blue or fighter("blue")), calibration) if calibration else SubmissionFinishModel(MatchupProfiles(red or fighter("red"), blue or fighter("blue")))


def test_submission_probability_traits_stamina_and_neutral_position():
    state = FightState(phase="ground", ground_controller="red")
    baseline = model().probability(state, Side.RED)[0]
    threat = model(red=fighter("red", submission_conversion=70)).probability(state, Side.RED)[0]
    resistance = model(blue=fighter("blue", submission_resistance=70)).probability(state, Side.RED)[0]
    bottom_result = model().probability(FightState(phase="ground", ground_controller="blue"), Side.RED)
    bottom = bottom_result[0]
    stamina = model().probability(FightState(phase="ground", ground_controller="red", red_stamina=1, blue_stamina=0), Side.RED)[0]
    extreme = model(red=fighter("red", submission_conversion=1e6)).probability(state, Side.RED)[0]
    assert threat > baseline > resistance
    top_result = model().probability(state, Side.RED)
    assert baseline == bottom
    assert top_result[4] == bottom_result[4] == 0.0
    assert stamina > baseline
    assert 0 < extreme < 1


def test_submission_sampling_is_reproducible_stochastic_and_structured():
    subject = model()
    attempt = ActionAttempt(Side.RED, "submission_attempt")
    values = [subject.resolve(FightState(ground_controller="red"), attempt, 4.5, np.random.default_rng(seed))[1].payload.finished for seed in range(40)]
    repeat = [subject.resolve(FightState(ground_controller="red"), attempt, 4.5, np.random.default_rng(seed))[1].payload.finished for seed in range(40)]
    assert values == repeat and len(set(values)) == 2
    class AlwaysFinish:
        def random(self): return 0.0
    delta, event = subject.resolve(FightState(ground_controller="red"), attempt, 4.5, AlwaysFinish())
    assert delta.finished and delta.winner == "red" and delta.finish_method == "SUB"
    assert isinstance(event.payload, SubmissionFinishOutcome) and event.payload.finished


def test_engine_accounts_finishing_submission_once_and_stops():
    class Candidate:
        candidate_id = "red_submission_attempt"
        rng_stream = RNGStream.SUBMISSION
        def resolve(self, state, context, rng):
            attempt = ActionAttempt(Side.RED, "submission_attempt", DynamicModifiers(1, 1))
            outcome = ConsequenceEvent(state.fight_time_seconds, "ActionOutcome", ActionOutcome(Side.RED, "submission_attempt", "attempted"))
            return Resolution(payload=attempt, consequence_events=(outcome,))
    class Provider:
        def candidates(self, state, context): return (EventRate(Candidate(), 1000),)
    class AlwaysFinishModel(SubmissionFinishModel):
        def resolve(self, state, attempt, timestamp, rng, *, pre_action_state=None):
            class Fixed:
                def random(self): return 0.0
            return super().resolve(state, attempt, timestamp, Fixed(), pre_action_state=pre_action_state)
    result = SimulationEngine(FightConfig(1, 10), Provider(), NoOpTimeAdvanceModel(), RNGManager(4), FullTraceEventSink(), submission_finish_model=AlwaysFinishModel(model().profiles)).run()
    events = [entry.payload for entry in result.sink_result if entry.kind == "event"]
    assert sum(isinstance(event, FightFinished) for event in events) == 1
    assert sum(isinstance(event, PrimaryEvent) for event in events) == 1
    assert sum(isinstance(event, ConsequenceEvent) and isinstance(event.payload, ActionOutcome) for event in events) == 1
    assert sum(isinstance(event, ConsequenceEvent) and isinstance(event.payload, SubmissionFinishOutcome) for event in events) == 1
    assert isinstance(events[-1], FightFinished)


def test_weight_class_override_reaches_submission_curve(tmp_path: Path):
    document = yaml.safe_load(Path("config/event_mc_v1.yaml").read_text())
    document["weight_classes"] = {"synthetic": {"submission_finish": {"intercept": -8.0}}}
    path = tmp_path / "config.yaml"; path.write_text(yaml.safe_dump(document))
    resolver = load_event_mc_config(path)
    default = model(calibration=resolver.for_weight_class())
    override = model(calibration=resolver.for_weight_class("synthetic"))
    state = FightState(ground_controller="red")
    assert override.probability(state, Side.RED)[0] < default.probability(state, Side.RED)[0]
    assert override.calibration.section("submission_finish")["rating_scale"] == default.calibration.section("submission_finish")["rating_scale"]


def test_current_attempt_uses_pre_action_stamina_and_cost_only_affects_later_attempts(tmp_path: Path):
    document = yaml.safe_load(Path("config/event_mc_v1.yaml").read_text())
    document["defaults"]["submission_finish"]["intercept"] = -10.0
    path = tmp_path / "config.yaml"; path.write_text(yaml.safe_dump(document))
    calibration = load_event_mc_config(path).for_weight_class()

    def run(first_cost):
        class Candidate:
            candidate_id = "red_submission_attempt"
            rng_stream = RNGStream.SUBMISSION
            calls = 0
            def resolve(self, state, context, rng):
                self.calls += 1
                cost = first_cost if self.calls == 1 else 0.0
                attempt = ActionAttempt(Side.RED, "submission_attempt", DynamicModifiers(1, 1))
                outcome = ConsequenceEvent(state.fight_time_seconds, "ActionOutcome", ActionOutcome(Side.RED, "submission_attempt", "attempted"))
                from pipeline.simulation.event_mc_v1.state import StateDelta
                return Resolution(payload=attempt, delta=StateDelta(red_stamina=state.red_stamina - cost), consequence_events=(outcome,))
        candidate = Candidate()
        class Provider:
            def candidates(self, state, context):
                return (EventRate(candidate, 1000),) if candidate.calls < 2 else ()
        sink = FullTraceEventSink()
        result = SimulationEngine(
            FightConfig(1, 10), Provider(), NoOpTimeAdvanceModel(), RNGManager(17), sink,
            submission_finish_model=model(calibration=calibration),
        ).run(FightState(fight_time_seconds=0.1, phase=Phase.GROUND, ground_controller="red"))
        checks = [
            entry.payload.payload
            for entry in result.sink_result
            if entry.kind == "event"
            and isinstance(entry.payload, ConsequenceEvent)
            and isinstance(entry.payload.payload, SubmissionFinishOutcome)
        ]
        return result, checks

    low_cost_result, low_cost_checks = run(0.1)
    high_cost_result, high_cost_checks = run(0.5)
    assert len(low_cost_checks) == len(high_cost_checks) == 2
    assert low_cost_checks[0].finish_probability == high_cost_checks[0].finish_probability
    assert low_cost_checks[1].finish_probability > high_cost_checks[1].finish_probability
    assert low_cost_result.state.red_stamina == 0.9
    assert high_cost_result.state.red_stamina == 0.5
