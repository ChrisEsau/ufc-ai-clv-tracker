from pipeline.simulation.event_mc_v1.components.actions import ActionAttempt
from pipeline.simulation.event_mc_v1.components.profiles import Side
from pipeline.simulation.event_mc_v1.diagnostics.phase7d_submission_decomposition import SubmissionDecompositionSink
from pipeline.simulation.event_mc_v1.events import ConsequenceEvent, PrimaryEvent
from pipeline.simulation.event_mc_v1.state import FightState, Phase
from pipeline.simulation.event_mc_v1.submission_finishes import SubmissionFinishOutcome


def test_sink_records_submission_round_position_conversion_and_ground_exposure():
    sink = SubmissionDecompositionSink()
    before = FightState(fight_time_seconds=305, phase=Phase.GROUND, ground_controller="red")
    after = FightState(fight_time_seconds=315, phase=Phase.GROUND, ground_controller="red")
    sink.on_time_advance(10, before, after)
    sink.on_event(PrimaryEvent(305, "red_submission", ActionAttempt(Side.RED, "submission_attempt")), before, after)
    outcome = SubmissionFinishOutcome(Side.RED, Side.BLUE, 60, 50, "top", .25, .2, True)
    sink.on_event(ConsequenceEvent(305, "SubmissionFinishOutcome", outcome), before, after)
    result = sink.finalize()
    assert result["attempts_by_round"] == {2: 1}
    assert result["attempts_by_position"] == {"top": 1}
    assert result["submission_finishes"] == 1
    assert result["submission_finishes_by_position"] == {"top": 1}
    assert result["ground_seconds"] == result["exposure_seconds"] == 10
    assert result["ground_control_seconds"] == {"red": 10}


def test_sink_classifies_bottom_attempt_without_counting_non_submission_action():
    sink = SubmissionDecompositionSink(); state = FightState(phase=Phase.GROUND, ground_controller="blue")
    sink.on_event(PrimaryEvent(2, "red_submission", ActionAttempt(Side.RED, "submission_attempt")), state, state)
    sink.on_event(PrimaryEvent(3, "red_strike", ActionAttempt(Side.RED, "ground_strike")), state, state)
    assert sink.finalize()["attempts_by_position"] == {"bottom": 1}
