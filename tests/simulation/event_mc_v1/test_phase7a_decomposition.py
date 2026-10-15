from pipeline.simulation.event_mc_v1.components.actions import ActionAttempt
from pipeline.simulation.event_mc_v1.components.profiles import Side
from pipeline.simulation.event_mc_v1.diagnostics.phase7a_decomposition import DecompositionSink, _distribution
from pipeline.simulation.event_mc_v1.events import ConsequenceEvent, PrimaryEvent
from pipeline.simulation.event_mc_v1.finishes import FinishOutcome
from pipeline.simulation.event_mc_v1.physiology import PhysiologyOutcome


class Snapshot:
    red_cumulative_trauma = 4.0
    blue_cumulative_trauma = 8.0
    red_acute_vulnerability = 0.0
    blue_acute_vulnerability = 0.5


def test_sink_pairs_compact_attempt_impact_and_finish_evidence():
    sink = DecompositionSink(); snapshot = Snapshot()
    sink.on_time_advance(12.0, snapshot, snapshot)
    sink.on_event(PrimaryEvent(12.0, "red_strike", ActionAttempt(Side.RED, "strike")), snapshot, snapshot)
    physiology = PhysiologyOutcome(Side.RED, Side.BLUE, "distance", 3.0, 2.0, 1.0, .2, True)
    sink.on_event(ConsequenceEvent(12.0, "PhysiologyOutcome", physiology), snapshot, snapshot)
    finish = FinishOutcome(Side.RED, Side.BLUE, 3.0, 1.0, .4, True, True)
    sink.on_event(ConsequenceEvent(12.0, "FinishOutcome", finish), snapshot, snapshot)
    result = sink.finalize()
    assert result["exposure_seconds"] == 12
    assert result["attempts"] == {"distance": 1}
    assert result["landed"] == {"distance": 1}
    assert result["impacts"][0] == {"impact":3.0,"kd":True,"finished":True,"phase":"distance","round":1,"trauma":8.0,"acute":.5}


def test_distribution_quantiles_are_exact_on_controlled_values():
    result = _distribution([1, 2, 3, 4])
    assert result["count"] == 4 and result["mean"] == 2.5
    assert result["median"] == 2.5 and result["max"] == 4
