from types import SimpleNamespace
from pipeline.simulation.event_clock_mc_v2.calibration.invariants import (
    inspect_path,
    status,
)
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase


def test_known_invalid_cases_are_detected():
    fighter = SimpleNamespace(
        stamina=2.0,
        cumulative_trauma=0.0,
        acute_vulnerability=0.0,
        knockdowns_suffered=0,
    )
    result = SimpleNamespace(
        timeline_segments=[SimpleNamespace(duration=-1.0, phase=Phase.GROUND)],
        reported_through_seconds=2.0,
        events=[
            SimpleNamespace(
                source_phase=Phase.GROUND,
                selected_action=ActionFamily.STAND_ATTACK,
                timestamp_seconds=3.0,
            )
        ],
        termination=object(),
        final_state=SimpleNamespace(
            fight_time_seconds=2.0,
            physiology=SimpleNamespace(red=fighter, blue=fighter),
        ),
    )
    counts = inspect_path(result)
    assert counts["illegal_cross_phase_actions"] == 1
    assert counts["timeline_exposure_mismatch"] == 1
    assert counts["post_finish_events"] == 1
    assert counts["invalid_state_transitions"] == 1
    assert counts["nan_or_non_finite_state"] == 0
    assert counts["impossible_physiology_state"] == 1
    assert status(counts)["status"] == "FAIL"
