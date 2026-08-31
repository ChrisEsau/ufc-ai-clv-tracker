from pipeline.simulation.event_mc_v1.components.action_rates import DistanceActionRateProvider
from pipeline.simulation.event_mc_v1.components.profiles import FighterProfile, MatchupProfiles
from pipeline.simulation.event_mc_v1.config import FightConfig
from pipeline.simulation.event_mc_v1.contracts import FightContext, NoOpTimeAdvanceModel
from pipeline.simulation.event_mc_v1.distance_stats import DistanceStatsSink
from pipeline.simulation.event_mc_v1.engine import SimulationEngine
from pipeline.simulation.event_mc_v1.rng import RNGManager
from pipeline.simulation.event_mc_v1.state import FightState, Phase


def profile(fighter_id: str, **changes) -> FighterProfile:
    values = dict(
        fighter_id=fighter_id,
        fighter_name=fighter_id,
        distance_striking_pressure=50,
        distance_striking_precision=50,
        distance_striking_defense=50,
        clinch_striking_pressure=50,
        wrestling_entry=50,
        wrestling_conversion=50,
        td_defense=50,
        control_imposition=50,
    )
    values.update(changes)
    return FighterProfile(**values)


def test_provider_exposes_exactly_six_distance_candidates_and_audits() -> None:
    provider = DistanceActionRateProvider(MatchupProfiles(profile("red"), profile("blue")))
    state = FightState()
    context = FightContext(FightConfig(), 0, 1)
    candidates = provider.candidates(state, context)
    assert {candidate.candidate.candidate_id for candidate in candidates} == {
        "red_strike", "blue_strike", "red_takedown", "blue_takedown",
        "red_clinch_entry", "blue_clinch_entry",
    }
    assert len(provider.audit_rows()) == 6
    state.phase = Phase.GROUND
    assert provider.candidates(state, context) == ()


def test_distance_engine_is_observable_and_stops_scheduling_after_transition() -> None:
    profiles = MatchupProfiles(
        profile("red", wrestling_conversion=100, wrestling_entry=100),
        profile("blue", td_defense=0),
    )
    sink = DistanceStatsSink()
    result = SimulationEngine(
        FightConfig(1, 30),
        DistanceActionRateProvider(profiles),
        NoOpTimeAdvanceModel(),
        RNGManager(4),
        sink,
    ).run()
    assert result.state.fight_time_seconds == 30
    assert result.state.finished
    assert sum(sum(counts.values()) for counts in result.sink_result["attempts"].values()) >= 1
    if result.state.phase == Phase.GROUND.value:
        assert result.state.ground_controller in {"red", "blue"}


def test_rate_audit_contains_legacy_blended_inputs() -> None:
    provider = DistanceActionRateProvider(MatchupProfiles(profile("red"), profile("blue")))
    td_rows = [row for row in provider.audit_rows() if row.action_family == "takedown"]
    assert all(row.interval_seconds == 10 for row in td_rows)
    assert all("legacy_wrestling_preference" in row.major_inputs for row in td_rows)
