import math

import pytest

from pipeline.simulation.event_mc_v1.components.action_rates import FightFlowRateProvider
from pipeline.simulation.event_mc_v1.components.actions import PhaseCandidate
from pipeline.simulation.event_mc_v1.components.formulas import (
    CLINCH_SEPARATE_BASE_10S,
    CLINCH_TD_ATTEMPT_BASE_10S,
    GROUND_EXIT_BASE_10S,
    clinch_separation_interval_probability,
    clinch_td_interval_probability,
    ground_exit_interval_probability,
    ground_exit_rates,
    phase_strike_rate_per_second,
    submission_attempt_interval_probability,
    td_attempt_interval_probability,
)
from pipeline.simulation.event_mc_v1.components.profiles import FighterProfile, MatchupProfiles, Side
from pipeline.simulation.event_mc_v1.config import FightConfig
from pipeline.simulation.event_mc_v1.contracts import FightContext, NoOpTimeAdvanceModel
from pipeline.simulation.event_mc_v1.engine import SimulationEngine
from pipeline.simulation.event_mc_v1.flow_stats import FlowStatsSink
from pipeline.simulation.event_mc_v1.rng import RNGManager
from pipeline.simulation.event_mc_v1.state import FightState, Phase


def fighter(name="fighter", **changes):
    values = dict(fighter_id=name, fighter_name=name, distance_striking_pressure=50,
                  distance_striking_precision=50, distance_striking_defense=50,
                  clinch_striking_pressure=50, wrestling_entry=50,
                  wrestling_conversion=50, td_defense=50, control_imposition=50)
    values.update(changes)
    return FighterProfile(**values)


def profiles(**red_changes):
    return MatchupProfiles(fighter("red", **red_changes), fighter("blue"))


def test_phase_candidates_are_phase_specific_and_controller_aware():
    provider = FightFlowRateProvider(profiles())
    context = FightContext(FightConfig(), 0, 1)
    clinch = provider.candidates(FightState(phase=Phase.CLINCH, clinch_controller="red"), context)
    assert {x.candidate.candidate_id for x in clinch} == {"red_clinch_strike", "blue_clinch_strike", "red_clinch_takedown", "blue_clinch_takedown", "blue_clinch_separation"}
    ground = provider.candidates(FightState(phase=Phase.GROUND, ground_controller="red"), context)
    assert {x.candidate.candidate_id for x in ground} == {"red_ground_strike", "blue_ground_strike", "red_submission_attempt", "blue_submission_attempt", "blue_ground_escape", "blue_ground_reversal"}


def test_ground_exit_is_exact_partition_not_two_exit_clocks():
    escape, reversal, total = ground_exit_rates(fighter("top"), fighter("bottom"))
    assert escape + reversal == pytest.approx(total)
    assert 0 < reversal < total


def test_neutral_phase_transition_consumers_match_v0_bases():
    neutral = fighter()
    assert clinch_td_interval_probability(neutral) == pytest.approx(CLINCH_TD_ATTEMPT_BASE_10S)
    assert clinch_separation_interval_probability(neutral, neutral) == pytest.approx(CLINCH_SEPARATE_BASE_10S)
    assert ground_exit_interval_probability(neutral, neutral) == pytest.approx(GROUND_EXIT_BASE_10S)


def test_phase2b_distance_initiation_remains_entry_only():
    base = td_attempt_interval_probability(fighter(wrestling_entry=55))
    assert td_attempt_interval_probability(fighter(wrestling_entry=55, control_imposition=90, clinch_striking_pressure=10)) == base


def test_poisson_rates_and_submission_attempt_mapping_move_correctly():
    assert phase_strike_rate_per_second(fighter(ground_striking_pressure=60), "ground") > phase_strike_rate_per_second(fighter(), "ground")
    assert phase_strike_rate_per_second(fighter(), "ground", bottom=True) == pytest.approx(phase_strike_rate_per_second(fighter(), "ground") * 0.20)
    assert submission_attempt_interval_probability(fighter(submission_pressure=60)) > submission_attempt_interval_probability(fighter())


def test_submission_attempt_position_multiplier_is_neutral():
    identical = fighter(submission_pressure=60)
    top = submission_attempt_interval_probability(identical, bottom=False)
    bottom = submission_attempt_interval_probability(identical, bottom=True)
    assert top == pytest.approx(bottom)
    assert top > 0


@pytest.mark.parametrize("family,outcome,phase,controller,expected_phase,expected_controller", [
    ("clinch_separation", "separated", Phase.CLINCH, None, Phase.DISTANCE, None),
    ("ground_escape", "escaped", Phase.GROUND, "red", Phase.DISTANCE, None),
    ("ground_reversal", "reversed", Phase.GROUND, "red", Phase.GROUND, "blue"),
    ("submission_attempt", "attempted", Phase.GROUND, "red", Phase.GROUND, "red"),
])
def test_transitions_are_nonterminal(family, outcome, phase, controller, expected_phase, expected_controller):
    p = profiles()
    side = Side.BLUE if family in {"ground_escape", "ground_reversal"} else Side.RED
    state = FightState(phase=phase, ground_controller=controller, clinch_controller="red" if phase is Phase.CLINCH else None)
    resolution = PhaseCandidate(side, family, p).resolve(state, FightContext(FightConfig(), 0, 1), RNGManager(1).stream(40 if family == "submission_attempt" else 10))
    SimulationEngine._apply_delta(state, resolution.delta)
    assert state.phase is expected_phase
    assert state.ground_controller == expected_controller
    assert not state.finished
    assert resolution.consequence_events[0].payload.outcome == outcome


def test_full_horizon_flow_accounts_for_every_second_and_is_deterministic():
    def run(sink):
        return SimulationEngine(FightConfig(3, 60), FightFlowRateProvider(profiles(wrestling_entry=60, wrestling_conversion=60)), NoOpTimeAdvanceModel(), RNGManager(77), sink).run()
    first, second = run(FlowStatsSink()), run(FlowStatsSink())
    assert first.state == second.state
    assert first.sink_result == second.sink_result
    assert sum(first.sink_result["phase_seconds"].values()) == pytest.approx(180)
    assert first.state.finish_reason == "scheduled_horizon"
    assert all(value >= 0 for value in first.sink_result["phase_seconds"].values())
