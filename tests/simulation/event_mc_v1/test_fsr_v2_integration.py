from dataclasses import FrozenInstanceError

import pytest

from pipeline.simulation.event_mc_v1.components.action_rates import FSRV2ActionRateProvider
from pipeline.simulation.event_mc_v1.components.fsr_v2 import FSRV2FighterInput, FSRV2Matchup, FSR_V2_SIMULATOR_FIELDS, FSR_V2_TRAIT_FIELDS
from pipeline.simulation.event_mc_v1.config import FightConfig
from pipeline.simulation.event_mc_v1.contracts import FightContext
from pipeline.simulation.event_mc_v1.flow_stats import FlowStatsSink
from pipeline.simulation.event_mc_v1.single_fight import build_engine, fight_from_fsr_v2_rows
from pipeline.simulation.event_mc_v1.state import FightState, Phase


def row(fighter_id="f", **changes):
    values = {name: 0.0 for name in FSR_V2_TRAIT_FIELDS}
    values.update({"fighter_id": fighter_id, "fighter_name": fighter_id,
        "standing_striking_tendency": .08, "takedown_tendency": .02,
        "ground_striking_tendency": .05, "submission_tendency": .01,
        "head_strike_tendency": .7, "body_strike_tendency": .3, "leg_strike_tendency": .2,
        "stamina_capacity": 100.0, "stamina_depletion_resistance": 61.0,
        "stamina_performance_resilience": 62.0, "striking_power": 63.0,
        "damage_durability": 64.0, "knockdown_resistance": 65.0})
    values.update(changes)
    return values


def test_adapter_requires_27_fields_and_passes_physical_values_unchanged():
    assert len(FSR_V2_SIMULATOR_FIELDS) == 27
    fighter = FSRV2FighterInput.from_mapping(row())
    profile = fighter.physical_profile()
    assert (profile.stamina_capacity, profile.stamina_depletion_resistance,
            profile.stamina_performance_resilience, profile.striking_power,
            profile.damage_durability, profile.knockdown_resistance) == (100, 61, 62, 63, 64, 65)
    with pytest.raises(FrozenInstanceError): fighter.striking_power = 50
    for name in FSR_V2_SIMULATOR_FIELDS:
        broken = row(); broken.pop(name)
        with pytest.raises(ValueError, match="missing required"): FSRV2FighterInput.from_mapping(broken)
    with pytest.raises(ValueError, match="stamina_capacity"):
        FSRV2FighterInput.from_mapping(row(stamina_capacity=99))


def test_two_state_clock_sets():
    matchup = FSRV2Matchup(FSRV2FighterInput.from_mapping(row("r")), FSRV2FighterInput.from_mapping(row("b")))
    provider = FSRV2ActionRateProvider(matchup, matchup.physical_profiles())
    context = FightContext(FightConfig(3), 0, 1)
    standing = provider.candidates(FightState(), context)
    assert {x.candidate.action_family for x in standing} == {"standing_strike", "takedown"}
    ground = provider.candidates(FightState(phase=Phase.GROUND, ground_controller="red"), context)
    assert {x.candidate.action_family for x in ground} == {"ground_strike", "submission_attempt", "ground_escape"}
    assert len(tuple(Phase)) == 2 and not hasattr(Phase, "CLINCH")
    assert all(x.rate_per_second >= 0 for x in (*standing, *ground))


def test_complete_fight_is_deterministic_and_uses_physical_input():
    fight = fight_from_fsr_v2_rows(row("r"), row("b", striking_power=72.0))
    assert fight.profiles.blue.striking_power == 72.0
    results = [build_engine(fight, 12345, FlowStatsSink())[0].run() for _ in range(2)]
    assert results[0] == results[1]
    assert results[0].state.finished and results[0].state.winner in {"red", "blue"}
    assert results[0].state.finish_method in {"KO_TKO", "SUB", "DEC"}
    assert sum(results[0].sink_result["phase_seconds"].values()) == pytest.approx(results[0].state.fight_time_seconds)
