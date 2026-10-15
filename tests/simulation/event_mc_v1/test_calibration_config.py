from pathlib import Path

import yaml

from pipeline.simulation.event_mc_v1.calibration import DEFAULT_CALIBRATION, load_event_mc_config
from pipeline.simulation.event_mc_v1.components.formulas import (
    DISTANCE_STRIKE_ACCURACY_BASE,
    DISTANCE_STRIKE_ATTEMPTS_PER_30S_BASE,
    DISTANCE_TD_ATTEMPT_BASE_30S,
)
from pipeline.simulation.event_mc_v1.components.profiles import FighterProfile, MatchupProfiles, Side
from pipeline.simulation.event_mc_v1.modifiers import DynamicModifierProvider
from pipeline.simulation.event_mc_v1.state import FightState
from pipeline.simulation.event_mc_v1.components.action_rates import FightFlowRateProvider
from pipeline.simulation.event_mc_v1.config import FightConfig
from pipeline.simulation.event_mc_v1.contracts import FightContext
from pipeline.simulation.event_mc_v1.physiology import ImpactTraumaKnockdownModel
from pipeline.simulation.event_mc_v1.stamina import StaminaModel


def profile():
    return FighterProfile("fighter", "fighter", 50, 50, 50, 50, 50, 50, 50, 50)


def test_yaml_defaults_are_active_values():
    distance = DEFAULT_CALIBRATION.section("distance")
    assert distance["strike_attempts_per_30s"] == DISTANCE_STRIKE_ATTEMPTS_PER_30S_BASE == 6.0
    assert distance["strike_accuracy"] == DISTANCE_STRIKE_ACCURACY_BASE == 0.40
    assert distance["td_attempt_base_30s"] == DISTANCE_TD_ATTEMPT_BASE_30S == 0.16
    submission = DEFAULT_CALIBRATION.section("submission_attempts")
    assert submission == {
        "base_30s": 0.045,
        "bottom_multiplier": 1.0,
        "modifier_scale": 10.0,
        "probability_cap": 0.35,
    }
    submission_finish = DEFAULT_CALIBRATION.section("submission_finish")
    assert submission_finish["top_position_bonus"] == 0.0
    assert submission_finish["bottom_position_bonus"] == 0.0
    assert submission_finish["intercept"] == -0.60


def test_empty_weight_class_mapping_is_neutral_and_fingerprint_stable():
    resolver = load_event_mc_config()
    assert resolver.for_weight_class("heavyweight") == resolver.for_weight_class()
    assert resolver.for_weight_class().fingerprint == resolver.for_weight_class().fingerprint


def test_partial_override_changes_only_requested_value(tmp_path: Path):
    document = yaml.safe_load(Path("config/event_mc_v1.yaml").read_text())
    document["weight_classes"] = {"synthetic": {"dynamic_modifiers": {"power_floor_low": 0.10}}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))
    resolver = load_event_mc_config(path)
    default, override = resolver.for_weight_class(), resolver.for_weight_class("synthetic")
    assert override.section("dynamic_modifiers")["power_floor_low"] == 0.10
    assert override.section("distance") == default.section("distance")
    state = FightState(red_stamina=0)
    assert DynamicModifierProvider(override).modifiers(profile(), state, Side.RED).power_multiplier > DynamicModifierProvider(default).modifiers(profile(), state, Side.RED).power_multiplier


def test_loading_config_does_not_consume_numpy_rng():
    import numpy as np
    expected = np.random.default_rng(77).random()
    rng = np.random.default_rng(77)
    load_event_mc_config()
    assert rng.random() == expected


def test_resolved_override_threads_through_distance_clinch_stamina_and_physiology(tmp_path: Path):
    document = yaml.safe_load(Path("config/event_mc_v1.yaml").read_text())
    document["weight_classes"] = {"synthetic": {
        "distance": {"strike_attempts_per_30s": 12.0},
        "clinch": {"strike_attempts_per_30s": 7.2},
        "stamina": {"action_costs": {"strike": 1.4}},
        "damage": {"impact_scale": 1.0},
    }}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))
    resolver = load_event_mc_config(path)
    default, override = resolver.for_weight_class(), resolver.for_weight_class("synthetic")
    profiles = MatchupProfiles(profile(), profile())
    context = FightContext(FightConfig(), 0, 1)
    def rates(calibration, state):
        stamina = StaminaModel(profiles, calibration=calibration)
        provider = FightFlowRateProvider(profiles, stamina, DynamicModifierProvider(calibration), calibration)
        return {item.candidate.candidate_id: item.rate_per_second for item in provider.candidates(state, context)}
    assert rates(override, FightState())["red_strike"] == 2 * rates(default, FightState())["red_strike"]
    clinch = FightState(phase=__import__("pipeline.simulation.event_mc_v1.state", fromlist=["Phase"]).Phase.CLINCH, clinch_controller="red")
    assert rates(override, clinch)["red_clinch_strike"] == 2 * rates(default, clinch)["red_clinch_strike"]
    assert StaminaModel(profiles, calibration=override).action_delta(FightState(), Side.RED, "strike").red_stamina < StaminaModel(profiles, calibration=default).action_delta(FightState(), Side.RED, "strike").red_stamina
    assert ImpactTraumaKnockdownModel(profiles, override).calibration.section("damage")["impact_scale"] == 1.0
