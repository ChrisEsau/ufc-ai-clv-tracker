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


def profile():
    return FighterProfile("fighter", "fighter", 50, 50, 50, 50, 50, 50, 50, 50)


def test_yaml_defaults_are_active_unchanged_values():
    distance = DEFAULT_CALIBRATION.section("distance")
    assert distance["strike_attempts_per_30s"] == DISTANCE_STRIKE_ATTEMPTS_PER_30S_BASE == 5.0
    assert distance["strike_accuracy"] == DISTANCE_STRIKE_ACCURACY_BASE == 0.40
    assert distance["td_attempt_base_30s"] == DISTANCE_TD_ATTEMPT_BASE_30S == 0.10


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
