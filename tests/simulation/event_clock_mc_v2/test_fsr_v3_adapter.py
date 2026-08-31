from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    SAMPLABLE_EPISTEMIC_TRAITS,
    derive_runtime_inputs,
    initialize_fighter_path_traits,
    initialize_path_matchup,
)


def _fighter(fid: str, **overrides):
    row = {
        "fighter_id": fid,
        "standing_striking_tendency": 180.0,
        "standing_striking_suppression": 1.0,
        "standing_striking_offense": 0.20,
        "standing_striking_defense": 0.10,
        "standing_accuracy_baseline": 0.45,
        "takedown_tendency": 6.0,
        "takedown_suppression": 1.0,
        "takedown_offense": 0.20,
        "takedown_defense": 0.10,
        "takedown_completion_baseline": 0.35,
        "ground_striking_tendency": 40.0,
        "ground_striking_suppression": 1.0,
        "ground_striking_offense": 0.20,
        "ground_accuracy_baseline": 0.70,
        "ground_striking_burst_baseline": 2.2,
    }
    row.update(overrides)
    return row


def _uncertainty(fid: str):
    rows = []
    means = {
        "takedown_tendency": 6.0,
        "takedown_suppression": 1.0,
        "standing_striking_tendency": 180.0,
        "standing_striking_suppression": 1.0,
        "takedown_offense": 0.2,
        "standing_striking_offense": 0.2,
        "ground_striking_tendency": 40.0,
    }
    sds = {
        "takedown_tendency": 2.0,
        "takedown_suppression": 0.25,
        "standing_striking_tendency": 30.0,
        "standing_striking_suppression": 0.15,
        "takedown_offense": 0.20,
        "standing_striking_offense": 0.20,
        "ground_striking_tendency": 15.0,
    }
    for trait, mean in means.items():
        enabled = trait in SAMPLABLE_EPISTEMIC_TRAITS
        rows.append(
            {
                "fighter_id": fid,
                "trait": trait,
                "posterior_mean": mean,
                "posterior_sd": sds[trait],
                "variance_multiplier": 1.0 if enabled else 0.0,
                "sampling_enabled": enabled,
            }
        )
    return pd.DataFrame(rows)


def test_means_only_uses_canonical_trait_values_exactly():
    row = _fighter("A")
    traits = initialize_fighter_path_traits(
        row,
        _uncertainty("A"),
        rng=np.random.default_rng(123),
        sample_epistemic=False,
    )
    assert not traits.epistemic_sampled
    for trait in SAMPLABLE_EPISTEMIC_TRAITS:
        assert traits[trait] == row[trait]


def test_epistemic_draws_are_positive_and_only_touch_validated_traits():
    row = _fighter("A")
    uncertainty = _uncertainty("A")
    traits = initialize_fighter_path_traits(
        row,
        uncertainty,
        rng=np.random.default_rng(20260820),
        sample_epistemic=True,
    )
    assert traits.epistemic_sampled
    for trait in SAMPLABLE_EPISTEMIC_TRAITS:
        assert traits[trait] > 0.0
    assert traits["takedown_offense"] == row["takedown_offense"]
    assert traits["standing_striking_offense"] == row["standing_striking_offense"]
    assert traits["ground_striking_tendency"] == row["ground_striking_tendency"]


def test_path_draw_is_deterministic_for_same_seed():
    row = _fighter("A")
    uncertainty = _uncertainty("A")
    a = initialize_fighter_path_traits(
        row,
        uncertainty,
        rng=np.random.default_rng(99),
        sample_epistemic=True,
    )
    b = initialize_fighter_path_traits(
        row,
        uncertainty,
        rng=np.random.default_rng(99),
        sample_epistemic=True,
    )
    for trait in SAMPLABLE_EPISTEMIC_TRAITS:
        assert a[trait] == b[trait]


def test_standing_suppression_is_multiplicative_and_lower_is_better():
    attacker = initialize_fighter_path_traits(
        _fighter("A", standing_striking_tendency=200.0),
        None,
        rng=np.random.default_rng(1),
        sample_epistemic=False,
    )
    neutral = initialize_fighter_path_traits(
        _fighter("B", standing_striking_suppression=1.0),
        None,
        rng=np.random.default_rng(2),
        sample_epistemic=False,
    )
    suppressor = initialize_fighter_path_traits(
        _fighter("C", standing_striking_suppression=0.6),
        None,
        rng=np.random.default_rng(3),
        sample_epistemic=False,
    )
    assert derive_runtime_inputs(attacker, neutral).standing_rate_15m == 200.0
    assert derive_runtime_inputs(attacker, suppressor).standing_rate_15m == 120.0


def test_takedown_suppression_is_multiplicative_and_lower_is_better():
    attacker = initialize_fighter_path_traits(
        _fighter("A", takedown_tendency=8.0), None,
        rng=np.random.default_rng(1), sample_epistemic=False,
    )
    defender = initialize_fighter_path_traits(
        _fighter("B", takedown_suppression=0.5), None,
        rng=np.random.default_rng(2), sample_epistemic=False,
    )
    assert derive_runtime_inputs(attacker, defender).takedown_rate_15m == 4.0


def test_ground_suppression_scales_slope_but_never_burst():
    attacker = initialize_fighter_path_traits(
        _fighter(
            "A",
            ground_striking_tendency=45.0,
            ground_striking_burst_baseline=2.2,
        ),
        None,
        rng=np.random.default_rng(1),
        sample_epistemic=False,
    )
    neutral = initialize_fighter_path_traits(
        _fighter("B", ground_striking_suppression=1.0), None,
        rng=np.random.default_rng(2), sample_epistemic=False,
    )
    suppressor = initialize_fighter_path_traits(
        _fighter("C", ground_striking_suppression=0.5), None,
        rng=np.random.default_rng(3), sample_epistemic=False,
    )
    n = derive_runtime_inputs(attacker, neutral)
    s = derive_runtime_inputs(attacker, suppressor)
    assert n.ground_burst_attempts == 2.2
    assert s.ground_burst_attempts == 2.2
    assert n.ground_slope_rate_15m_own_control == 45.0
    assert s.ground_slope_rate_15m_own_control == 22.5
    assert n.ground_expected_attempts(0.0) == 2.2
    assert s.ground_expected_attempts(0.0) == 2.2
    assert n.ground_expected_attempts(900.0) == 47.2
    assert s.ground_expected_attempts(900.0) == 24.7


def test_ground_accuracy_has_no_defender_effectiveness_term():
    attacker = initialize_fighter_path_traits(
        _fighter("A", ground_striking_offense=0.4), None,
        rng=np.random.default_rng(1), sample_epistemic=False,
    )
    defender1 = initialize_fighter_path_traits(
        _fighter("B", standing_striking_defense=-1.0, takedown_defense=-1.0), None,
        rng=np.random.default_rng(2), sample_epistemic=False,
    )
    defender2 = initialize_fighter_path_traits(
        _fighter("C", standing_striking_defense=1.0, takedown_defense=1.0), None,
        rng=np.random.default_rng(3), sample_epistemic=False,
    )
    p1 = derive_runtime_inputs(attacker, defender1).ground_accuracy
    p2 = derive_runtime_inputs(attacker, defender2).ground_accuracy
    assert p1 == p2


def test_offense_and_defense_move_standing_and_td_success_in_expected_directions():
    attacker_low = initialize_fighter_path_traits(
        _fighter("A", standing_striking_offense=-0.2, takedown_offense=-0.2), None,
        rng=np.random.default_rng(1), sample_epistemic=False,
    )
    attacker_high = initialize_fighter_path_traits(
        _fighter("B", standing_striking_offense=0.5, takedown_offense=0.5), None,
        rng=np.random.default_rng(2), sample_epistemic=False,
    )
    defender_low = initialize_fighter_path_traits(
        _fighter("C", standing_striking_defense=-0.2, takedown_defense=-0.2), None,
        rng=np.random.default_rng(3), sample_epistemic=False,
    )
    defender_high = initialize_fighter_path_traits(
        _fighter("D", standing_striking_defense=0.5, takedown_defense=0.5), None,
        rng=np.random.default_rng(4), sample_epistemic=False,
    )
    low_off = derive_runtime_inputs(attacker_low, defender_low)
    high_off = derive_runtime_inputs(attacker_high, defender_low)
    assert high_off.standing_accuracy > low_off.standing_accuracy
    assert high_off.takedown_completion > low_off.takedown_completion
    weak_def = derive_runtime_inputs(attacker_low, defender_low)
    strong_def = derive_runtime_inputs(attacker_low, defender_high)
    assert strong_def.standing_accuracy < weak_def.standing_accuracy
    assert strong_def.takedown_completion < weak_def.takedown_completion


def test_path_matchup_draws_once_then_holds_immutable_values():
    path = initialize_path_matchup(
        _fighter("R"),
        _fighter("B"),
        _uncertainty("R"),
        _uncertainty("B"),
        rng=np.random.default_rng(77),
        sample_epistemic=True,
    )
    first = path.red["takedown_tendency"]
    second = path.red["takedown_tendency"]
    assert first == second
    assert path.red.epistemic_sampled
    assert path.blue.epistemic_sampled
