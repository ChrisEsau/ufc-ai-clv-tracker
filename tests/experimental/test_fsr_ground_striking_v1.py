from __future__ import annotations

import pandas as pd

from scripts.experimental import fsr_ground_striking_v1 as ground


def _row(**overrides):
    base = {
        "rfs_phase_base_fight_ground_attempts_per_round": 4.0,
        "rfs_phase_base_fight_ground_attempt_share": 0.20,
        "rfs_phase_interact_fight_ground_attempts": 12.0,
        "rfs_phase_interact_fight_ground_accuracy": 0.55,
        "rfs_phase_interact_fight_opp_ground_attempts": 10.0,
        "rfs_phase_interact_fight_ground_accuracy_allowed": 0.45,
    }
    base.update(overrides)
    return pd.Series(base)


def test_observation_bundle_requires_ground_opportunity():
    pools = {key: [] for key in ground.POOL_KEYS}
    bundle = ground.observation_bundle(
        _row(
            rfs_phase_interact_fight_ground_attempts=0.0,
            rfs_phase_interact_fight_opp_ground_attempts=0.0,
        ),
        pools,
    )

    assert bundle["ground_striking_pressure"] == (None, 0.0)
    assert bundle["ground_striking_precision"] == (None, 0.0)
    assert bundle["ground_striking_defense"] == (None, 0.0)


def test_pressure_combines_rate_and_share():
    pools = {
        "ground_attempts_per_round": [1.0, 2.0, 3.0, 4.0, 5.0],
        "ground_attempt_share": [0.05, 0.10, 0.15, 0.20, 0.25],
        "ground_accuracy": [0.30, 0.40, 0.50, 0.60, 0.70],
        "ground_accuracy_allowed": [0.30, 0.40, 0.50, 0.60, 0.70],
    }
    observation, quality = ground.observation_bundle(_row(), pools)[
        "ground_striking_pressure"
    ]

    # Both components are at the 80th percentile in this pool.
    assert observation == 0.8
    assert 0.0 < quality < 1.0


def test_precision_and_defense_move_in_opposite_allowed_directions():
    pools = {
        "ground_attempts_per_round": [1.0],
        "ground_attempt_share": [0.1],
        "ground_accuracy": [0.30, 0.40, 0.50, 0.60],
        "ground_accuracy_allowed": [0.30, 0.40, 0.50, 0.60],
    }

    good = ground.observation_bundle(
        _row(
            rfs_phase_interact_fight_ground_accuracy=0.60,
            rfs_phase_interact_fight_ground_accuracy_allowed=0.30,
        ),
        pools,
    )
    poor = ground.observation_bundle(
        _row(
            rfs_phase_interact_fight_ground_accuracy=0.30,
            rfs_phase_interact_fight_ground_accuracy_allowed=0.60,
        ),
        pools,
    )

    assert good["ground_striking_precision"][0] > poor["ground_striking_precision"][0]
    assert good["ground_striking_defense"][0] > poor["ground_striking_defense"][0]


def test_equal_ratings_preserve_population_baseline():
    baseline = 0.37
    expected = ground.expected_matchup(50.0, 50.0, baseline)
    assert abs(expected - baseline) < 1e-12
