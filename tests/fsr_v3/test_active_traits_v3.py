import numpy as np
import pandas as pd

from pipeline.fsr_v3.active_config import ActiveTraitConfig
from pipeline.simulation.event_clock_mc_v2.canonical_c import (
    legacy_kdres_equivalent,
    sample_kd_resistance_latent,
)


def test_active_trait_locked_constants():
    c = ActiveTraitConfig()
    assert c.escape_prior_entries == 8.0
    assert c.escape_variance_multiplier == 0.0
    assert c.kd_resistance_rho == 0.005
    assert c.kd_resistance_sigma == 0.70
    assert c.kd_resistance_variance_multiplier == 1.0


def test_kd_resistance_coordinate_preserves_native_logit_effect():
    c = ActiveTraitConfig()
    native = 0.14421
    rating = float(legacy_kdres_equivalent(native, c))
    frozen_effect = c.frozen_event_clock_kdres_beta * (rating - 50.0)
    assert np.isclose(frozen_effect, -native, atol=1e-12)


def test_kd_resistance_zero_latent_is_neutral_rating():
    assert np.isclose(float(legacy_kdres_equivalent(0.0)), 50.0)


def test_kd_resistance_sampling_uses_native_normal_posterior_once():
    row = pd.Series(
        {
            "pre_rating": 0.25,
            "pre_posterior_sd": 0.4,
            "variance_multiplier": 1.0,
            "validated_regime": True,
        }
    )
    expected = np.random.default_rng(123).normal(0.25, 0.4)
    actual = sample_kd_resistance_latent(row, np.random.default_rng(123))
    assert np.isclose(actual, expected)


def test_kd_resistance_unvalidated_history_is_mean_only():
    row = pd.Series(
        {
            "pre_rating": 0.25,
            "pre_posterior_sd": 0.4,
            "variance_multiplier": 1.0,
            "validated_regime": False,
        }
    )
    assert sample_kd_resistance_latent(row, np.random.default_rng(123)) == 0.25
