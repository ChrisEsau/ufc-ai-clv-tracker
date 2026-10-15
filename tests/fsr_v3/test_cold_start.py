from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.fsr_v3.cold_start.features import build_external_feature_snapshots
from pipeline.fsr_v3.cold_start.mma_global import add_leakage_safe_elo
from pipeline.fsr_v3.cold_start.model import ColdStartNB2RateModel, calibrate_extra_evidence_seconds
from pipeline.fsr_v3.cold_start.priors import combine_positive_rate_prior


def _bout(*, fight_id: str, date: str, organization: str, fighter: str, opponent: str) -> dict:
    return {
        "fight_id": fight_id,
        "event_date": date,
        "organization": organization,
        "is_major_org": False,
        "fighter_name": fighter,
        "opponent_name": opponent,
        "result": "W",
        "method_class": "Decision",
        "round_num": 3,
        "time_finish_seconds": 300,
        "fighter_height_cm": 180.0,
        "fighter_weight_kg": 77.0,
        "opponent_height_cm": 178.0,
        "opponent_weight_kg": 77.0,
        "fighter_pre_elo": 1500.0,
        "opponent_pre_elo": 1500.0,
        "fighter_post_elo": 1512.0,
    }


def test_zero_external_strength_is_exact_population_prior():
    prior = combine_positive_rate_prior(
        population_mean_rate_15m=37.5,
        population_seconds=468.48,
        external_mean_rate_15m=100.0,
        extra_seconds=0.0,
    )
    assert prior.mean_rate_15m == 37.5
    assert prior.total_seconds == 468.48
    assert prior.external_seconds == 0.0
    assert np.isclose(prior.shape, 37.5 * 468.48 / 900.0)
    assert np.isclose(prior.rate, 468.48 / 900.0)


def test_external_strength_reduces_prior_sd_and_moves_mean():
    base = combine_positive_rate_prior(population_mean_rate_15m=30.0, population_seconds=90.0)
    cold = combine_positive_rate_prior(
        population_mean_rate_15m=30.0,
        population_seconds=90.0,
        external_mean_rate_15m=60.0,
        extra_seconds=180.0,
    )
    assert 30.0 < cold.mean_rate_15m < 60.0
    assert cold.total_seconds == 270.0
    assert cold.sd_rate_15m < base.sd_rate_15m


def test_feature_snapshots_exclude_ufc_and_future_bouts():
    bouts = pd.DataFrame(
        [
            _bout(fight_id="old", date="2020-01-01", organization="LFA", fighter="A Fighter", opponent="Old Opp"),
            _bout(fight_id="ufc", date="2020-06-01", organization="UFC", fighter="A Fighter", opponent="UFC Opp"),
            _bout(fight_id="future", date="2022-01-01", organization="PFL", fighter="A Fighter", opponent="Future Opp"),
        ]
    )
    targets = pd.DataFrame([{"fighter_name": "A Fighter", "as_of_date": "2021-01-01", "fight_id": "target"}])
    row = build_external_feature_snapshots(targets, bouts).iloc[0]
    assert row["ext_bouts"] == 1
    assert row["ext_wins"] == 1
    assert row["ext_lfa_bouts"] == 1
    assert row["ext_pfl_bouts"] == 0
    assert row["evidence_bucket"] == "1_2"


def test_cross_promotion_elo_uses_same_date_delayed_states():
    wide = pd.DataFrame(
        [
            {"fight_id": "a", "event_date": "2020-01-01", "fighter_1": "Fighter X", "fighter_2": "Fighter Y", "winner": "Fighter X"},
            {"fight_id": "b", "event_date": "2020-01-01", "fighter_1": "Fighter X", "fighter_2": "Fighter Z", "winner": "Fighter X"},
            {"fight_id": "c", "event_date": "2020-02-01", "fighter_1": "Fighter X", "fighter_2": "Fighter Q", "winner": "Fighter Q"},
        ]
    )
    scored = add_leakage_safe_elo(wide)
    same_day = scored[scored["event_date"] == pd.Timestamp("2020-01-01")]
    assert np.allclose(same_day["f1_pre_elo"], 1500.0)
    later = scored[scored["fight_id"] == "c"].iloc[0]
    assert later["f1_pre_elo"] > 1500.0


def test_nb2_external_model_learns_positive_rate_signal():
    rng = np.random.default_rng(17)
    n = 240
    x = np.linspace(0.0, 10.0, n)
    q_pop = np.full(n, 20.0)
    rate = q_pop * np.exp(0.07 * (x - x.mean()))
    y = rng.poisson(rate)
    frame = pd.DataFrame(
        {
            "ext_bouts": x,
            "numerator": y,
            "exposure_seconds": 900.0,
            "population_rate_15m": q_pop,
            "observation_alpha": 0.05,
        }
    )
    model = ColdStartNB2RateModel(feature_columns=("ext_bouts",), ridge_alpha=1.0).fit(frame)
    predicted = model.predict_rate(frame)
    assert model.coefficients_[0] > 0.0
    assert predicted[-1] > predicted[0]


def test_strength_calibration_returns_only_tested_strength():
    frame = pd.DataFrame(
        {
            "evidence_bucket": ["3_5"] * 40,
            "external_predicted_rate_15m": [20.0] * 40,
            "numerator": [20.0] * 40,
            "exposure_seconds": [900.0] * 40,
            "population_rate_15m": [20.0] * 40,
            "observation_alpha": [0.2] * 40,
        }
    )
    chosen, table = calibrate_extra_evidence_seconds(
        frame,
        population_seconds=90.0,
        grid=np.linspace(0.1, 80.0, 800),
        candidates=(0.0, 90.0, 180.0),
    )
    assert chosen["3_5"] in {0.0, 90.0, 180.0}
    assert set(table["extra_seconds"]) == {0.0, 90.0, 180.0}
