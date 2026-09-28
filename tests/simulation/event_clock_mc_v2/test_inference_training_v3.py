from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.inference import (
    _add_fitted_direct_predictions,
)


class _AttemptModel:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, x, exposure):
        return np.full(len(x), self.value, dtype=float) * np.asarray(exposure, dtype=float)


class _CompletionModel:
    def __init__(self, probability):
        self.probability = float(probability)

    def predict_probability(self, x):
        return np.full(len(x), self.probability, dtype=float)


class _ControlModel:
    def predict(self, x, exposure):
        # Deliberately overpredict control so the fight-level cap is exercised.
        expected = np.full(len(x), 800.0, dtype=float)
        return expected, np.ones(len(x)), np.ones(len(x))


def test_pair_training_predictions_are_attached_and_control_is_physically_capped():
    train = pd.DataFrame(
        {
            "fight_id": ["f1", "f1"],
            "side": ["red", "blue"],
            "duration": [900.0, 900.0],
        }
    )
    x = np.zeros((2, 1), dtype=float)
    exposure = np.ones(2, dtype=float)
    families = ("distance", "clinch", "ground", "td")
    attempts = {fam: _AttemptModel(10.0) for fam in families}
    completions = {fam: _CompletionModel(0.5) for fam in families}

    out = _add_fitted_direct_predictions(
        train,
        x,
        exposure,
        attempts,
        completions,
        _ControlModel(),
    )

    for fam in families:
        assert f"pred_{fam}_attempted" in out.columns
        assert f"pred_{fam}_landed" in out.columns
        assert np.allclose(out[f"pred_{fam}_attempted"], 10.0)
        assert np.allclose(out[f"pred_{fam}_landed"], 5.0)

    assert "pred_qualified_control_inflicted_seconds" in out.columns
    # 800 + 800 would exceed the 900-second fight, so the pair is rescaled.
    assert np.isclose(out["pred_qualified_control_inflicted_seconds"].sum(), 900.0)
    assert np.allclose(out["pred_qualified_control_inflicted_seconds"], 450.0)
    assert np.allclose(out["pred_standing_attempted"], 20.0)
    assert np.allclose(out["pred_standing_landed"], 10.0)
