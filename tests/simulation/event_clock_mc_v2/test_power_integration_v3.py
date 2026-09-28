from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v1.ko_kd_shadow import ShadowKOKDCalibration
from pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle import (
    legacy_power_equivalent,
    overlay_v3_power_on_frozen_profiles,
)


def test_v3_power_translation_preserves_native_kd_logit_effect():
    latent = np.array([-0.50, 0.0, 0.75])
    translated = legacy_power_equivalent(latent)
    beta = ShadowKOKDCalibration().kd_power_beta

    # Frozen KD hazard must see exactly the native V3 logit effect.
    implied = beta * (translated - 50.0)
    assert np.allclose(implied, latent, atol=1e-12)


def test_power_overlay_changes_only_striking_power():
    frozen = pd.DataFrame(
        [
            {
                "fight_id": "f1",
                "fighter_id": "A",
                "striking_power": 81.0,
                "knockdown_resistance": 63.0,
                "stamina_output": 55.0,
            },
            {
                "fight_id": "f1",
                "fighter_id": "B",
                "striking_power": 42.0,
                "knockdown_resistance": 48.0,
                "stamina_output": 61.0,
            },
        ]
    )
    fsr_v3 = pd.DataFrame(
        [
            {"fight_id": "f1", "fighter_id": "A", "striking_power_v3": 0.40},
            {"fight_id": "f1", "fighter_id": "B", "striking_power_v3": -0.25},
        ]
    )

    out = overlay_v3_power_on_frozen_profiles(frozen, fsr_v3)
    assert "striking_power_v3" not in out.columns
    assert np.allclose(out["knockdown_resistance"], frozen["knockdown_resistance"])
    assert np.allclose(out["stamina_output"], frozen["stamina_output"])
    assert not np.allclose(out["striking_power"], frozen["striking_power"])

    beta = ShadowKOKDCalibration().kd_power_beta
    expected = 50.0 + fsr_v3["striking_power_v3"].to_numpy(float) / beta
    assert np.allclose(out["striking_power"], expected)
