"""Canonical Event Clock C helpers for validated V3 KD resistance.

The frozen detailed KD hazard is not retuned.  Native V3 resistance is a logit
latent whose positive direction means more resistance.  We translate it into
the legacy profile coordinate so the frozen coefficient produces exactly the
same -resistance latent contribution, and sample the validated posterior once
per Monte Carlo path.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from pipeline.fsr_v3.paths import KD_RESISTANCE_HISTORY_PATH
from pipeline.simulation.event_mc_v1.components.profiles import MatchupProfiles
from pipeline.simulation.event_clock_mc_v2.physiology_adapter import legacy_kdres_equivalent


def load_kd_resistance_history() -> pd.DataFrame:
    frame = pd.read_parquet(KD_RESISTANCE_HISTORY_PATH).copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise").dt.normalize()
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    key = ["event_date", "fight_id", "fighter_id"]
    if frame.duplicated(key).any():
        raise ValueError("duplicate V3 KD-resistance history rows")
    return frame


def historical_kd_resistance_row(
    history: pd.DataFrame,
    *,
    event_date,
    fight_id: str,
    fighter_id: str,
) -> pd.Series:
    date = pd.Timestamp(event_date).normalize()
    matched = history[
        history["event_date"].eq(date)
        & history["fight_id"].eq(str(fight_id))
        & history["fighter_id"].eq(str(fighter_id))
    ]
    if len(matched) != 1:
        raise KeyError(
            f"expected one V3 KD-resistance row for fight={fight_id} fighter={fighter_id}; "
            f"found {len(matched)}"
        )
    return matched.iloc[0]


def sample_kd_resistance_latent(
    row: pd.Series,
    rng: np.random.Generator,
) -> float:
    mean = float(row["pre_rating"])
    sd = float(row["pre_posterior_sd"])
    multiplier = float(row.get("variance_multiplier", 1.0))
    validated = bool(row.get("validated_regime", True))
    if not validated or multiplier <= 0.0 or sd <= 0.0:
        return mean
    return float(rng.normal(mean, sd * np.sqrt(multiplier)))


def fight_with_kd_resistance(
    fight,
    *,
    red_native_resistance: float,
    blue_native_resistance: float,
):
    red = replace(
        fight.profiles.red,
        knockdown_resistance=float(legacy_kdres_equivalent(red_native_resistance)),
    )
    blue = replace(
        fight.profiles.blue,
        knockdown_resistance=float(legacy_kdres_equivalent(blue_native_resistance)),
    )
    profiles = MatchupProfiles(red=red, blue=blue)
    return replace(fight, profiles=profiles)
