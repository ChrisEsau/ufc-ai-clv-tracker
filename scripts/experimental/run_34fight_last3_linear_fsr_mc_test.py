"""Run the frozen 34-fight card with last-3 linear Kline FSR forecasts.

Research-only trajectory diagnostic.

For every fighter/trait:
- include the target-fight PREFIGHT FSR snapshot (leakage-safe state entering fight)
- take the latest 3 prefight FSR observations through that target snapshot
- fit a simple linear trend over sequence x = 1,2,3
- extrapolate x = 4 as the fight-night FSR estimate
- clip only to valid FSR [10, 90]
- if fewer than 3 observations exist, keep the stored target prefight FSR

The frozen 34 bouts, simulator, seeds, paths, and no-age setup are inherited from
run_34fight_poly2_fsr_mc_test.py. Outputs are written separately.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_canonical_database as canonical
from scripts.experimental import run_34fight_poly2_fsr_mc_test as experiment


OUTPUT_DIR = Path("data/experimental/validation_last3_linear_fsr_mc")
experiment.OUTPUT_DIR = OUTPUT_DIR
experiment.FORECAST_AUDIT_PATH = OUTPUT_DIR / "fighter_trait_forecasts.csv"
experiment.OUTPUT_PATH = OUTPUT_DIR / "fsr_mc_card_validation_last3_linear_v1.csv"
experiment.SUMMARY_PATH = OUTPUT_DIR / "fsr_mc_card_validation_last3_linear_v1_summary.csv"


def _fit_next_last3_linear(values: np.ndarray) -> float:
    """Fit a line through exactly the latest 3 sequence observations and predict N+1."""
    vals = np.asarray(values, dtype=float)
    if len(vals) < 3:
        raise ValueError("last-3 linear fit requires at least 3 observations")
    vals = vals[-3:]
    x = np.array([1.0, 2.0, 3.0])
    coeff = np.polyfit(x, vals, deg=1)
    return float(np.poly1d(coeff)(4.0))


def _forecast_profile_last3(
    profile: pd.Series,
    fsr: pd.DataFrame,
    target_fight_id: str,
    target_date: pd.Timestamp,
    fighter_name: str,
) -> tuple[pd.Series, list[dict[str, object]]]:
    fighter_id = experiment._fighter_id(profile)

    hist = fsr.loc[
        fsr["fighter_id"].eq(fighter_id)
        & (
            fsr["date"].lt(target_date)
            | fsr["fight_id"].eq(str(target_fight_id))
        )
    ].copy()
    hist = hist.sort_values(["date", "fight_id"]).reset_index(drop=True)

    target_rows = hist.loc[hist["fight_id"].eq(str(target_fight_id))]
    if len(target_rows) != 1:
        raise RuntimeError(
            f"{fighter_name}: expected exactly one target prefight FSR row for "
            f"{target_fight_id}, found {len(target_rows)}"
        )

    predicted = profile.copy(deep=True)
    audit: list[dict[str, object]] = []

    for trait in canonical.CANONICAL_RATINGS:
        current = pd.to_numeric(pd.Series([profile.get(trait)]), errors="coerce").iloc[0]
        if pd.isna(current):
            raise RuntimeError(f"{fighter_name} profile missing numeric {trait}")
        current = float(current)

        vals = pd.to_numeric(hist[trait], errors="coerce").dropna().to_numpy(dtype=float)
        method = "latest"
        raw = current
        mc_value = current
        clipped = 0

        if len(vals) >= 3 and np.isfinite(vals).all():
            raw = _fit_next_last3_linear(vals)
            if np.isfinite(raw):
                mc_value = float(np.clip(raw, experiment.FSR_MIN, experiment.FSR_MAX))
                clipped = int(mc_value != raw)
                # Keep the inherited summary counter populated; audit explicitly
                # identifies the actual forecast method below.
                method = "poly2"
            else:
                raw = current

        predicted[trait] = mc_value
        audit.append({
            "target_fight_id": str(target_fight_id),
            "target_date": target_date,
            "fighter_id": fighter_id,
            "fighter_name": fighter_name,
            "trait": trait,
            "history_n": int(len(vals)),
            "target_prefight_included": 1,
            "forecast_method": "last3_linear" if len(vals) >= 3 else "latest",
            "method": method,
            "aligned_latest_fsr": current,
            "raw_poly2_forecast": float(raw),
            "mc_fsr": float(mc_value),
            "raw_delta": float(raw - current),
            "mc_delta": float(mc_value - current),
            "clipped_to_fsr_range": clipped,
        })

    return predicted, audit


experiment._forecast_profile = _forecast_profile_last3


if __name__ == "__main__":
    print(
        "[last3-linear-34] using latest 3 PREFIGHT FSR observations through the target fight; "
        "linear N+1 forecast is used as fight-night FSR.",
        flush=True,
    )
    print(
        "[last3-linear-34] NOTE: inherited console labels may still say 'poly2'; "
        "all forecasts in this run are last-3 linear and outputs are isolated.",
        flush=True,
    )
    experiment.main()
