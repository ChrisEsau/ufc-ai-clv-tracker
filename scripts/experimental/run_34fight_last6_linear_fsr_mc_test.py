"""Run the frozen 34-fight card with recent-window linear Kline FSR forecasts.

Research-only trajectory diagnostic.

For every fighter/trait:
- include the target-fight PREFIGHT FSR snapshot (leakage-safe state entering fight)
- use up to the latest 6 prefight FSR observations through that target snapshot
- require at least 3 observations; with 3-5, use all available
- fit a simple linear trend over sequence order
- extrapolate one additional sequence point as the fight-night FSR estimate
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


OUTPUT_DIR = Path("data/experimental/validation_last6_linear_fsr_mc")
experiment.OUTPUT_DIR = OUTPUT_DIR
experiment.FORECAST_AUDIT_PATH = OUTPUT_DIR / "fighter_trait_forecasts.csv"
experiment.OUTPUT_PATH = OUTPUT_DIR / "fsr_mc_card_validation_last6_linear_v1.csv"
experiment.SUMMARY_PATH = OUTPUT_DIR / "fsr_mc_card_validation_last6_linear_v1_summary.csv"

MAX_RECENT = 6
MIN_HISTORY = 3


def _fit_next_recent6_linear(values: np.ndarray) -> tuple[float, int]:
    """Fit a line through the latest up-to-6 observations and predict one step ahead."""
    vals = np.asarray(values, dtype=float)
    if len(vals) < MIN_HISTORY:
        raise ValueError("recent-6 linear fit requires at least 3 observations")

    vals = vals[-MAX_RECENT:]
    n = len(vals)
    x = np.arange(1, n + 1, dtype=float)
    coeff = np.polyfit(x, vals, deg=1)
    predicted = float(np.poly1d(coeff)(float(n + 1)))
    return predicted, n


def _forecast_profile_last6(
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
        forecast_method = "latest"
        raw = current
        mc_value = current
        clipped = 0
        fit_n = 0

        if len(vals) >= MIN_HISTORY and np.isfinite(vals).all():
            raw, fit_n = _fit_next_recent6_linear(vals)
            if np.isfinite(raw):
                mc_value = float(np.clip(raw, experiment.FSR_MIN, experiment.FSR_MAX))
                clipped = int(mc_value != raw)
                # Keep inherited summary/console counters populated.
                method = "poly2"
                forecast_method = f"last{fit_n}_linear"
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
            "fit_n": int(fit_n),
            "target_prefight_included": 1,
            "forecast_method": forecast_method,
            "method": method,
            "aligned_latest_fsr": current,
            "raw_poly2_forecast": float(raw),
            "mc_fsr": float(mc_value),
            "raw_delta": float(raw - current),
            "mc_delta": float(mc_value - current),
            "clipped_to_fsr_range": clipped,
        })

    return predicted, audit


experiment._forecast_profile = _forecast_profile_last6


if __name__ == "__main__":
    print(
        "[last6-linear-34] using up to latest 6 PREFIGHT FSR observations through the target fight; "
        "linear N+1 forecast is used as fight-night FSR.",
        flush=True,
    )
    print(
        "[last6-linear-34] fighters with 3-5 observations use all available; fewer than 3 fall back to latest. "
        "Inherited console labels may still say 'poly2'; outputs are isolated.",
        flush=True,
    )
    experiment.main()
