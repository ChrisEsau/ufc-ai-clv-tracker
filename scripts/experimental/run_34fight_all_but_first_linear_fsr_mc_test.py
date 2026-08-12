"""Run frozen 34 fights with linear FSR trends excluding the initial 50 anchor.

Research-only trajectory diagnostic.

For every fighter/trait:
- include all PREFIGHT FSR snapshots through the target-fight prefight row
- drop the fighter's first FSR observation (the universal 50 initialization anchor)
- fit a linear trend over every remaining observation
- extrapolate one sequence step as fight-night FSR
- clip only to valid FSR [10, 90]
- if fewer than 2 observations remain after dropping the first point, keep stored prefight FSR

The frozen 34 bouts, simulator, paths, seed stream, and no-age setup are inherited
from run_34fight_poly2_fsr_mc_test.py. Outputs are isolated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_canonical_database as canonical
from scripts.experimental import run_34fight_poly2_fsr_mc_test as experiment


OUTPUT_DIR = Path("data/experimental/validation_all_but_first_linear_fsr_mc")
experiment.OUTPUT_DIR = OUTPUT_DIR
experiment.FORECAST_AUDIT_PATH = OUTPUT_DIR / "fighter_trait_forecasts.csv"
experiment.OUTPUT_PATH = OUTPUT_DIR / "fsr_mc_card_validation_all_but_first_linear_v1.csv"
experiment.SUMMARY_PATH = OUTPUT_DIR / "fsr_mc_card_validation_all_but_first_linear_v1_summary.csv"


def _fit_next_all_but_first_linear(values: np.ndarray) -> float:
    """Drop the first observation, fit a line to all remaining values, predict N+1."""
    vals = np.asarray(values, dtype=float)
    if len(vals) < 3:
        raise ValueError("need at least 3 total observations so 2 remain after dropping first")
    vals = vals[1:]
    x = np.arange(1, len(vals) + 1, dtype=float)
    coeff = np.polyfit(x, vals, deg=1)
    return float(np.poly1d(coeff)(float(len(vals) + 1)))


def _forecast_profile_all_but_first(
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
            raw = _fit_next_all_but_first_linear(vals)
            if np.isfinite(raw):
                mc_value = float(np.clip(raw, experiment.FSR_MIN, experiment.FSR_MAX))
                clipped = int(mc_value != raw)
                # Preserve inherited summary machinery; explicit forecast_method is authoritative.
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
            "history_n_total": int(len(vals)),
            "history_n_fit": int(max(len(vals) - 1, 0)),
            "initial_anchor_dropped": int(len(vals) >= 1),
            "target_prefight_included": 1,
            "forecast_method": "all_but_first_linear" if len(vals) >= 3 else "latest",
            "method": method,
            "aligned_latest_fsr": current,
            "raw_poly2_forecast": float(raw),
            "mc_fsr": float(mc_value),
            "raw_delta": float(raw - current),
            "mc_delta": float(mc_value - current),
            "clipped_to_fsr_range": clipped,
        })

    return predicted, audit


experiment._forecast_profile = _forecast_profile_all_but_first


if __name__ == "__main__":
    print(
        "[all-but-first-linear-34] dropping each fighter's first PREFIGHT FSR point "
        "(universal 50 initialization), fitting a line through all remaining points "
        "through the target prefight state, then predicting N+1.",
        flush=True,
    )
    print(
        "[all-but-first-linear-34] inherited console labels may still say 'poly2'; "
        "this run is linear and outputs are isolated.",
        flush=True,
    )
    experiment.main()
