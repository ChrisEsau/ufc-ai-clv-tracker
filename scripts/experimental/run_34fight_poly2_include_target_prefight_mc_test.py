"""Run frozen 34 fights using Kline-style poly2 forecasts through target prefight state.

This is the intended Kline-style diagnostic. The FSR row labeled with the target
fight is a PREFIGHT snapshot: it represents demonstrated state entering that
fight and contains no target-fight outcome information. Therefore that target
prefight point is included in the degree-2 fit, and one additional fight-sequence
point is extrapolated as the estimated fight-night FSR.

This launcher deliberately preserves the earlier strict-before-target diagnostic
and writes separate outputs.

Research only. No stored FSR, age modifier, or simulator configuration is changed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_canonical_database as canonical
from scripts.experimental import run_34fight_poly2_fsr_mc_test as experiment


OUTPUT_DIR = Path("data/experimental/validation_poly2_include_target_prefight_mc")
experiment.OUTPUT_DIR = OUTPUT_DIR
experiment.FORECAST_AUDIT_PATH = OUTPUT_DIR / "fighter_trait_forecasts.csv"
experiment.OUTPUT_PATH = OUTPUT_DIR / "fsr_mc_card_validation_poly2_include_target_v1.csv"
experiment.SUMMARY_PATH = OUTPUT_DIR / "fsr_mc_card_validation_poly2_include_target_v1_summary.csv"


def _forecast_profile_include_target(
    profile: pd.Series,
    fsr: pd.DataFrame,
    target_fight_id: str,
    target_date: pd.Timestamp,
    fighter_name: str,
) -> tuple[pd.Series, list[dict[str, object]]]:
    """Fit through the target-fight PREFIGHT snapshot and extrapolate one step."""
    fighter_id = experiment._fighter_id(profile)

    # The target fight's FSR row is a leakage-safe PREFIGHT state. Include it,
    # along with every earlier snapshot, exactly like the original Kline plot.
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
            raw = experiment._fit_next_poly2(vals)
            if np.isfinite(raw):
                mc_value = float(np.clip(raw, experiment.FSR_MIN, experiment.FSR_MAX))
                clipped = int(mc_value != raw)
                method = "poly2_include_target_prefight"
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
            "method": method,
            "aligned_latest_fsr": current,
            "raw_poly2_forecast": float(raw),
            "mc_fsr": float(mc_value),
            "raw_delta": float(raw - current),
            "mc_delta": float(mc_value - current),
            "clipped_to_fsr_range": clipped,
        })

    return predicted, audit


# Patch only the profile forecast contract. The frozen bouts, seed stream,
# simulator, scoring, and summary logic stay identical to the first diagnostic.
experiment._forecast_profile = _forecast_profile_include_target


if __name__ == "__main__":
    print(
        "[poly2-34 include-target] target-labeled PREFIGHT FSR is INCLUDED in fit; "
        "forecasting one additional sequence point for fight-night state.",
        flush=True,
    )
    experiment.main()
