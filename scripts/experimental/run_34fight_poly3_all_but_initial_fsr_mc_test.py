"""Run frozen 34-fight MC with degree-3 Kline forecasts excluding the initial FSR point.

For every fighter/trait:
- include the target-fight PREFIGHT FSR snapshot
- use all prefight FSR observations through the target fight except the fighter's
  first chronological initialization point
- fit degree-3 over UFC fight-sequence index
- extrapolate one N+1 point as fight-night FSR
- clip only to [10, 90] for MC input
- if fewer than 4 usable observations remain, fall back to stored target prefight FSR
- no age modifier

Research only; stored FSR and simulator configuration are unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_canonical_database as canonical
from scripts.experimental import run_34fight_poly2_fsr_mc_test as experiment


OUTPUT_DIR = Path("data/experimental/validation_poly3_all_but_initial_fsr_mc")
experiment.OUTPUT_DIR = OUTPUT_DIR
experiment.FORECAST_AUDIT_PATH = OUTPUT_DIR / "fighter_trait_forecasts.csv"
experiment.OUTPUT_PATH = OUTPUT_DIR / "fsr_mc_card_validation_all_but_initial_poly3_v1.csv"
experiment.SUMMARY_PATH = OUTPUT_DIR / "fsr_mc_card_validation_all_but_initial_poly3_v1_summary.csv"


def _fit_next_poly3(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    if len(vals) < 4:
        raise ValueError("degree-3 fit requires at least 4 observations")
    x = np.arange(1, len(vals) + 1, dtype=float)
    coeff = np.polyfit(x, vals, deg=3)
    return float(np.poly1d(coeff)(float(len(vals) + 1)))


def _forecast_profile_poly3_all_but_initial(
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

        all_vals = pd.to_numeric(hist[trait], errors="coerce").dropna().to_numpy(dtype=float)
        usable = all_vals[1:]

        method = "latest"
        raw = current
        mc_value = current
        clipped = 0

        if len(usable) >= 4 and np.isfinite(usable).all():
            raw = _fit_next_poly3(usable)
            if np.isfinite(raw):
                mc_value = float(np.clip(raw, experiment.FSR_MIN, experiment.FSR_MAX))
                clipped = int(mc_value != raw)
                # inherited driver counts method == 'poly2'; preserve counter semantics
                # while the audit records the actual cubic method explicitly.
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
            "history_n": int(len(all_vals)),
            "fit_n": int(len(usable)),
            "target_prefight_included": 1,
            "forecast_variant": "all_but_initial",
            "forecast_method": "poly3" if method == "poly2" else "latest",
            "method": method,
            "aligned_latest_fsr": current,
            "raw_poly2_forecast": float(raw),
            "mc_fsr": float(mc_value),
            "raw_delta": float(raw - current),
            "mc_delta": float(mc_value - current),
            "clipped_to_fsr_range": clipped,
        })

    return predicted, audit


experiment._forecast_profile = _forecast_profile_poly3_all_but_initial


if __name__ == "__main__":
    print(
        "[poly3-all-but-initial-34] all prefight points except first initialization point; "
        "degree-3 fit -> N+1 fight-night FSR.",
        flush=True,
    )
    print(
        "[poly3-all-but-initial-34] target-fight prefight point INCLUDED; no age modifier; "
        "fewer than 4 usable observations fall back to stored prefight FSR.",
        flush=True,
    )
    experiment.main()
