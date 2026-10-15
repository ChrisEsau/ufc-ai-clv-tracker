"""Run frozen 34-fight MC with degree-2 Kline forecasts over selectable history windows.

Variants
--------
last3
    Use exactly the latest 3 PREFIGHT FSR observations through the target-fight
    prefight snapshot, fit degree-2, extrapolate one sequence point.

last6
    Use up to the latest 6 PREFIGHT FSR observations through the target-fight
    prefight snapshot. If 3-5 observations exist, use all available. Fit degree-2
    and extrapolate one sequence point.

all_but_initial
    Use all PREFIGHT FSR observations through the target-fight prefight snapshot
    except the fighter's first chronological FSR observation (the initialization
    anchor at 50). Fit degree-2 and extrapolate one sequence point.

For all variants, fewer than 3 usable observations falls back to the stored target
prefight FSR. Forecasts are clipped only to [10, 90] for MC input. Target prefight
FSR is included and is leakage-safe by project contract. No age modifier is used.
Research only; stored FSR and simulator configuration are not changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_canonical_database as canonical
from scripts.experimental import run_34fight_poly2_fsr_mc_test as experiment


VARIANTS = {
    "last3": "validation_poly2_last3_fsr_mc",
    "last6": "validation_poly2_last6_fsr_mc",
    "all_but_initial": "validation_poly2_all_but_initial_fsr_mc",
}


def _fit_next_poly2(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    if len(vals) < 3:
        raise ValueError("degree-2 fit requires at least 3 observations")
    x = np.arange(1, len(vals) + 1, dtype=float)
    coeff = np.polyfit(x, vals, deg=2)
    return float(np.poly1d(coeff)(float(len(vals) + 1)))


def _select_values(vals: np.ndarray, variant: str) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    if variant == "last3":
        return vals[-3:]
    if variant == "last6":
        return vals[-6:]
    if variant == "all_but_initial":
        return vals[1:]
    raise ValueError(f"unknown variant: {variant}")


def _make_forecaster(variant: str):
    def _forecast_profile(
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
            current = pd.to_numeric(
                pd.Series([profile.get(trait)]), errors="coerce"
            ).iloc[0]
            if pd.isna(current):
                raise RuntimeError(f"{fighter_name} profile missing numeric {trait}")
            current = float(current)

            all_vals = pd.to_numeric(hist[trait], errors="coerce").dropna().to_numpy(dtype=float)
            usable = _select_values(all_vals, variant)

            method = "latest"
            raw = current
            mc_value = current
            clipped = 0

            if len(usable) >= 3 and np.isfinite(usable).all():
                raw = _fit_next_poly2(usable)
                if np.isfinite(raw):
                    mc_value = float(np.clip(raw, experiment.FSR_MIN, experiment.FSR_MAX))
                    clipped = int(mc_value != raw)
                    # The inherited driver counts method == 'poly2'. Keep that contract
                    # while recording the exact variant separately.
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
                "forecast_variant": variant,
                "forecast_method": "poly2" if method == "poly2" else "latest",
                "method": method,
                "aligned_latest_fsr": current,
                "raw_poly2_forecast": float(raw),
                "mc_fsr": float(mc_value),
                "raw_delta": float(raw - current),
                "mc_delta": float(mc_value - current),
                "clipped_to_fsr_range": clipped,
            })

        return predicted, audit

    return _forecast_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        required=True,
        choices=sorted(VARIANTS),
        help="Quadratic history window to use.",
    )
    args, remaining = parser.parse_known_args()

    variant = args.variant
    output_dir = Path("data/experimental") / VARIANTS[variant]
    experiment.OUTPUT_DIR = output_dir
    experiment.FORECAST_AUDIT_PATH = output_dir / "fighter_trait_forecasts.csv"
    experiment.OUTPUT_PATH = output_dir / f"fsr_mc_card_validation_{variant}_poly2_v1.csv"
    experiment.SUMMARY_PATH = output_dir / f"fsr_mc_card_validation_{variant}_poly2_v1_summary.csv"
    experiment._forecast_profile = _make_forecaster(variant)

    # Remove our argument before delegating to inherited argparse.
    import sys
    sys.argv = [sys.argv[0], *remaining]

    descriptions = {
        "last3": "latest 3 prefight points",
        "last6": "up to latest 6 prefight points",
        "all_but_initial": "all prefight points except first initialization point",
    }
    print(
        f"[poly2-window-34] variant={variant}: {descriptions[variant]}; "
        "degree-2 fit -> N+1 fight-night FSR.",
        flush=True,
    )
    print(
        "[poly2-window-34] target-fight prefight point INCLUDED; no age modifier; "
        "inherited console labels may simply say 'poly2'.",
        flush=True,
    )
    experiment.main()


if __name__ == "__main__":
    main()
