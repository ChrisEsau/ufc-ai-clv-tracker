"""Targeted Ricci vs Kline FSR-32 trajectory forecast diagnostic.

Runs the existing leakage-safe raw + shrunk trajectory forecast logic only for
Tabatha Ricci and Fatima Kline, then prints the target-fight values.

Shadow/research only.
"""
from __future__ import annotations

import time

import pandas as pd

from scripts.experimental.fsr_32_trajectory_forecast_audit import (
    FSR_PATH,
    build_forecasts,
)

TARGET_FIGHT_ID = "52ddf20a10890b41"
RICCI_ID = "d25240135aee03e5"
KLINE_ID = "745fa7b605f8e2da"
FIGHTER_IDS = {RICCI_ID, KLINE_ID}


def main() -> None:
    started = time.perf_counter()
    print("[ricci-kline] loading FSR-32...", flush=True)
    fsr = pd.read_parquet(FSR_PATH)
    print(f"[ricci-kline] loaded {len(fsr):,} total fighter-fight rows", flush=True)

    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    fsr["fight_id"] = fsr["fight_id"].astype(str)
    subset = fsr.loc[fsr["fighter_id"].isin(FIGHTER_IDS)].copy()

    print(
        f"[ricci-kline] filtered to {len(subset):,} rows "
        f"({subset['fighter_id'].nunique()} fighters)",
        flush=True,
    )
    for fighter_id, g in subset.groupby("fighter_id"):
        name = g["fighter_name"].dropna().iloc[-1] if "fighter_name" in g.columns and g["fighter_name"].notna().any() else fighter_id
        print(f"[ricci-kline] {name}: {len(g)} prefight snapshots", flush=True)

    print("[ricci-kline] fitting raw + shrunk forecasts...", flush=True)
    raw, shrunk, audit, traits = build_forecasts(subset)
    print(
        f"[ricci-kline] forecast complete: {len(traits)} traits, "
        f"elapsed={time.perf_counter() - started:.2f}s",
        flush=True,
    )

    target = audit.loc[audit["fight_id"].eq(TARGET_FIGHT_ID)].copy()
    if target.empty:
        raise RuntimeError(f"Target fight {TARGET_FIGHT_ID} not found for Ricci/Kline")

    target = target.sort_values(["fighter_name", "trait"])
    target["current_fsr"] = pd.to_numeric(target["current_fsr"], errors="coerce")
    target["raw_forecast_fsr"] = pd.to_numeric(target["raw_forecast_fsr"], errors="coerce")
    target["shrunk_forecast_fsr"] = pd.to_numeric(target["shrunk_forecast_fsr"], errors="coerce")
    target["raw_delta"] = pd.to_numeric(target["raw_delta"], errors="coerce")
    target["shrunk_delta"] = pd.to_numeric(target["shrunk_delta"], errors="coerce")
    target["trend_slope_per_year"] = pd.to_numeric(target["trend_slope_per_year"], errors="coerce")

    print("\n" + "=" * 126)
    print("RICCI vs KLINE — TARGET-FIGHT FSR-32 TRAJECTORY FORECAST")
    print("=" * 126)

    cols = [
        "fighter_name",
        "trait",
        "current_fsr",
        "raw_forecast_fsr",
        "shrunk_forecast_fsr",
        "raw_delta",
        "shrunk_delta",
        "trend_slope_per_year",
        "history_n",
        "trend_r2",
        "shrink_lambda",
    ]
    print(
        target[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    summary = target.groupby("fighter_name", as_index=False).agg(
        mean_raw_delta=("raw_delta", "mean"),
        mean_shrunk_delta=("shrunk_delta", "mean"),
        mean_slope_per_year=("trend_slope_per_year", "mean"),
    )
    print("\nFIGHTER TRAJECTORY SUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n[ricci-kline] done in {time.perf_counter() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
