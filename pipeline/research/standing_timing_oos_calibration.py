"""Leakage-safe calibration of the standing action clock against historical attempts.

Research only. Fits one population timing scale mapping matchup-effective prefight
FSR standing attempt pace to observed UFCStats distance significant-strike attempts.
The scale is estimated strictly pre-2025 and evaluated untouched on 2025-2026.

This deliberately calibrates timing/cadence only. It does not change action choice,
accuracy, damage, KO, submissions, wrestling mechanics, or production code.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.diagnostics.stage8_structural_population import (
    ROUND_STATS, actual_side_totals, pick_col, side_rows,
)
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import load_prefight_snapshots

OUTDIR = Path("data/research/standing_timing_oos_calibration")
CUTOFF = pd.Timestamp("2025-01-01")
EPS = 1e-12


def build_frame() -> pd.DataFrame:
    rounds = pd.read_parquet(ROUND_STATS).copy()
    fc = pick_col(rounds, "fight_id", "bout_id")
    rounds[fc] = rounds[fc].astype(str)

    # One row per fighter-fight with actual distance significant-strike attempts.
    keys = []
    fight_ids = rounds[fc].dropna().astype(str).unique()
    fighter_col = pick_col(rounds, "fighter_id", "fighter")
    side_col = pick_col(rounds, "side", "corner")
    for fid in fight_ids:
        fr = rounds[rounds[fc].astype(str).eq(fid)]
        for fighter_id, g in fr.groupby(fighter_col, dropna=False):
            if pd.isna(fighter_id):
                continue
            sides = g[side_col].dropna().astype(str).str.lower().unique().tolist()
            if not sides:
                continue
            side = sides[0]
            try:
                vals = actual_side_totals(side_rows(rounds, fid, str(fighter_id), side))
            except Exception:
                continue
            # Fight elapsed exposure is shared by both sides; derive from round rows.
            elapsed = 0.0
            if "fight_elapsed_seconds" in g.columns:
                elapsed = float(pd.to_numeric(g["fight_elapsed_seconds"], errors="coerce").max())
            if not np.isfinite(elapsed) or elapsed <= 0:
                # Sum completed round exposure from per-round duration if available.
                for c in ("round_time_sec", "round_elapsed_seconds", "elapsed_seconds"):
                    if c in g.columns:
                        elapsed = float(pd.to_numeric(g[c], errors="coerce").fillna(0).sum())
                        break
            keys.append({"fight_id": fid, "fighter_id": str(fighter_id),
                         "actual_distance_attempts": float(vals["distance_att"]),
                         "elapsed_seconds_raw": elapsed})
    actual = pd.DataFrame(keys)

    snaps = load_prefight_snapshots().copy()
    snaps["fight_id"] = snaps["fight_id"].astype(str)
    snaps["fighter_id"] = snaps["fighter_id"].astype(str)
    snaps["event_date"] = pd.to_datetime(snaps["event_date"], errors="coerce").dt.normalize()

    # Locate the matchup-effective standing-rate column used by the simulator.
    candidates = [
        "standing_rate_15m", "standing_attempt_rate_15m", "standing_attempt_rate",
        "standing_rate", "sig_strike_attempt_rate_15m",
    ]
    rate_col = next((c for c in candidates if c in snaps.columns), None)
    if rate_col is None:
        # FSR V3 snapshot commonly stores component trait names rather than adapter names.
        # Try any unambiguous standing+rate column.
        matches = [c for c in snaps.columns if "standing" in c.lower() and "rate" in c.lower()]
        if len(matches) != 1:
            raise RuntimeError(f"could not identify standing-rate column; candidates={matches}")
        rate_col = matches[0]

    keep = snaps[["event_date", "fight_id", "fighter_id", rate_col]].rename(columns={rate_col:"fsr_standing_rate_15m"})
    f = actual.merge(keep, on=["fight_id", "fighter_id"], how="inner", validate="many_to_one")
    f["fsr_standing_rate_15m"] = pd.to_numeric(f["fsr_standing_rate_15m"], errors="coerce")

    # Prefer explicit elapsed exposure from snapshot if available; otherwise recover from round stats.
    elapsed_candidates = [c for c in snaps.columns if c in {"fight_elapsed_seconds", "elapsed_seconds"}]
    if elapsed_candidates:
        ex = snaps[["fight_id", "fighter_id", elapsed_candidates[0]]].rename(columns={elapsed_candidates[0]:"snapshot_elapsed_seconds"})
        f = f.merge(ex, on=["fight_id","fighter_id"], how="left")
        f["elapsed_seconds"] = pd.to_numeric(f["snapshot_elapsed_seconds"], errors="coerce")
    else:
        f["elapsed_seconds"] = f["elapsed_seconds_raw"]

    # If exposure is still missing, infer from number of round rows and last-round time fields.
    # Rows without trustworthy exposure are excluded rather than fabricated.
    f = f.replace([np.inf, -np.inf], np.nan).dropna(subset=["event_date", "fsr_standing_rate_15m", "elapsed_seconds"])
    f = f[(f.fsr_standing_rate_15m > 0) & (f.elapsed_seconds > 0)].copy()
    f["raw_expected_attempts"] = f.fsr_standing_rate_15m * f.elapsed_seconds / 900.0
    return f.sort_values(["event_date","fight_id","fighter_id"]).reset_index(drop=True), rate_col


def metrics(df: pd.DataFrame, scale: float) -> dict:
    pred = scale * df.raw_expected_attempts.to_numpy(float)
    y = df.actual_distance_attempts.to_numpy(float)
    err = pred - y
    return {
        "rows": int(len(df)),
        "actual_attempts": float(y.sum()),
        "predicted_attempts": float(pred.sum()),
        "E_over_O": float(pred.sum()/max(y.sum(), EPS)),
        "actual_mean_per_fighter_fight": float(y.mean()),
        "predicted_mean_per_fighter_fight": float(pred.mean()),
        "MAE_attempts": float(np.mean(np.abs(err))),
        "RMSE_attempts": float(np.sqrt(np.mean(err**2))),
        "correlation": float(np.corrcoef(y,pred)[0,1]) if len(y)>1 and np.std(y)>0 and np.std(pred)>0 else None,
        "actual_attempts_per_15_exposure": float(y.sum()/df.elapsed_seconds.sum()*900.0),
        "predicted_attempts_per_15_exposure": float(pred.sum()/df.elapsed_seconds.sum()*900.0),
    }


def main():
    f, rate_col = build_frame()
    train = f[f.event_date < CUTOFF].copy()
    holdout = f[f.event_date >= CUTOFF].copy()
    if train.empty or holdout.empty:
        raise RuntimeError(f"empty split train={len(train)} holdout={len(holdout)}")
    scale = float(train.actual_distance_attempts.sum()/max(train.raw_expected_attempts.sum(), EPS))
    result = {
        "study": "standing timing OOS calibration",
        "production_changed": False,
        "target": "fighter-fight UFCStats distance significant-strike attempts",
        "fit_cutoff": str(CUTOFF.date()),
        "fsr_rate_column": rate_col,
        "standing_rate_scale": scale,
        "mean_delay_multiplier_equivalent": 1.0/scale,
        "train": metrics(train, scale),
        "holdout_2025_2026": metrics(holdout, scale),
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR/"results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    f.assign(split=np.where(f.event_date < CUTOFF,"train","holdout")).to_csv(OUTDIR/"fighter_fight_predictions.csv", index=False)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
