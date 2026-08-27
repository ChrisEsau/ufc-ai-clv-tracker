"""Leakage-safe calibration of the standing action clock against historical attempts.

Research only. Fits one population timing scale mapping the exact matchup-effective
prefight FSR V3 standing rate (attacker tendency x defender suppression) to observed
UFCStats distance significant-strike attempts. Fit is strictly pre-2025; 2025-2026
is untouched holdout.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.diagnostics.stage8_structural_population import (
    MASTER, ROUND_STATS, actual_side_totals, elapsed_seconds, pick_col, side_rows,
)
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import load_prefight_snapshots

OUTDIR = Path("data/research/standing_timing_oos_calibration")
CUTOFF = pd.Timestamp("2025-01-01")
EPS = 1e-12


def build_frame() -> pd.DataFrame:
    master = pd.read_parquet(MASTER).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    dc = pick_col(master, "date", "event_date")
    master["event_date"] = pd.to_datetime(master[dc], errors="coerce").dt.normalize()
    master = master.dropna(subset=["event_date"])

    rounds = pd.read_parquet(ROUND_STATS).copy()
    fc = pick_col(rounds, "fight_id", "bout_id")
    rounds[fc] = rounds[fc].astype(str)
    available = set(rounds[fc].unique())

    snaps = load_prefight_snapshots().copy()
    required = {"standing_striking_tendency", "standing_striking_suppression"}
    missing = required.difference(snaps.columns)
    if missing:
        raise RuntimeError(f"missing FSR standing traits: {sorted(missing)}")
    snap_groups = {fid: g.set_index("fighter_id", drop=False) for fid, g in snaps.groupby("fight_id")}

    rows_out = []
    for fight in master.itertuples(index=False):
        fid = str(fight.fight_id)
        if fid not in available or fid not in snap_groups:
            continue
        rid, bid = str(fight.r_id), str(fight.b_id)
        sg = snap_groups[fid]
        if rid not in sg.index or bid not in sg.index:
            continue
        red, blue = sg.loc[rid], sg.loc[bid]
        # Guard against accidental same-fight-id duplicates across dates.
        if isinstance(red, pd.DataFrame) or isinstance(blue, pd.DataFrame):
            continue
        horizon = elapsed_seconds(pd.Series(fight._asdict()))
        if not np.isfinite(horizon) or horizon <= 0:
            continue
        try:
            red_actual = actual_side_totals(side_rows(rounds, fid, rid, "red"))["distance_att"]
            blue_actual = actual_side_totals(side_rows(rounds, fid, bid, "blue"))["distance_att"]
        except Exception:
            continue
        red_rate = max(float(red.standing_striking_tendency) * float(blue.standing_striking_suppression), 0.0)
        blue_rate = max(float(blue.standing_striking_tendency) * float(red.standing_striking_suppression), 0.0)
        date = pd.Timestamp(fight.event_date).normalize()
        for side, fighter_id, opp_id, rate, actual in (
            ("red", rid, bid, red_rate, red_actual),
            ("blue", bid, rid, blue_rate, blue_actual),
        ):
            if np.isfinite(actual) and rate > 0:
                rows_out.append({
                    "event_date": date,
                    "fight_id": fid,
                    "fighter_id": fighter_id,
                    "opponent_id": opp_id,
                    "side": side,
                    "elapsed_seconds": float(horizon),
                    "fsr_matchup_standing_rate_15m": float(rate),
                    "actual_distance_attempts": float(actual),
                    "raw_expected_attempts": float(rate * horizon / 900.0),
                })
    f = pd.DataFrame(rows_out)
    if f.empty:
        raise RuntimeError("no complete fighter-fight observations")
    return f.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def metrics(df: pd.DataFrame, scale: float) -> dict:
    pred = scale * df.raw_expected_attempts.to_numpy(float)
    y = df.actual_distance_attempts.to_numpy(float)
    err = pred - y
    total_seconds = float(df.elapsed_seconds.sum())
    return {
        "rows": int(len(df)),
        "fights": int(df.fight_id.nunique()),
        "actual_attempts": float(y.sum()),
        "predicted_attempts": float(pred.sum()),
        "E_over_O": float(pred.sum()/max(y.sum(), EPS)),
        "actual_mean_per_fighter_fight": float(y.mean()),
        "predicted_mean_per_fighter_fight": float(pred.mean()),
        "MAE_attempts": float(np.mean(np.abs(err))),
        "RMSE_attempts": float(np.sqrt(np.mean(err**2))),
        "correlation": float(np.corrcoef(y,pred)[0,1]) if len(y)>1 and np.std(y)>0 and np.std(pred)>0 else None,
        "actual_attempts_per_15_exposure": float(y.sum()/max(total_seconds, EPS)*900.0),
        "predicted_attempts_per_15_exposure": float(pred.sum()/max(total_seconds, EPS)*900.0),
    }


def main():
    f = build_frame()
    train = f[f.event_date < CUTOFF].copy()
    holdout = f[f.event_date >= CUTOFF].copy()
    if train.empty or holdout.empty:
        raise RuntimeError(f"empty split train={len(train)} holdout={len(holdout)}")

    # Population cadence scale chosen ONLY from pre-2025 aggregate attempt exposure.
    scale = float(train.actual_distance_attempts.sum()/max(train.raw_expected_attempts.sum(), EPS))
    result = {
        "study": "standing timing OOS calibration",
        "production_changed": False,
        "target": "fighter-fight UFCStats distance significant-strike attempts",
        "runtime_rate": "attacker standing_striking_tendency * defender standing_striking_suppression",
        "fit_cutoff": str(CUTOFF.date()),
        "standing_rate_scale": scale,
        "mean_delay_multiplier_equivalent": 1.0/scale,
        "train": metrics(train, scale),
        "holdout_2025_2026": metrics(holdout, scale),
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR/"results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    out = f.copy()
    out["split"] = np.where(out.event_date < CUTOFF, "train", "holdout")
    out["predicted_distance_attempts"] = scale * out.raw_expected_attempts
    out.to_csv(OUTDIR/"fighter_fight_predictions.csv", index=False)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
