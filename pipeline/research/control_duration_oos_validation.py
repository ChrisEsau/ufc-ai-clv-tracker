"""Research-only OOS validation for round-level control duration conditional on TD success.

Purpose:
- Build a defensible empirical target from UFCStats round-level control seconds.
- Condition on rounds with at least one successful takedown.
- Test whether simple point-in-time fighter retention + opponent control-allowed
  features predict held-out control duration.
- Do not modify production simulator mechanics.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA = Path("data/fight_details/ufc_round_stats.parquet")
CUTOFF = pd.Timestamp("2025-01-01")
SHRINK_N = 10.0


def _paired(df: pd.DataFrame) -> pd.DataFrame:
    key = ["fight_id", "round"]
    cols = key + ["fighter_name", "opponent_name", "event_date", "td_landed", "td_attempted", "ctrl_sec"]
    x = df[cols].copy()
    x["event_date"] = pd.to_datetime(x["event_date"], errors="coerce")
    for c in ["td_landed", "td_attempted", "ctrl_sec"]:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    # one fighter-row observation; opponent fields already supplied in raw table
    x = x.sort_values(["event_date", "fight_id", "round", "fighter_name"]).reset_index(drop=True)
    return x


def _ewm_prior(values: list[float], alpha: float = 0.50) -> float | None:
    if not values:
        return None
    s = float(values[0])
    for v in values[1:]:
        s = alpha * float(v) + (1.0 - alpha) * s
    return s


def build_point_in_time_rows(df: pd.DataFrame) -> pd.DataFrame:
    global_train = df[(df["event_date"] < CUTOFF) & (df["td_landed"] > 0)]
    global_mean = float(global_train["ctrl_sec"].mean())
    global_median = float(global_train["ctrl_sec"].median())

    retention_hist: dict[str, list[float]] = {}
    allowed_hist: dict[str, list[float]] = {}
    rows = []

    for r in df.itertuples(index=False):
        fighter = str(r.fighter_name)
        opponent = str(r.opponent_name)
        td = float(r.td_landed)
        ctrl = float(r.ctrl_sec)
        date = pd.Timestamp(r.event_date)

        if td > 0:
            top_hist = retention_hist.get(fighter, [])
            bottom_hist = allowed_hist.get(opponent, [])
            top_mean = float(np.mean(top_hist)) if top_hist else global_mean
            bottom_mean = float(np.mean(bottom_hist)) if bottom_hist else global_mean
            top_n = len(top_hist)
            bottom_n = len(bottom_hist)
            top_shrunk = (top_n * top_mean + SHRINK_N * global_mean) / (top_n + SHRINK_N)
            bottom_shrunk = (bottom_n * bottom_mean + SHRINK_N * global_mean) / (bottom_n + SHRINK_N)
            top_ewm = _ewm_prior(top_hist)
            bottom_ewm = _ewm_prior(bottom_hist)
            top_ewm = global_mean if top_ewm is None else top_ewm
            bottom_ewm = global_mean if bottom_ewm is None else bottom_ewm

            rows.append({
                "event_date": date,
                "fight_id": r.fight_id,
                "round": int(r.round),
                "fighter_name": fighter,
                "opponent_name": opponent,
                "td_landed": td,
                "td_attempted": float(r.td_attempted),
                "actual_ctrl_sec": ctrl,
                "global_mean": global_mean,
                "global_median": global_median,
                "top_prior_mean": top_mean,
                "bottom_allowed_prior_mean": bottom_mean,
                "top_prior_shrunk": top_shrunk,
                "bottom_allowed_prior_shrunk": bottom_shrunk,
                "top_prior_ewm": top_ewm,
                "bottom_allowed_prior_ewm": bottom_ewm,
                "top_prior_n": top_n,
                "bottom_prior_n": bottom_n,
            })

            retention_hist.setdefault(fighter, []).append(ctrl)
            allowed_hist.setdefault(opponent, []).append(ctrl)

    out = pd.DataFrame(rows)
    # candidate matchup predictions; none use target-round information
    out["pred_global_mean"] = out["global_mean"]
    out["pred_shrunk_arith"] = 0.5 * (out["top_prior_shrunk"] + out["bottom_allowed_prior_shrunk"])
    out["pred_shrunk_geom"] = np.sqrt(np.maximum(out["top_prior_shrunk"], 0.1) * np.maximum(out["bottom_allowed_prior_shrunk"], 0.1))
    out["pred_ewm_arith"] = 0.5 * (out["top_prior_ewm"] + out["bottom_allowed_prior_ewm"])
    out["pred_ewm_geom"] = np.sqrt(np.maximum(out["top_prior_ewm"], 0.1) * np.maximum(out["bottom_allowed_prior_ewm"], 0.1))
    return out


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    return {
        "n": int(len(y)),
        "mean_actual": float(np.mean(y)),
        "mean_pred": float(np.mean(p)),
        "mean_error": float(np.mean(err)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "corr": float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 and np.std(p) > 0 else None,
    }


def main():
    raw = pd.read_parquet(DATA)
    df = _paired(raw)
    pit = build_point_in_time_rows(df)
    holdout = pit[pit["event_date"] >= CUTOFF].copy()

    preds = ["pred_global_mean", "pred_shrunk_arith", "pred_shrunk_geom", "pred_ewm_arith", "pred_ewm_geom"]
    report = {
        "study": "Round-level control duration conditional on >=1 TD landed — point-in-time OOS validation",
        "production_changed": False,
        "source": str(DATA),
        "cutoff": str(CUTOFF.date()),
        "target_definition": "fighter round ctrl_sec, restricted to td_landed > 0",
        "note": "UFCStats control is round aggregate, not individual control-spell duration.",
        "train_positive_td_rounds": int((pit["event_date"] < CUTOFF).sum()),
        "holdout_positive_td_rounds": int(len(holdout)),
        "models": {c: metrics(holdout["actual_ctrl_sec"].to_numpy(float), holdout[c].to_numpy(float)) for c in preds},
    }

    # Calibration by predicted quartile for best simple model candidate selected by MAE.
    best = min(preds, key=lambda c: report["models"][c]["mae"])
    report["best_by_mae"] = best
    q = pd.qcut(holdout[best], 4, duplicates="drop")
    cal = holdout.assign(bin=q).groupby("bin", observed=True).agg(
        n=("actual_ctrl_sec", "size"),
        mean_pred=(best, "mean"),
        mean_actual=("actual_ctrl_sec", "mean"),
        median_actual=("actual_ctrl_sec", "median"),
    ).reset_index()
    report["quartile_calibration"] = [
        {"bin": str(r.bin), "n": int(r.n), "mean_pred": float(r.mean_pred), "mean_actual": float(r.mean_actual), "median_actual": float(r.median_actual)}
        for r in cal.itertuples(index=False)
    ]

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
