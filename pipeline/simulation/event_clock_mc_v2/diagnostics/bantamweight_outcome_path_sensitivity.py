"""Counterfactual attribution test for bantamweight favorite compression.

Measurement only; no simulator, FSR, event-generation, or fitted-model changes.
Consumes the completed fight-level matchup decomposition and preserves each
fight's total DEC / KO / SUB probability while changing only which fighter
receives that method probability.

Arms:
  baseline          observed i10_b0 posterior-mean allocation
  decision_sharp2   double the conditional decision winner log-odds
  finish_sharp2     double conditional KO and SUB winner log-odds
  submission_neutral set conditional SUB winner probability to 50/50

These are deliberately strong diagnostic interventions, not proposed production
coefficients. They answer how much ML compression is attributable to soft
winner allocation within each outcome family.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0 - 1e-9)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _reallocate(fav: pd.Series, dog: pd.Series, mode: str) -> tuple[np.ndarray, np.ndarray]:
    f = fav.to_numpy(float)
    d = dog.to_numpy(float)
    total = f + d
    q = np.divide(f, total, out=np.full_like(f, 0.5), where=total > 0)
    if mode == "sharp2":
        q2 = _sigmoid(2.0 * _logit(q))
    elif mode == "neutral":
        q2 = np.full_like(q, 0.5)
    elif mode == "baseline":
        q2 = q
    else:
        raise ValueError(mode)
    return total * q2, total * (1.0 - q2)


def _arm_frame(x: pd.DataFrame, arm: str) -> pd.DataFrame:
    y = x.copy()
    for m in ("dec", "ko", "sub"):
        y[f"cf_fav_{m}"] = y[f"fav_{m}"].astype(float)
        y[f"cf_dog_{m}"] = y[f"dog_{m}"].astype(float)

    if arm == "decision_sharp2":
        f, d = _reallocate(y["fav_dec"], y["dog_dec"], "sharp2")
        y["cf_fav_dec"], y["cf_dog_dec"] = f, d
    elif arm == "finish_sharp2":
        for m in ("ko", "sub"):
            f, d = _reallocate(y[f"fav_{m}"], y[f"dog_{m}"], "sharp2")
            y[f"cf_fav_{m}"], y[f"cf_dog_{m}"] = f, d
    elif arm == "submission_neutral":
        f, d = _reallocate(y["fav_sub"], y["dog_sub"], "neutral")
        y["cf_fav_sub"], y["cf_dog_sub"] = f, d
    elif arm != "baseline":
        raise ValueError(arm)

    y["arm"] = arm
    y["cf_fav_win"] = y[["cf_fav_dec", "cf_fav_ko", "cf_fav_sub"]].sum(axis=1)
    y["cf_dog_win"] = y[["cf_dog_dec", "cf_dog_ko", "cf_dog_sub"]].sum(axis=1)
    # Numerical guard: original method probabilities sum to one modulo MC rounding.
    total = y["cf_fav_win"] + y["cf_dog_win"]
    y["cf_fav_win"] = y["cf_fav_win"] / total
    y["cf_dog_win"] = 1.0 - y["cf_fav_win"]
    y["compression_pp"] = 100.0 * (y["market_favorite_fair_p"] - y["cf_fav_win"])
    return y


def _metrics(y: pd.DataFrame, subset: str) -> dict:
    z = y.copy()
    if subset == "high_evidence":
        z = z[z["min_prior_ufc_fights"] >= 3].copy()
    p = z["cf_fav_win"].to_numpy(float)
    obs = z["favorite_won"].astype(int).to_numpy()
    auc = float(roc_auc_score(obs, p)) if len(np.unique(obs)) == 2 else np.nan
    return {
        "arm": str(y["arm"].iloc[0]),
        "subset": subset,
        "fights": len(z),
        "mean_market_favorite_fair_p": float(z["market_favorite_fair_p"].mean()),
        "mean_cf_favorite_p": float(p.mean()),
        "actual_favorite_win_rate": float(obs.mean()),
        "mean_compression_pp": float(z["compression_pp"].mean()),
        "favorite_accuracy": float(((p >= 0.5).astype(int) == obs).mean()),
        "favorite_auc": auc,
        "favorite_brier": float(np.mean((p - obs) ** 2)),
        "favorite_logloss": float(-np.mean(obs*np.log(np.clip(p,1e-9,1)) + (1-obs)*np.log(np.clip(1-p,1e-9,1)))),
        "mean_fav_dec": float(z["cf_fav_dec"].mean()),
        "mean_fav_ko": float(z["cf_fav_ko"].mean()),
        "mean_fav_sub": float(z["cf_fav_sub"].mean()),
        "mean_dog_dec": float(z["cf_dog_dec"].mean()),
        "mean_dog_ko": float(z["cf_dog_ko"].mean()),
        "mean_dog_sub": float(z["cf_dog_sub"].mean()),
    }


def _buckets(y: pd.DataFrame) -> pd.DataFrame:
    z = y.copy()
    z["bucket"] = pd.cut(
        z["market_favorite_fair_p"], [0.5,0.6,0.7,0.8,0.9,1.01],
        labels=["50-60","60-70","70-80","80-90","90+"], right=False,
    )
    cols = ["market_favorite_fair_p","cf_fav_win","favorite_won","compression_pp",
            "cf_fav_dec","cf_fav_ko","cf_fav_sub","cf_dog_dec","cf_dog_ko","cf_dog_sub"]
    out = z.groupby("bucket", observed=True)[cols].mean().reset_index()
    out.insert(0, "arm", str(y["arm"].iloc[0]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decomposition-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    x = pd.read_csv(args.decomposition_path)
    required = {
        "fight_id","market_favorite_fair_p","favorite_won","min_prior_ufc_fights",
        "fav_dec","fav_ko","fav_sub","dog_dec","dog_ko","dog_sub",
    }
    missing = sorted(required - set(x.columns))
    if missing:
        raise RuntimeError(f"missing required columns: {missing}")

    arms = ["baseline", "decision_sharp2", "finish_sharp2", "submission_neutral"]
    frames, metrics, buckets = [], [], []
    for arm in arms:
        y = _arm_frame(x, arm)
        frames.append(y)
        for subset in ("all", "high_evidence"):
            metrics.append(_metrics(y, subset))
        buckets.append(_buckets(y))

    all_frames = pd.concat(frames, ignore_index=True)
    m = pd.DataFrame(metrics)
    b = pd.concat(buckets, ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_frames.to_csv(args.out_dir / "fight_level_counterfactuals.csv", index=False)
    m.to_csv(args.out_dir / "counterfactual_metrics.csv", index=False)
    b.to_csv(args.out_dir / "counterfactual_market_buckets.csv", index=False)

    print("BANTAMWEIGHT OUTCOME-PATH CAUSAL SENSITIVITY")
    print(f"fights={len(x)} | no resimulation; total method mass preserved per fight")
    print("\nMETRICS")
    print(m.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nMARKET-STRENGTH BUCKETS")
    print(b.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\nOUTPUT: {args.out_dir}")


if __name__ == "__main__":
    main()
