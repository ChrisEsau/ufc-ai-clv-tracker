"""Leakage-safe OOS validation of prefight submission-attempt opportunity.

Research only; production simulator unchanged.

Tests whether prefight submission tendency and opponent suppression predict future
UFCStats effective submission attempts, including exposure/reliability buckets.
"""
from __future__ import annotations

import json
from math import lgamma
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from pipeline.fsr_v2.replay.engine import ReplayEngine, aggregate_fights
from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v2.traits.registry import GROUPS
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH

OUTDIR = Path("data/research/submission_attempt_opportunity_oos_validation")
EPS = 1e-12


def build_frame() -> pd.DataFrame:
    fights = aggregate_fights(build_paired_rounds()).copy()
    fights["event_date"] = pd.to_datetime(fights["event_date"]).dt.normalize()
    for c in ("fight_id", "fighter_id", "opponent_id"):
        fights[c] = fights[c].astype(str)

    snaps = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    snaps["event_date"] = pd.to_datetime(snaps["event_date"]).dt.normalize()
    for c in ("fight_id", "fighter_id"):
        snaps[c] = snaps[c].astype(str)

    own = snaps[["event_date", "fight_id", "fighter_id", "submission_tendency"]].copy()
    opp = snaps[["event_date", "fight_id", "fighter_id", "submission_suppression"]].rename(
        columns={"fighter_id": "opponent_id", "submission_suppression": "opp_submission_suppression"}
    )
    f = fights.merge(own, on=["event_date", "fight_id", "fighter_id"], how="inner", validate="one_to_one")
    f = f.merge(opp, on=["event_date", "fight_id", "opponent_id"], how="left", validate="one_to_one")

    engine = ReplayEngine()
    th = engine.replay(GROUPS["submission_tendency"], fights).history.copy()
    th["event_date"] = pd.to_datetime(th["event_date"]).dt.normalize()
    for c in ("fight_id", "fighter_id"):
        th[c] = th[c].astype(str)
    th = th[["event_date", "fight_id", "fighter_id", "fighter_prior_attempts", "fighter_prior_exposure_seconds"]]
    f = f.merge(th, on=["event_date", "fight_id", "fighter_id"], how="left", validate="one_to_one")

    for c in ("submission_tendency", "opp_submission_suppression", "effective_submission_attempts",
              "fight_elapsed_seconds", "fighter_prior_attempts", "fighter_prior_exposure_seconds"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f = f.dropna(subset=["submission_tendency", "opp_submission_suppression", "effective_submission_attempts",
                         "fight_elapsed_seconds", "fighter_prior_exposure_seconds"]).copy()
    f["effective_submission_attempts"] = np.maximum(f["effective_submission_attempts"], 0.0)
    f["rate_tendency"] = np.maximum(f["submission_tendency"], 0.0)
    f["rate_matchup"] = f["rate_tendency"] * np.maximum(f["opp_submission_suppression"], 0.0)
    return f.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def fit_scale(train: pd.DataFrame, col: str) -> float:
    raw_mu = train[col].to_numpy(float) * train["fight_elapsed_seconds"].to_numpy(float)
    y = train["effective_submission_attempts"].to_numpy(float)
    return float(y.sum() / max(raw_mu.sum(), EPS))


def poisson_nll(y, mu):
    y = np.asarray(y, float)
    mu = np.maximum(np.asarray(mu, float), EPS)
    return float(np.mean(mu - y * np.log(mu) + np.vectorize(lgamma)(y + 1.0)))


def metrics(df: pd.DataFrame, mu: np.ndarray) -> dict:
    y = df["effective_submission_attempts"].to_numpy(float)
    mu = np.maximum(np.asarray(mu, float), 0.0)
    any_y = (y > 0).astype(int)
    p_any = np.clip(1.0 - np.exp(-mu), 1e-9, 1 - 1e-9)
    actual = float(y.sum())
    expected = float(mu.sum())
    return {
        "rows": int(len(df)),
        "actual_attempts": actual,
        "expected_attempts": expected,
        "E_over_O": expected / actual if actual > 0 else None,
        "actual_attempts_per_15": float(actual / max(df["fight_elapsed_seconds"].sum(), EPS) * 900.0),
        "pred_attempts_per_15": float(expected / max(df["fight_elapsed_seconds"].sum(), EPS) * 900.0),
        "poisson_nll_per_row": poisson_nll(y, mu),
        "any_attempt_auc": float(roc_auc_score(any_y, p_any)) if np.unique(any_y).size == 2 else None,
        "any_attempt_brier": float(brier_score_loss(any_y, p_any)),
        "any_attempt_log_loss": float(log_loss(any_y, p_any, labels=[0,1])),
    }


def eval_period(f: pd.DataFrame, start_year: int, end_year: int, cutoff: str) -> dict:
    cut = pd.Timestamp(cutoff)
    train = f[f.event_date < cut].copy()
    test = f[f.event_date.dt.year.between(start_year, end_year)].copy()
    out = {"train_rows": int(len(train)), "test_rows": int(len(test)), "models": {}, "exposure_buckets": {}}
    mus = {}
    for name, col in (("tendency_only", "rate_tendency"), ("tendency_x_suppression", "rate_matchup")):
        s = fit_scale(train, col)
        mu = s * test[col].to_numpy(float) * test["fight_elapsed_seconds"].to_numpy(float)
        mus[name] = mu
        out["models"][name] = {"scale": s, **metrics(test, mu)}

    # Reliability check by prior UFC exposure entering the fight.
    bins = [-1, 900, 2700, 5400, np.inf]
    labels = ["<15m", "15-45m", "45-90m", "90m+"]
    test = test.copy()
    test["prior_exposure_bucket"] = pd.cut(test["fighter_prior_exposure_seconds"], bins=bins, labels=labels)
    test["mu_matchup"] = mus["tendency_x_suppression"]
    for label in labels:
        g = test[test["prior_exposure_bucket"].astype(str).eq(label)]
        if len(g):
            out["exposure_buckets"][label] = metrics(g, g["mu_matchup"].to_numpy(float))

    # High-rate tails: do extreme prefight opportunity signals remain high in the next fight?
    for q in (0.80, 0.90, 0.95):
        threshold = float(test["rate_matchup"].quantile(q))
        g = test[test["rate_matchup"] >= threshold].copy()
        out.setdefault("high_predicted_rate", {})[f"top_{int((1-q)*100)}pct"] = {
            "threshold_attempts_per_15_pre_scale": threshold * 900.0,
            **metrics(g, mus["tendency_x_suppression"][test.index.get_indexer(g.index)] if False else g.assign(_mu=out.get('_dummy', 0)).get('_mu', pd.Series(index=g.index, dtype=float)).to_numpy())
        }
    # Recompute high-tail metrics safely from the trained scale.
    s = out["models"]["tendency_x_suppression"]["scale"]
    out["high_predicted_rate"] = {}
    for q in (0.80, 0.90, 0.95):
        threshold = float(test["rate_matchup"].quantile(q))
        g = test[test["rate_matchup"] >= threshold].copy()
        mu = s * g["rate_matchup"].to_numpy(float) * g["fight_elapsed_seconds"].to_numpy(float)
        out["high_predicted_rate"][f"top_{int(round((1-q)*100))}pct"] = {
            "threshold_attempts_per_15_pre_scale": threshold * 900.0,
            **metrics(g, mu),
        }
    return out


def main():
    f = build_frame()
    result = {
        "study": "submission attempt opportunity OOS validation",
        "production_changed": False,
        "selection_2020_2024": eval_period(f, 2020, 2024, "2020-01-01"),
        "confirmation_2025_2026": eval_period(f, 2025, 2026, "2025-01-01"),
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
