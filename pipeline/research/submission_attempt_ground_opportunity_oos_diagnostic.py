"""Research-only OOS diagnostic for submission-attempt opportunity denominator.

Production simulator and submission conversion are untouched.

Tests the same validated prefight submission signal under two exposure choices:
1) total fight exposure;
2) the repository's existing modeled true-ground exposure.

No new ground-time labels or finish mechanics are introduced.
"""
from __future__ import annotations

import json
from math import lgamma
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from pipeline.research.submission_attempt_opportunity_oos_validation import build_frame

OUTDIR = Path("data/research/submission_attempt_ground_opportunity_oos_diagnostic")
CUTOFF = pd.Timestamp("2025-01-01")
GROUND_ACTION_SECONDS = 4.4
ALLEN_FIGHT_ID = "419fff06f338f5c6"
EPS = 1e-12


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
        "poisson_nll_per_row": poisson_nll(y, mu),
        "any_attempt_auc": float(roc_auc_score(any_y, p_any)) if np.unique(any_y).size == 2 else None,
        "any_attempt_brier": float(brier_score_loss(any_y, p_any)),
        "any_attempt_log_loss": float(log_loss(any_y, p_any, labels=[0, 1])),
    }


def build_ground_frame() -> pd.DataFrame:
    f = build_frame().copy()
    required = [
        "modeled_ground_exposure_seconds", "ground_attempted", "td_landed",
        "sub_att", "rev",
    ]
    missing = [c for c in required if c not in f.columns]
    if missing:
        raise RuntimeError(f"submission frame missing existing ground fields: {missing}")
    f[required] = f[required].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    f["has_ground_opportunity"] = f["modeled_ground_exposure_seconds"] > 0
    f["ground_share"] = f["modeled_ground_exposure_seconds"] / np.maximum(
        f["fight_elapsed_seconds"], EPS
    )
    return f


def fit_exposure_scale(train: pd.DataFrame, exposure_col: str) -> float:
    raw_mu = train["rate_matchup"].to_numpy(float) * train[exposure_col].to_numpy(float)
    y = train["effective_submission_attempts"].to_numpy(float)
    return float(y.sum() / max(raw_mu.sum(), EPS))


def evaluate(train: pd.DataFrame, test: pd.DataFrame, exposure_col: str) -> dict:
    scale = fit_exposure_scale(train, exposure_col)
    mu = scale * test["rate_matchup"].to_numpy(float) * test[exposure_col].to_numpy(float)
    return {"scale": scale, "exposure_col": exposure_col, **metrics(test, mu)}


def bucket_table(test: pd.DataFrame, ground_scale: float) -> pd.DataFrame:
    x = test.copy()
    bins = [-1e-9, 0, 30, 60, 120, 240, np.inf]
    labels = ["0s", "1-30s", "31-60s", "61-120s", "121-240s", "240s+"]
    x["ground_exposure_bucket"] = pd.cut(x["modeled_ground_exposure_seconds"], bins=bins, labels=labels)
    x["ground_mu"] = ground_scale * x["rate_matchup"] * x["modeled_ground_exposure_seconds"]
    rows = []
    for label in labels:
        g = x[x["ground_exposure_bucket"].astype(str).eq(label)]
        if g.empty:
            continue
        actual = float(g["effective_submission_attempts"].sum())
        expected = float(g["ground_mu"].sum())
        exposure = float(g["modeled_ground_exposure_seconds"].sum())
        rows.append({
            "ground_exposure_bucket": label,
            "fighter_fights": int(len(g)),
            "actual_submission_attempts": actual,
            "expected_submission_attempts_ground_model": expected,
            "E_over_O": expected / actual if actual > 0 else None,
            "total_ground_exposure_seconds": exposure,
            "actual_attempts_per_15m_ground": actual / max(exposure, EPS) * 900.0,
            "mean_prefight_matchup_signal_pre_scale_per_15": float(g["rate_matchup"].mean() * 900.0),
        })
    return pd.DataFrame(rows)


def main():
    f = build_ground_frame()
    train = f[f.event_date < CUTOFF].copy()
    test = f[f.event_date >= CUTOFF].copy()

    total_model = evaluate(train, test, "fight_elapsed_seconds")
    ground_model = evaluate(train, test, "modeled_ground_exposure_seconds")

    train_ground = train[train.has_ground_opportunity].copy()
    test_ground = test[test.has_ground_opportunity].copy()
    ground_conditional = evaluate(train_ground, test_ground, "modeled_ground_exposure_seconds")

    ground_scale = float(ground_model["scale"])
    test = test.copy()
    test["mu_total_exposure"] = float(total_model["scale"]) * test["rate_matchup"] * test["fight_elapsed_seconds"]
    test["mu_ground_exposure"] = ground_scale * test["rate_matchup"] * test["modeled_ground_exposure_seconds"]
    test["ground_hazard_attempts_per_second"] = ground_scale * test["rate_matchup"]
    test["p_sub_per_4p4s_from_ground_model"] = 1.0 - np.exp(
        -test["ground_hazard_attempts_per_second"] * GROUND_ACTION_SECONDS
    )

    allen = test[test.fight_id.astype(str).eq(ALLEN_FIGHT_ID)].copy()
    allen_cols = [
        "event_date", "fight_id", "fighter_id", "opponent_id",
        "effective_submission_attempts", "fight_elapsed_seconds",
        "modeled_ground_exposure_seconds", "ground_share",
        "submission_tendency", "opp_submission_suppression", "rate_matchup",
        "mu_total_exposure", "mu_ground_exposure", "p_sub_per_4p4s_from_ground_model",
        "fighter_prior_attempts", "fighter_prior_exposure_seconds",
    ]

    buckets = bucket_table(test, ground_scale)
    result = {
        "study": "submission attempt ground-opportunity denominator OOS diagnostic",
        "production_changed": False,
        "submission_conversion_changed": False,
        "prefight_signal_changed": False,
        "cutoff": str(CUTOFF.date()),
        "ground_opportunity_definition": "existing modeled_ground_exposure_seconds from aggregate_fights(build_paired_rounds())",
        "ground_action_seconds_for_interpretation_only": GROUND_ACTION_SECONDS,
        "train_rows": int(len(train)),
        "holdout_rows": int(len(test)),
        "holdout_rows_with_ground_opportunity": int(test.has_ground_opportunity.sum()),
        "holdout_actual_attempts_without_modeled_ground_opportunity": float(
            test.loc[~test.has_ground_opportunity, "effective_submission_attempts"].sum()
        ),
        "models": {
            "unconditional_total_fight_exposure": total_model,
            "ground_opportunity_exposure_all_rows": ground_model,
            "ground_opportunity_exposure_conditional_on_ground": ground_conditional,
        },
        "allen_shahbazyan": allen[allen_cols].to_dict("records"),
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, default=str)
    (OUTDIR / "results.json").write_text(payload, encoding="utf-8")
    test.to_csv(OUTDIR / "fighter_fight_holdout.csv", index=False)
    buckets.to_csv(OUTDIR / "ground_exposure_buckets.csv", index=False)
    allen[allen_cols].to_csv(OUTDIR / "allen_shahbazyan.csv", index=False)

    print(payload)
    print("GROUND_EXPOSURE_BUCKETS")
    print(buckets.to_string(index=False))


if __name__ == "__main__":
    main()
