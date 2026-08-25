from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import FSR_V3_PREFIGHT_PATH

OUT = Path("data/diagnostics/submission_signal_audit.json")
KEYS = ["event_date", "fight_id", "fighter_id"]
TRAITS = [
    "submission_tendency",
    "submission_offense",
    "submission_defense",
    "submission_conversion_baseline",
]


def _safe_auc(y, p):
    y = np.asarray(y, dtype=int)
    if np.unique(y).size < 2:
        return None
    return float(roc_auc_score(y, p))


def _fit_eval(train, test, features, target):
    xtr = train[features].astype(float).to_numpy()
    xte = test[features].astype(float).to_numpy()
    ytr = train[target].astype(int).to_numpy()
    yte = test[target].astype(int).to_numpy()
    model = LogisticRegression(C=1.0, max_iter=5000)
    model.fit(xtr, ytr)
    p = model.predict_proba(xte)[:, 1]
    return {
        "features": features,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "positive_rate_test": float(yte.mean()),
        "auc": _safe_auc(yte, p),
        "log_loss": float(log_loss(yte, p, labels=[0, 1])),
        "coefficients": {name: float(v) for name, v in zip(features, model.coef_[0])},
        "intercept": float(model.intercept_[0]),
    }


def _quintiles(frame, column, target):
    work = frame[[column, target]].dropna().copy()
    try:
        work["bucket"] = pd.qcut(work[column], 5, duplicates="drop")
    except ValueError:
        return []
    out = []
    for bucket, part in work.groupby("bucket", observed=True):
        out.append({
            "bucket": str(bucket),
            "n": int(len(part)),
            "mean_signal": float(part[column].mean()),
            "target_rate": float(part[target].mean()),
        })
    return out


def main():
    paired = build_paired_rounds()
    snapshots = pd.read_parquet(FSR_V3_PREFIGHT_PATH).copy()
    snapshots["event_date"] = pd.to_datetime(snapshots["event_date"]).dt.normalize()
    paired["event_date"] = pd.to_datetime(paired["event_date"]).dt.normalize()

    missing = [c for c in TRAITS if c not in snapshots.columns]
    if missing:
        raise RuntimeError(f"missing submission traits in canonical prefight snapshots: {missing}")

    ff = paired.groupby(KEYS, as_index=False).agg(
        ground_exposure_seconds=("modeled_ground_exposure_seconds", "sum"),
        submission_attempts=("effective_submission_attempts", "sum"),
        submission_finish=("submission_finish", "max"),
        ground_entries=("ground_entries", "sum"),
        opponent_id=("opponent_id", "first"),
    )
    ff["any_submission_attempt"] = ff["submission_attempts"].gt(0).astype(int)
    ff["ground_minutes"] = ff["ground_exposure_seconds"] / 60.0
    ff["attempts_per_ground_min"] = np.where(
        ff["ground_minutes"] > 0,
        ff["submission_attempts"] / ff["ground_minutes"],
        np.nan,
    )
    ff["log_ground_seconds"] = np.log1p(ff["ground_exposure_seconds"])
    ff["log_attempts"] = np.log1p(ff["submission_attempts"])

    own = snapshots[KEYS + TRAITS].copy()
    frame = ff.merge(own, on=KEYS, how="inner", validate="one_to_one")

    opp = snapshots[KEYS + ["fighter_id", "submission_defense"]].copy()
    opp = opp.rename(columns={"fighter_id": "opponent_id", "submission_defense": "opponent_submission_defense"})
    frame = frame.merge(
        opp,
        on=["event_date", "fight_id", "opponent_id"],
        how="left",
        validate="one_to_one",
    )
    frame["offense_defense_edge"] = frame["submission_offense"] - frame["opponent_submission_defense"]
    frame["baseline_logit"] = np.log(
        np.clip(frame["submission_conversion_baseline"], 1e-8, 1-1e-8)
        / (1 - np.clip(frame["submission_conversion_baseline"], 1e-8, 1-1e-8))
    )

    frame = frame.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)
    cutoff = frame["event_date"].quantile(0.70, interpolation="nearest")
    train = frame[frame.event_date < cutoff].copy()
    test = frame[frame.event_date >= cutoff].copy()

    # Attempt propensity: only rows with explicit modeled ground opportunity.
    a_train = train[train.ground_exposure_seconds > 0].dropna(subset=["submission_tendency"])
    a_test = test[test.ground_exposure_seconds > 0].dropna(subset=["submission_tendency"])
    attempt_base = _fit_eval(a_train, a_test, ["log_ground_seconds"], "any_submission_attempt")
    attempt_trait = _fit_eval(a_train, a_test, ["log_ground_seconds", "submission_tendency"], "any_submission_attempt")

    rate_rows = frame[(frame.ground_exposure_seconds > 0) & frame.submission_tendency.notna()]
    rho, rho_p = spearmanr(rate_rows["submission_tendency"], rate_rows["attempts_per_ground_min"], nan_policy="omit")

    # Conversion discrimination: UFCStats lacks attempt-level outcomes, so use
    # attempt-positive fighter-fights and control for number of attempts.
    c_train = train[train.submission_attempts > 0].dropna(subset=["baseline_logit", "offense_defense_edge"])
    c_test = test[test.submission_attempts > 0].dropna(subset=["baseline_logit", "offense_defense_edge"])
    conv_attempt_only = _fit_eval(c_train, c_test, ["log_attempts"], "submission_finish")
    conv_baseline = _fit_eval(c_train, c_test, ["log_attempts", "baseline_logit"], "submission_finish")
    conv_matchup = _fit_eval(c_train, c_test, ["log_attempts", "baseline_logit", "offense_defense_edge"], "submission_finish")

    result = {
        "scope": {
            "fighter_fights": int(len(frame)),
            "first_date": str(frame.event_date.min().date()),
            "last_date": str(frame.event_date.max().date()),
            "chronological_holdout_cutoff": str(pd.Timestamp(cutoff).date()),
            "ground_opportunity_rows": int((frame.ground_exposure_seconds > 0).sum()),
            "attempt_positive_rows": int((frame.submission_attempts > 0).sum()),
            "submission_finish_rows": int(frame.submission_finish.sum()),
        },
        "attempt_propensity": {
            "baseline_exposure_only": attempt_base,
            "plus_submission_tendency": attempt_trait,
            "delta_auc": None if attempt_base["auc"] is None or attempt_trait["auc"] is None else float(attempt_trait["auc"] - attempt_base["auc"]),
            "delta_log_loss": float(attempt_trait["log_loss"] - attempt_base["log_loss"]),
            "spearman_tendency_vs_attempts_per_ground_min": float(rho),
            "spearman_p": float(rho_p),
            "tendency_quintiles": _quintiles(rate_rows, "submission_tendency", "any_submission_attempt"),
        },
        "conversion": {
            "limitation": "UFCStats has submission-attempt counts but not attempt-level timestamps/outcomes; models predict fighter-fight submission finish among attempt-positive rows and control for attempt count.",
            "attempt_count_only": conv_attempt_only,
            "plus_conversion_baseline": conv_baseline,
            "plus_offense_minus_opponent_defense": conv_matchup,
            "delta_auc_matchup_vs_baseline": None if conv_baseline["auc"] is None or conv_matchup["auc"] is None else float(conv_matchup["auc"] - conv_baseline["auc"]),
            "delta_log_loss_matchup_vs_baseline": float(conv_matchup["log_loss"] - conv_baseline["log_loss"]),
            "edge_quintiles": _quintiles(c_test, "offense_defense_edge", "submission_finish"),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
