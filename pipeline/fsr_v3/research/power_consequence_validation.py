from __future__ import annotations

"""Final measurement-only consequence gate for the selected FSR V3 power trait.

The direct trait target is future knockdowns per landed significant strike.  The
selected sequential specification is fixed from development research:

    sigma = 0.50
    rho   = 0.01
    c     = 0.00

This script does two things without changing production FSR or Event Clock:

1. Reports reserved-outer KD likelihood by prior-UFC-fight evidence bucket.
2. Fits a simple landed-strike -> KO/TKO-win consequence mapping on 2020-2023
   and scores it untouched on 2024+ for population, frozen FSR V2 power, and
   selected V3 latent power.

Age is intentionally omitted here because it remains a separate matchup-time
predictor in the current Event Clock shadow hazard.  This gate asks whether the
persisted power trait itself adds consequence information.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from pipeline.common.paths import FSR_V2_PREFIGHT_SNAPSHOTS_PATH
from pipeline.fsr_v3.research.power_sequential_shrinkage_study import (
    _bb_loglik_p,
    _fit_population_beta,
    _prepare,
    _sequential_score,
    _training_evidence,
)

SIGMA = 0.50
RHO = 0.01
TRAIN_STATE_CUTOFF = pd.Timestamp("2020-01-01")
KO_MAP_OUTER_CUTOFF = pd.Timestamp("2024-01-01")
EPS = 1e-12


def _bucket(n: int) -> str:
    if n <= 0:
        return "debut"
    if n == 1:
        return "1_prior"
    if n == 2:
        return "2_prior"
    return "3plus_prior"


def _safe_auc(y, p) -> float:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def _ko_probability(n_landed: np.ndarray, intercept: float, slope: float, score: np.ndarray) -> np.ndarray:
    strike_p = expit(float(intercept) + float(slope) * np.asarray(score, float))
    q = 1.0 - np.power(1.0 - strike_p, np.asarray(n_landed, float))
    return np.clip(q, EPS, 1.0 - EPS)


def _fit_ko_population(train: pd.DataFrame) -> tuple[float, float]:
    y = train["ko_win"].to_numpy(float)
    n = train["sig_landed"].to_numpy(float)

    def objective(intercept: float) -> float:
        q = _ko_probability(n, float(intercept), 0.0, np.zeros_like(n))
        return -float(np.sum(y * np.log(q) + (1.0 - y) * np.log1p(-q)))

    fit = minimize_scalar(objective, bounds=(-12.0, -1.0), method="bounded")
    if not fit.success:
        raise RuntimeError("KO population fit failed")
    return float(fit.x), 0.0


def _fit_ko_power(train: pd.DataFrame, score_col: str) -> tuple[float, float]:
    y = train["ko_win"].to_numpy(float)
    n = train["sig_landed"].to_numpy(float)
    score = train[score_col].to_numpy(float)

    def objective(theta: np.ndarray) -> float:
        q = _ko_probability(n, float(theta[0]), float(theta[1]), score)
        return -float(np.sum(y * np.log(q) + (1.0 - y) * np.log1p(-q)))

    fit = minimize(
        objective,
        x0=np.array([-6.0, 0.20]),
        method="L-BFGS-B",
        bounds=[(-12.0, -1.0), (-4.0, 4.0)],
    )
    if not fit.success:
        raise RuntimeError(f"KO power fit failed for {score_col}: {fit.message}")
    return float(fit.x[0]), float(fit.x[1])


def _score_ko(frame: pd.DataFrame, *, model: str, intercept: float, slope: float, score_col: str | None) -> tuple[dict[str, float], pd.DataFrame]:
    score = np.zeros(len(frame), float) if score_col is None else frame[score_col].to_numpy(float)
    q = _ko_probability(frame["sig_landed"].to_numpy(float), intercept, slope, score)
    y = frame["ko_win"].to_numpy(float)
    ll_row = y * np.log(q) + (1.0 - y) * np.log1p(-q)
    brier_row = np.square(q - y)

    detail = frame[["fight_id", "date", "fighter_id", "prior_ufc_fights", "ko_win", "sig_landed"]].copy()
    detail["model"] = model
    detail["ko_probability"] = q
    detail["log_likelihood"] = ll_row
    detail["brier"] = brier_row

    metrics = {
        "model": model,
        "intercept": intercept,
        "slope": slope,
        "fighter_fights": len(frame),
        "fights": frame["fight_id"].nunique(),
        "ko_wins": int(frame["ko_win"].sum()),
        "log_likelihood": float(ll_row.sum()),
        "log_loss": float(-ll_row.mean()),
        "brier": float(brier_row.mean()),
        "auc": _safe_auc(y, q),
        "mean_predicted_ko_win": float(q.mean()),
        "actual_ko_win_rate": float(y.mean()),
    }
    return metrics, detail


def _bootstrap_differences(detail: pd.DataFrame, reference: str = "population", reps: int = 5000) -> pd.DataFrame:
    pivot_ll = detail.pivot_table(index=["fight_id", "fighter_id"], columns="model", values="log_likelihood", aggfunc="first")
    pivot_br = detail.pivot_table(index=["fight_id", "fighter_id"], columns="model", values="brier", aggfunc="first")
    fight_ids = np.asarray(sorted(detail["fight_id"].astype(str).unique()))
    rng = np.random.default_rng(20260821)
    rows = []

    for model in sorted(set(detail["model"]) - {reference}):
        ll_by_fight = (pivot_ll[model] - pivot_ll[reference]).groupby(level=0).sum()
        br_by_fight = (pivot_br[model] - pivot_br[reference]).groupby(level=0).mean()
        ll_diffs, br_diffs = [], []
        for _ in range(reps):
            sample = rng.choice(fight_ids, size=len(fight_ids), replace=True)
            ll_diffs.append(float(ll_by_fight.reindex(sample).sum()))
            br_diffs.append(float(br_by_fight.reindex(sample).mean()))
        rows.append({
            "model": model,
            "reference": reference,
            "observed_ll_gain": float(ll_by_fight.sum()),
            "ll_gain_ci_low": float(np.percentile(ll_diffs, 2.5)),
            "ll_gain_ci_high": float(np.percentile(ll_diffs, 97.5)),
            "observed_brier_delta": float(br_by_fight.mean()),
            "brier_delta_ci_low": float(np.percentile(br_diffs, 2.5)),
            "brier_delta_ci_high": float(np.percentile(br_diffs, 97.5)),
        })
    return pd.DataFrame(rows)


def main(out_dir: str = "data/diagnostics/fsr_v3_power") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    obs = _prepare()
    state_train = obs[obs["date"] < TRAIN_STATE_CUTOFF].copy()
    forward = obs[obs["date"] >= TRAIN_STATE_CUTOFF].copy()

    beta = _fit_population_beta(state_train, RHO)
    evidence = _training_evidence(state_train, beta, RHO)
    result = _sequential_score(forward, beta=beta, rho=RHO, sigma=SIGMA, train_evidence=evidence)
    detail = result.detail.copy()

    fsr2 = pd.read_parquet(FSR_V2_PREFIGHT_SNAPSHOTS_PATH).copy()
    fsr2["fight_id"] = fsr2["fight_id"].astype(str)
    fsr2["fighter_id"] = fsr2["fighter_id"].astype(str)
    detail = detail.merge(
        fsr2[["fight_id", "fighter_id", "striking_power"]],
        on=["fight_id", "fighter_id"], how="left", validate="one_to_one",
    )
    if detail["striking_power"].isna().any():
        raise RuntimeError("Missing frozen V2 power in consequence validation")
    detail["v2_power_centered_10"] = (detail["striking_power"] - 50.0) / 10.0
    detail["evidence_bucket"] = detail["prior_ufc_fights"].astype(int).map(_bucket)

    outer = detail[detail["date"] >= KO_MAP_OUTER_CUTOFF].copy()

    # Direct KD likelihood by evidence bucket on the untouched 2024+ outer.
    kd_rows = []
    pop_p = float(expit(beta))
    for bucket, g in outer.groupby("evidence_bucket", sort=False):
        k = g["kd_scored"].to_numpy(float)
        n = g["sig_landed"].to_numpy(float)
        mean = g["posterior_mean_logit_power"].to_numpy(float)
        ll_pop = float(np.sum(_bb_loglik_p(k, n, pop_p, RHO)))
        ll_v3 = float(np.sum(_bb_loglik_p(k, n, expit(beta + mean), RHO)))
        kd_rows.append({
            "evidence_bucket": bucket,
            "fighter_fights": len(g),
            "ko_wins": int(g["ko_win"].sum()),
            "landed_sig": float(n.sum()),
            "kd_ll_population": ll_pop,
            "kd_ll_v3": ll_v3,
            "kd_ll_gain_v3": ll_v3 - ll_pop,
            "v3_ko_winner_auc": _safe_auc(g["ko_win"], g["posterior_mean_logit_power"]),
            "v2_ko_winner_auc": _safe_auc(g["ko_win"], g["striking_power"]),
            "mean_v3_posterior_sd": float(g["posterior_sd_logit_power"].mean()),
        })
    kd_bucket = pd.DataFrame(kd_rows)

    # Fit KO consequence mapping on development only, then freeze for outer.
    ko_train = detail[(detail["date"] >= TRAIN_STATE_CUTOFF) & (detail["date"] < KO_MAP_OUTER_CUTOFF)].copy()
    pop_par = _fit_ko_population(ko_train)
    v3_par = _fit_ko_power(ko_train, "posterior_mean_logit_power")
    v2_par = _fit_ko_power(ko_train, "v2_power_centered_10")

    metric_rows, scored = [], []
    for model, par, col in (
        ("population", pop_par, None),
        ("v3_power", v3_par, "posterior_mean_logit_power"),
        ("v2_power", v2_par, "v2_power_centered_10"),
    ):
        m, s = _score_ko(outer, model=model, intercept=par[0], slope=par[1], score_col=col)
        metric_rows.append(m)
        scored.append(s)
    ko_metrics = pd.DataFrame(metric_rows)
    ko_detail = pd.concat(scored, ignore_index=True)
    bootstrap = _bootstrap_differences(ko_detail)

    print("=" * 120)
    print("FSR V3 POWER — FINAL CONSEQUENCE VALIDATION")
    print("=" * 120)
    print(f"fixed selected trait: sigma={SIGMA:.2f} rho={RHO:.2f} c=0")
    print("age omitted from this trait-only gate; Event Clock retains age separately")
    print("\n2024+ DIRECT KD LIKELIHOOD BY EVIDENCE")
    print(kd_bucket.to_string(index=False))
    print("\n2024+ KO/TKO CONSEQUENCE PROPER SCORES")
    print(ko_metrics.to_string(index=False))
    print("\nFIGHT-LEVEL BOOTSTRAP VS POPULATION")
    print(bootstrap.to_string(index=False))

    kd_bucket.to_csv(out / "power_final_kd_by_evidence.csv", index=False)
    ko_metrics.to_csv(out / "power_final_ko_metrics.csv", index=False)
    bootstrap.to_csv(out / "power_final_ko_bootstrap.csv", index=False)
    ko_detail.to_csv(out / "power_final_ko_detail.csv", index=False)
    print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()
