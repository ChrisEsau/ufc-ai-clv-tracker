"""Research-only KO denominator study.

Compare candidate exposure units for fighter-side KO/TKO incidence without changing
Brain or production mechanics. All fighter histories are same-date delayed.
Selection: 2020-2024. Untouched confirmation: 2025-2026.

Candidate denominators:
  * elapsed fight seconds
  * significant strikes landed
  * significant strikes attempted
  * head significant strikes landed
  * distance significant strikes landed

For each denominator we estimate leakage-safe attacker KO-production and defender
KO-loss hazards with empirical-Bayes shrinkage toward the pre-year population rate,
then combine them symmetrically on the log-hazard scale. Realized exposure in the
test fight is used only for this denominator-identification study to ask which unit
best behaves like a stable survival exposure. It is NOT a deployable prefight model.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from pipeline.common.fight_time import repair_elapsed_match_time
from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH

OUT = Path("data/research/ko_denominator_study")
SELECTION_YEARS = tuple(range(2020, 2025))
CONFIRMATION_YEARS = (2025, 2026)
PRIOR_STRENGTH_EVENTS = (2.0, 5.0, 10.0, 20.0)

ROUND_COLS = [
    "event_date", "fight_id", "fighter_id", "fighter_name", "opponent_id",
    "sig_str_landed", "sig_str_attempted", "head_landed", "distance_landed",
]

DENOMS = {
    "fight_seconds": ("fight_seconds", "fight_seconds"),
    "sig_landed": ("sig_landed", "sig_absorbed"),
    "sig_attempted": ("sig_attempted", "sig_attempted_absorbed"),
    "head_landed": ("head_landed", "head_absorbed"),
    "distance_landed": ("distance_landed", "distance_absorbed"),
}


def _ko_tko(s: pd.Series) -> pd.Series:
    t = s.fillna("").astype(str).str.upper()
    return t.str.contains(r"KO/TKO|\bTKO\b|\bKO\b", regex=True)


def load_fighter_fights() -> pd.DataFrame:
    r = pd.read_parquet(ROUND_STATS_PATH, columns=ROUND_COLS).copy()
    r["event_date"] = pd.to_datetime(r["event_date"]).dt.normalize()
    for c in ["sig_str_landed", "sig_str_attempted", "head_landed", "distance_landed"]:
        r[c] = pd.to_numeric(r[c], errors="coerce").fillna(0.0)
    agg = r.groupby(["event_date", "fight_id", "fighter_id"], as_index=False).agg(
        fighter_name=("fighter_name", "first"), opponent_id=("opponent_id", "first"),
        sig_landed=("sig_str_landed", "sum"), sig_attempted=("sig_str_attempted", "sum"),
        head_landed=("head_landed", "sum"), distance_landed=("distance_landed", "sum"),
    )
    opp = agg[["fight_id", "fighter_id", "sig_landed", "sig_attempted", "head_landed", "distance_landed"]].rename(columns={
        "fighter_id": "opponent_id", "sig_landed": "sig_absorbed",
        "sig_attempted": "sig_attempted_absorbed", "head_landed": "head_absorbed",
        "distance_landed": "distance_absorbed",
    })
    agg = agg.merge(opp, on=["fight_id", "opponent_id"], how="left", validate="one_to_one")

    m = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    m = repair_elapsed_match_time(m)
    m["date"] = pd.to_datetime(m["date"]).dt.normalize()
    cols = ["fight_id", "date", "method", "winner_id", "r_id", "b_id", "division", "match_time_sec"]
    agg = agg.merge(m[cols], on="fight_id", how="left", validate="many_to_one")
    agg["event_date"] = agg["date"]
    agg["fight_seconds"] = pd.to_numeric(agg["match_time_sec"], errors="coerce")
    agg["won"] = agg["fighter_id"].astype(str).eq(agg["winner_id"].astype(str))
    agg["ko_win"] = (agg["won"] & _ko_tko(agg["method"])).astype(int)
    agg["ko_loss"] = ((~agg["won"]) & _ko_tko(agg["method"])).astype(int)
    agg["test_year"] = agg["event_date"].dt.year
    return agg.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def add_prefight_history(ff: pd.DataFrame) -> pd.DataFrame:
    # same-date delayed: snapshot entire event date before updating any row from it
    state = defaultdict(lambda: defaultdict(float))
    rows = []
    fields = ["ko_win", "ko_loss", "fight_seconds", "sig_landed", "sig_absorbed",
              "sig_attempted", "sig_attempted_absorbed", "head_landed", "head_absorbed",
              "distance_landed", "distance_absorbed"]
    for _, batch in ff.groupby("event_date", sort=True):
        for row in batch.itertuples(index=False):
            d = row._asdict()
            s = state[str(row.fighter_id)]
            for f in fields:
                d[f"prior_{f}"] = float(s[f])
            rows.append(d)
        for row in batch.itertuples(index=False):
            s = state[str(row.fighter_id)]
            for f in fields:
                s[f] += float(getattr(row, f))
    return pd.DataFrame(rows)


def _safe_log(x):
    return np.log(np.clip(np.asarray(x, float), 1e-12, None))


def predict_year(g: pd.DataFrame, train: pd.DataFrame, denom: str, prior_events: float) -> pd.DataFrame:
    att_col, def_col = DENOMS[denom]
    # Population event rate per unit exposure, learned strictly before test year.
    total_exp = float(train[att_col].sum())
    p0 = float(train["ko_win"].sum() / total_exp) if total_exp > 0 else np.nan
    if not np.isfinite(p0) or p0 <= 0:
        raise RuntimeError(f"invalid population rate for {denom}")
    # prior strength expressed as equivalent population events, converted to exposure units
    prior_exp = prior_events / p0
    att_rate = (g["prior_ko_win"].to_numpy(float) + prior_events) / (
        g[f"prior_{att_col}"].to_numpy(float) + prior_exp
    )
    def_rate = (g["prior_ko_loss"].to_numpy(float) + prior_events) / (
        g[f"prior_{def_col}"].to_numpy(float) + prior_exp
    )
    # symmetric attacker/defender combination centered on population
    log_rate = _safe_log(att_rate) + _safe_log(def_rate) - np.log(p0)
    rate = np.exp(np.clip(log_rate, -30, 5))
    exposure = g[att_col].to_numpy(float)
    # constant-hazard survival mapping over the realized candidate exposure
    p = 1.0 - np.exp(-rate * exposure)
    out = g[["event_date", "fight_id", "fighter_id", "fighter_name", "opponent_id", "division", "ko_win", att_col]].copy()
    out["denominator"] = denom
    out["prior_events"] = prior_events
    out["population_rate_per_unit"] = p0
    out["attacker_rate"] = att_rate
    out["defender_rate"] = def_rate
    out["combined_rate"] = rate
    out["p_ko"] = np.clip(p, 1e-9, 1 - 1e-9)
    out["test_year"] = g["test_year"].to_numpy(int)
    return out


def metrics(g: pd.DataFrame) -> dict:
    y = g["ko_win"].to_numpy(int)
    p = g["p_ko"].to_numpy(float)
    return {
        "n": int(len(g)),
        "actual_rate": float(y.mean()),
        "predicted_rate": float(p.mean()),
        "eo": float(y.sum() / p.sum()) if p.sum() > 0 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "mean_p_actual_ko": float(p[y == 1].mean()) if (y == 1).any() else np.nan,
        "mean_p_non_ko": float(p[y == 0].mean()) if (y == 0).any() else np.nan,
        "extreme_fp_ge_050": int(((p >= 0.50) & (y == 0)).sum()),
        "extreme_fp_ge_030": int(((p >= 0.30) & (y == 0)).sum()),
    }


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ff = add_prefight_history(load_fighter_fights())
    preds = []
    years = sorted(y for y in ff.test_year.unique() if y >= 2020)
    for year in years:
        train = ff[ff.event_date < pd.Timestamp(f"{year}-01-01")].copy()
        test = ff[ff.test_year.eq(year)].copy()
        if len(train) < 500 or len(test) < 20:
            continue
        for denom in DENOMS:
            for pe in PRIOR_STRENGTH_EVENTS:
                preds.append(predict_year(test, train, denom, pe))
    pred = pd.concat(preds, ignore_index=True)
    rows = []
    for period, yrs in [("selection_2020_2024", SELECTION_YEARS), ("confirmation_2025_2026", CONFIRMATION_YEARS)]:
        for (denom, pe), g in pred[pred.test_year.isin(yrs)].groupby(["denominator", "prior_events"]):
            m = metrics(g)
            m.update(period=period, denominator=denom, prior_events=float(pe))
            rows.append(m)
    summary = pd.DataFrame(rows)
    selection = summary[summary.period.eq("selection_2020_2024")].sort_values(["log_loss", "brier"])
    best = selection.groupby("denominator", as_index=False).first().sort_values(["log_loss", "brier"])
    chosen = best[["denominator", "prior_events"]]
    confirm = summary[summary.period.eq("confirmation_2025_2026")].merge(chosen, on=["denominator", "prior_events"], how="inner").sort_values(["log_loss", "brier"])

    pred.to_csv(OUT / "predictions.csv", index=False)
    summary.to_csv(OUT / "all_summary.csv", index=False)
    best.to_csv(OUT / "selection_best_by_denominator.csv", index=False)
    confirm.to_csv(OUT / "confirmation_selected.csv", index=False)
    report = {
        "study": "KO denominator identification",
        "production_changed": False,
        "brain_used": False,
        "same_date_delayed": True,
        "selection_years": list(SELECTION_YEARS),
        "confirmation_years": list(CONFIRMATION_YEARS),
        "important_limitation": "Realized test-fight exposure is intentionally used to identify the statistical denominator; this is not a deployable prefight predictor.",
        "unsupported_denominator": "phase time (UFCStats has phase strike counts, not standing/clinch/ground time)",
        "confirmation_selected": confirm.to_dict(orient="records"),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("\nKO DENOMINATOR STUDY — SELECTION BEST BY DENOMINATOR")
    print(best.to_string(index=False))
    print("\nKO DENOMINATOR STUDY — UNTOUCHED 2025-2026 CONFIRMATION")
    print(confirm.to_string(index=False))


if __name__ == "__main__":
    run()
