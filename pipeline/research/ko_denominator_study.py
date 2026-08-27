"""Research-only KO denominator study, corrected prefight OOS design.

Question: which historical exposure denominator produces the most useful leakage-safe
prefight KO offense/defense signal?

No current-fight realized exposure is used in prediction. This avoids informative
censoring (KO itself shortens fight time and strike accumulation). Every candidate
uses the same model form, same-date-delayed histories, expanding-year OOS fitting,
2020-2024 selection, and untouched 2025-2026 confirmation.

Candidates:
  * prior fights (fight-level KO rate)
  * prior elapsed fight seconds
  * prior significant strikes landed
  * prior significant strikes attempted
  * prior head significant strikes landed
  * prior distance significant strikes landed

This identifies the best normalization for PREFIGHT KO strength. A separate survival
study is still required before turning a time-based winner into an in-fight clock.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline.common.fight_time import repair_elapsed_match_time
from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH

OUT = Path("data/research/ko_denominator_study")
SELECTION_YEARS = tuple(range(2020, 2025))
CONFIRMATION_YEARS = (2025, 2026)
PRIOR_EVENTS = (1.0, 2.0, 5.0, 10.0)

ROUND_COLS = [
    "event_date", "fight_id", "fighter_id", "fighter_name", "opponent_id",
    "sig_str_landed", "sig_str_attempted", "head_landed", "distance_landed",
]

DENOMS = {
    "fight_count": ("fight_count", "fight_count"),
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
    cols = ["fight_id", "date", "method", "winner_id", "division", "match_time_sec"]
    agg = agg.merge(m[cols], on="fight_id", how="left", validate="many_to_one")
    agg["event_date"] = agg["date"]
    agg["fight_seconds"] = pd.to_numeric(agg["match_time_sec"], errors="coerce")
    agg["fight_count"] = 1.0
    agg["won"] = agg["fighter_id"].astype(str).eq(agg["winner_id"].astype(str))
    agg["ko_win"] = (agg["won"] & _ko_tko(agg["method"])).astype(int)
    agg["ko_loss"] = ((~agg["won"]) & _ko_tko(agg["method"])).astype(int)
    agg["test_year"] = agg["event_date"].dt.year
    needed = ["fight_seconds", "sig_absorbed", "sig_attempted_absorbed", "head_absorbed", "distance_absorbed"]
    agg = agg.dropna(subset=needed).copy()
    return agg.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def add_prefight_history(ff: pd.DataFrame) -> pd.DataFrame:
    state = defaultdict(lambda: defaultdict(float))
    rows = []
    fields = [
        "ko_win", "ko_loss", "fight_count", "fight_seconds", "sig_landed", "sig_absorbed",
        "sig_attempted", "sig_attempted_absorbed", "head_landed", "head_absorbed",
        "distance_landed", "distance_absorbed",
    ]
    for _, batch in ff.groupby("event_date", sort=True):
        # snapshot first: no same-event leakage
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


def population_rate(train: pd.DataFrame, denom: str) -> float:
    att_col, _ = DENOMS[denom]
    exp = float(train[att_col].sum())
    return float(train["ko_win"].sum() / exp) if exp > 0 else np.nan


def make_features(frame: pd.DataFrame, denom: str, p0: float, prior_events: float) -> pd.DataFrame:
    att_col, def_col = DENOMS[denom]
    prior_exp = prior_events / p0
    att_exp = frame[f"prior_{att_col}"].to_numpy(float)
    def_exp = frame[f"prior_{def_col}"].to_numpy(float)
    att = (frame["prior_ko_win"].to_numpy(float) + prior_events) / (att_exp + prior_exp)
    deff = (frame["prior_ko_loss"].to_numpy(float) + prior_events) / (def_exp + prior_exp)
    x = pd.DataFrame(index=frame.index)
    # Rate ratios make units comparable across candidate denominators.
    x["att_log_rate_ratio"] = np.log(np.clip(att / p0, 1e-9, 1e9))
    x["def_log_rate_ratio"] = np.log(np.clip(deff / p0, 1e-9, 1e9))
    # Exposure confidence; standardized by equivalent prior exposure.
    x["att_log_confidence"] = np.log1p(att_exp / prior_exp)
    x["def_log_confidence"] = np.log1p(def_exp / prior_exp)
    # Fight count is included for sample-size context identically in every arm.
    x["log_prior_fights"] = np.log1p(frame["prior_fight_count"].to_numpy(float))
    x["division"] = frame["division"].fillna("UNKNOWN").astype(str).to_numpy()
    return x


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, denom: str, pe: float) -> np.ndarray:
    p0 = population_rate(train, denom)
    if not np.isfinite(p0) or p0 <= 0:
        raise RuntimeError(f"invalid population rate {denom}")
    xtr = make_features(train, denom, p0, pe)
    xte = make_features(test, denom, p0, pe)
    nums = ["att_log_rate_ratio", "def_log_rate_ratio", "att_log_confidence", "def_log_confidence", "log_prior_fights"]
    prep = ColumnTransformer([
        ("num", StandardScaler(), nums),
        ("div", OneHotEncoder(handle_unknown="ignore"), ["division"]),
    ])
    model = Pipeline([
        ("prep", prep),
        ("logit", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")),
    ])
    model.fit(xtr, train["ko_win"].to_numpy(int))
    return model.predict_proba(xte)[:, 1]


def metrics(g: pd.DataFrame) -> dict:
    y = g["ko_win"].to_numpy(int)
    p = np.clip(g["p_ko"].to_numpy(float), 1e-9, 1 - 1e-9)
    return {
        "n": int(len(g)), "actual_rate": float(y.mean()), "predicted_rate": float(p.mean()),
        "eo": float(y.sum() / p.sum()), "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "mean_p_actual_ko": float(p[y == 1].mean()), "mean_p_non_ko": float(p[y == 0].mean()),
        "extreme_fp_ge_050": int(((p >= .50) & (y == 0)).sum()),
        "extreme_fp_ge_030": int(((p >= .30) & (y == 0)).sum()),
    }


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ff = add_prefight_history(load_fighter_fights())
    preds = []
    for year in sorted(y for y in ff.test_year.unique() if y >= 2020):
        train = ff[ff.event_date < pd.Timestamp(f"{year}-01-01")].copy()
        test = ff[ff.test_year.eq(year)].copy()
        if len(train) < 500 or len(test) < 20:
            continue
        for denom in DENOMS:
            for pe in PRIOR_EVENTS:
                p = fit_predict(train, test, denom, pe)
                d = test[["event_date", "fight_id", "fighter_id", "fighter_name", "opponent_id", "division", "ko_win", "test_year"]].copy()
                d["denominator"] = denom
                d["prior_events"] = pe
                d["p_ko"] = p
                preds.append(d)
    pred = pd.concat(preds, ignore_index=True)
    rows = []
    periods = [("selection_2020_2024", SELECTION_YEARS), ("confirmation_2025_2026", CONFIRMATION_YEARS)]
    for period, yrs in periods:
        q = pred[pred.test_year.isin(yrs)]
        for (denom, pe), g in q.groupby(["denominator", "prior_events"]):
            m = metrics(g); m.update(period=period, denominator=denom, prior_events=float(pe)); rows.append(m)
    summary = pd.DataFrame(rows)
    sel = summary[summary.period.eq("selection_2020_2024")].sort_values(["log_loss", "brier"])
    best = sel.groupby("denominator", as_index=False).first().sort_values(["log_loss", "brier"])
    chosen = best[["denominator", "prior_events"]]
    conf = summary[summary.period.eq("confirmation_2025_2026")].merge(chosen, on=["denominator", "prior_events"], how="inner").sort_values(["log_loss", "brier"])

    pred.to_csv(OUT / "predictions.csv", index=False)
    summary.to_csv(OUT / "all_summary.csv", index=False)
    best.to_csv(OUT / "selection_best_by_denominator.csv", index=False)
    conf.to_csv(OUT / "confirmation_selected.csv", index=False)
    report = {
        "study": "KO denominator identification — prefight OOS",
        "production_changed": False, "brain_used": False, "same_date_delayed": True,
        "uses_current_fight_exposure": False,
        "selection_years": list(SELECTION_YEARS), "confirmation_years": list(CONFIRMATION_YEARS),
        "interpretation": "Ranks which historical exposure normalization carries the strongest prefight KO offense/defense signal. Does not yet prove the correct in-fight event clock.",
        "unsupported": "phase-time denominators; UFCStats has phase strike counts but not phase time",
        "confirmation_selected": conf.to_dict(orient="records"),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("\nKO DENOMINATOR STUDY V2 — SELECTION BEST BY DENOMINATOR")
    print(best.to_string(index=False))
    print("\nKO DENOMINATOR STUDY V2 — UNTOUCHED 2025-2026 CONFIRMATION")
    print(conf.to_string(index=False))


if __name__ == "__main__":
    run()
