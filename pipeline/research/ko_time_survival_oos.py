"""Research-only censored KO/TKO time-survival study.

Question: should KO be modeled as a time clock rather than a per-strike roll?

Each fighter-side is a survival process:
  * KO/TKO winner: event at observed finish time
  * opponent in that KO/TKO fight: censored at the same time
  * all other outcomes: censored at observed fight end

All fighter histories are same-date delayed. No Brain or production mechanics are changed.
Selection years: 2020-2024. Untouched confirmation: 2025-2026.

Models:
  1) population constant time hazard
  2) attacker/defender EB constant time hazard
  3) attacker/defender EB + piecewise round baseline hazard
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline.common.fight_time import repair_elapsed_match_time
from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH

OUT = Path("data/research/ko_time_survival_oos")
SELECTION_YEARS = tuple(range(2020, 2025))
CONFIRMATION_YEARS = (2025, 2026)
PRIOR_EVENTS = (0.5, 1.0, 2.0, 5.0, 10.0)
CUTS = np.array([0.0, 300.0, 600.0, 900.0, 1200.0, 1500.0])


def _ko_tko(s: pd.Series) -> pd.Series:
    t = s.fillna("").astype(str).str.upper()
    return t.str.contains(r"KO/TKO|\\bTKO\\b|\\bKO\\b", regex=True)


def load_fighter_fights() -> pd.DataFrame:
    # Use round stats for canonical fighter/opponent IDs; no strike features are used.
    r = pd.read_parquet(ROUND_STATS_PATH, columns=["event_date", "fight_id", "fighter_id", "fighter_name", "opponent_id"])
    r["event_date"] = pd.to_datetime(r["event_date"]).dt.normalize()
    ff = r.groupby(["event_date", "fight_id", "fighter_id"], as_index=False).agg(
        fighter_name=("fighter_name", "first"), opponent_id=("opponent_id", "first")
    )
    m = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    m = repair_elapsed_match_time(m)
    m["date"] = pd.to_datetime(m["date"]).dt.normalize()
    cols = ["fight_id", "date", "method", "winner_id", "r_id", "b_id", "division", "match_time_sec"]
    ff = ff.merge(m[cols], on="fight_id", how="left", validate="many_to_one")
    ff["event_date"] = ff["date"]
    ff["fight_seconds"] = pd.to_numeric(ff["match_time_sec"], errors="coerce")
    ff = ff[ff.fight_seconds.gt(0) & ff.fight_seconds.le(1500)].copy()
    ff["won"] = ff.fighter_id.astype(str).eq(ff.winner_id.astype(str))
    ff["ko_event"] = (ff.won & _ko_tko(ff.method)).astype(int)
    ff["ko_loss"] = ((~ff.won) & _ko_tko(ff.method)).astype(int)
    ff["test_year"] = ff.event_date.dt.year
    return ff.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def add_prefight(ff: pd.DataFrame) -> pd.DataFrame:
    state = defaultdict(lambda: {"seconds": 0.0, "ko_win": 0.0, "ko_loss": 0.0})
    rows = []
    for _, batch in ff.groupby("event_date", sort=True):
        for row in batch.itertuples(index=False):
            d = row._asdict()
            s = state[str(row.fighter_id)]
            d["prior_seconds"] = s["seconds"]
            d["prior_ko_win"] = s["ko_win"]
            d["prior_ko_loss"] = s["ko_loss"]
            rows.append(d)
        for row in batch.itertuples(index=False):
            s = state[str(row.fighter_id)]
            s["seconds"] += float(row.fight_seconds)
            s["ko_win"] += float(row.ko_event)
            s["ko_loss"] += float(row.ko_loss)
    x = pd.DataFrame(rows)
    opp = x[["event_date", "fight_id", "fighter_id", "prior_seconds", "prior_ko_loss"]].rename(columns={
        "fighter_id": "opponent_id", "prior_seconds": "opp_prior_seconds", "prior_ko_loss": "opp_prior_ko_loss"
    })
    x = x.merge(opp, on=["event_date", "fight_id", "opponent_id"], how="left", validate="one_to_one")
    return x


def interval_exposure(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float)
    return np.stack([np.clip(t - CUTS[i], 0.0, CUTS[i+1] - CUTS[i]) for i in range(len(CUTS)-1)], axis=1)


def interval_event_index(t: np.ndarray) -> np.ndarray:
    # Event exactly at 300 belongs to first interval, 600 to second, etc.
    return np.clip(np.searchsorted(CUTS[1:], np.asarray(t, float), side="left"), 0, len(CUTS)-2)


def train_baselines(train: pd.DataFrame) -> tuple[float, np.ndarray]:
    sec = float(train.fight_seconds.sum())
    ev = float(train.ko_event.sum())
    p0 = ev / sec
    exp = interval_exposure(train.fight_seconds.to_numpy(float)).sum(axis=0)
    events = np.zeros(len(CUTS)-1, float)
    idx = interval_event_index(train.loc[train.ko_event.eq(1), "fight_seconds"].to_numpy(float))
    for j in idx:
        events[int(j)] += 1.0
    # modest Gamma-style stabilization toward overall rate: 2 equivalent events per interval
    prior_e = 2.0
    prior_x = prior_e / p0
    piece = (events + prior_e) / (exp + prior_x)
    return p0, piece


def fighter_rr(test: pd.DataFrame, p0: float, prior_events: float) -> np.ndarray:
    prior_sec = prior_events / p0
    att = (test.prior_ko_win.to_numpy(float) + prior_events) / (test.prior_seconds.to_numpy(float) + prior_sec)
    deff = (test.opp_prior_ko_loss.to_numpy(float) + prior_events) / (test.opp_prior_seconds.to_numpy(float) + prior_sec)
    return np.clip(att * deff / (p0 * p0), 0.05, 20.0)


def survival_nll_constant(event: np.ndarray, t: np.ndarray, hazard: np.ndarray) -> float:
    h = np.clip(np.asarray(hazard,float), 1e-12, None)
    return float(np.mean(h*t - event*np.log(h)))


def survival_nll_piecewise(event: np.ndarray, t: np.ndarray, rr: np.ndarray, base: np.ndarray) -> tuple[float, np.ndarray]:
    exp = interval_exposure(t)
    cumhaz = (exp * (rr[:,None] * base[None,:])).sum(axis=1)
    idx = interval_event_index(t)
    event_h = rr * base[idx]
    nll = cumhaz - event*np.log(np.clip(event_h,1e-12,None))
    return float(np.mean(nll)), cumhaz


def score_year(test: pd.DataFrame, train: pd.DataFrame, prior_events: float) -> list[dict]:
    p0, piece = train_baselines(train)
    y = test.ko_event.to_numpy(int)
    t = test.fight_seconds.to_numpy(float)
    rr = fighter_rr(test, p0, prior_events)
    rows = []

    for name, haz in [
        ("population_constant", np.full(len(test), p0)),
        ("fighter_constant", p0*rr),
    ]:
        nll = survival_nll_constant(y,t,haz)
        # Ranking uses prefight hazard only, never realized test duration.
        auc = roc_auc_score(y,haz) if np.unique(y).size==2 else np.nan
        rows.append({"model":name,"survival_nll":nll,"event_auc_prefight_hazard":float(auc),
                     "population_hazard_per_second":p0,"prior_events":prior_events})

    nll, _ = survival_nll_piecewise(y,t,rr,piece)
    # Piecewise baseline is common to every fighter, so prefight ranking is rr.
    auc = roc_auc_score(y,rr) if np.unique(y).size==2 else np.nan
    rows.append({"model":"fighter_piecewise_round","survival_nll":nll,"event_auc_prefight_hazard":float(auc),
                 "population_hazard_per_second":p0,"prior_events":prior_events,
                 **{f"baseline_r{i+1}_per_second":float(v) for i,v in enumerate(piece)}})
    return rows


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ff = add_prefight(load_fighter_fights())
    rows = []
    for year in sorted(y for y in ff.test_year.unique() if y >= 2020):
        train = ff[ff.event_date < pd.Timestamp(f"{year}-01-01")].copy()
        test = ff[ff.test_year.eq(year)].copy()
        if len(train)<500 or len(test)<20: continue
        for pe in PRIOR_EVENTS:
            for r in score_year(test,train,pe):
                r.update(test_year=int(year), n=int(len(test)), events=int(test.ko_event.sum()))
                rows.append(r)
    by_year = pd.DataFrame(rows)

    pooled=[]
    for period,years in [("selection_2020_2024",SELECTION_YEARS),("confirmation_2025_2026",CONFIRMATION_YEARS)]:
        g=by_year[by_year.test_year.isin(years)]
        for (model,pe),z in g.groupby(["model","prior_events"]):
            # weighted by fighter-side rows
            w=z.n.to_numpy(float)
            pooled.append({"period":period,"model":model,"prior_events":float(pe),"n":int(w.sum()),"events":int(z.events.sum()),
                           "survival_nll":float(np.average(z.survival_nll,weights=w)),
                           "event_auc_prefight_hazard":float(np.average(z.event_auc_prefight_hazard,weights=w))})
    pooled=pd.DataFrame(pooled)
    sel=pooled[pooled.period.eq("selection_2020_2024")].sort_values(["survival_nll","event_auc_prefight_hazard"],ascending=[True,False])
    best=sel.groupby("model",as_index=False).first().sort_values("survival_nll")
    chosen=best[["model","prior_events"]]
    confirm=pooled[pooled.period.eq("confirmation_2025_2026")].merge(chosen,on=["model","prior_events"],how="inner").sort_values("survival_nll")

    # Descriptive empirical event-time distribution, not used for fitting selection.
    ev=ff[ff.ko_event.eq(1)].copy()
    event_bins=pd.cut(ev.fight_seconds,bins=CUTS,right=True,include_lowest=True)
    event_dist=ev.groupby(event_bins,observed=False).size().reset_index(name="ko_events")
    event_dist["share"] = event_dist.ko_events / max(event_dist.ko_events.sum(),1)

    by_year.to_csv(OUT/"by_year.csv",index=False)
    pooled.to_csv(OUT/"pooled.csv",index=False)
    best.to_csv(OUT/"selection_best.csv",index=False)
    confirm.to_csv(OUT/"confirmation_selected.csv",index=False)
    event_dist.to_csv(OUT/"ko_event_time_distribution.csv",index=False)
    report={"study":"censored KO/TKO time survival","production_changed":False,"brain_used":False,
            "same_date_delayed":True,"selection_years":list(SELECTION_YEARS),"confirmation_years":list(CONFIRMATION_YEARS),
            "models":["population_constant","fighter_constant","fighter_piecewise_round"],
            "confirmation":confirm.to_dict(orient="records")}
    (OUT/"report.json").write_text(json.dumps(report,indent=2,default=str)+"\n")
    print("KO TIME SURVIVAL — SELECTION BEST")
    print(best.to_string(index=False))
    print("\nKO TIME SURVIVAL — UNTOUCHED 2025-2026 CONFIRMATION")
    print(confirm.to_string(index=False))
    print("\nKO EVENT TIME DISTRIBUTION")
    print(event_dist.to_string(index=False))

if __name__ == "__main__":
    run()
