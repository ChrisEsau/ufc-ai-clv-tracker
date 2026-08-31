from pathlib import Path
import json
import numpy as np
import pandas as pd

from pipeline.research import xgboost_method_market_offset as method
from pipeline.research.xgboost_method_hierarchical_v5_oof import _fit_conditional, _metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/research/prop_mispricing"
V5_TEST = OUT / "xgboost_v5_exact_reproduction_test_predictions.csv"
FEATURE_LIST = OUT / "xgboost_method_market_offset__feature_list.json"
MARKET = ROOT / "data/market/historical_market_outcomes.parquet"
PRED = OUT / "xgboost_method_hierarchical_v5_holdout_2025_2026_predictions.csv"
LEDGER = OUT / "xgboost_method_hierarchical_v5_holdout_2025_2026_betting_ledger.csv"
SUMMARY = OUT / "xgboost_method_hierarchical_v5_holdout_2025_2026_betting_summary.json"

THRESHOLD = 0.30
EPS = 1e-12
SLUGS = method.SLUGS
CLASS_META = {
    "red_ko": ("red", "win_by_ko_tko_dq", 0),
    "red_sub": ("red", "win_by_submission", 1),
    "red_dec": ("red", "win_by_decision", 2),
    "blue_ko": ("blue", "win_by_ko_tko_dq", 3),
    "blue_sub": ("blue", "win_by_submission", 4),
    "blue_dec": ("blue", "win_by_decision", 5),
}

def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))

def summarize(d):
    if d.empty:
        return {"bets":0,"wins":0,"losses":0,"stake_units":0.0,"profit_units":0.0,"roi":None,"hit_rate":None,"fights_bet":0,"bets_per_fight":None}
    stake=float(d.stake_units.sum()); profit=float(d.profit_units.sum())
    return {"bets":int(len(d)),"wins":int(d.won.sum()),"losses":int((1-d.won).sum()),"stake_units":stake,"profit_units":profit,"roi":profit/stake,"hit_rate":float(d.won.mean()),"fights_bet":int(d.fight_id.nunique()),"bets_per_fight":float(len(d)/d.fight_id.nunique())}

features = json.loads(FEATURE_LIST.read_text())["features"]
df, _, _ = method._build_rows(False, True, forced_features=features)
df["date"] = pd.to_datetime(df["date"])
train = df[df.date <= "2024-12-31"].copy()
test = df[(df.date >= "2025-01-01") & (df.date < "2027-01-01")].copy()

v5 = pd.read_csv(V5_TEST)
v5["fight_id"] = v5.fight_id.astype(str)
v5["date"] = pd.to_datetime(v5.date)
v5 = v5[(v5.market_key == "moneyline") & (v5.bookmaker == "legacy_consensus") & (v5.canonical_side == "red") & (v5.result_status == "graded")].copy()
v5 = v5[(v5.date >= "2025-01-01") & (v5.date < "2027-01-01")][["fight_id","model_p_red"]].drop_duplicates("fight_id")

test["fight_id"] = test.fight_id.astype(str)
test = test.merge(v5, on="fight_id", how="inner")
if test.empty:
    raise RuntimeError("no hierarchical/V5 2025-2026 overlap")

red_cond, red_train_n, red_fc = _fit_conditional(train, test, features, "red")
blue_cond, blue_train_n, blue_fc = _fit_conditional(train, test, features, "blue")
p_red = np.clip(test.model_p_red.to_numpy(float), EPS, 1-EPS)
p = np.concatenate([p_red[:,None]*red_cond, (1-p_red)[:,None]*blue_cond], axis=1)
p = p / p.sum(axis=1, keepdims=True)

pred = test[["fight_id","date","event_name","red_fighter","blue_fighter","target"]].copy()
pred["v5_model_p_red"] = p_red
for j, slug in enumerate(SLUGS):
    pred[f"hier_{slug}"] = p[:,j]
    pred[f"market_{slug}"] = test[f"market_{slug}"].to_numpy(float)
pred.to_csv(PRED, index=False)

m = pd.read_parquet(MARKET).copy()
m["fight_id"] = m.fight_id.astype(str)
m = m[(m.bookmaker == "legacy_consensus") & m.outcome_side.astype(str).isin(["red","blue"])].copy()
m["implied_probability"] = pd.to_numeric(m.implied_probability, errors="coerce")
m = m[np.isfinite(m.implied_probability) & (m.implied_probability > 0) & (m.implied_probability < 1)].copy()
price_map={}
for slug,(side,key,_) in CLASS_META.items():
    z=m[(m.outcome_side.astype(str)==side)&(m.market_key==key)].copy()
    counts=z.groupby("fight_id").size()
    good=counts[counts==1].index
    z=z[z.fight_id.isin(good)]
    for r in z[["fight_id","implied_probability"]].itertuples(index=False):
        price_map[(str(r.fight_id),slug)] = float(r.implied_probability)

rows=[]
for r in pred.itertuples(index=False):
    side = "red" if float(r.v5_model_p_red) >= 0.5 else "blue"
    for slug in SLUGS:
        bet_side,_,class_idx = CLASS_META[slug]
        if bet_side != side: continue
        model_p=float(getattr(r,f"hier_{slug}")); market_p=float(getattr(r,f"market_{slug}"))
        resid=float(logit(model_p)-logit(market_p))
        if resid < THRESHOLD: continue
        raw=price_map.get((str(r.fight_id),slug))
        if raw is None: continue
        dec=1.0/raw; won=int(int(r.target)==class_idx); profit=(dec-1.0) if won else -1.0
        rows.append({"fight_id":str(r.fight_id),"date":r.date,"year":int(pd.Timestamp(r.date).year),"event_name":r.event_name,"red_fighter":r.red_fighter,"blue_fighter":r.blue_fighter,"predicted_side":side,"bet_slug":slug,"model_probability":model_p,"normalized_market_probability":market_p,"signed_logit_residual":resid,"raw_implied_probability":raw,"decimal_odds":dec,"won":won,"stake_units":1.0,"profit_units":profit})
ledger=pd.DataFrame(rows)
if not ledger.empty: ledger=ledger.sort_values(["date","fight_id","bet_slug"]).reset_index(drop=True)
ledger.to_csv(LEDGER,index=False)

year_metrics={}
for y,g in ledger.groupby("year") if not ledger.empty else []:
    year_metrics[str(int(y))]=summarize(g)

metrics_all=_metrics(pred.target.to_numpy(int), p)
metrics_by_year={}
for y,g in pred.groupby(pred.date.dt.year):
    pp=g[[f"hier_{s}" for s in SLUGS]].to_numpy(float)
    metrics_by_year[str(int(y))]=_metrics(g.target.to_numpy(int),pp)

counts=ledger.groupby("fight_id").size() if not ledger.empty else pd.Series(dtype=int)
summary={
    "experiment":"hierarchical_v5_frozen_2025_2026_holdout_betting_v1",
    "model":"new hierarchical V5 model only",
    "training_cutoff":"2024-12-31",
    "evaluation_period":f"{pred.date.min().date()} through {pred.date.max().date()}",
    "holdout_fights":int(len(pred)),
    "red_conditional_train_n":int(red_train_n),"blue_conditional_train_n":int(blue_train_n),"feature_count":int(len(features)),
    "rule":{"winner_side":"red if frozen V5 P(red)>=0.5 else blue","eligible_methods":"KO/SUB/DEC on predicted winner side only","signed_logit_residual_threshold":THRESHOLD,"multiple_bets_per_fight_allowed":True,"stake_per_bet_units":1.0,"payout_price":"raw legacy-consensus implied probability as decimal 1/p"},
    "model_metrics":metrics_all,"model_metrics_by_year":metrics_by_year,
    "pooled":summarize(ledger),"by_year":year_metrics,
    "bet_count_distribution":{"fights_with_1_bet":int((counts==1).sum()),"fights_with_2_bets":int((counts==2).sum()),"fights_with_3_bets":int((counts==3).sum())}
}
SUMMARY.write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
