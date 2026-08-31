from __future__ import annotations

import subprocess
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.features.views.moneyline import build_moneyline_feature_view

OUT = Path("data/research/prop_mispricing")
OUTPUT = OUT / "ufc_shanghai_v5_moneyline_20260829.csv"
SNAP = "7df1b61126be1f4e036b256d1c774c531b8a281f"
FIGHT_DATE = pd.Timestamp("2026-08-29")

FIGHTS = [
    ("Umar Nurmagomedov", "Song Yadong", -550, 400, 5),
    ("Yan Xiaonan", "Denise Gomes", -150, 125, 3),
    ("Aoriqileng", "Kai Asakura", 350, -450, 3),
    ("Alex Perez", "Sumudaerji", 180, -220, 3),
    ("Liu Ce", "Levi Rodrigues Jr.", -200, 165, 3),
    ("Bilal Hasan", "Nilson Rojas", -800, 550, 3),
    ("Namsrai Batbayar", "Andre Lima", 220, -270, 3),
    ("Rei Tsuruya", "Kevin Borjas", -650, 475, 3),
    ("Jack Jenkins", "Sean Woodson", 125, -150, 3),
    ("Xiao Long", "Francesco Nuzzi", -175, 145, 3),
    ("Lawrence Lui", "Hector Santiago", -275, 225, 3),
    ("Xiong Jingnan", "Julia Polastri", 200, -245, 3),
    ("Ding Meng", "Cam Nelson", -150, 125, 3),
]

SELECTED = [
    "reach_diff","recent_form_recent_avg_fight_time_diff","age_diff","ewm_sapm_diff","ewm_recent_sapm_diff",
    "style_ko_finisher_score_diff","ewm_td_acc_diff","recent_finish_rate_diff","chin_risk_diff","recent_form_avg_opponent_elo_diff",
    "recent_avg_fight_time_diff","aggression_index_diff","age_squared_diff","sapm_diff","ewm_kd_avg_diff",
    "style_all_round_finisher_score_diff","recent_form_kd_absorbed_avg_diff","ewm_recent_splm_diff","elo_diff","ewm_elo_diff",
    "ewm_recent_td_avg_diff","days_since_last_fight_diff","td_avg_diff","style_score_spread_diff","ko_dependency_diff",
    "recent_form_avg_fight_time_diff","wrestling_mismatch_diff","win_pct_diff","recent_form_ko_rate_diff","recent_form_worst_loss_elo_diff",
    "age_x_career_ko_losses_diff","ewm_str_def_diff","losses_diff","ewm_recent_win_pct_diff","avg_opponent_elo_diff",
    "ewm_td_avg_diff","avg_fight_time_diff","ewm_days_since_last_fight_diff","pressure_striking_adv_diff","weight_diff",
    "ctrl_against_per_min_diff","ewm_finish_loss_rate_diff","ewm_win_pct_diff","victory_concentration_index_diff","recent_form_td_acc_diff",
    "sub_avg_diff","recent_form_best_win_elo_diff","ewm_best_win_elo_diff","style_primary_score_diff","recent_form_recent_finish_rate_diff",
]


def norm(s: object) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(ch)).replace("’", "'").lower().strip()


def imp(ao: float) -> float:
    return 100.0 / (ao + 100.0) if ao > 0 else (-ao) / ((-ao) + 100.0)


def clip_p(p): return np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
def logit(p):
    p = clip_p(p); return np.log(p / (1 - p))
def sigmoid(z):
    z = np.clip(np.asarray(z, float), -30, 30); return 1 / (1 + np.exp(-z))


def add_live_engineered(livefv: pd.DataFrame) -> pd.DataFrame:
    def _num(c): return pd.to_numeric(livefv[c], errors="coerce") if c in livefv.columns else pd.Series(np.nan, index=livefv.index)
    def _rate(c):
        s = _num(c); return pd.Series(np.where(s > 1, s / 100.0, s), index=livefv.index)
    livefv["chin_risk_diff"] = _num("r_pre_sapm") * (1 - _rate("r_pre_str_def")) - _num("b_pre_sapm") * (1 - _rate("b_pre_str_def"))
    livefv["aggression_index_diff"] = (_num("r_pre_splm") + _num("r_pre_td_avg")) - (_num("b_pre_splm") + _num("b_pre_td_avg"))
    livefv["age_squared_diff"] = _num("r_pre_age") ** 2 - _num("b_pre_age") ** 2
    livefv["wrestling_mismatch_diff"] = _num("r_pre_td_avg") * (1 - _rate("b_pre_td_def")) - _num("b_pre_td_avg") * (1 - _rate("r_pre_td_def"))
    livefv["pressure_striking_adv_diff"] = _num("r_pre_splm") * (1 - _rate("b_pre_str_def")) - _num("b_pre_splm") * (1 - _rate("r_pre_str_def"))
    livefv["age_x_career_ko_losses_diff"] = _num("r_pre_age") * _num("r_pre_career_ko_losses") - _num("b_pre_age") * _num("b_pre_career_ko_losses")
    return livefv


def fit_frozen_v5():
    OUT.mkdir(parents=True, exist_ok=True)
    for repo_path, out_path in [("data/market/historical_market_outcomes.parquet", "/tmp/v5_market.parquet"),("data/features/moneyline_feature_view.parquet", "/tmp/v5_fv.parquet")]:
        with open(out_path, "wb") as f: subprocess.run(["git", "show", f"{SNAP}:{repo_path}"], stdout=f, check=True)
    market = pd.read_parquet("/tmp/v5_market.parquet").copy()
    market = market[(market.bookmaker == "legacy_consensus") & (market.result_status == "graded") & market.won.notna()].copy()
    market["date"] = pd.to_datetime(market.date, errors="coerce"); market["won"] = market.won.astype(bool).astype(int); market["implied_probability"] = pd.to_numeric(market.implied_probability, errors="coerce")
    market = market.dropna(subset=["date", "implied_probability"]).copy(); ml = market[market.market_key == "moneyline"].copy(); good = ml.groupby("fight_id").size(); ml = ml[ml.fight_id.isin(good[good == 2].index)].copy()
    ml["market_overround"] = ml.groupby("fight_id").implied_probability.transform("sum"); ml["fair_market_p"] = ml.implied_probability / ml.market_overround; red = ml[ml.outcome_side.astype(str).eq("red")].copy(); fv = pd.read_parquet("/tmp/v5_fv.parquet").copy()
    feature_cols = SELECTED + ["market_overround"]; df = red.merge(fv[["fight_id"] + [c for c in SELECTED if c in fv.columns]], on="fight_id", how="inner").sort_values(["date", "fight_id"]).copy()
    missing_train = [c for c in SELECTED if c not in df.columns]
    if missing_train: raise RuntimeError(f"Frozen V5 training snapshot missing selected features: {missing_train}")
    xraw = df[feature_cols].replace([np.inf, -np.inf], np.nan); tr = df.date <= "2024-12-31"; valid = [c for c in feature_cols if xraw.loc[tr, c].notna().any()]; med = xraw.loc[tr, valid].median(numeric_only=True); xtr = xraw.loc[tr, valid].fillna(med).fillna(0.0); ytr = df.loc[tr, "won"].astype(int).to_numpy(); mtr = logit(df.loc[tr, "fair_market_p"])
    params = {"max_depth":1,"eta":0.03,"subsample":0.8,"colsample_bytree":0.7,"min_child_weight":10,"lambda":8.0,"alpha":1.0,"objective":"binary:logistic","eval_metric":"logloss","seed":42,"nthread":2}; dtr = xgb.DMatrix(xtr, label=ytr, base_margin=mtr, feature_names=valid); model = xgb.train(params, dtr, num_boost_round=300, verbose_eval=False)
    return model, valid, med


def build_prefight_view():
    hist = pd.read_parquet("data/features/fighter_state_history.parquet").copy()
    if "fighter_name" not in hist.columns or "fighter_id" not in hist.columns or "date" not in hist.columns: raise RuntimeError("fighter_state_history lacks fighter_name/fighter_id/date")
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce"); hist["fighter_id"] = hist.fighter_id.astype(str); hist["_norm_name"] = hist.fighter_name.map(norm)
    prep_rows, state_rows, meta = [], [], {}
    for i, (a, b, ao_a, ao_b, rounds) in enumerate(FIGHTS, 1):
        fid = f"shanghai_v5_20260829_{i:02d}"; ids, found, state_dates = [], [], []
        for nm in (a, b):
            hit = hist[(hist._norm_name.eq(norm(nm))) & (hist.date < FIGHT_DATE)].sort_values("date")
            if len(hit): rec = hit.iloc[-1].drop(labels=["_norm_name"]).to_dict(); fighter_id = str(rec["fighter_id"]); ok = True; sd = pd.Timestamp(rec["date"]).strftime("%Y-%m-%d")
            else: rec = {c: np.nan for c in hist.columns if c != "_norm_name"}; fighter_id = f"missing::{norm(nm)}"; rec["fighter_id"] = fighter_id; rec["fighter_name"] = nm; ok = False; sd = None
            rec["fight_id"] = fid; state_rows.append(rec); ids.append(fighter_id); found.append(ok); state_dates.append(sd)
        prep_rows.append({"fight_id":fid,"r_id":ids[0],"b_id":ids[1],"r_name":a,"b_name":b,"date":FIGHT_DATE,"title_fight":False,"total_rounds":rounds}); meta[fid] = {"r":(a, ao_a, found[0], state_dates[0]), "b":(b, ao_b, found[1], state_dates[1])}
    livefv = add_live_engineered(build_moneyline_feature_view(prepared_fights_df=pd.DataFrame(prep_rows), fighter_state_history_df=pd.DataFrame(state_rows)))
    missing_live = [c for c in SELECTED if c not in livefv.columns]
    if missing_live: raise RuntimeError(f"Prefight feature view missing frozen V5 features: {missing_live}")
    return livefv, meta


def main():
    model, valid, med = fit_frozen_v5(); livefv, meta = build_prefight_view(); rows = []
    for _, r in livefv.iterrows():
        a, ao_a, found_a, date_a = meta[r.fight_id]["r"]; b, ao_b, found_b, date_b = meta[r.fight_id]["b"]; ipa, ipb = imp(ao_a), imp(ao_b); over = ipa + ipb; fair_a, fair_b = ipa / over, ipb / over; vals = {c: r[c] for c in SELECTED}; vals["market_overround"] = over
        xlive = pd.DataFrame([{c: vals.get(c, np.nan) for c in valid}], columns=valid).replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0); mlive = logit([fair_a]); dlive = xgb.DMatrix(xlive, base_margin=mlive, feature_names=valid); full_margin = model.predict(dlive, output_margin=True); correction = full_margin - mlive; p_a = float(sigmoid(mlive + correction)[0]); p_b = 1.0 - p_a; cold = not (found_a and found_b); common = {"market_source":"UFC event page 2026-08-29","fight_cold_start":cold,"selected_feature_count":len(SELECTED)+1,"market_overround":over}
        rows += [{"fight_id":r.fight_id,"fighter":a,"opponent":b,"side":"red","american_odds":ao_a,"fighter_state_found":found_a,"prefight_state_date":date_a,"fair_market_p":fair_a,"v5_model_p":p_a,"edge":p_a-fair_a,**common},{"fight_id":r.fight_id,"fighter":b,"opponent":a,"side":"blue","american_odds":ao_b,"fighter_state_found":found_b,"prefight_state_date":date_b,"fair_market_p":fair_b,"v5_model_p":p_b,"edge":p_b-fair_b,**common}]
    out = pd.DataFrame(rows).sort_values(["fight_id","side"]); out.to_csv(OUTPUT, index=False); print(out.to_string(index=False)); print("missing_states=", out.loc[~out.fighter_state_found, "fighter"].tolist()); print("eligible_fights=", int((~out.groupby("fight_id").fight_cold_start.first()).sum()), "/", out.fight_id.nunique()); print("output=", OUTPUT)

if __name__ == "__main__": main()
