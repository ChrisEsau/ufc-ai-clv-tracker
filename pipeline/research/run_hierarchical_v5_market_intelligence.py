from __future__ import annotations

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.research import xgboost_method_market_offset as method
from pipeline.research.xgboost_method_hierarchical_v5_oof import _fit_conditional
from pipeline.research import xgboost_market_offset_v5_frozen as v5

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/research/prop_mispricing"
HIST = ROOT / "data/market/market_intelligence_history.parquet"
HIST_OUTCOMES = ROOT / "data/market/historical_market_outcomes.parquet"
FEATURES = ROOT / "data/features/moneyline_feature_view.parquet"
PRED_OUT = OUT / "hierarchical_v5_market_intelligence_predictions.csv"
BET_OUT = OUT / "hierarchical_v5_market_intelligence_bets.csv"
SKIP_OUT = OUT / "hierarchical_v5_market_intelligence_unscored.csv"
SUMMARY_OUT = OUT / "hierarchical_v5_market_intelligence_summary.json"

REQ_METHOD = ["win_by_ko_tko_dq", "win_by_submission", "win_by_decision"]
SLUGS = method.SLUGS
THRESHOLD = 0.30
EPS = 1e-12


def norm_name(x):
    if x is None or pd.isna(x):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1-EPS)
    return np.log(p/(1-p))


def find_col(cols, candidates):
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def split_display(s):
    s = str(s or "")
    for sep in [" vs. ", " vs ", " v ", " versus "]:
        if sep in s.lower():
            # case-insensitive split preserving text
            m = re.split(re.escape(sep), s, maxsplit=1, flags=re.I)
            if len(m) == 2:
                return m[0].strip(), m[1].strip()
    return "", ""


def classify_side(row, red_name, blue_name):
    vals = [row.get("fighter_name"), row.get("outcome_display"), row.get("outcome_key"), row.get("comparison_key")]
    rn, bn = norm_name(red_name), norm_name(blue_name)
    for x in vals:
        nx = norm_name(x)
        if not nx:
            continue
        if rn and (rn in nx or nx in rn):
            return "red"
        if bn and (bn in nx or nx in bn):
            return "blue"
    return None


def train_v5():
    df, xraw, signed = v5.build_v5_frame(HIST_OUTCOMES, FEATURES)
    feature_cols = v5.frozen_feature_order(df, xraw, signed)
    tr = pd.to_datetime(df["date"]) <= pd.Timestamp("2024-12-31")
    valid = [c for c in feature_cols if xraw.loc[tr, c].notna().any()]
    med = xraw.loc[tr, valid].median(numeric_only=True)
    xtr = xraw.loc[tr, valid].fillna(med).fillna(0.0)
    ytr = df.loc[tr, "won"].astype(int).to_numpy()
    mtr = v5._logit(df.loc[tr, "fair_market_p"])
    dtr = xgb.DMatrix(xtr, label=ytr, base_margin=mtr, feature_names=valid)
    model = xgb.train(v5.PARAMS, dtr, num_boost_round=v5.ROUNDS, verbose_eval=False)
    return model, valid, med


def prep_latest_complete():
    h = pd.read_parquet(HIST).copy()
    h["fight_id"] = h["fight_id"].astype(str)
    h["refresh_timestamp"] = pd.to_datetime(h["refresh_timestamp"], errors="coerce", utc=True)
    h["implied_probability"] = pd.to_numeric(h["implied_probability"], errors="coerce")
    h["american_odds"] = pd.to_numeric(h["american_odds"], errors="coerce")
    h = h[h["bookmaker"].astype(str).str.contains("DraftKings", case=False, na=False)].copy()
    h = h[h["market_key"].isin(["moneyline"] + REQ_METHOD)].copy()
    h = h[h["implied_probability"].gt(0) & np.isfinite(h["implied_probability"])].copy()

    # For every fight choose the latest refresh containing 2 ML + 2 rows for each winner-by-method market.
    counts = h.groupby(["fight_id", "refresh_timestamp", "market_key"]).size().unstack(fill_value=0)
    for k in ["moneyline"] + REQ_METHOD:
        if k not in counts.columns:
            counts[k] = 0
    good = counts[(counts[["moneyline"] + REQ_METHOD] >= 2).all(axis=1)].reset_index()
    latest = good.sort_values("refresh_timestamp").groupby("fight_id", as_index=False).tail(1)[["fight_id", "refresh_timestamp"]]
    chosen = h.merge(latest, on=["fight_id", "refresh_timestamp"], how="inner")
    return h, chosen, latest


def build_score_rows(chosen, fv):
    fv = fv.copy()
    fv["fight_id"] = fv["fight_id"].astype(str)
    red_col = find_col(fv.columns, ["red_fighter", "r_fighter", "red_fighter_name", "r_fighter_name", "fighter_red"])
    blue_col = find_col(fv.columns, ["blue_fighter", "b_fighter", "blue_fighter_name", "b_fighter_name", "fighter_blue"])
    fmeta = chosen.sort_values("refresh_timestamp").groupby("fight_id", as_index=False).last()[["fight_id", "event_name", "fight_display", "refresh_timestamp"]]
    merged = fmeta.merge(fv, on="fight_id", how="left", indicator=True)
    rows, skips = [], []

    for r in merged.itertuples(index=False):
        fid = str(r.fight_id)
        z = chosen[chosen["fight_id"].eq(fid)].copy()
        if getattr(r, "_merge") != "both":
            skips.append({"fight_id":fid,"event_name":r.event_name,"fight_display":r.fight_display,"reason":"no_feature_view_match"})
            continue
        red_name = getattr(r, red_col) if red_col else ""
        blue_name = getattr(r, blue_col) if blue_col else ""
        if not norm_name(red_name) or not norm_name(blue_name):
            a,b = split_display(r.fight_display)
            red_name = red_name if norm_name(red_name) else a
            blue_name = blue_name if norm_name(blue_name) else b
        if not norm_name(red_name) or not norm_name(blue_name):
            skips.append({"fight_id":fid,"event_name":r.event_name,"fight_display":r.fight_display,"reason":"cannot_resolve_fighter_orientation"})
            continue

        raw = {}
        ok = True
        for mk, suffix in [("moneyline","ml"),("win_by_ko_tko_dq","ko"),("win_by_submission","sub"),("win_by_decision","dec")]:
            zz = z[z["market_key"].eq(mk)].copy()
            vals = {"red":[], "blue":[]}
            for _, rr in zz.iterrows():
                side = classify_side(rr, red_name, blue_name)
                if side:
                    vals[side].append(rr)
            if len(vals["red"]) < 1 or len(vals["blue"]) < 1:
                ok = False
                skips.append({"fight_id":fid,"event_name":r.event_name,"fight_display":r.fight_display,"reason":f"cannot_map_{mk}_to_red_blue","red_fighter":red_name,"blue_fighter":blue_name})
                break
            # duplicate provider rows inside same refresh: retain last deterministically
            for side in ["red","blue"]:
                rr = vals[side][-1]
                raw[f"{side}_{suffix}_raw_p"] = float(rr["implied_probability"])
                raw[f"{side}_{suffix}_american"] = float(rr["american_odds"]) if pd.notna(rr["american_odds"]) else np.nan
        if not ok:
            continue

        ml_sum = raw["red_ml_raw_p"] + raw["blue_ml_raw_p"]
        raw["market_overround"] = ml_sum
        raw["market_p_red_ml"] = raw["red_ml_raw_p"] / ml_sum
        method_raw = np.array([raw["red_ko_raw_p"],raw["red_sub_raw_p"],raw["red_dec_raw_p"],raw["blue_ko_raw_p"],raw["blue_sub_raw_p"],raw["blue_dec_raw_p"]], float)
        method_norm = method_raw / method_raw.sum()
        base = {"fight_id":fid,"event_name":r.event_name,"fight_display":r.fight_display,"refresh_timestamp":r.refresh_timestamp,"red_fighter":str(red_name),"blue_fighter":str(blue_name),**raw}
        for j,slug in enumerate(SLUGS):
            base[f"market_{slug}"] = float(method_norm[j])
        # carry all feature columns from merged row
        sr = pd.Series(r._asdict())
        for c in fv.columns:
            if c != "fight_id" and c in sr.index:
                base[c] = sr[c]
        rows.append(base)
    return pd.DataFrame(rows), pd.DataFrame(skips), red_col, blue_col


def attach_results(pred):
    try:
        m = pd.read_parquet(HIST_OUTCOMES).copy()
        m["fight_id"] = m["fight_id"].astype(str)
        m = m[(m["result_status"] == "graded") & m["won"].notna() & m["market_key"].isin(REQ_METHOD) & m["outcome_side"].astype(str).isin(["red","blue"])].copy()
        m["won"] = m["won"].astype(bool).astype(int)
        target_rows = []
        for fid,g in m.groupby("fight_id"):
            wins=[]
            for j,(slug,_,side,key,_) in enumerate(method.CLASS_SPECS):
                zz=g[(g["outcome_side"].astype(str)==side)&(g["market_key"]==key)]
                if len(zz) and int(zz.iloc[0]["won"])==1:
                    wins.append(j)
            if len(wins)==1:
                target_rows.append((fid,wins[0]))
        t=pd.DataFrame(target_rows,columns=["fight_id","target"])
        return pred.merge(t,on="fight_id",how="left")
    except Exception:
        pred["target"] = np.nan
        return pred


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h, chosen, latest = prep_latest_complete()
    fv = pd.read_parquet(FEATURES).copy()
    score, skips, red_col, blue_col = build_score_rows(chosen, fv)
    if score.empty:
        skips.to_csv(SKIP_OUT,index=False)
        raise RuntimeError("No market-intelligence fights could be mapped into score rows")

    # Frozen V5 moneyline trained only through 2024.
    ml_model, ml_features, ml_med = train_v5()
    missing_ml = [c for c in ml_features if c not in score.columns and c != "market_overround"]
    if missing_ml:
        raise RuntimeError(f"score feature view missing frozen V5 features: {missing_ml[:20]}")
    xsc = score[ml_features].replace([np.inf,-np.inf],np.nan).fillna(ml_med).fillna(0.0)
    msc = v5._logit(score["market_p_red_ml"].to_numpy(float))
    dsc = xgb.DMatrix(xsc, base_margin=msc, feature_names=ml_features)
    full_margin = ml_model.predict(dsc, output_margin=True)
    p_red = v5._sigmoid(msc + (full_margin - msc))
    score["v5_model_p_red"] = p_red

    # Frozen hierarchical method training: all chronological development data <= 2024, no 2025+ target use.
    train, method_features, _ = method._build_rows(True, True)
    red_cond, _, _ = _fit_conditional(train, score, method_features, "red")
    blue_cond, _, _ = _fit_conditional(train, score, method_features, "blue")
    hp = np.concatenate([p_red[:,None]*red_cond, (1-p_red)[:,None]*blue_cond], axis=1)
    hp = hp / hp.sum(axis=1, keepdims=True)
    for j,slug in enumerate(SLUGS):
        score[f"hier_{slug}"] = hp[:,j]
    score["predicted_side"] = np.where(score["v5_model_p_red"] >= 0.5, "red", "blue")
    score = attach_results(score)

    bet_rows=[]
    for r in score.itertuples(index=False):
        for slug in SLUGS:
            side = "red" if slug.startswith("red_") else "blue"
            if side != r.predicted_side:
                continue
            model_p=float(getattr(r,f"hier_{slug}")); market_p=float(getattr(r,f"market_{slug}"))
            resid=float(logit(model_p)-logit(market_p))
            if resid < THRESHOLD:
                continue
            meth = slug.split("_",1)[1]
            raw_p=float(getattr(r,f"{side}_{meth}_raw_p"))
            american=float(getattr(r,f"{side}_{meth}_american"))
            class_idx=SLUGS.index(slug)
            target=getattr(r,"target")
            won=np.nan; profit=np.nan
            if pd.notna(target):
                won=int(int(target)==class_idx)
                dec=1.0/raw_p
                profit=(dec-1.0) if won else -1.0
            bet_rows.append({"fight_id":r.fight_id,"event_name":r.event_name,"fight_display":r.fight_display,"refresh_timestamp":r.refresh_timestamp,"red_fighter":r.red_fighter,"blue_fighter":r.blue_fighter,"predicted_side":r.predicted_side,"bet_slug":slug,"model_probability":model_p,"normalized_market_probability":market_p,"signed_logit_residual":resid,"raw_implied_probability":raw_p,"american_odds":american,"decimal_odds":1.0/raw_p,"target":target,"won":won,"profit_units":profit})

    bets=pd.DataFrame(bet_rows)
    pred_cols=["fight_id","event_name","fight_display","refresh_timestamp","red_fighter","blue_fighter","v5_model_p_red","predicted_side","target"]+[f"market_{s}" for s in SLUGS]+[f"hier_{s}" for s in SLUGS]
    score[pred_cols].to_csv(PRED_OUT,index=False)
    bets.to_csv(BET_OUT,index=False)
    skips.to_csv(SKIP_OUT,index=False)

    graded=bets[bets["profit_units"].notna()].copy() if not bets.empty else bets
    event_summary=[]
    for ev,g in score.groupby("event_name",dropna=False):
        gb=bets[bets["event_name"].eq(ev)] if not bets.empty else bets
        gg=gb[gb["profit_units"].notna()] if not gb.empty else gb
        event_summary.append({"event_name":str(ev),"fights_scored":int(len(g)),"bets":int(len(gb)),"graded_bets":int(len(gg)),"profit_units":float(gg["profit_units"].sum()) if len(gg) else None,"roi":float(gg["profit_units"].sum()/len(gg)) if len(gg) else None})

    summary={
        "experiment":"frozen_hierarchical_v5_all_market_intelligence_events_v1",
        "source":"data/market/market_intelligence_history.parquet",
        "source_rows":int(len(pd.read_parquet(HIST))),
        "draftkings_required_market_rows":int(len(h)),
        "unique_fights_in_source":int(pd.read_parquet(HIST,columns=["fight_id"])["fight_id"].astype(str).nunique()),
        "fights_with_complete_latest_snapshot":int(len(latest)),
        "fights_scored":int(len(score)),
        "events_scored":int(score["event_name"].nunique()),
        "unscored_rows":int(len(skips)),
        "feature_orientation_columns":{"red":red_col,"blue":blue_col},
        "training_cutoff":"2024-12-31",
        "selection_or_tuning_on_2025_plus":False,
        "bet_rule":{"winner_side_only":True,"signed_logit_residual_threshold":THRESHOLD,"multiple_bets_per_fight_allowed":True,"stake_units":1.0},
        "bets":int(len(bets)),
        "graded_bets":int(len(graded)),
        "graded_profit_units":float(graded["profit_units"].sum()) if len(graded) else None,
        "graded_roi":float(graded["profit_units"].sum()/len(graded)) if len(graded) else None,
        "event_summary":event_summary,
    }
    SUMMARY_OUT.write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))

if __name__ == "__main__":
    main()
