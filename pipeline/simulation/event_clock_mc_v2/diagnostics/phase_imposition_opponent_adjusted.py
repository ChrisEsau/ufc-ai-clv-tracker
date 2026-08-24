"""Measurement-only opponent-adjusted phase-imposition validation.

No simulator or FSR changes.  Uses only information available before each fight.
The hypothesis is that a fighter has a persistent ability to change what an
opponent is normally able to do: attempt takedowns, accumulate control, or force
ground/clinch striking.  Each observed fight first computes the opponent's
realized phase usage minus that opponent's own prefight baseline.  A fighter's
prefight imposition estimate is then the shrunk mean of those residuals from
prior opponents.

Positive suppression means the fighter historically made opponents do LESS of
the action than those opponents normally did.  Validation is chronological and
compares frozen FSR deltas with FSR plus each phase feature and all phase
features.  The historical market is diagnostic only.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v2.diagnostics.phase_imposition_falsification import (
    load_fight_aggregates, MASTER_PATH, MARKET_AUDIT_PATH, _find,
)

OUT = Path("data/diagnostics/event_clock_mc_v2/phase_imposition_opponent_adjusted")
MIN_PRIOR = 3
SHRINK_K = 3.0
COMPS = ("td_att", "control_share", "ground_share", "clinch_share")


def _mean(hist, fid, comp):
    v = hist.get(str(fid), {}).get(comp, [])
    return float(np.mean(v)) if v else np.nan


def _n(hist, fid):
    z = hist.get(str(fid), {})
    return max((len(v) for v in z.values()), default=0)


def _add(hist, fid, comp, val):
    hist.setdefault(str(fid), {}).setdefault(comp, []).append(float(val))


def _shrunk_residual_mean(hist, fid, comp):
    v = hist.get(str(fid), {}).get(comp, [])
    if not v:
        return np.nan
    n = len(v)
    return float(np.mean(v)) * n / (n + SHRINK_K)


def build_features(fights: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe opponent-adjusted suppression features."""
    own, induced = {}, {}
    rows = []
    for _, r in fights.sort_values(["event_date", "fight_id"]).iterrows():
        rid, bid = str(r.r_id), str(r.b_id)
        rec = {
            "fight_id": str(r.fight_id), "event_date": r.event_date,
            "r_id": rid, "b_id": bid, "r_prior": _n(own, rid), "b_prior": _n(own, bid),
        }
        # snapshot all prefight baselines before any current-fight updates
        pre = {}
        for side, fid, opp in (("r", rid, bid), ("b", bid, rid)):
            for c in COMPS:
                pre[(side, c)] = _mean(own, opp, c)
                # induced residual is opponent realized - opponent expected; negate so + = suppression
                rec[f"{side}_suppress_{c}"] = -_shrunk_residual_mean(induced, fid, c) if pd.notna(_shrunk_residual_mean(induced, fid, c)) else np.nan
        rows.append(rec)

        # update only after the snapshot
        for side, fid in (("r", rid), ("b", bid)):
            for c in COMPS:
                _add(own, fid, c, float(r[f"{side}_{c}"]))
        for side, fid, oside in (("r", rid, "b"), ("b", bid, "r")):
            for c in COMPS:
                base = pre[(side, c)]
                if pd.notna(base):
                    _add(induced, fid, c, float(r[f"{oside}_{c}"]) - float(base))
    return pd.DataFrame(rows)


def attach_labels_fsr(feat: pd.DataFrame) -> pd.DataFrame:
    master = pd.read_parquet(MASTER_PATH).copy(); cols = master.columns
    mf = _find(cols,["fight_id","bout_id"]); mr = _find(cols,["r_id","red_id","r_fighter_id"]); mb = _find(cols,["b_id","blue_id","b_fighter_id"])
    win = _find(cols,["winner_id","winner_fighter_id","winner"],False)
    m = master[[mf,mr,mb,win]].drop_duplicates(mf).copy(); m.columns=["fight_id","mr_id","mb_id","winner"]
    for c in m.columns: m[c] = m[c].astype(str)
    z = feat.merge(m,on="fight_id",how="left")
    z["red_won"] = (z.winner.eq(z.r_id) | z.winner.str.lower().eq("red")).astype(int)

    fsr = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy(); fsr["fight_id"] = fsr.fight_id.astype(str); fsr["fighter_id"] = fsr.fighter_id.astype(str)
    id_like={"fight_id","fighter_id","event_date","fighter_name","name","side"}
    nums=[c for c in fsr.columns if c not in id_like and pd.api.types.is_numeric_dtype(fsr[c])]
    red=fsr.rename(columns={"fighter_id":"r_id", **{c:f"r__{c}" for c in nums}})
    blue=fsr.rename(columns={"fighter_id":"b_id", **{c:f"b__{c}" for c in nums}})
    z=z.merge(red[["fight_id","r_id"]+[f"r__{c}" for c in nums]],on=["fight_id","r_id"],how="left")
    z=z.merge(blue[["fight_id","b_id"]+[f"b__{c}" for c in nums]],on=["fight_id","b_id"],how="left")
    for c in nums: z[f"fsr_delta__{c}"] = z[f"r__{c}"] - z[f"b__{c}"]
    for c in COMPS: z[f"phase_delta__{c}"] = z[f"r_suppress_{c}"] - z[f"b_suppress_{c}"]
    return z


def score(y, p):
    return {
        "n":len(y), "accuracy":accuracy_score(y,p>=.5), "auc":roc_auc_score(y,p),
        "brier":brier_score_loss(y,p), "log_loss":log_loss(y,np.clip(p,1e-6,1-1e-6)),
    }


def fit_predict(tr, te, cols):
    model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(C=1.0,max_iter=4000))
    model.fit(tr[cols],tr.red_won)
    return model.predict_proba(te[cols])[:,1]


def main():
    f=load_fight_aggregates(); feat=build_features(f); z=attach_labels_fsr(feat)
    z=z[(z.r_prior>=MIN_PRIOR)&(z.b_prior>=MIN_PRIOR)&z.winner.notna()].sort_values(["event_date","fight_id"]).copy()
    fsr_cols=[c for c in z if c.startswith("fsr_delta__")]
    phase_cols=[c for c in z if c.startswith("phase_delta__")]
    z=z[z[fsr_cols].notna().any(axis=1)].copy()
    cut=int(.70*len(z)); tr=z.iloc[:cut]; te=z.iloc[cut:]

    specs=[("fsr_only",fsr_cols)]
    for c in phase_cols: specs.append((f"fsr_plus_{c.replace('phase_delta__','')}",fsr_cols+[c]))
    specs += [("fsr_plus_all_phase",fsr_cols+phase_cols),("phase_only",phase_cols)]
    rows=[]
    for label,cols in specs:
        p=fit_predict(tr,te,cols); rows.append({"model":label,**score(te.red_won.to_numpy(),p)})
    pred=pd.DataFrame(rows)
    base=pred[pred.model.eq("fsr_only")].iloc[0]
    pred["delta_auc_vs_fsr"]=pred.auc-base.auc; pred["delta_brier_vs_fsr"]=pred.brier-base.brier; pred["delta_logloss_vs_fsr"]=pred.log_loss-base.log_loss

    # Non-overlapping reliability: compare first-half vs second-half induced residual means per fighter.
    rel=[]
    # reconstruct from realized fight residuals for mature histories
    own={}; obs={}
    for _,r in f.sort_values(["event_date","fight_id"]).iterrows():
        rid,bid=str(r.r_id),str(r.b_id)
        preb={c:_mean(own,bid,c) for c in COMPS}; prebr={c:_mean(own,rid,c) for c in COMPS}
        for c in COMPS:
            if pd.notna(preb[c]): _add(obs,rid,c,float(r[f"b_{c}"])-preb[c])
            if pd.notna(prebr[c]): _add(obs,bid,c,float(r[f"r_{c}"])-prebr[c])
            _add(own,rid,c,float(r[f"r_{c}"])); _add(own,bid,c,float(r[f"b_{c}"]))
    for c in COMPS:
        a=[]; b=[]
        for fid,d in obs.items():
            v=d.get(c,[])
            if len(v)>=6:
                h=len(v)//2; a.append(-float(np.mean(v[:h]))); b.append(-float(np.mean(v[h:])))
        rel.append({"feature":c,"fighters":len(a),"split_half_corr":float(pd.Series(a).corr(pd.Series(b))) if len(a)>=5 else np.nan})
    reliability=pd.DataFrame(rel)

    market_summary=pd.DataFrame(); market_rows=pd.DataFrame()
    if MARKET_AUDIT_PATH.exists():
        b=pd.read_csv(MARKET_AUDIT_PATH); ml=b[b.market_key.eq("moneyline")].copy(); rr=[]
        for fid,g in ml.groupby(ml.fight_id.astype(str)):
            if len(g)<2: continue
            g=g.sort_values("market_fair_probability",ascending=False); fav=g.iloc[0]
            rr.append({"fight_id":str(fid),"favorite_side":str(fav.outcome_side),"market_p":float(fav.market_fair_probability),"mc_p":float(fav.model_probability),"residual":float(fav.market_fair_probability-fav.model_probability),"rp":float(fav.red_prior_ufc_fights),"bp":float(fav.blue_prior_ufc_fights)})
        mm=pd.DataFrame(rr); mm=mm[(mm.rp>=MIN_PRIOR)&(mm.bp>=MIN_PRIOR)].merge(z[["fight_id"]+phase_cols],on="fight_id",how="left")
        for c in phase_cols: mm[f"fav__{c}"]=np.where(mm.favorite_side.eq("red"),mm[c],-mm[c])
        xcols=[f"fav__{c}" for c in phase_cols]; valid=mm.dropna(subset=["residual"]).copy()
        pipe=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=10.0)); pipe.fit(valid[xcols],valid.residual); ph=pipe.predict(valid[xcols])
        d={"mature_priced_fights":len(valid),"mean_market_p":valid.market_p.mean(),"mean_mc_p":valid.mc_p.mean(),"mean_residual_pp":100*valid.residual.mean(),"phase_r2_insample":1-float(np.sum((valid.residual-ph)**2)/np.sum((valid.residual-valid.residual.mean())**2)),"phase_mae_pp":100*mean_absolute_error(valid.residual,ph)}
        for c,x in zip(COMPS,xcols): d[f"corr_residual__{c}"]=valid[x].corr(valid.residual)
        market_summary=pd.DataFrame([d]); market_rows=valid

    OUT.mkdir(parents=True,exist_ok=True)
    pred.to_csv(OUT/"winner_prediction_incremental.csv",index=False); reliability.to_csv(OUT/"split_half_reliability.csv",index=False); market_summary.to_csv(OUT/"market_residual_summary.csv",index=False); market_rows.to_csv(OUT/"mature_market_rows.csv",index=False); z.to_csv(OUT/"mature_prefight_features.csv",index=False)
    print("OPPONENT-ADJUSTED PHASE IMPOSITION")
    print(f"mature={len(z)} train={len(tr)} test={len(te)} shrink_k={SHRINK_K}")
    print("\nWINNER PREDICTION")
    print(pred.to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    print("\nSPLIT-HALF RELIABILITY")
    print(reliability.to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    if not market_summary.empty:
        print("\nMATURE MARKET RESIDUAL")
        print(market_summary.to_string(index=False,float_format=lambda x:f"{x:.5f}"))

if __name__ == "__main__": main()
