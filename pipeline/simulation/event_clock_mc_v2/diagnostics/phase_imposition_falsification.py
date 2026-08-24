"""Measurement-only falsification of a prefight phase-imposition latent.

Question: do fighters consistently distort an opponent's normal phase usage, and
is that distortion useful out of sample beyond frozen FSR V3?  No simulator or
FSR mechanics are modified.

The diagnostic builds chronological fighter-fight aggregates from UFC round
stats.  For each completed fight, it measures how much each fighter changed the
opponent's takedown-attempt rate, ground-strike share, clinch-strike share and
control share relative to that opponent's own prior-UFC baseline.  A fighter's
prefight imposition rating is the trailing mean of the distortions they induced
in previous opponents.  Only information available before each fight is used.

Validation:
  1. test-retest / next-fight stability of the imposition components;
  2. chronological 70/30 winner prediction: FSR-only vs FSR+imposition;
  3. mature priced fights (>=3 prior UFC fights each): association with the
     historical market-favorite minus frozen-MC residual.
"""
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH

ROUND_PATH = Path("data/fight_details/ufc_round_stats.parquet")
MASTER_PATH = Path("data/master/ufc_master.parquet")
MARKET_AUDIT_PATH = Path("/tmp/market_edge/bet_level_audit.csv")
OUT = Path("data/diagnostics/event_clock_mc_v2/phase_imposition_falsification")
MIN_PRIOR = 3


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _find(cols, aliases, required=True):
    norm = {_norm(c): c for c in cols}
    for a in aliases:
        if _norm(a) in norm:
            return norm[_norm(a)]
    # suffix/contains fallback, but only for fairly specific aliases
    for a in aliases:
        aa = _norm(a)
        hits = [c for c in cols if aa and aa in _norm(c)]
        if len(hits) == 1:
            return hits[0]
    if required:
        raise KeyError(f"none of aliases found: {aliases}; columns={list(cols)}")
    return None


def _num(x):
    return pd.to_numeric(x, errors="coerce")


def _parse_clock(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)): return float(x)
    s = str(x).strip()
    m = re.match(r"^(\d+):(\d{1,2})$", s)
    if m: return 60.0*float(m.group(1))+float(m.group(2))
    try: return float(s)
    except Exception: return np.nan


def _ratio(a, b):
    a=float(a) if pd.notna(a) else 0.0; b=float(b) if pd.notna(b) else 0.0
    d=a+b
    return a/d if d>0 else 0.5


def load_fight_aggregates():
    rd = pd.read_parquet(ROUND_PATH)
    cols = rd.columns
    fight = _find(cols,["fight_id","bout_id"])
    date = _find(cols,["event_date","date"])
    rid = _find(cols,["r_id","red_id","r_fighter_id"])
    bid = _find(cols,["b_id","blue_id","b_fighter_id"])

    # The source has changed names across scraper generations, so resolve aliases.
    fields = {}
    for side,pfx in [("r","red"),("b","blue")]:
        fields[(side,"td_att")] = _find(cols,[f"{side}_td_att",f"{pfx}_td_att",f"{side}_takedown_att"],False)
        fields[(side,"ground_att")] = _find(cols,[f"{side}_ground_att",f"{pfx}_ground_att",f"{side}_ground_sig_str_att"],False)
        fields[(side,"clinch_att")] = _find(cols,[f"{side}_clinch_att",f"{pfx}_clinch_att",f"{side}_clinch_sig_str_att"],False)
        fields[(side,"distance_att")] = _find(cols,[f"{side}_distance_att",f"{pfx}_distance_att",f"{side}_distance_sig_str_att"],False)
        fields[(side,"sig_att")] = _find(cols,[f"{side}_sig_str_att",f"{pfx}_sig_str_att",f"{side}_sig_att"],False)
        fields[(side,"control")] = _find(cols,[f"{side}_ctrl",f"{pfx}_ctrl",f"{side}_control",f"{side}_control_time"],False)
    if fields[("r","td_att")] is None or fields[("b","td_att")] is None:
        raise RuntimeError("TD attempt fields are required")

    keep=[fight,date,rid,bid]+[c for c in fields.values() if c]
    x=rd[keep].copy()
    x[date]=pd.to_datetime(x[date],errors="coerce").dt.normalize()
    for c in set(fields.values())-{None}:
        if "ctrl" in _norm(c) or "control" in _norm(c): x[c]=x[c].map(_parse_clock)
        else: x[c]=_num(x[c]).fillna(0.0)

    agg={}
    for side in ("r","b"):
        for stat in ("td_att","ground_att","clinch_att","distance_att","sig_att","control"):
            c=fields[(side,stat)]
            if c: agg[f"{side}_{stat}"]=(c,"sum")
    f=x.groupby(fight,as_index=False).agg(event_date=(date,"first"),r_id=(rid,"first"),b_id=(bid,"first"),**agg)
    for side in ("r","b"):
        sig=f.get(f"{side}_sig_att",pd.Series(0.0,index=f.index)).astype(float)
        gr=f.get(f"{side}_ground_att",pd.Series(0.0,index=f.index)).astype(float)
        cl=f.get(f"{side}_clinch_att",pd.Series(0.0,index=f.index)).astype(float)
        ds=f.get(f"{side}_distance_att",pd.Series(np.nan,index=f.index)).astype(float)
        if ds.isna().all(): ds=(sig-gr-cl).clip(lower=0)
        denom=(ds+cl+gr).replace(0,np.nan)
        f[f"{side}_ground_share"]=(gr/denom).fillna(0.0)
        f[f"{side}_clinch_share"]=(cl/denom).fillna(0.0)
    ctrl_r=f.get("r_control",pd.Series(0.0,index=f.index)).fillna(0.0)
    ctrl_b=f.get("b_control",pd.Series(0.0,index=f.index)).fillna(0.0)
    d=(ctrl_r+ctrl_b).replace(0,np.nan)
    f["r_control_share"]=(ctrl_r/d).fillna(0.5); f["b_control_share"]=(ctrl_b/d).fillna(0.5)
    # TD attempts are per-fight counts here; baseline comparisons remain on same unit.
    return f.sort_values(["event_date",fight]).rename(columns={fight:"fight_id"})


def build_chronological_features(f):
    # Store each fighter's own historical phase usage, and distortions induced in opponents.
    own={}; induced={}; rows=[]
    comps=("td_att","ground_share","clinch_share","control_share")
    def hist_mean(d,key,comp):
        vals=d.get(str(key),{}).get(comp,[])
        return float(np.mean(vals)) if vals else np.nan
    def count(d,key):
        z=d.get(str(key),{})
        return max([len(v) for v in z.values()], default=0)
    def add(d,key,comp,val):
        d.setdefault(str(key),{}).setdefault(comp,[]).append(float(val))

    for _,r in f.iterrows():
        rid,bid=str(r.r_id),str(r.b_id)
        rec={"fight_id":str(r.fight_id),"event_date":r.event_date,"r_id":rid,"b_id":bid,
             "r_prior":count(own,rid),"b_prior":count(own,bid)}
        for side,fid in [("r",rid),("b",bid)]:
            opp=bid if side=="r" else rid
            for c in comps:
                rec[f"{side}_own_prior_{c}"]=hist_mean(own,fid,c)
                rec[f"{side}_induced_{c}"]=hist_mean(induced,fid,c)
                rec[f"{side}_opp_prior_{c}"]=hist_mean(own,opp,c)
        rows.append(rec)

        # after snapshot, observe this fight and update histories
        for side,fid,opp in [("r",rid,bid),("b",bid,rid)]:
            oside="b" if side=="r" else "r"
            for c in comps:
                val=float(r[f"{side}_{c}"])
                add(own,fid,c,val)
                opp_val=float(r[f"{oside}_{c}"])
                opp_base=hist_mean(own,opp,c)  # includes current opponent update if same loop order; fix below via pre-cache
        # redo induced using truly prefight baselines cached in rec
        for side,fid,oside in [("r",rid,"b"),("b",bid,"r")]:
            for c in comps:
                base=rec[f"{side}_opp_prior_{c}"]
                if pd.notna(base): add(induced,fid,c,float(r[f"{oside}_{c}"])-float(base))
    return pd.DataFrame(rows)


def attach_outcome_and_fsr(feat):
    master=pd.read_parquet(MASTER_PATH).copy(); cols=master.columns
    mf=_find(cols,["fight_id","bout_id"]); mr=_find(cols,["r_id","red_id","r_fighter_id"]); mb=_find(cols,["b_id","blue_id","b_fighter_id"])
    win=_find(cols,["winner_id","winner_fighter_id","winner"],False)
    if win is None: raise RuntimeError("winner id/name field required in master")
    m=master[[mf,mr,mb,win]].drop_duplicates(mf).copy(); m.columns=["fight_id","mr_id","mb_id","winner"]
    for c in ["fight_id","mr_id","mb_id","winner"]: m[c]=m[c].astype(str)
    z=feat.merge(m,on="fight_id",how="left")
    z["red_won"]=(z["winner"].eq(z["r_id"]) | z["winner"].str.lower().eq("red")).astype(int)

    fsr=pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy(); fsr["fight_id"]=fsr["fight_id"].astype(str)
    id_like={"fight_id","fighter_id","event_date","fighter_name","name","side"}
    nums=[c for c in fsr.columns if c not in id_like and pd.api.types.is_numeric_dtype(fsr[c])]
    fr=fsr.copy(); fr["fighter_id"]=fr["fighter_id"].astype(str)
    red=fr.rename(columns={"fighter_id":"r_id",**{c:f"r_fsr__{c}" for c in nums}})
    blue=fr.rename(columns={"fighter_id":"b_id",**{c:f"b_fsr__{c}" for c in nums}})
    z=z.merge(red[["fight_id","r_id"]+[f"r_fsr__{c}" for c in nums]],on=["fight_id","r_id"],how="left")
    z=z.merge(blue[["fight_id","b_id"]+[f"b_fsr__{c}" for c in nums]],on=["fight_id","b_id"],how="left")
    for c in nums: z[f"fsr_delta__{c}"]=z[f"r_fsr__{c}"]-z[f"b_fsr__{c}"]
    for c in ("td_att","ground_share","clinch_share","control_share"):
        # lower opponent use is suppression; negate induced difference so positive means red suppresses more / imposes more.
        z[f"imp_delta__{c}"]=-(z[f"r_induced_{c}"]-z[f"b_induced_{c}"])
    return z


def metrics(y,p):
    pred=(p>=.5).astype(int)
    return {"n":len(y),"accuracy":accuracy_score(y,pred),"auc":roc_auc_score(y,p),"brier":brier_score_loss(y,p),"log_loss":log_loss(y,np.clip(p,1e-6,1-1e-6))}


def main():
    f=load_fight_aggregates(); feat=build_chronological_features(f); z=attach_outcome_and_fsr(feat)
    mature=z[(z.r_prior>=MIN_PRIOR)&(z.b_prior>=MIN_PRIOR)&z.winner.notna()].sort_values(["event_date","fight_id"]).copy()
    fsr_cols=[c for c in mature if c.startswith("fsr_delta__")]
    imp_cols=[c for c in mature if c.startswith("imp_delta__")]
    usable=mature[mature[fsr_cols].notna().any(axis=1)].copy()
    cut=max(1,int(.70*len(usable))); tr=usable.iloc[:cut]; te=usable.iloc[cut:]
    rows=[]
    for label,cols in [("fsr_only",fsr_cols),("fsr_plus_imposition",fsr_cols+imp_cols),("imposition_only",imp_cols)]:
        model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(C=1.0,max_iter=4000))
        model.fit(tr[cols],tr.red_won); p=model.predict_proba(te[cols])[:,1]
        rows.append({"model":label,**metrics(te.red_won.to_numpy(),p)})
    pred=pd.DataFrame(rows)

    stability=[]
    for c in imp_cols:
        a=usable[c]; future=usable.groupby("r_id")[c].shift(-1)
        mask=a.notna()&future.notna()
        stability.append({"feature":c,"n":int(mask.sum()),"next_fight_corr":float(a[mask].corr(future[mask])) if mask.sum()>=5 else np.nan,
                          "mean":float(a.mean()),"sd":float(a.std())})
    stab=pd.DataFrame(stability)

    market_rows=[]; market_summary=pd.DataFrame()
    if MARKET_AUDIT_PATH.exists():
        b=pd.read_csv(MARKET_AUDIT_PATH); ml=b[b.market_key.eq("moneyline")].copy()
        for fid,g in ml.groupby(ml.fight_id.astype(str)):
            if len(g)<2: continue
            g=g.sort_values("market_fair_probability",ascending=False); fav=g.iloc[0]
            market_rows.append({"fight_id":str(fid),"favorite_side":str(fav.outcome_side),"market_p":float(fav.market_fair_probability),
                                "mc_p":float(fav.model_probability),"residual":float(fav.market_fair_probability-fav.model_probability),
                                "red_prior_market":float(fav.red_prior_ufc_fights),"blue_prior_market":float(fav.blue_prior_ufc_fights)})
        mm=pd.DataFrame(market_rows); mm=mm[(mm.red_prior_market>=MIN_PRIOR)&(mm.blue_prior_market>=MIN_PRIOR)]
        mm=mm.merge(usable[["fight_id"]+imp_cols],on="fight_id",how="left")
        # orient red-blue features favorite-dog
        for c in imp_cols: mm[f"fav_{c}"]=np.where(mm.favorite_side.eq("red"),mm[c],-mm[c])
        xcols=[f"fav_{c}" for c in imp_cols]
        valid=mm.dropna(subset=["residual"]).copy()
        corrs=[abs(valid[x].corr(valid.residual)) if valid[x].notna().sum()>=5 else np.nan for x in xcols]
        # small sample: ridge LOOCV-style fitted correlation is descriptive only; report individual correlations and in-sample R2.
        X=valid[xcols]
        pipe=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=10.0)); pipe.fit(X,valid.residual); ph=pipe.predict(X)
        market_summary=pd.DataFrame([{"mature_priced_fights":len(valid),"mean_market_p":valid.market_p.mean(),"mean_mc_p":valid.mc_p.mean(),
                                     "mean_residual_pp":100*valid.residual.mean(),"phase_fit_r2_insample":1-float(np.sum((valid.residual-ph)**2)/np.sum((valid.residual-valid.residual.mean())**2)),
                                     "phase_fit_mae_pp":100*mean_absolute_error(valid.residual,ph),
                                     **{f"corr_residual__{x.replace('fav_imp_delta__','')}":valid[x].corr(valid.residual) for x in xcols}}])
        valid.to_csv(OUT/"mature_market_rows.csv",index=False)

    OUT.mkdir(parents=True,exist_ok=True)
    pred.to_csv(OUT/"winner_prediction.csv",index=False); stab.to_csv(OUT/"phase_imposition_stability.csv",index=False)
    market_summary.to_csv(OUT/"market_residual_summary.csv",index=False); usable.to_csv(OUT/"mature_prefight_features.csv",index=False)
    print("PHASE IMPOSITION FALSIFICATION")
    print(f"fight aggregates={len(f)} mature usable={len(usable)} train={len(tr)} test={len(te)}")
    print("\nWINNER PREDICTION")
    print(pred.to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    print("\nSTABILITY")
    print(stab.to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    if not market_summary.empty:
        print("\nMATURE MARKET RESIDUAL")
        print(market_summary.to_string(index=False,float_format=lambda x:f"{x:.5f}"))

if __name__=="__main__": main()
