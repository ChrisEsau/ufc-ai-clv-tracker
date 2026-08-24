"""Measurement-only prototype for a tiny dynamic fighter-policy layer.

This does NOT modify Event Clock, FSR V3, or production inference.

Question
--------
Can a generic shared policy that reacts to live fight state, while using FSR V3
for fighter-specific capabilities, add chronological OOS winner signal and
reduce mature-favorite probability compression?

Prototype states: DISTANCE, CLINCH_RED, CLINCH_BLUE, GROUND_RED, GROUND_BLUE.
The policy is deliberately tiny and shared by every fighter.  FSR determines
strike/wrestling preference and contest success.  The only live adaptation is
that the fighter currently behind becomes modestly more willing to seek a phase
change.  No market information enters policy construction or fitting.

This is a falsification screen, not a candidate simulator implementation.
"""
from __future__ import annotations

from hashlib import blake2b
from math import exp, log
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH

MASTER_PATH = Path("data/master/ufc_master.parquet")
MARKET_AUDIT_PATH = Path("/tmp/market_edge/bet_level_audit.csv")
OUT = Path("data/diagnostics/event_clock_mc_v2/dynamic_fighter_policy_prototype")
MIN_PRIOR = 3
PATHS = 96
TICKS = 48
SEED = 20260824
EPS = 1e-9


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _find(cols, aliases, required=True):
    norm = {_norm(c): c for c in cols}
    for a in aliases:
        if _norm(a) in norm:
            return norm[_norm(a)]
    for a in aliases:
        aa = _norm(a)
        hits = [c for c in cols if aa and aa in _norm(c)]
        if len(hits) == 1:
            return hits[0]
    if required:
        raise KeyError(f"none of aliases found: {aliases}; columns={list(cols)}")
    return None


def sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return 1.0 / (1.0 + exp(-x))


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return log(p / (1.0 - p))


def stable_seed(fight_id: str) -> int:
    h = blake2b(str(fight_id).encode(), digest_size=8).digest()
    return (int.from_bytes(h, "little") + SEED) % (2**63 - 1)


def matchup_accuracy(att: pd.Series, deff: pd.Series, prefix: str) -> float:
    if prefix == "standing":
        base = float(att["standing_accuracy_baseline"])
        off = float(att["standing_striking_offense"])
        defense = float(deff["standing_striking_defense"])
    elif prefix == "td":
        base = float(att["takedown_completion_baseline"])
        off = float(att["takedown_offense"])
        defense = float(deff["takedown_defense"])
    else:
        base = float(att["ground_accuracy_baseline"])
        off = float(att["ground_striking_offense"])
        defense = 0.0
    return sigmoid(logit(base) + off - defense)


def wrestle_propensity(me: pd.Series, opp: pd.Series, score_gap: float) -> float:
    """Generic policy: compare own TD appetite with own standing appetite.

    The scale conversion (standing / 25) only puts the two positive tendencies
    into comparable numerical range; it is fixed a priori and never market-fit.
    A fighter who is behind gets a small generic phase-change pressure boost.
    """
    td = max(float(me["takedown_tendency"]) * float(opp["takedown_suppression"]), 0.02)
    st = max(float(me["standing_striking_tendency"]) * float(opp["standing_striking_suppression"]) / 25.0, 0.02)
    base = log(td / st)
    desperation = 0.35 if score_gap < -1.0 else (0.15 if score_gap < 0.0 else 0.0)
    return float(np.clip(sigmoid(base + desperation), 0.03, 0.80))


def entry_success(att: pd.Series, deff: pd.Series) -> float:
    a = max(float(att["takedown_tendency"]) * float(deff["takedown_suppression"]), 0.05)
    b = max(float(deff["standing_striking_tendency"]) * float(att["standing_striking_suppression"]) / 25.0, 0.05)
    return float(np.clip(sigmoid(log(a / b)), 0.10, 0.90))


def ground_escape_prob(bottom: pd.Series, top: pd.Series, score_gap: float) -> float:
    """Minimal generic escape policy; uses existing ground tendencies only.

    This is intentionally weak so the prototype does not invent a hidden escape
    FSR trait.  It merely lets a live state persist or reset instead of using a
    pre-drawn fight budget.
    """
    top_hold = max(float(top["ground_striking_tendency"]) * float(bottom["ground_striking_suppression"]), 0.1)
    bottom_reset = max(float(bottom["standing_striking_tendency"]) / 25.0, 0.1)
    p = sigmoid(-0.5 + 0.45 * log(bottom_reset / top_hold))
    if score_gap < -1.0:
        p = min(0.85, p + 0.08)
    return float(np.clip(p, 0.08, 0.75))


def simulate_dynamic_features(red: pd.Series, blue: pd.Series, fight_id: str) -> dict[str, float]:
    rng = np.random.default_rng(stable_seed(fight_id))
    red_scores = []
    red_distance_wins = []
    red_control_ticks = []
    blue_control_ticks = []
    phase_changes = []

    r_st_acc = matchup_accuracy(red, blue, "standing")
    b_st_acc = matchup_accuracy(blue, red, "standing")
    r_td = matchup_accuracy(red, blue, "td")
    b_td = matchup_accuracy(blue, red, "td")
    r_gr = matchup_accuracy(red, blue, "ground")
    b_gr = matchup_accuracy(blue, red, "ground")

    for _ in range(PATHS):
        state = "D"
        score = 0.0
        r_ctrl = b_ctrl = changes = dist_wins = 0
        for _tick in range(TICKS):
            if state == "D":
                rp = wrestle_propensity(red, blue, score)
                bp = wrestle_propensity(blue, red, -score)
                ra = rng.random() < rp
                ba = rng.random() < bp
                if ra or ba:
                    if ra and ba:
                        # simultaneous phase change: stronger entry side gets initiative
                        er = entry_success(red, blue)
                        eb = entry_success(blue, red)
                        p_red = er / max(er + eb, EPS)
                        red_initiates = rng.random() < p_red
                    else:
                        red_initiates = ra
                    if red_initiates:
                        if rng.random() < entry_success(red, blue):
                            state = "CR"; changes += 1; score += 0.15
                        else:
                            score -= 0.05
                    else:
                        if rng.random() < entry_success(blue, red):
                            state = "CB"; changes += 1; score -= 0.15
                        else:
                            score += 0.05
                else:
                    # shared striking action; paired FSR controls who wins exchange
                    rr = r_st_acc * max(float(red["standing_striking_tendency"]), 1.0)
                    bb = b_st_acc * max(float(blue["standing_striking_tendency"]), 1.0)
                    p_red = rr / max(rr + bb, EPS)
                    if rng.random() < p_red:
                        score += 0.20; dist_wins += 1
                    else:
                        score -= 0.20

            elif state == "CR":
                r_ctrl += 1
                shoot = rng.random() < (0.68 + (0.10 if score < 0 else 0.0))
                if shoot and rng.random() < r_td:
                    state = "GR"; changes += 1; score += 0.30
                elif rng.random() < 0.28:
                    state = "D"; changes += 1
                else:
                    score += 0.05

            elif state == "CB":
                b_ctrl += 1
                shoot = rng.random() < (0.68 + (0.10 if score > 0 else 0.0))
                if shoot and rng.random() < b_td:
                    state = "GB"; changes += 1; score -= 0.30
                elif rng.random() < 0.28:
                    state = "D"; changes += 1
                else:
                    score -= 0.05

            elif state == "GR":
                r_ctrl += 1
                if rng.random() < r_gr:
                    score += 0.12
                if rng.random() < ground_escape_prob(blue, red, -score):
                    state = "D"; changes += 1

            else:  # GB
                b_ctrl += 1
                if rng.random() < b_gr:
                    score -= 0.12
                if rng.random() < ground_escape_prob(red, blue, score):
                    state = "D"; changes += 1

        red_scores.append(score)
        red_distance_wins.append(dist_wins / TICKS)
        red_control_ticks.append(r_ctrl / TICKS)
        blue_control_ticks.append(b_ctrl / TICKS)
        phase_changes.append(changes / TICKS)

    rs = np.asarray(red_scores, dtype=float)
    return {
        "dyn_mean_score": float(rs.mean()),
        "dyn_score_sd": float(rs.std(ddof=0)),
        "dyn_red_path_win_share": float(np.mean(rs > 0.0)),
        "dyn_red_distance_win_rate": float(np.mean(red_distance_wins)),
        "dyn_red_control_share": float(np.mean(red_control_ticks)),
        "dyn_blue_control_share": float(np.mean(blue_control_ticks)),
        "dyn_control_edge": float(np.mean(red_control_ticks) - np.mean(blue_control_ticks)),
        "dyn_phase_change_rate": float(np.mean(phase_changes)),
    }


def load_dataset() -> pd.DataFrame:
    fsr = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    fsr["event_date"] = pd.to_datetime(fsr["event_date"], errors="coerce").dt.normalize()

    master = pd.read_parquet(MASTER_PATH).copy(); cols = master.columns
    mf = _find(cols,["fight_id","bout_id"]); mr = _find(cols,["r_id","red_id","r_fighter_id"])
    mb = _find(cols,["b_id","blue_id","b_fighter_id"]); mw = _find(cols,["winner_id","winner_fighter_id","winner"])
    md = _find(cols,["event_date","date"])
    m = master[[mf,mr,mb,mw,md]].drop_duplicates(mf).copy()
    m.columns=["fight_id","r_id","b_id","winner","event_date"]
    for c in ["fight_id","r_id","b_id","winner"]: m[c]=m[c].astype(str)
    m["event_date"] = pd.to_datetime(m["event_date"], errors="coerce").dt.normalize()

    # Chronological UFC prior counts using only fights represented in canonical FSR snapshots.
    fight_order = m[["fight_id","event_date","r_id","b_id"]].sort_values(["event_date","fight_id"]).copy()
    counts={}; prior=[]
    for _,r in fight_order.iterrows():
        rid,bid=str(r.r_id),str(r.b_id)
        prior.append((str(r.fight_id),counts.get(rid,0),counts.get(bid,0)))
        counts[rid]=counts.get(rid,0)+1; counts[bid]=counts.get(bid,0)+1
    pr=pd.DataFrame(prior,columns=["fight_id","r_prior","b_prior"])
    m=m.merge(pr,on="fight_id",how="left")

    id_like={"fight_id","fighter_id","event_date","fighter_name","name","side"}
    nums=[c for c in fsr.columns if c not in id_like and pd.api.types.is_numeric_dtype(fsr[c])]
    rows=[]
    needed={
        "standing_striking_tendency","standing_striking_suppression","standing_striking_offense","standing_striking_defense","standing_accuracy_baseline",
        "takedown_tendency","takedown_suppression","takedown_offense","takedown_defense","takedown_completion_baseline",
        "ground_striking_tendency","ground_striking_suppression","ground_striking_offense","ground_accuracy_baseline",
    }
    for _,r in m.iterrows():
        if int(r.r_prior) < MIN_PRIOR or int(r.b_prior) < MIN_PRIOR: continue
        fr=fsr[(fsr.fight_id.eq(str(r.fight_id))) & (fsr.fighter_id.eq(str(r.r_id)))]
        fb=fsr[(fsr.fight_id.eq(str(r.fight_id))) & (fsr.fighter_id.eq(str(r.b_id)))]
        if len(fr)!=1 or len(fb)!=1: continue
        rr=fr.iloc[0]; bb=fb.iloc[0]
        if not needed.issubset(set(rr.index)): continue
        rec={"fight_id":str(r.fight_id),"event_date":r.event_date,"r_id":str(r.r_id),"b_id":str(r.b_id),
             "r_prior":int(r.r_prior),"b_prior":int(r.b_prior),"red_won":int(str(r.winner)==str(r.r_id) or str(r.winner).lower()=="red")}
        for c in nums:
            try: rec[f"fsr_delta__{c}"]=float(rr[c])-float(bb[c])
            except Exception: pass
        rec.update(simulate_dynamic_features(rr,bb,str(r.fight_id)))
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["event_date","fight_id"]).reset_index(drop=True)


def metrics(y, p):
    return {
        "n":len(y),"accuracy":accuracy_score(y,p>=0.5),"auc":roc_auc_score(y,p),
        "brier":brier_score_loss(y,p),"log_loss":log_loss(y,np.clip(p,1e-6,1-1e-6)),
    }


def fit_eval(tr, te, cols, label):
    model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(C=1.0,max_iter=4000))
    model.fit(tr[cols],tr.red_won); p=model.predict_proba(te[cols])[:,1]
    return model,p,{"model":label,**metrics(te.red_won.to_numpy(),p)}


def main():
    z=load_dataset(); cut=max(1,int(.70*len(z))); tr=z.iloc[:cut].copy(); te=z.iloc[cut:].copy()
    fsr_cols=[c for c in z if c.startswith("fsr_delta__")]
    dyn_cols=[c for c in z if c.startswith("dyn_")]
    rows=[]
    m_fsr,p_fsr,r=fit_eval(tr,te,fsr_cols,"fsr_only"); rows.append(r)
    m_dyn,p_dyn,r=fit_eval(tr,te,dyn_cols,"dynamic_only"); rows.append(r)
    m_both,p_both,r=fit_eval(tr,te,fsr_cols+dyn_cols,"fsr_plus_dynamic"); rows.append(r)
    pred=pd.DataFrame(rows)
    base=pred[pred.model.eq("fsr_only")].iloc[0]
    pred["delta_auc_vs_fsr"]=pred.auc-float(base.auc)
    pred["delta_brier_vs_fsr"]=pred.brier-float(base.brier)
    pred["delta_logloss_vs_fsr"]=pred.log_loss-float(base.log_loss)

    teout=te[["fight_id","event_date","red_won"]+dyn_cols].copy(); teout["p_fsr"]=p_fsr; teout["p_dynamic_only"]=p_dyn; teout["p_fsr_plus_dynamic"]=p_both

    market_summary=pd.DataFrame(); market_rows=pd.DataFrame()
    if MARKET_AUDIT_PATH.exists():
        b=pd.read_csv(MARKET_AUDIT_PATH); ml=b[b.market_key.eq("moneyline")].copy(); out=[]
        for fid,g in ml.groupby(ml.fight_id.astype(str)):
            if len(g)<2: continue
            g=g.sort_values("market_fair_probability",ascending=False); fav=g.iloc[0]
            if float(fav.red_prior_ufc_fights)<MIN_PRIOR or float(fav.blue_prior_ufc_fights)<MIN_PRIOR: continue
            out.append({"fight_id":str(fid),"favorite_side":str(fav.outcome_side),"market_p":float(fav.market_fair_probability),"mc_p":float(fav.model_probability)})
        mm=pd.DataFrame(out).merge(z[["fight_id"]+dyn_cols],on="fight_id",how="inner")
        # Fit dynamic probability model using chronological winner training only, then score mature priced fights.
        pdyn=m_both.predict_proba(mm[fsr_cols+dyn_cols] if set(fsr_cols).issubset(mm.columns) else pd.DataFrame())[:,1] if False else None
        # We cannot score combined model without FSR columns in this compact merge; market test therefore uses direct dynamic path-win share as a pure diagnostic.
        mm["dyn_fav_path_p"]=np.where(mm.favorite_side.eq("red"),mm.dyn_red_path_win_share,1.0-mm.dyn_red_path_win_share)
        mm["market_minus_mc_pp"]=100*(mm.market_p-mm.mc_p)
        mm["market_minus_dynamic_pp"]=100*(mm.market_p-mm.dyn_fav_path_p)
        market_rows=mm
        market_summary=pd.DataFrame([{
            "mature_priced_fights":len(mm),"mean_market_p":mm.market_p.mean(),"mean_mc_p":mm.mc_p.mean(),
            "mean_dynamic_path_p":mm.dyn_fav_path_p.mean(),"mean_market_minus_mc_pp":mm.market_minus_mc_pp.mean(),
            "mean_market_minus_dynamic_pp":mm.market_minus_dynamic_pp.mean(),
            "corr_dynamic_with_market":mm.dyn_fav_path_p.corr(mm.market_p),
            "corr_dynamic_with_market_mc_residual":mm.dyn_fav_path_p.corr(mm.market_p-mm.mc_p),
        }])

    OUT.mkdir(parents=True,exist_ok=True)
    pred.to_csv(OUT/"winner_prediction.csv",index=False)
    teout.to_csv(OUT/"chronological_holdout_predictions.csv",index=False)
    z.to_csv(OUT/"mature_dynamic_features.csv",index=False)
    market_rows.to_csv(OUT/"mature_market_dynamic.csv",index=False)
    market_summary.to_csv(OUT/"market_summary.csv",index=False)

    print("DYNAMIC FIGHTER POLICY PROTOTYPE")
    print(f"mature={len(z)} train={len(tr)} test={len(te)} paths={PATHS} ticks={TICKS}")
    print("\nWINNER PREDICTION")
    print(pred.to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    if not market_summary.empty:
        print("\nMATURE MARKET DIAGNOSTIC")
        print(market_summary.to_string(index=False,float_format=lambda x:f"{x:.5f}"))


if __name__=="__main__":
    main()
