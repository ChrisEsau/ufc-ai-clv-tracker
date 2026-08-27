"""Test chronological KO-history conditional priors for KO V3.

Research only. Production unchanged.

Question: should fighters with zero prior KO/TKO wins use a different population
prior and shrinkage strength than fighters with >=1 prior KO/TKO win?

We test this separately for:
- attacker KO/Sig offense (grouped by attacker prior KO wins ==0 vs >=1)
- defender KO-loss/Sig susceptibility (grouped by defender prior KO losses ==0 vs >=1)
- both sides conditional

All group priors are strictly chronological: for each event date, the group prior is
built only from fighter-fight rows on earlier event dates, where those historical
rows are classified by their own prefight group at that time.

Selection: 2020-2024. Confirmation: untouched 2025-2026.
Primary selection metric: strike-weighted grouped Bernoulli log loss.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline.research import ko_v3_from_scratch_stage1 as s1

OUTDIR = Path("data/research/ko_v3_conditional_ko_history_priors")
STRENGTHS = (25.0, 50.0, 100.0, 200.0, 400.0, 800.0)
EPS = 1e-9


def clip(p): return np.clip(np.asarray(p, dtype=float), EPS, 1.0-EPS)
def logit(p):
    p=clip(p); return np.log(p/(1.0-p))
def sigmoid(z):
    z=np.clip(np.asarray(z,dtype=float),-30.0,30.0); return 1.0/(1.0+np.exp(-z))


def add_chronological_priors(frame: pd.DataFrame) -> pd.DataFrame:
    df=frame.copy(); df["event_date"]=pd.to_datetime(df["event_date"]).dt.normalize()
    valid=df[df.sig_landed>0].copy()
    valid["att_group"]=np.where(valid.prior_ko_wins.to_numpy(float)>0,"has_ko","zero_ko")
    valid["def_group"]=np.where(valid.opp_prior_ko_losses.to_numpy(float)>0,"has_loss","zero_loss")

    dates=pd.DataFrame({"event_date":sorted(valid.event_date.unique())})

    # Universal population prior.
    daily=(valid.groupby("event_date",as_index=False).agg(k=("ko_win","sum"),n=("sig_landed","sum")).sort_values("event_date"))
    daily["prior_k"]=daily.k.cumsum().shift(1,fill_value=0.0); daily["prior_n"]=daily.n.cumsum().shift(1,fill_value=0.0)
    daily["p_univ"]=np.where(daily.prior_n>0,daily.prior_k/daily.prior_n,np.nan)
    dates=dates.merge(daily[["event_date","p_univ"]],on="event_date",how="left")

    def group_prior(group_col: str, prefix: str):
        g=(valid.groupby(["event_date",group_col],as_index=False).agg(k=("ko_win","sum"),n=("sig_landed","sum")))
        pivk=g.pivot(index="event_date",columns=group_col,values="k").fillna(0.0).sort_index()
        pivn=g.pivot(index="event_date",columns=group_col,values="n").fillna(0.0).sort_index()
        out=pd.DataFrame(index=dates.event_date)
        for grp in sorted(valid[group_col].unique()):
            k=pivk.get(grp,pd.Series(0.0,index=pivk.index)).reindex(out.index,fill_value=0.0)
            n=pivn.get(grp,pd.Series(0.0,index=pivn.index)).reindex(out.index,fill_value=0.0)
            pk=k.cumsum().shift(1,fill_value=0.0); pn=n.cumsum().shift(1,fill_value=0.0)
            out[f"{prefix}_{grp}"]=np.where(pn>0,pk/pn,np.nan)
        return out.reset_index(names="event_date")

    dates=dates.merge(group_prior("att_group","p_att"),on="event_date",how="left")
    dates=dates.merge(group_prior("def_group","p_def"),on="event_date",how="left")
    df=df.merge(dates,on="event_date",how="left",validate="many_to_one")
    df["att_group"]=np.where(df.prior_ko_wins.to_numpy(float)>0,"has_ko","zero_ko")
    df["def_group"]=np.where(df.opp_prior_ko_losses.to_numpy(float)>0,"has_loss","zero_loss")
    # Fall back to universal prior only when a historical subgroup has not yet accumulated exposure.
    for c in ["p_att_has_ko","p_att_zero_ko","p_def_has_loss","p_def_zero_loss"]:
        df[c]=df[c].fillna(df.p_univ)
    return df


def choose_prior(df, side):
    if side=="att":
        return np.where(df.att_group.eq("has_ko"),df.p_att_has_ko,df.p_att_zero_ko).astype(float)
    return np.where(df.def_group.eq("has_loss"),df.p_def_has_loss,df.p_def_zero_loss).astype(float)


def hazards(df, s_zero, s_pos, cond_att=True, cond_def=True):
    p0=df.p_univ.to_numpy(float)
    att_k=df.prior_ko_wins.to_numpy(float); att_n=df.prior_sig_landed.to_numpy(float)
    def_k=df.opp_prior_ko_losses.to_numpy(float); def_n=df.opp_prior_sig_absorbed.to_numpy(float)

    att_has=att_k>0; def_has=def_k>0
    sa=np.where(att_has,s_pos,s_zero); sd=np.where(def_has,s_pos,s_zero)
    pa0=choose_prior(df,"att") if cond_att else p0
    pd0=choose_prior(df,"def") if cond_def else p0
    p_att=(att_k+sa*pa0)/(att_n+sa)
    p_def=(def_k+sd*pd0)/(def_n+sd)

    # Express each component as a deviation from its own prior, then apply that
    # deviation to the universal population center. This prevents double-counting
    # the elevated has-KO subgroup base rate.
    da=logit(p_att)-logit(pa0)
    dd=logit(p_def)-logit(pd0)
    both=sigmoid(logit(p0)+da+dd)
    return p_att,p_def,both


def universal(df,s=400.0):
    p0=df.p_univ.to_numpy(float); ak=df.prior_ko_wins.to_numpy(float); an=df.prior_sig_landed.to_numpy(float); dk=df.opp_prior_ko_losses.to_numpy(float); dn=df.opp_prior_sig_absorbed.to_numpy(float)
    pa=(ak+s*p0)/(an+s); pdv=(dk+s*p0)/(dn+s)
    return sigmoid(logit(p0)+(logit(pa)-logit(p0))+(logit(pdv)-logit(p0)))


def metrics(df,h):
    h=np.asarray(h,float); y=df.ko_win.to_numpy(int); n=df.sig_landed.to_numpy(float); hc=clip(h)
    ll=-float(np.sum(y*np.log(hc)+(n-y)*np.log(1.0-hc))/np.sum(n)); exp=float(np.sum(h*n)); act=float(np.sum(y)); auc=float(roc_auc_score(y,h))
    tmp=df[["fight_id","ko_win"]].copy(); tmp["h"]=h; correct=[]
    for _,g in tmp.groupby("fight_id"):
        if len(g)==2 and int(g.ko_win.sum())==1:
            wh=float(g.loc[g.ko_win==1,"h"].iloc[0]); lh=float(g.loc[g.ko_win==0,"h"].iloc[0]); correct.append(1.0 if wh>lh else 0.5 if wh==lh else 0.0)
    return {"strike_ll":ll,"e_o":exp/act,"auc":auc,"correct_side":float(np.mean(correct)),"correct_n":len(correct),"zeros":int(np.sum(h<=0)),"mean_h":float(np.sum(h*n)/np.sum(n))}


def period(df,a,b): return df[df.event_date.dt.year.between(a,b)&(df.sig_landed>0)&df.p_univ.notna()].copy().reset_index(drop=True)


def sweep(sel,conf,cond_att,cond_def):
    rows=[]
    for sz in STRENGTHS:
        for sp in STRENGTHS:
            _,_,hs=hazards(sel,sz,sp,cond_att,cond_def); _,_,hc=hazards(conf,sz,sp,cond_att,cond_def)
            rows.append({"s_zero":sz,"s_positive":sp,"selection":metrics(sel,hs),"confirmation":metrics(conf,hc)})
    rows.sort(key=lambda r:r["selection"]["strike_ll"])
    return rows


def main():
    ff,audit=s1.load_raw_fighter_fights(); frame=s1.build_matchup_frame(s1.build_prefight_states(ff)).copy(); frame["fight_id"]=frame.fight_id.astype(str); frame=add_chronological_priors(frame)
    sel=period(frame,2020,2024); conf=period(frame,2025,2026)
    arms={"attacker_conditional":(True,False),"defender_conditional":(False,True),"both_conditional":(True,True)}
    result={"study":"KO-history conditional priors","production_changed":False,"selection":"2020-2024","confirmation":"2025-2026","strengths":list(STRENGTHS),"baseline_s400":{"selection":metrics(sel,universal(sel)),"confirmation":metrics(conf,universal(conf))},"arms":{},"stage1_audit":audit}
    for name,(ca,cd) in arms.items():
        sw=sweep(sel,conf,ca,cd); result["arms"][name]={"selected":sw[0],"top_selection":sw[:8]}
    # Audit subgroup base rates on each period using observed current-fight outcomes/exposure.
    def subgroup_rates(df):
        out={}
        for col in ["att_group","def_group"]:
            out[col]={}
            for g,x in df.groupby(col): out[col][str(g)]={"rows":len(x),"ko":int(x.ko_win.sum()),"sig":float(x.sig_landed.sum()),"ko_per_sig":float(x.ko_win.sum()/x.sig_landed.sum())}
        return out
    result["subgroup_observed"]={"selection":subgroup_rates(sel),"confirmation":subgroup_rates(conf)}
    OUTDIR.mkdir(parents=True,exist_ok=True); (OUTDIR/"results.json").write_text(json.dumps(result,indent=2,sort_keys=True))

    print("KO V3 CONDITIONAL KO-HISTORY PRIORS")
    print("="*96)
    for p in ["selection","confirmation"]:
        m=result["baseline_s400"][p]; print(f"baseline S400 {p:12s}: LL={m['strike_ll']:.8f} E/O={m['e_o']:.3f} AUC={m['auc']:.4f} correct={m['correct_side']:.4f}")
    print()
    for name,b in result["arms"].items():
        r=b["selected"]; s=r["selection"]; c=r["confirmation"]
        print(f"{name}: SELECT s_zero={int(r['s_zero'])} s_pos={int(r['s_positive'])}")
        print(f"  selection    LL={s['strike_ll']:.8f} E/O={s['e_o']:.3f} AUC={s['auc']:.4f} correct={s['correct_side']:.4f}")
        print(f"  confirmation LL={c['strike_ll']:.8f} E/O={c['e_o']:.3f} AUC={c['auc']:.4f} correct={c['correct_side']:.4f}")
    print("\nObserved subgroup KO/Sig rates:")
    for p,block in result["subgroup_observed"].items():
        print(p)
        for col,groups in block.items():
            print(" ",col,", ".join(f"{g}={v['ko_per_sig']:.6f} ({v['ko']}/{int(v['sig'])})" for g,v in groups.items()))
    print(f"\nWrote {OUTDIR/'results.json'}")

if __name__=='__main__': main()
