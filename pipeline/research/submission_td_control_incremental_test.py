#!/usr/bin/env python3
"""Leakage-safe TD/control incremental test for side-specific SUB wins.

Research-only. No Brain, FSR, or market inputs.

Baseline is the previously validated incremental SUB signal:
    Glicko SUB O/D + prefight SUB-attempt opportunity + age delta.

This script asks whether leakage-safe prefight TD opportunity and control-time
opportunity add independent signal. All logistic coefficients are fitted only
on pre-2025 observations and evaluated untouched on 2025+ fights. Any six-way
integration changes only R_SUB/B_SUB logits of the frozen 0.25 partial-evidence
joint Glicko-6 probabilities.
"""
from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.fsr_v2.replay.engine import aggregate_fights
from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.research.prefight_strength_elo import build_bouts
from pipeline.research.standalone_glicko_six_way_partial_sweep import run as run_glicko, SIX_COLS
from pipeline.research.submission_incremental_signal_test import (
    EPS, add_age, binary_metrics, build_opportunity, fit_logistic, logit,
    predict_logistic, six_metrics,
)
from pipeline.research.submission_stat_auc_screen import method_family


def build_td_control_opportunity(bouts: pd.DataFrame) -> pd.DataFrame:
    """Chronological prefight TD/control matchup signals from prior UFC fights.

    TD signal = own prior TD-attempt rate/15m + opponent prior TD-attempts-allowed rate/15m.
    Control signal = own prior control share + opponent prior control-allowed share.
    Same-event updates are delayed to avoid leakage between bouts on one event date.
    """
    f=aggregate_fights(build_paired_rounds()).copy()
    f['event_date']=pd.to_datetime(f.event_date).dt.normalize()
    f['fighter_key']=f.fighter_name.astype(str).str.strip().str.lower()
    f['opp_key']=f.opponent_name.astype(str).str.strip().str.lower()
    for c in ['td_attempted','opponent_td_attempted','ctrl_sec','opponent_ctrl_sec','fight_elapsed_seconds']:
        f[c]=pd.to_numeric(f[c],errors='coerce').fillna(0.0)
    lookup={(r.event_date,r.fighter_key,r.opp_key):(float(r.td_attempted),float(r.opponent_td_attempted),float(r.ctrl_sec),float(r.opponent_ctrl_sec),float(r.fight_elapsed_seconds)) for r in f.itertuples(index=False)}

    # own TD, allowed TD, own control, allowed control, exposure seconds
    state=defaultdict(lambda:[0.0,0.0,0.0,0.0,0.0])
    rows=[]
    bsort=bouts.sort_values(['date','bout_id']).copy(); bsort['date_norm']=pd.to_datetime(bsort.date).dt.normalize()
    for dt,batch in bsort.groupby('date_norm',sort=True):
        pending=[]
        for b in batch.itertuples(index=False):
            rk=str(b.red_fighter).strip().lower(); bk=str(b.blue_fighter).strip().lower(); sr=state[rk]; sb=state[bk]
            def rates(s):
                ex=s[4]
                if ex<=0: return (np.nan,np.nan,np.nan,np.nan)
                return (s[0]/ex*900.0,s[1]/ex*900.0,s[2]/ex,s[3]/ex)
            rtd,ratd,rctrl,ractrl=rates(sr); btd,batd,bctrl,bactrl=rates(sb)
            rows.append({
                'bout_id':b.bout_id,
                'r_tdopp':rtd+batd if np.isfinite(rtd) and np.isfinite(batd) else np.nan,
                'b_tdopp':btd+ratd if np.isfinite(btd) and np.isfinite(ratd) else np.nan,
                'r_ctrlopp':rctrl+bactrl if np.isfinite(rctrl) and np.isfinite(bactrl) else np.nan,
                'b_ctrlopp':bctrl+ractrl if np.isfinite(bctrl) and np.isfinite(ractrl) else np.nan,
            })
            rv=lookup.get((dt,rk,bk)); bv=lookup.get((dt,bk,rk))
            # Each paired fighter row already contains own and opponent values. Use each side's own row once.
            if rv is not None and bv is not None:
                pending.append((rk,rv[0],rv[1],rv[2],rv[3],rv[4]))
                pending.append((bk,bv[0],bv[1],bv[2],bv[3],bv[4]))
        for key,td,atd,ctrl,actrl,ex in pending:
            if ex>0:
                s=state[key]; s[0]+=td; s[1]+=atd; s[2]+=ctrl; s[3]+=actrl; s[4]+=ex
    return pd.DataFrame(rows)


def make_side_frame(pred, subopp, age, tc):
    d=pred.merge(subopp,on='bout_id',how='left').merge(age,on='bout_id',how='left').merge(tc,on='bout_id',how='left')
    rows=[]
    for b in d.itertuples(index=False):
        am=method_family(getattr(b,'method',''))
        rows.append({'date':b.date,'bout_id':b.bout_id,'side':'R','y':int(am=='SUB' and b.winner==b.red_fighter),
                     'od':float(logit(b.q_r_sub)),'opp':b.r_subopp,'age':b.age_delta,'td':b.r_tdopp,'ctrl':b.r_ctrlopp})
        rows.append({'date':b.date,'bout_id':b.bout_id,'side':'B','y':int(am=='SUB' and b.winner==b.blue_fighter),
                     'od':float(logit(b.q_b_sub)),'opp':b.b_subopp,'age':-b.age_delta,'td':b.b_tdopp,'ctrl':b.b_ctrlopp})
    return pd.DataFrame(rows),d


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--holdout-from',default='2025-01-01'); ap.add_argument('--output-dir',type=Path,default=Path('data/diagnostics/submission_td_control_incremental')); args=ap.parse_args()
    bouts=build_bouts(pd.read_parquet('data/master/ufc_master.parquet'))
    pred=run_glicko(bouts,0.25)
    subopp=build_opportunity(bouts,Path('data/fight_details/ufc_round_stats.parquet'))
    age=add_age(bouts); tc=build_td_control_opportunity(bouts)
    side,fights=make_side_frame(pred,subopp,age,tc); side['date']=pd.to_datetime(side.date); cutoff=pd.Timestamp(args.holdout_from)

    specs={
        'BASE_OD_OPP_AGE':['od','opp','age'],
        '+TD':['od','opp','age','td'],
        '+CTRL':['od','opp','age','ctrl'],
        '+TD+CTRL':['od','opp','age','td','ctrl'],
    }
    models={}; results=[]
    # Common complete-case cohort for a fair side-level comparison across all four specs.
    common_cols=['od','opp','age','td','ctrl']
    tr_common=side[(side.date<cutoff)].dropna(subset=common_cols)
    ho_common=side[(side.date>=cutoff)].dropna(subset=common_cols)
    for name,cols in specs.items():
        tr=tr_common; ho=ho_common
        m=fit_logistic(tr[cols].to_numpy(),tr.y.to_numpy(),l2=1.0); models[name]=m
        p=predict_logistic(m,ho[cols].to_numpy()); bm=binary_metrics(ho.y,p)
        results.append({'model':name,'features':'+'.join(cols),'train_n':len(tr),'holdout_n':len(ho),**bm,
                        'coef_intercept':float(m['coef'][0]),**{f'coef_{c}':float(m['coef'][i+1]) for i,c in enumerate(cols)}})
    res=pd.DataFrame(results).sort_values(['log_loss','auc'],ascending=[True,False])

    # Six-way: identical complete-case fights; each candidate contributes only its incremental
    # logit delta relative to BASE_OD_OPP_AGE. KO/DEC logits stay untouched.
    h=fights[pd.to_datetime(fights.date)>=cutoff].copy()
    common=h[['r_subopp','b_subopp','age_delta','r_tdopp','b_tdopp','r_ctrlopp','b_ctrlopp']].notna().all(axis=1).to_numpy()
    hb=h.loc[common].copy(); P0=hb[list(SIX_COLS)].to_numpy(float); six=[]
    six.append({'model':'BASE_0.25_RAW','complete_case_n':len(hb),**six_metrics(hb,P0)})
    base=models['BASE_OD_OPP_AGE']
    basecols=specs['BASE_OD_OPP_AGE']
    def X(side_name,cols):
        if side_name=='R':
            frame=pd.DataFrame({'od':logit(hb.q_r_sub.to_numpy()),'opp':hb.r_subopp.to_numpy(),'age':hb.age_delta.to_numpy(),'td':hb.r_tdopp.to_numpy(),'ctrl':hb.r_ctrlopp.to_numpy()})
        else:
            frame=pd.DataFrame({'od':logit(hb.q_b_sub.to_numpy()),'opp':hb.b_subopp.to_numpy(),'age':-hb.age_delta.to_numpy(),'td':hb.b_tdopp.to_numpy(),'ctrl':hb.b_ctrlopp.to_numpy()})
        return frame[cols].to_numpy()
    base_pr=predict_logistic(base,X('R',basecols)); base_pb=predict_logistic(base,X('B',basecols))
    # First include validated BASE incremental correction itself vs an O/D-only train fit on same common rows.
    odfit=fit_logistic(tr_common[['od']].to_numpy(),tr_common.y.to_numpy(),l2=1.0)
    odr=predict_logistic(odfit,X('R',['od'])); odb=predict_logistic(odfit,X('B',['od']))
    variants=[('BASE_OD_OPP_AGE',base_pr,base_pb,odr,odb)]
    for name in ['+TD','+CTRL','+TD+CTRL']:
        pr=predict_logistic(models[name],X('R',specs[name])); pb=predict_logistic(models[name],X('B',specs[name]))
        variants.append((name,pr,pb,odr,odb))
    for name,pr,pb,qr,qb in variants:
        dr=logit(pr)-logit(qr); db=logit(pb)-logit(qb)
        L=np.log(np.clip(P0,EPS,1)); L[:,1]+=dr; L[:,4]+=db; L-=L.max(axis=1,keepdims=True); E=np.exp(L); P=E/E.sum(axis=1,keepdims=True)
        six.append({'model':name,'complete_case_n':len(hb),**six_metrics(hb,P)})
    sixdf=pd.DataFrame(six).sort_values('log_loss')

    args.output_dir.mkdir(parents=True,exist_ok=True)
    res.to_csv(args.output_dir/'side_td_control_incremental.csv',index=False)
    sixdf.to_csv(args.output_dir/'sixway_td_control_incremental.csv',index=False)
    side.to_csv(args.output_dir/'side_features.csv',index=False)
    tc.to_csv(args.output_dir/'prefight_td_control_features.csv',index=False)
    summary={'holdout_from':args.holdout_from,'negative_evidence_weight':0.25,'protocol':'pre-2025 fit; untouched 2025+ holdout; common complete cases','side_results':res.to_dict('records'),'sixway_results':sixdf.to_dict('records')}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('\nSIDE-SPECIFIC SUB WIN — COMMON COMPLETE CASE\n'+res.to_string(index=False))
    print('\nSIX-WAY — COMMON COMPLETE CASE\n'+sixdf.to_string(index=False))

if __name__=='__main__': main()
