"""Trace where standing-offense separation is compressed in FSR V3.

Measurement only. Replays the validated standing effectiveness filter and exposes,
at every bantamweight prefight snapshot, the accumulated likelihood-only offense
estimate, the Normal-prior posterior mean used by production, opponent defense,
raw prior accuracy, and realized inverse-needed offense.
"""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.special import expit

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.fsr_v3.replay.paired_effectiveness import (
    build_effectiveness_fighter_fights, standing_effectiveness_spec,
    _normal_prior, _fit_population_beta,
)
from pipeline.fsr_v3.replay.math import beta_binomial_log_likelihood, normalize_log_weights, weighted_mean_sd
from pipeline.simulation.event_clock_mc_v2.diagnostics.fsr_market_residual_audit import build_two_way_market, MARKET_PATH
from pipeline.simulation.event_clock_mc_v2.diagnostics.bantamweight_fsr_population_inverse_market_audit import ROUND_STATS, fighter_rows, round_totals, inverse_side

OUT=Path('data/diagnostics/event_clock_mc_v2/bantamweight_standing_offense_evidence_trace')
DIVISION='bantamweight'


def raw_prior_states(fights: pd.DataFrame) -> pd.DataFrame:
    x=fights.sort_values(['fighter_id','event_date','fight_id']).copy()
    x['prior_landed']=x.groupby('fighter_id')['landed'].transform(lambda s:s.cumsum().shift(1)).fillna(0.0)
    x['prior_attempted']=x.groupby('fighter_id')['attempted'].transform(lambda s:s.cumsum().shift(1)).fillna(0.0)
    x['prior_fights']=x.groupby('fighter_id').cumcount()
    x['raw_prior_accuracy']=np.where(x.prior_attempted>0,x.prior_landed/x.prior_attempted,np.nan)
    return x[['event_date','fight_id','fighter_id','prior_landed','prior_attempted','prior_fights','raw_prior_accuracy']]


def likelihood_trace(fights: pd.DataFrame, spec) -> pd.DataFrame:
    grid=np.linspace(spec.grid_min,spec.grid_max,spec.grid_points)
    prior=_normal_prior(grid,spec.sigma_offense)
    off_states={}; def_states={}; py=[]; pn=[]; beta=None; rows=[]
    for event_date,batch in fights.groupby('event_date',sort=True):
        beta=_fit_population_beta(py,pn,spec.rho,beta)
        pending=[]
        for r in batch.to_dict('records'):
            a=str(r['fighter_id']); d=str(r['opponent_id']); y=float(r['landed']); n=float(r['attempted'])
            olp=prior.copy() + (off_states[a] if a in off_states else 0.0)
            dlp=prior.copy() + (def_states[d] if d in def_states else 0.0)
            ow=normalize_log_weights(olp); dw=normalize_log_weights(dlp)
            opre,osd=weighted_mean_sd(grid,ow); dpre,dsd=weighted_mean_sd(grid,dw)
            if a in off_states:
                lw=normalize_log_weights(off_states[a]); lmean,lsd=weighted_mean_sd(grid,lw); lmle=float(grid[np.argmax(off_states[a])])
            else:
                lmean=0.0; lsd=np.nan; lmle=0.0
            expected=float(expit(beta+opre-dpre))
            rows.append({'event_date':event_date,'fight_id':str(r['fight_id']),'fighter_id':a,'fighter_name':r['fighter_name'],
                         'opponent_id':d,'opponent_name':r['opponent_name'],'population_beta':beta,'population_accuracy':float(expit(beta)),
                         'likelihood_only_mean':lmean,'likelihood_only_mle':lmle,'likelihood_only_sd':lsd,
                         'posterior_offense_mean':opre,'posterior_offense_sd':osd,'opponent_defense_mean':dpre,'opponent_defense_sd':dsd,
                         'matchup_expected_accuracy':expected,'current_landed':y,'current_attempted':n})
            oll=dll=None
            if n>0:
                oll=beta_binomial_log_likelihood(y,n,expit(beta+grid-dpre),spec.rho)
                dll=beta_binomial_log_likelihood(y,n,expit(beta+opre-grid),spec.rho)
            pending.append((a,d,y,n,oll,dll))
        for a,d,y,n,oll,dll in pending:
            if n<=0 or oll is None: continue
            off_states[a]=(off_states[a]+oll if a in off_states else oll.copy()); off_states[a]-=np.max(off_states[a])
            def_states[d]=(def_states[d]+dll if d in def_states else dll.copy()); def_states[d]-=np.max(def_states[d])
            py.append(y); pn.append(n)
    return pd.DataFrame(rows)


def main():
    cfg=FSRV3Config(); spec=standing_effectiveness_spec(cfg)
    fights=build_effectiveness_fighter_fights(spec)
    trace=likelihood_trace(fights,spec).merge(raw_prior_states(fights),on=['event_date','fight_id','fighter_id'],how='left',validate='one_to_one')

    master=pd.read_parquet(MASTER_PATH).drop_duplicates('fight_id').copy(); master['fight_id']=master['fight_id'].astype(str)
    master['event_date']=pd.to_datetime(master['date'],errors='coerce').dt.normalize()
    master=master[master['division'].astype(str).str.lower().str.strip().eq(DIVISION)].copy()
    fsr=pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy(); fsr['fight_id']=fsr['fight_id'].astype(str); fsr['fighter_id']=fsr['fighter_id'].astype(str)
    trace=trace.merge(fsr[['fight_id','fighter_id','standing_striking_offense','standing_striking_defense']],on=['fight_id','fighter_id'],how='left')
    trace['published_minus_replay']=trace['standing_striking_offense']-trace['posterior_offense_mean']
    trace['prior_shrinkage']=trace['posterior_offense_mean']-trace['likelihood_only_mean']

    rs=pd.read_parquet(ROUND_STATS).copy(); fcol='fight_id' if 'fight_id' in rs.columns else 'bout_id'; rs[fcol]=rs[fcol].astype(str)
    inverse=[]
    fsr_idx=fsr.set_index(['fight_id','fighter_id'])
    for _,mr in master.iterrows():
        fid=str(mr['fight_id']); fr=rs[rs[fcol].eq(fid)]; rr0,bb0=fighter_rows(fr,mr)
        if rr0.empty or bb0.empty: continue
        actual={'red':round_totals(rr0),'blue':round_totals(bb0)}; exposure=float(mr['match_time_sec'])
        try: r=fsr_idx.loc[(fid,str(mr['r_id']))]; b=fsr_idx.loc[(fid,str(mr['b_id']))]
        except KeyError: continue
        rn=inverse_side(r.to_dict(),b.to_dict(),actual['red'],exposure)['standing_striking_offense_needed']
        bn=inverse_side(b.to_dict(),r.to_dict(),actual['blue'],exposure)['standing_striking_offense_needed']
        inverse += [{'fight_id':fid,'fighter_id':str(mr['r_id']),'needed_offense':rn},{'fight_id':fid,'fighter_id':str(mr['b_id']),'needed_offense':bn}]
    trace=trace.merge(pd.DataFrame(inverse),on=['fight_id','fighter_id'],how='left')
    trace['needed_minus_published']=trace['needed_offense']-trace['standing_striking_offense']

    market=build_two_way_market(MARKET_PATH).copy(); market['fight_id']=market['fight_id'].astype(str)
    mk=market[['fight_id','favorite_id','market_favorite_fair_p']].drop_duplicates('fight_id')
    trace=trace.merge(mk,on='fight_id',how='left'); trace['is_market_favorite']=trace['fighter_id'].eq(trace['favorite_id'].astype(str))

    # Fight-level favorite-vs-dog gaps for priced bantam fights.
    rows=[]
    for fid,g in trace[trace['fight_id'].isin(master['fight_id'])].groupby('fight_id'):
        if len(g)!=2 or g['market_favorite_fair_p'].isna().all(): continue
        fav=g[g.is_market_favorite]; dog=g[~g.is_market_favorite]
        if len(fav)!=1 or len(dog)!=1: continue
        f=fav.iloc[0]; d=dog.iloc[0]
        rec={'fight_id':fid,'event_date':f.event_date,'favorite':f.fighter_name,'dog':d.fighter_name,'market_favorite_fair_p':f.market_favorite_fair_p}
        for c in ['raw_prior_accuracy','likelihood_only_mean','posterior_offense_mean','standing_striking_offense','needed_offense','prior_shrinkage']:
            rec[f'favorite_{c}']=f[c]; rec[f'dog_{c}']=d[c]; rec[f'{c}_gap']=f[c]-d[c] if pd.notna(f[c]) and pd.notna(d[c]) else np.nan
        rec['evidence_to_posterior_gap_loss']=rec['likelihood_only_mean_gap']-rec['posterior_offense_mean_gap']
        rec['posterior_to_needed_gap']=rec['needed_offense_gap']-rec['posterior_offense_mean_gap']
        rows.append(rec)
    fights_out=pd.DataFrame(rows).sort_values('market_favorite_fair_p',ascending=False)

    summary=[]
    for label,g in [('all_priced',fights_out),('fav70+',fights_out[fights_out.market_favorite_fair_p>=.70]),('fav80+',fights_out[fights_out.market_favorite_fair_p>=.80])]:
        summary.append({'group':label,'n':len(g),
                        'mean_likelihood_gap':g.likelihood_only_mean_gap.mean(),'mean_posterior_gap':g.posterior_offense_mean_gap.mean(),
                        'mean_needed_gap':g.needed_offense_gap.mean(),'mean_gap_loss_prior':g.evidence_to_posterior_gap_loss.mean(),
                        'corr_likelihood_needed':g[['likelihood_only_mean_gap','needed_offense_gap']].corr().iloc[0,1] if len(g)>2 else np.nan,
                        'corr_posterior_needed':g[['posterior_offense_mean_gap','needed_offense_gap']].corr().iloc[0,1] if len(g)>2 else np.nan})
    summary=pd.DataFrame(summary)

    OUT.mkdir(parents=True,exist_ok=True)
    trace.to_csv(OUT/'fighter_fight_trace.csv',index=False); fights_out.to_csv(OUT/'priced_fight_gap_trace.csv',index=False); summary.to_csv(OUT/'summary.csv',index=False)
    print('BANTAMWEIGHT STANDING OFFENSE EVIDENCE TRACE')
    print(f'trace_rows={len(trace)} priced_fights={len(fights_out)} prior_sigma={spec.sigma_offense} rho={spec.rho}')
    print('\nSUMMARY'); print(summary.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    show=fights_out[fights_out.market_favorite_fair_p>=.80].copy()
    cols=['favorite','dog','market_favorite_fair_p','raw_prior_accuracy_gap','likelihood_only_mean_gap','posterior_offense_mean_gap','needed_offense_gap','evidence_to_posterior_gap_loss','posterior_to_needed_gap']
    print('\n80%+ FAVORITES'); print(show[cols].to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nInterpretation: likelihood->posterior loss isolates Normal-prior shrinkage after opponent-conditioned likelihood attribution; disagreement already present in likelihood-only estimates implicates the historical evidence/attribution rather than the prior.')

if __name__=='__main__': main()
