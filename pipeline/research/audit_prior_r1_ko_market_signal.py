from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.research.analyze_ko_market_residual_features import _side_rows

MASTER = Path('data/master/ufc_master.parquet')
FEATURE = Path('data/features/moneyline_feature_view.parquet')
FREEZE = Path('data/research/prop_mispricing/ko_market_archetype_2026_freeze.json')
OUT = Path('data/research/prop_mispricing')


def is_ko(x):
    s = str(x).upper()
    return ('KO' in s) or ('TKO' in s)


def build_prefight_r1():
    m = pd.read_parquet(MASTER).copy()
    m['date'] = pd.to_datetime(m['date'], errors='coerce')
    m = m.dropna(subset=['date','fight_id','r_id','b_id']).sort_values(['date','fight_id'])
    stats = defaultdict(lambda: {'fights':0,'r1_ko_wins':0,'r1_ko_losses':0,'ko_wins':0,'recent5':deque(maxlen=5)})
    rows=[]
    for r in m.itertuples(index=False):
        fid=str(r.fight_id); rid=str(r.r_id); bid=str(r.b_id)
        for side, fighter, opp in [('red',rid,bid),('blue',bid,rid)]:
            s=stats[fighter]; o=stats[opp]
            rows.append({
                'fight_id':fid,'side':side,'date':r.date,
                'prior_ufc_fights':s['fights'],
                'prior_r1_ko_wins':s['r1_ko_wins'],
                'prior_r1_ko_win_rate':s['r1_ko_wins']/s['fights'] if s['fights'] else 0.0,
                'prior_ko_wins':s['ko_wins'],
                'prior_r1_share_of_ko_wins':s['r1_ko_wins']/s['ko_wins'] if s['ko_wins'] else 0.0,
                'recent5_r1_ko_wins':sum(s['recent5']),
                'opp_prior_r1_ko_losses':o['r1_ko_losses'],
                'opp_prior_r1_ko_loss_rate':o['r1_ko_losses']/o['fights'] if o['fights'] else 0.0,
            })
        winner=str(getattr(r,'winner_id',None))
        ko=is_ko(getattr(r,'method',None))
        r1=ko and pd.to_numeric(getattr(r,'finish_round',np.nan),errors='coerce')==1
        for fighter in [rid,bid]:
            stats[fighter]['fights'] += 1
            stats[fighter]['recent5'].append(1 if (r1 and fighter==winner) else 0)
        if ko and winner in (rid,bid):
            loser=bid if winner==rid else rid
            stats[winner]['ko_wins'] += 1
            if r1:
                stats[winner]['r1_ko_wins'] += 1
                stats[loser]['r1_ko_losses'] += 1
    return pd.DataFrame(rows)


def summarize(x, label, period):
    if x.empty:
        return {'period':period,'group':label,'n':0}
    return {
        'period':period,'group':label,'n':len(x),
        'actual_exact_ko_rate':x.actual_ko_win.mean(),
        'market_exact_ko_p':x.market_exact_ko_p.mean(),
        'market_residual':x.market_residual.mean(),
        'roi_diag':x.profit_units.mean(),
    }


def hist_audit(rows):
    out=[]
    for period, years in [('dev_2021_2024',range(2021,2025)),('validation_2025',[2025])]:
        z=rows[rows.year.isin(years) & rows.betting_eligible].copy()
        masks={
            'no_prior_r1_ko_win': z.prior_r1_ko_wins.eq(0),
            'any_prior_r1_ko_win': z.prior_r1_ko_wins.ge(1),
            'prior_r1_ko_wins_1': z.prior_r1_ko_wins.eq(1),
            'prior_r1_ko_wins_2plus': z.prior_r1_ko_wins.ge(2),
            'recent5_any_r1_ko_win': z.recent5_r1_ko_wins.ge(1),
            'opp_no_prior_r1_ko_loss': z.opp_prior_r1_ko_losses.eq(0),
            'opp_any_prior_r1_ko_loss': z.opp_prior_r1_ko_losses.ge(1),
            'fighter_r1_and_opp_r1_vulnerable': z.prior_r1_ko_wins.ge(1) & z.opp_prior_r1_ko_losses.ge(1),
            'fighter_r1_but_opp_no_r1_loss': z.prior_r1_ko_wins.ge(1) & z.opp_prior_r1_ko_losses.eq(0),
        }
        for label,mask in masks.items(): out.append(summarize(z[mask],label,period))
        if period=='dev_2021_2024':
            for y in range(2021,2025):
                zy=z[z.year.eq(y)]
                for label in ['any_prior_r1_ko_win','fighter_r1_and_opp_r1_vulnerable']:
                    mask = zy.prior_r1_ko_wins.ge(1) if label=='any_prior_r1_ko_win' else (zy.prior_r1_ko_wins.ge(1)&zy.opp_prior_r1_ko_losses.ge(1))
                    out.append(summarize(zy[mask],label,f'year_{y}'))
    return pd.DataFrame(out)


def archetype_2026(r1):
    freeze=json.loads(FREEZE.read_text())['rule']
    f=pd.read_parquet(FEATURE).copy(); f['date']=pd.to_datetime(f.date,errors='coerce'); f['fight_id']=f.fight_id.astype(str)
    f=f[f.date.between('2026-06-01','2026-08-22')].drop_duplicates('fight_id')
    parts=[]
    for side,sign in [('red',1),('blue',-1)]:
        x=f[['fight_id','date','r_pre_fights','b_pre_fights','height_diff','ewm_str_acc_diff','aggression_index_diff','recent_form_win_streak_diff']].copy(); x['side']=side
        for c in ['height_diff','ewm_str_acc_diff','aggression_index_diff','recent_form_win_streak_diff']: x[c]=pd.to_numeric(x[c],errors='coerce')*sign
        x['eligible']=pd.concat([pd.to_numeric(x.r_pre_fights,errors='coerce'),pd.to_numeric(x.b_pre_fights,errors='coerce')],axis=1).min(axis=1).ge(2)
        parts.append(x)
    x=pd.concat(parts,ignore_index=True); x=x[x.eligible]
    x['p_height']=x.height_diff.ge(float(freeze['height_diff_min_cm'])-1e-9)
    x['p_acc']=x.ewm_str_acc_diff.ge(float(freeze['ewm_str_acc_diff_min']))
    x['p_aggr']=x.aggression_index_diff.ge(float(freeze['aggression_index_diff_min']))
    x['p_streak']=x.recent_form_win_streak_diff.gt(float(freeze['recent_form_win_streak_diff_min_exclusive']))
    x['conditions_passed']=x[['p_height','p_acc','p_aggr','p_streak']].sum(axis=1)
    x=x[x.conditions_passed.ge(3)].merge(r1,on=['fight_id','side'],how='left')
    # attach names/results from master only after selection
    m=pd.read_parquet(MASTER).copy(); m['fight_id']=m.fight_id.astype(str)
    m=m[['fight_id','r_name','b_name','r_id','b_id','winner_id','method','finish_round']].drop_duplicates('fight_id')
    x=x.merge(m,on='fight_id',how='left')
    x['fighter']=np.where(x.side.eq('red'),x.r_name,x.b_name); x['opponent']=np.where(x.side.eq('red'),x.b_name,x.r_name)
    x['fighter_id']=np.where(x.side.eq('red'),x.r_id.astype(str),x.b_id.astype(str)); x['winner_id']=x.winner_id.astype(str)
    x['actual_result']=np.where(x.fighter_id.eq(x.winner_id),np.where(x.method.map(is_ko),'WIN_KO',np.where(x.method.astype(str).str.upper().str.contains('SUB'),'WIN_SUB','WIN_DEC')),'LOSS')
    return x.sort_values(['date','fight_id','side'])


def main():
    r1=build_prefight_r1()
    rows,_=_side_rows(); rows=rows.merge(r1.drop(columns='date'),on=['fight_id','side'],how='left',validate='one_to_one')
    hist=hist_audit(rows)
    cohort=archetype_2026(r1)
    hist.to_csv(OUT/'prior_r1_ko_market_signal_summary.csv',index=False)
    cohort.to_csv(OUT/'prior_r1_ko_june_aug_2026_archetype.csv',index=False)
    rows[['fight_id','date','year','fighter','side','actual_ko_win','market_exact_ko_p','market_residual','prior_r1_ko_wins','prior_r1_ko_win_rate','recent5_r1_ko_wins','opp_prior_r1_ko_losses','opp_prior_r1_ko_loss_rate']].to_csv(OUT/'prior_r1_ko_market_signal_rows.csv',index=False)
    print(hist.to_string(index=False))
    print('\n2026 exact/near cohort:')
    print(cohort[['date','fighter','opponent','conditions_passed','actual_result','prior_r1_ko_wins','prior_r1_ko_win_rate','recent5_r1_ko_wins','opp_prior_r1_ko_losses','opp_prior_r1_ko_loss_rate']].to_string(index=False))

if __name__=='__main__': main()
