from __future__ import annotations

from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/research/prop_mispricing'
BETS = OUT / 'hierarchical_v5_market_intelligence_bets.csv'
FEATURES = ROOT / 'data/features/moneyline_feature_view.parquet'
LEDGER_OUT = OUT / 'hierarchical_v5_market_intelligence_graded_bets.csv'
UNMATCHED_OUT = OUT / 'hierarchical_v5_market_intelligence_ungraded_bets.csv'
SUMMARY_OUT = OUT / 'hierarchical_v5_market_intelligence_grading_summary.json'

CLASS_IDX = {'red_ko':0,'red_sub':1,'red_dec':2,'blue_ko':3,'blue_sub':4,'blue_dec':5}

def norm_name(x):
    if x is None or pd.isna(x): return ''
    return re.sub(r'[^a-z0-9]+','',str(x).lower())

def method_bucket(x):
    s=str(x or '').lower()
    if 'ko' in s or 'tko' in s: return 'ko'
    if 'sub' in s: return 'sub'
    if 'dec' in s: return 'dec'
    return None

def summarize(d):
    if d.empty:
        return {'bets':0,'wins':0,'losses':0,'stake_units':0.0,'profit_units':0.0,'roi':None,'hit_rate':None,'fights_bet':0}
    stake=float(d['stake_units'].sum()); profit=float(d['profit_units'].sum())
    return {'bets':int(len(d)),'wins':int(d['won'].sum()),'losses':int((1-d['won']).sum()),'stake_units':stake,'profit_units':profit,'roi':float(profit/stake) if stake else None,'hit_rate':float(d['won'].mean()),'fights_bet':int(d['fight_id'].nunique())}

def derive_target(r):
    red=norm_name(r.get('r_name')); blue=norm_name(r.get('b_name'))
    winner=norm_name(r.get('winner'))
    wid=str(r.get('winner_id') or '')
    rid=str(r.get('r_id') or '')
    bid=str(r.get('b_id') or '')
    side=None
    if winner and red and winner==red: side='red'
    elif winner and blue and winner==blue: side='blue'
    elif wid and rid and wid==rid: side='red'
    elif wid and bid and wid==bid: side='blue'
    meth=method_bucket(r.get('method'))
    if side is None or meth is None: return None
    return {'red_ko':0,'red_sub':1,'red_dec':2,'blue_ko':3,'blue_sub':4,'blue_dec':5}[f'{side}_{meth}']

def main():
    bets=pd.read_csv(BETS)
    bets['fight_id']=bets['fight_id'].astype(str)
    fv=pd.read_parquet(FEATURES).copy()
    fv['fight_id']=fv['fight_id'].astype(str)
    # One canonical row per fight id. If duplicate perspectives exist, prefer row_perspective='canonical' then first.
    if 'row_perspective' in fv.columns:
        fv['_rank']=(fv['row_perspective'].astype(str).str.lower()!='canonical').astype(int)
        fv=fv.sort_values(['fight_id','_rank']).drop_duplicates('fight_id',keep='first')
    else:
        fv=fv.drop_duplicates('fight_id',keep='first')
    fmap=fv.set_index('fight_id').to_dict(orient='index')

    rows=[]; misses=[]
    clean_cols=list(bets.columns)
    reasons={}
    for _,b in bets.iterrows():
        fr=fmap.get(str(b['fight_id']))
        if fr is None:
            reason='no_feature_result_row'
            misses.append({**{c:b[c] for c in clean_cols},'ungraded_reason':reason}); reasons[reason]=reasons.get(reason,0)+1; continue
        target=derive_target(fr)
        if target is None:
            reason='result_not_yet_available_or_unrecognized_method'
            misses.append({**{c:b[c] for c in clean_cols},'ungraded_reason':reason}); reasons[reason]=reasons.get(reason,0)+1; continue
        slug=str(b['bet_slug'])
        won=int(CLASS_IDX[slug]==target)
        dec=float(b['decimal_odds']) if pd.notna(b['decimal_odds']) else 1.0/float(b['raw_implied_probability'])
        profit=(dec-1.0) if won else -1.0
        row={c:b[c] for c in clean_cols}
        row.update({'target':int(target),'won':won,'stake_units':1.0,'profit_units':profit,'actual_method':fr.get('method'),'actual_winner':fr.get('winner')})
        rows.append(row)

    graded=pd.DataFrame(rows); ungraded=pd.DataFrame(misses)
    graded.to_csv(LEDGER_OUT,index=False); ungraded.to_csv(UNMATCHED_OUT,index=False)
    by_event={}; by_method={}
    if len(graded):
        for ev,g in graded.groupby('event_name',dropna=False): by_event[str(ev)]=summarize(g)
        for m,g in graded.groupby('bet_slug'): by_method[str(m)]=summarize(g)
    summary={
        'experiment':'grade_frozen_hierarchical_v5_market_intelligence_bets_v2',
        'bet_source':str(BETS.relative_to(ROOT)),
        'result_source':str(FEATURES.relative_to(ROOT)),
        'total_bets':int(len(bets)),
        'graded_bets':int(len(graded)),
        'ungraded_bets':int(len(ungraded)),
        'ungraded_reasons':reasons,
        'pooled':summarize(graded),
        'by_event':by_event,
        'by_method':by_method,
        'grading_rule':'canonical fight_id joined to moneyline_feature_view; realized winner + method mapped to six-way target',
    }
    SUMMARY_OUT.write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))

if __name__=='__main__': main()
