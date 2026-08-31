from __future__ import annotations

from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/research/prop_mispricing'
BETS = OUT / 'hierarchical_v5_market_intelligence_bets.csv'
HIST = ROOT / 'data/market/historical_market_outcomes.parquet'
LEDGER_OUT = OUT / 'hierarchical_v5_market_intelligence_graded_bets.csv'
UNMATCHED_OUT = OUT / 'hierarchical_v5_market_intelligence_ungraded_bets.csv'
SUMMARY_OUT = OUT / 'hierarchical_v5_market_intelligence_grading_summary.json'

CLASS_META = {
    'red_ko': ('red','win_by_ko_tko_dq',0),
    'red_sub': ('red','win_by_submission',1),
    'red_dec': ('red','win_by_decision',2),
    'blue_ko': ('blue','win_by_ko_tko_dq',3),
    'blue_sub': ('blue','win_by_submission',4),
    'blue_dec': ('blue','win_by_decision',5),
}

def norm_name(x):
    if x is None or pd.isna(x): return ''
    return re.sub(r'[^a-z0-9]+','',str(x).lower())

def pair_key(a,b):
    x=sorted([norm_name(a),norm_name(b)])
    return '||'.join(x) if all(x) else ''

def summarize(d):
    if d.empty:
        return {'bets':0,'wins':0,'losses':0,'stake_units':0.0,'profit_units':0.0,'roi':None,'hit_rate':None,'fights_bet':0}
    stake=float(d['stake_units'].sum()); profit=float(d['profit_units'].sum())
    return {'bets':int(len(d)),'wins':int(d['won'].sum()),'losses':int((1-d['won']).sum()),'stake_units':stake,'profit_units':profit,'roi':float(profit/stake) if stake else None,'hit_rate':float(d['won'].mean()),'fights_bet':int(d['fight_id'].nunique())}

def main():
    bets=pd.read_csv(BETS)
    bets['fight_id']=bets['fight_id'].astype(str)
    bets['_pair']=bets.apply(lambda r: pair_key(r['red_fighter'],r['blue_fighter']),axis=1)

    h=pd.read_parquet(HIST).copy()
    h['fight_id']=h['fight_id'].astype(str)
    req=['win_by_ko_tko_dq','win_by_submission','win_by_decision']
    h=h[(h['result_status'].astype(str)=='graded') & h['market_key'].isin(req) & h['outcome_side'].astype(str).isin(['red','blue']) & h['won'].notna()].copy()
    h['won']=h['won'].astype(bool).astype(int)

    targets=[]
    for fid,g in h.groupby('fight_id'):
        wins=[]
        for slug,(side,key,idx) in CLASS_META.items():
            z=g[(g['outcome_side'].astype(str)==side)&(g['market_key']==key)]
            if len(z) and int(z.iloc[0]['won'])==1: wins.append(idx)
        if len(wins)!=1: continue
        rname=''; bname=''; ev=''; date=pd.NaT
        for c in ['red_fighter','r_name','fighter_red']:
            if c in g.columns and g[c].notna().any(): rname=str(g.loc[g[c].notna(),c].iloc[0]); break
        for c in ['blue_fighter','b_name','fighter_blue']:
            if c in g.columns and g[c].notna().any(): bname=str(g.loc[g[c].notna(),c].iloc[0]); break
        if not rname or not bname:
            for side in ['red','blue']:
                z=g[g['outcome_side'].astype(str)==side]
                for c in ['fighter_name','outcome_display']:
                    if c in z.columns and z[c].notna().any():
                        if side=='red': rname=str(z.loc[z[c].notna(),c].iloc[0])
                        else: bname=str(z.loc[z[c].notna(),c].iloc[0])
                        break
        if 'event_name' in g.columns and g['event_name'].notna().any(): ev=str(g.loc[g['event_name'].notna(),'event_name'].iloc[0])
        for c in ['date','event_date']:
            if c in g.columns and g[c].notna().any(): date=pd.to_datetime(g.loc[g[c].notna(),c].iloc[0],errors='coerce'); break
        if rname and bname:
            targets.append({'historical_fight_id':fid,'_pair':pair_key(rname,bname),'hist_red_fighter':rname,'hist_blue_fighter':bname,'hist_event_name':ev,'hist_date':date,'target':wins[0]})
    t=pd.DataFrame(targets)
    unique=t.groupby('_pair').filter(lambda x: len(x)==1).copy() if len(t) else t
    lookup=unique.set_index('_pair').to_dict(orient='index') if len(unique) else {}

    rows=[]; misses=[]
    clean_cols=[c for c in bets.columns if not c.startswith('_')]
    for _,r in bets.iterrows():
        tr=lookup.get(r['_pair'])
        if tr is None:
            misses.append({c:r[c] for c in clean_cols})
            continue
        slug=str(r['bet_slug']); class_idx=CLASS_META[slug][2]
        won=int(int(tr['target'])==class_idx)
        raw_p=float(r['raw_implied_probability'])
        dec=float(r['decimal_odds']) if pd.notna(r['decimal_odds']) else 1.0/raw_p
        profit=(dec-1.0) if won else -1.0
        row={c:r[c] for c in clean_cols}
        row.update({'historical_fight_id':tr['historical_fight_id'],'historical_event_name':tr['hist_event_name'],'target':int(tr['target']),'won':won,'stake_units':1.0,'profit_units':profit})
        rows.append(row)
    graded=pd.DataFrame(rows); ungraded=pd.DataFrame(misses)
    graded.to_csv(LEDGER_OUT,index=False); ungraded.to_csv(UNMATCHED_OUT,index=False)

    by_event={}
    if len(graded):
        for ev,g in graded.groupby('event_name',dropna=False): by_event[str(ev)]=summarize(g)
    by_method={}
    if len(graded):
        for m,g in graded.groupby('bet_slug'): by_method[str(m)]=summarize(g)

    summary={
        'experiment':'grade_frozen_hierarchical_v5_market_intelligence_bets_v1',
        'bet_source':str(BETS.relative_to(ROOT)),
        'result_source':str(HIST.relative_to(ROOT)),
        'total_bets':int(len(bets)),
        'graded_bets':int(len(graded)),
        'ungraded_bets':int(len(ungraded)),
        'unique_historical_pair_matches':int(unique['_pair'].nunique()) if len(unique) else 0,
        'pooled':summarize(graded),
        'by_event':by_event,
        'by_method':by_method,
        'grading_rule':'normalized unordered fighter pair; only unique historical graded pair matches accepted',
    }
    SUMMARY_OUT.write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))

if __name__=='__main__': main()
