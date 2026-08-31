from __future__ import annotations

from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/research/prop_mispricing'
PRED = OUT / 'hierarchical_v5_market_intelligence_predictions.csv'
FEATURES = ROOT / 'data/features/moneyline_feature_view.parquet'
MARKET = ROOT / 'data/market/market_intelligence_history.parquet'
LEDGER = OUT / 'hierarchical_v5_top_method_2026_dk_ledger.csv'
SUMMARY = OUT / 'hierarchical_v5_top_method_2026_dk_summary.json'

SIDE_METHODS = {
    'red': [('red_ko','hier_red_ko','market_red_ko','win_by_ko_tko_dq',0),('red_sub','hier_red_sub','market_red_sub','win_by_submission',1),('red_dec','hier_red_dec','market_red_dec','win_by_decision',2)],
    'blue': [('blue_ko','hier_blue_ko','market_blue_ko','win_by_ko_tko_dq',3),('blue_sub','hier_blue_sub','market_blue_sub','win_by_submission',4),('blue_dec','hier_blue_dec','market_blue_dec','win_by_decision',5)],
}

def norm_name(x):
    if x is None or pd.isna(x): return ''
    return re.sub(r'[^a-z0-9]+','',str(x).lower())

def method_class(side: str, method: str):
    m = str(method).lower()
    if 'ko' in m or 'tko' in m: suffix='ko'
    elif 'sub' in m: suffix='sub'
    elif 'decision' in m: suffix='dec'
    else: return None
    return {'red_ko':0,'red_sub':1,'red_dec':2,'blue_ko':3,'blue_sub':4,'blue_dec':5}[f'{side}_{suffix}']

def decimal_from_american(a: float):
    return 1 + (100/abs(a) if a < 0 else a/100)

def summarize(d: pd.DataFrame):
    if d.empty:
        return {'bets':0,'wins':0,'losses':0,'stake_units':0.0,'profit_units':0.0,'roi':None,'hit_rate':None,'fights_bet':0}
    stake=float(d['stake_units'].sum()); profit=float(d['profit_units'].sum())
    return {'bets':int(len(d)),'wins':int(d['won'].sum()),'losses':int((1-d['won']).sum()),'stake_units':stake,'profit_units':profit,'roi':profit/stake if stake else None,'hit_rate':float(d['won'].mean()),'fights_bet':int(d['fight_id'].nunique())}

def main():
    p = pd.read_csv(PRED)
    p['fight_id']=p['fight_id'].astype(str)
    p['refresh_timestamp']=pd.to_datetime(p['refresh_timestamp'],utc=True)

    f = pd.read_parquet(FEATURES)
    f['fight_id']=f['fight_id'].astype(str)
    f=f[['fight_id','r_name','b_name','winner','method']].drop_duplicates('fight_id')
    df=p.merge(f,on='fight_id',how='left')

    m=pd.read_parquet(MARKET).copy()
    # Source rows used in the scorer are DraftKings. Keep only exact method markets.
    if 'sportsbook' in m.columns: m=m[m['sportsbook'].astype(str).str.lower().eq('draftkings')]
    elif 'source' in m.columns: m=m[m['source'].astype(str).str.lower().eq('draftkings')]
    m=m[m['market_key'].isin(['win_by_ko_tko_dq','win_by_submission','win_by_decision'])].copy()
    m['refresh_timestamp']=pd.to_datetime(m['refresh_timestamp'],utc=True)
    m['_fighter_norm']=m['outcome_display'].map(norm_name)

    rows=[]; skipped=[]
    for _,r in df.iterrows():
        side=str(r['predicted_side'])
        if side not in SIDE_METHODS: continue
        slug, model_col, market_col, market_key, class_idx=max(SIDE_METHODS[side],key=lambda x: float(r[x[1]]))
        fighter = str(r['red_fighter'] if side=='red' else r['blue_fighter'])
        z=m[(m['fight_display'].astype(str)==str(r['fight_display'])) &
            (m['refresh_timestamp']==r['refresh_timestamp']) &
            (m['market_key']==market_key) &
            (m['_fighter_norm']==norm_name(fighter))]
        if len(z)!=1:
            skipped.append({'fight_id':r['fight_id'],'fight_display':r['fight_display'],'bet_slug':slug,'reason':f'raw_price_rows={len(z)}'})
            continue
        q=z.iloc[0]
        raw_p=float(q['implied_probability'])
        american=float(q['american_odds'])
        dec=decimal_from_american(american)

        winner=str(r['winner']) if pd.notna(r['winner']) else ''
        red=str(r['r_name']) if pd.notna(r['r_name']) else str(r['red_fighter'])
        blue=str(r['b_name']) if pd.notna(r['b_name']) else str(r['blue_fighter'])
        actual_side='red' if norm_name(winner)==norm_name(red) else ('blue' if norm_name(winner)==norm_name(blue) else None)
        actual_class=method_class(actual_side,r['method']) if actual_side else None
        won=int(actual_class==class_idx) if actual_class is not None else None
        profit=(dec-1.0) if won==1 else (-1.0 if won==0 else None)
        rows.append({
            'fight_id':r['fight_id'],'event_name':r['event_name'],'fight_display':r['fight_display'],'refresh_timestamp':r['refresh_timestamp'],
            'red_fighter':r['red_fighter'],'blue_fighter':r['blue_fighter'],'v5_model_p_red':float(r['v5_model_p_red']),
            'predicted_side':side,'bet_fighter':fighter,'bet_slug':slug,'model_probability':float(r[model_col]),
            'normalized_market_probability':float(r[market_col]),'raw_implied_probability':raw_p,'american_odds':american,'decimal_odds':dec,
            'actual_winner':winner,'actual_method':r['method'],'target':actual_class,'won':won,'stake_units':1.0,'profit_units':profit,
        })

    led=pd.DataFrame(rows); led.to_csv(LEDGER,index=False)
    graded=led[led['won'].notna()].copy()
    by_event={str(ev):summarize(g) for ev,g in graded.groupby('event_name')}
    by_method={str(k):summarize(g) for k,g in graded.groupby('bet_slug')}
    summary={
        'experiment':'frozen_hierarchical_v5_top_method_on_v5_winner_2026_dk_v1',
        'rule':'For every scored fight, choose V5 projected winner side and bet exactly one outcome: highest hierarchical KO/SUB/DEC probability on that side. No edge threshold.',
        'max_bets_per_fight':1,'threshold':None,'prediction_rows':int(len(df)),'priced_bets':int(len(led)),'price_match_skips':int(len(skipped)),'graded_bets':int(len(graded)),
        'pooled':summarize(graded),'by_event':by_event,'by_method':by_method,'skipped':skipped,
        'pricing_note':'ROI uses actual DraftKings American odds from the exact market_intelligence_history snapshot used for each prediction.'
    }
    SUMMARY.write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))

if __name__=='__main__': main()
