from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/research/prop_mispricing'
PRED = OUT / 'hierarchical_v5_market_intelligence_predictions.csv'
FEATURES = ROOT / 'data/features/moneyline_feature_view.parquet'
LEDGER = OUT / 'hierarchical_v5_top_method_2026_dk_ledger.csv'
SUMMARY = OUT / 'hierarchical_v5_top_method_2026_dk_summary.json'

SIDE_METHODS = {
    'red': [('red_ko','hier_red_ko','market_red_ko',0),('red_sub','hier_red_sub','market_red_sub',1),('red_dec','hier_red_dec','market_red_dec',2)],
    'blue': [('blue_ko','hier_blue_ko','market_blue_ko',3),('blue_sub','hier_blue_sub','market_blue_sub',4),('blue_dec','hier_blue_dec','market_blue_dec',5)],
}

def method_class(side: str, method: str):
    m = str(method).lower()
    if 'ko' in m or 'tko' in m: suffix='ko'
    elif 'sub' in m: suffix='sub'
    elif 'decision' in m: suffix='dec'
    else: return None
    return {'red_ko':0,'red_sub':1,'red_dec':2,'blue_ko':3,'blue_sub':4,'blue_dec':5}[f'{side}_{suffix}']

def american_from_prob(p: float):
    if p <= 0 or p >= 1: return None
    return -100*p/(1-p) if p >= 0.5 else 100*(1-p)/p

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
    f = pd.read_parquet(FEATURES)
    f['fight_id']=f['fight_id'].astype(str)
    result_cols=['fight_id','r_name','b_name','winner','method']
    f=f[result_cols].drop_duplicates('fight_id')
    df=p.merge(f,on='fight_id',how='left')

    rows=[]
    for _,r in df.iterrows():
        side=str(r['predicted_side'])
        if side not in SIDE_METHODS: continue
        choices=SIDE_METHODS[side]
        best=max(choices,key=lambda x: float(r[x[1]]))
        slug, model_col, market_col, class_idx = best
        market_p=float(r[market_col])
        # These are normalized six-way market probabilities. Recover displayed odds only if present indirectly is impossible,
        # so use the corresponding raw market probability fields if available; otherwise normalized probability is used as a conservative proxy.
        raw_col = market_col.replace('market_','raw_market_')
        raw_p=float(r[raw_col]) if raw_col in df.columns and pd.notna(r[raw_col]) else market_p
        dec=1.0/raw_p
        winner=str(r['winner']) if pd.notna(r['winner']) else ''
        red=str(r['r_name']) if pd.notna(r['r_name']) else str(r['red_fighter'])
        blue=str(r['b_name']) if pd.notna(r['b_name']) else str(r['blue_fighter'])
        actual_side = 'red' if winner.strip().lower()==red.strip().lower() else ('blue' if winner.strip().lower()==blue.strip().lower() else None)
        actual_class = method_class(actual_side, r['method']) if actual_side else None
        won = int(actual_class == class_idx) if actual_class is not None else None
        profit = (dec-1.0) if won==1 else (-1.0 if won==0 else None)
        rows.append({
            'fight_id':r['fight_id'],'event_name':r['event_name'],'fight_display':r['fight_display'],
            'red_fighter':r['red_fighter'],'blue_fighter':r['blue_fighter'],'v5_model_p_red':float(r['v5_model_p_red']),
            'predicted_side':side,'bet_slug':slug,'model_probability':float(r[model_col]),
            'normalized_market_probability':market_p,'pricing_probability_used':raw_p,'decimal_odds_used':dec,
            'actual_winner':winner,'actual_method':r['method'],'target':actual_class,'won':won,'stake_units':1.0,'profit_units':profit,
        })
    led=pd.DataFrame(rows)
    led.to_csv(LEDGER,index=False)
    graded=led[led['won'].notna()].copy()
    by_event={str(ev):summarize(g) for ev,g in graded.groupby('event_name')}
    by_method={str(m):summarize(g) for m,g in graded.groupby('bet_slug')}
    summary={
        'experiment':'frozen_hierarchical_v5_top_method_on_v5_winner_2026_dk_v1',
        'rule':'For each scored fight, choose V5 projected winner side; bet exactly one method: highest hierarchical model probability among KO/SUB/DEC on that side.',
        'threshold':None,'max_bets_per_fight':1,'total_scored_fights':int(len(led)),'graded_bets':int(len(graded)),
        'pooled':summarize(graded),'by_event':by_event,'by_method':by_method,
        'pricing_note':'Prediction artifact currently stores normalized six-way market probabilities, not raw displayed probabilities for every outcome; ROI here uses inverse normalized probability unless raw_market_* columns are present.'
    }
    SUMMARY.write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))

if __name__=='__main__': main()
