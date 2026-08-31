from __future__ import annotations
from pathlib import Path
import json,re
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/research/prop_mispricing'
PRED=OUT/'hierarchical_v5_market_intelligence_predictions.csv'
FEATURES=ROOT/'data/features/moneyline_feature_view.parquet'
MARKET=ROOT/'data/market/market_intelligence_history.parquet'
LEDGER=OUT/'hierarchical_v5_ml_dog_top_method_2026_dk_ledger.csv'
SUMMARY=OUT/'hierarchical_v5_ml_dog_top_method_2026_dk_summary.json'

DOGS=['Damian Rzepecki','Santiago Ponzinibbio','Brendson Ribeiro','Saygid Izagakhmaev','Tyrell Fortune','Alden Coria','Anna Melisano','Tabatha Ricci','Jean-Paul Lebosnoyani']
SIDE_METHODS={
'red':[('red_ko','hier_red_ko','win_by_ko_tko_dq',0),('red_sub','hier_red_sub','win_by_submission',1),('red_dec','hier_red_dec','win_by_decision',2)],
'blue':[('blue_ko','hier_blue_ko','win_by_ko_tko_dq',3),('blue_sub','hier_blue_sub','win_by_submission',4),('blue_dec','hier_blue_dec','win_by_decision',5)]}

def norm(x): return re.sub(r'[^a-z0-9]+','',str(x).lower()) if pd.notna(x) else ''
def dec_odds(a): return 1+(100/abs(a) if a<0 else a/100)
def method_class(side,m):
    s=str(m).lower()
    suf='ko' if ('ko' in s or 'tko' in s) else ('sub' if 'sub' in s else ('dec' if 'decision' in s else None))
    if not suf:return None
    return {'red_ko':0,'red_sub':1,'red_dec':2,'blue_ko':3,'blue_sub':4,'blue_dec':5}[f'{side}_{suf}']

def main():
    p=pd.read_csv(PRED); p['refresh_timestamp']=pd.to_datetime(p['refresh_timestamp'],utc=True)
    f=pd.read_parquet(FEATURES)[['fight_id','r_name','b_name','winner','method']].drop_duplicates('fight_id')
    f['fight_id']=f['fight_id'].astype(str); p['fight_id']=p['fight_id'].astype(str)
    df=p.merge(f,on='fight_id',how='left')
    m=pd.read_parquet(MARKET).copy()
    if 'sportsbook' in m.columns:m=m[m['sportsbook'].astype(str).str.lower().eq('draftkings')]
    elif 'source' in m.columns:m=m[m['source'].astype(str).str.lower().eq('draftkings')]
    m['refresh_timestamp']=pd.to_datetime(m['refresh_timestamp'],utc=True)
    m['_fighter']=m['outcome_display'].map(norm)
    dogset={norm(x):x for x in DOGS}
    rows=[]; skips=[]
    for _,r in df.iterrows():
        redn,bluen=norm(r['red_fighter']),norm(r['blue_fighter'])
        dog_side='red' if redn in dogset else ('blue' if bluen in dogset else None)
        if dog_side is None: continue
        dog=str(r['red_fighter'] if dog_side=='red' else r['blue_fighter'])
        slug,model_col,key,class_idx=max(SIDE_METHODS[dog_side],key=lambda x:float(r[x[1]]))
        z=m[(m['fight_display'].astype(str)==str(r['fight_display']))&(m['refresh_timestamp']==r['refresh_timestamp'])&(m['market_key']==key)&(m['_fighter']==norm(dog))]
        if len(z)!=1:
            skips.append({'fight_display':r['fight_display'],'dog':dog,'bet_slug':slug,'reason':f'raw_price_rows={len(z)}'});continue
        q=z.iloc[0]; a=float(q['american_odds']); d=dec_odds(a)
        winner=str(r['winner']) if pd.notna(r['winner']) else ''
        actual_side='red' if norm(winner)==norm(r['r_name']) else ('blue' if norm(winner)==norm(r['b_name']) else None)
        actual_class=method_class(actual_side,r['method']) if actual_side else None
        won=int(actual_class==class_idx) if actual_class is not None else None
        profit=(d-1) if won==1 else (-1 if won==0 else None)
        rows.append({'fight_id':r['fight_id'],'event_name':r['event_name'],'fight_display':r['fight_display'],'dog_fighter':dog,'dog_side':dog_side,'v5_model_p_red':r['v5_model_p_red'],'bet_slug':slug,'model_probability':float(r[model_col]),'american_odds':a,'decimal_odds':d,'actual_winner':winner,'actual_method':r['method'],'won':won,'stake_units':1.0,'profit_units':profit})
    led=pd.DataFrame(rows);led.to_csv(LEDGER,index=False)
    g=led[led['won'].notna()].copy(); stake=float(g['stake_units'].sum()); profit=float(g['profit_units'].sum())
    summary={'experiment':'2026_ml_bet_dog_then_top_method','requested_dogs':DOGS,'priced_bets':len(led),'skips':skips,'bets':len(g),'wins':int(g['won'].sum()),'losses':int((1-g['won']).sum()),'hit_rate':float(g['won'].mean()) if len(g) else None,'profit_units':profit,'roi':profit/stake if stake else None}
    SUMMARY.write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2)); print(led.to_string(index=False))
if __name__=='__main__':main()
