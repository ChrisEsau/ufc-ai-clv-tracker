from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/research/prop_mispricing'
PRED=OUT/'xgboost_method_hierarchical_v5_oof_predictions.csv'
MARKET=ROOT/'data/market/historical_market_outcomes.parquet'
CSV=OUT/'xgboost_method_hierarchical_v5_bet_structure.csv'
JSON=OUT/'xgboost_method_hierarchical_v5_bet_structure.json'
EPS=1e-12
SLUGS=['red_ko','red_sub','red_dec','blue_ko','blue_sub','blue_dec']
META={'red_ko':('red','win_by_ko_tko_dq',0,'KO'),'red_sub':('red','win_by_submission',1,'SUB'),'red_dec':('red','win_by_decision',2,'DEC'),'blue_ko':('blue','win_by_ko_tko_dq',3,'KO'),'blue_sub':('blue','win_by_submission',4,'SUB'),'blue_dec':('blue','win_by_decision',5,'DEC')}

def logit(p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS); return np.log(p/(1-p))
def summ(d):
    if len(d)==0:return {'bets':0,'wins':0,'profit':0.0,'roi':None,'hit_rate':None,'avg_residual':None,'avg_model_p':None,'avg_odds':None}
    return {'bets':int(len(d)),'wins':int(d.won.sum()),'profit':float(d.profit.sum()),'roi':float(d.profit.mean()),'hit_rate':float(d.won.mean()),'avg_residual':float(d.residual.mean()),'avg_model_p':float(d.model_p.mean()),'avg_odds':float(d.american_odds.mean())}

p=pd.read_csv(PRED); p['fight_id']=p.fight_id.astype(str); p['date']=pd.to_datetime(p.date)
m=pd.read_parquet(MARKET); m['fight_id']=m.fight_id.astype(str)
m=m[(m.bookmaker=='legacy_consensus') & m.outcome_side.astype(str).isin(['red','blue'])].copy()
m['implied_probability']=pd.to_numeric(m.implied_probability,errors='coerce')
price={}
for slug,(side,key,_,_) in META.items():
    z=m[(m.outcome_side.astype(str)==side)&(m.market_key==key)].sort_values('fight_id').drop_duplicates('fight_id',keep=False)
    for r in z.itertuples(index=False): price[(str(r.fight_id),slug)]=float(r.implied_probability)
rows=[]
for r in p.itertuples(index=False):
    ps='red' if float(r.v5_model_p_red)>=.5 else 'blue'
    for slug in SLUGS:
        side,_,idx,meth=META[slug]
        if side!=ps:continue
        mp=float(getattr(r,f'market_{slug}')); hp=float(getattr(r,f'hier_{slug}'))
        res=float(logit(hp)-logit(mp)); raw=price.get((str(r.fight_id),slug))
        if raw is None or not (0<raw<1):continue
        dec=1/raw; won=int(int(r.target)==idx); prof=(dec-1) if won else -1
        rows.append({'fight_id':str(r.fight_id),'year':int(pd.Timestamp(r.date).year),'method':meth,'slug':slug,'model_p':hp,'market_p':mp,'residual':res,'raw_implied':raw,'decimal_odds':dec,'american_odds':(100*(dec-1) if dec>=2 else -100/(dec-1)),'won':won,'profit':prof})
d=pd.DataFrame(rows)
d['residual_band']=pd.cut(d.residual,[-np.inf,.2,.3,.4,.5,.75,np.inf],right=False,labels=['<.20','.20-.299','.30-.399','.40-.499','.50-.749','>=.75'])
# one max-residual method per fight is a pre-specified simplification diagnostic.
max1=d.sort_values(['fight_id','residual'],ascending=[True,False]).drop_duplicates('fight_id')
report={'all_winner_side_candidates':summ(d),'threshold_030':summ(d[d.residual>=.3]),'threshold_030_by_method':{},'threshold_030_by_year':{},'method_by_year':{},'residual_bands':{},'max_one_per_fight_030':summ(max1[max1.residual>=.3]),'max_one_per_fight_030_by_year':{}}
for meth,g in d[d.residual>=.3].groupby('method'): report['threshold_030_by_method'][meth]=summ(g)
for yr,g in d[d.residual>=.3].groupby('year'): report['threshold_030_by_year'][str(yr)]=summ(g)
for (meth,yr),g in d[d.residual>=.3].groupby(['method','year']): report['method_by_year'][f'{meth}_{yr}']=summ(g)
for band,g in d.groupby('residual_band',observed=True): report['residual_bands'][str(band)]=summ(g)
for yr,g in max1[max1.residual>=.3].groupby('year'): report['max_one_per_fight_030_by_year'][str(yr)]=summ(g)
d.sort_values(['year','fight_id','residual'],ascending=[True,True,False]).to_csv(CSV,index=False)
JSON.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
print('CSV=',CSV);print('JSON=',JSON)
