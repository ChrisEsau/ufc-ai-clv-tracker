from __future__ import annotations

import subprocess, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from pipeline.features.views.moneyline import build_moneyline_feature_view

OUT=Path('data/research/prop_mispricing')
OUTPUT=OUT/'ufc330_v5_moneyline_20260815.csv'
BET_OUTPUT=OUT/'ufc330_v5_betting_logit030_no_cold_exclusion_20260815.csv'
SNAP='7df1b61126be1f4e036b256d1c774c531b8a281f'
FIGHT_DATE=pd.Timestamp('2026-08-15')
# fighter A, fighter B, A odds, B odds, rounds, title_fight
FIGHTS=[
 ('Islam Makhachev','Ian Machado Garry',-370,270,5,True),
 ('Mackenzie Dern','Gillian Robertson',-218,180,5,True),
 ('Jalin Turner','Kauê Fernandes',-120,100,3,False),
 ('Mansur Abdul-Malik','Dustin Stoltzfus',-700,500,3,False),
 ('Edson Barboza','Esteban Ribovics',525,-750,3,False),
 ('Chidi Njokuani','Joel Álvarez',270,-370,3,False),
 ('Charles Johnson','Eduardo Chapolin',-115,-105,3,False),
 ('Donte Johnson','Eric McConico',-325,260,3,False),
 ('Vicente Luque','Tresean Gore',-105,-115,3,False),
 ('Rafael Tobias','Lucas Fernando',230,-280,3,False),
 ('Neil Magny','Ramiz Brahimaj',120,-145,3,False),
 ('Jeremiah Wells','Myktybek Orolbai',675,-1300,3,False),
]
SELECTED=[
'reach_diff','recent_form_recent_avg_fight_time_diff','age_diff','ewm_sapm_diff','ewm_recent_sapm_diff','style_ko_finisher_score_diff','ewm_td_acc_diff','recent_finish_rate_diff','chin_risk_diff','recent_form_avg_opponent_elo_diff','recent_avg_fight_time_diff','aggression_index_diff','age_squared_diff','sapm_diff','ewm_kd_avg_diff','style_all_round_finisher_score_diff','recent_form_kd_absorbed_avg_diff','ewm_recent_splm_diff','elo_diff','ewm_elo_diff','ewm_recent_td_avg_diff','days_since_last_fight_diff','td_avg_diff','style_score_spread_diff','ko_dependency_diff','recent_form_avg_fight_time_diff','wrestling_mismatch_diff','win_pct_diff','recent_form_ko_rate_diff','recent_form_worst_loss_elo_diff','age_x_career_ko_losses_diff','ewm_str_def_diff','losses_diff','ewm_recent_win_pct_diff','avg_opponent_elo_diff','ewm_td_avg_diff','avg_fight_time_diff','ewm_days_since_last_fight_diff','pressure_striking_adv_diff','weight_diff','ctrl_against_per_min_diff','ewm_finish_loss_rate_diff','ewm_win_pct_diff','victory_concentration_index_diff','recent_form_td_acc_diff','sub_avg_diff','recent_form_best_win_elo_diff','ewm_best_win_elo_diff','style_primary_score_diff','recent_form_recent_finish_rate_diff']

def norm(s): return ''.join(ch for ch in unicodedata.normalize('NFKD',str(s)) if not unicodedata.combining(ch)).replace('’',"'").lower().strip()
def imp(o): return 100/(o+100) if o>0 else (-o)/((-o)+100)
def clip_p(p): return np.clip(np.asarray(p,float),1e-6,1-1e-6)
def logit(p):
 p=clip_p(p); return np.log(p/(1-p))
def sigmoid(z):
 z=np.clip(np.asarray(z,float),-30,30); return 1/(1+np.exp(-z))

def add_engineered(v):
 def n(c): return pd.to_numeric(v[c],errors='coerce') if c in v.columns else pd.Series(np.nan,index=v.index)
 def rate(c):
  s=n(c); return pd.Series(np.where(s>1,s/100,s),index=v.index)
 v['chin_risk_diff']=n('r_pre_sapm')*(1-rate('r_pre_str_def'))-n('b_pre_sapm')*(1-rate('b_pre_str_def'))
 v['aggression_index_diff']=(n('r_pre_splm')+n('r_pre_td_avg'))-(n('b_pre_splm')+n('b_pre_td_avg'))
 v['age_squared_diff']=n('r_pre_age')**2-n('b_pre_age')**2
 v['wrestling_mismatch_diff']=n('r_pre_td_avg')*(1-rate('b_pre_td_def'))-n('b_pre_td_avg')*(1-rate('r_pre_td_def'))
 v['pressure_striking_adv_diff']=n('r_pre_splm')*(1-rate('b_pre_str_def'))-n('b_pre_splm')*(1-rate('r_pre_str_def'))
 v['age_x_career_ko_losses_diff']=n('r_pre_age')*n('r_pre_career_ko_losses')-n('b_pre_age')*n('b_pre_career_ko_losses')
 return v

def fit_model():
 OUT.mkdir(parents=True,exist_ok=True)
 for rp,op in [('data/market/historical_market_outcomes.parquet','/tmp/v5_market.parquet'),('data/features/moneyline_feature_view.parquet','/tmp/v5_fv.parquet')]:
  with open(op,'wb') as f: subprocess.run(['git','show',f'{SNAP}:{rp}'],stdout=f,check=True)
 m=pd.read_parquet('/tmp/v5_market.parquet').copy()
 m=m[(m.bookmaker=='legacy_consensus')&(m.result_status=='graded')&m.won.notna()].copy(); m['date']=pd.to_datetime(m.date,errors='coerce'); m['won']=m.won.astype(bool).astype(int); m['implied_probability']=pd.to_numeric(m.implied_probability,errors='coerce'); m=m.dropna(subset=['date','implied_probability'])
 ml=m[m.market_key=='moneyline'].copy(); good=ml.groupby('fight_id').size(); ml=ml[ml.fight_id.isin(good[good==2].index)].copy(); ml['market_overround']=ml.groupby('fight_id').implied_probability.transform('sum'); ml['fair_market_p']=ml.implied_probability/ml.market_overround; red=ml[ml.outcome_side.astype(str).eq('red')].copy()
 fv=pd.read_parquet('/tmp/v5_fv.parquet'); feats=SELECTED+['market_overround']; d=red.merge(fv[['fight_id']+[c for c in SELECTED if c in fv.columns]],on='fight_id',how='inner').sort_values(['date','fight_id'])
 miss=[c for c in SELECTED if c not in d.columns]
 if miss: raise RuntimeError(f'missing frozen train features: {miss}')
 xr=d[feats].replace([np.inf,-np.inf],np.nan); tr=d.date<='2024-12-31'; valid=[c for c in feats if xr.loc[tr,c].notna().any()]; med=xr.loc[tr,valid].median(numeric_only=True); x=xr.loc[tr,valid].fillna(med).fillna(0); y=d.loc[tr,'won'].astype(int).to_numpy(); bm=logit(d.loc[tr,'fair_market_p'])
 params={'max_depth':1,'eta':.03,'subsample':.8,'colsample_bytree':.7,'min_child_weight':10,'lambda':8.,'alpha':1.,'objective':'binary:logistic','eval_metric':'logloss','seed':42,'nthread':2}
 model=xgb.train(params,xgb.DMatrix(x,label=y,base_margin=bm,feature_names=valid),num_boost_round=300,verbose_eval=False)
 return model,valid,med

def build_view():
 h=pd.read_parquet('data/features/fighter_state_history.parquet').copy(); h['date']=pd.to_datetime(h.date,errors='coerce'); h['fighter_id']=h.fighter_id.astype(str); h['_norm_name']=h.fighter_name.map(norm)
 prep=[]; states=[]; meta={}
 for i,(a,b,oa,ob,rounds,title) in enumerate(FIGHTS,1):
  fid=f'ufc330_v5_20260815_{i:02d}'; ids=[]; found=[]; dates=[]
  for nm in (a,b):
   hit=h[(h._norm_name.eq(norm(nm)))&(h.date<FIGHT_DATE)].sort_values('date')
   if len(hit): rec=hit.iloc[-1].drop(labels=['_norm_name']).to_dict(); fighter_id=str(rec['fighter_id']); ok=True; sd=pd.Timestamp(rec['date']).strftime('%Y-%m-%d')
   else: rec={c:np.nan for c in h.columns if c!='_norm_name'}; fighter_id=f'missing::{norm(nm)}'; rec['fighter_id']=fighter_id; rec['fighter_name']=nm; ok=False; sd=None
   rec['fight_id']=fid; states.append(rec); ids.append(fighter_id); found.append(ok); dates.append(sd)
  prep.append({'fight_id':fid,'r_id':ids[0],'b_id':ids[1],'r_name':a,'b_name':b,'date':FIGHT_DATE,'title_fight':title,'total_rounds':rounds}); meta[fid]={'r':(a,oa,found[0],dates[0]),'b':(b,ob,found[1],dates[1])}
 v=add_engineered(build_moneyline_feature_view(prepared_fights_df=pd.DataFrame(prep),fighter_state_history_df=pd.DataFrame(states)))
 miss=[c for c in SELECTED if c not in v.columns]
 if miss: raise RuntimeError(f'missing live features: {miss}')
 return v,meta

def main():
 model,valid,med=fit_model(); v,meta=build_view(); rows=[]
 for _,r in v.iterrows():
  a,oa,fa,da=meta[r.fight_id]['r']; b,ob,fb,db=meta[r.fight_id]['b']; ia,ib=imp(oa),imp(ob); over=ia+ib; ma,mb=ia/over,ib/over; vals={c:r[c] for c in SELECTED}; vals['market_overround']=over; x=pd.DataFrame([{c:vals.get(c,np.nan) for c in valid}],columns=valid).replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0); base=logit([ma]); dm=xgb.DMatrix(x,base_margin=base,feature_names=valid); full=model.predict(dm,output_margin=True); pa=float(sigmoid(full)[0]); pb=1-pa; cold=not(fa and fb)
  for fighter,opp,side,odds,found,sd,market,p in [(a,b,'red',oa,fa,da,ma,pa),(b,a,'blue',ob,fb,db,mb,pb)]:
   mr=float(logit([market])[0]); vr=float(logit([p])[0]); rows.append({'fight_id':r.fight_id,'fighter':fighter,'opponent':opp,'side':side,'american_odds':odds,'market_source':'UFC official event page 2026-08-15','fighter_state_found':found,'fight_cold_start':cold,'prefight_state_date':sd,'fair_market_p':market,'v5_model_p':p,'edge':p-market,'market_logit':mr,'v5_logit':vr,'logit_residual':vr-mr,'qualifies_logit_030':(vr-mr)>=.30,'selected_feature_count':51})
 out=pd.DataFrame(rows).sort_values(['fight_id','side']); out.to_csv(OUTPUT,index=False)
 pref=out.sort_values(['fight_id','logit_residual'],ascending=[True,False]).groupby('fight_id',as_index=False).first(); pref.to_csv(BET_OUTPUT,index=False)
 print(out.to_string(index=False)); print('\nPREFERRED SIDES'); print(pref[['fighter','opponent','american_odds','fight_cold_start','fair_market_p','v5_model_p','logit_residual','qualifies_logit_030']].to_string(index=False)); print('QUALIFIERS=',int(pref.qualifies_logit_030.sum())); print('OUTPUT=',OUTPUT); print('BET_OUTPUT=',BET_OUTPUT)
if __name__=='__main__': main()
