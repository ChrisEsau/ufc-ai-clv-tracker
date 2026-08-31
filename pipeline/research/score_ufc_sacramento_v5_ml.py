from __future__ import annotations

import subprocess, unicodedata
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from pipeline.features.views.moneyline import build_moneyline_feature_view

OUT=Path('data/research/prop_mispricing')
OUTPUT=OUT/'ufc_sacramento_v5_moneyline_20260822.csv'
BET_OUTPUT=OUT/'ufc_sacramento_v5_betting_logit030_no_cold_exclusion_20260822.csv'
SNAP='7df1b61126be1f4e036b256d1c774c531b8a281f'
FIGHT_DATE=pd.Timestamp('2026-08-22')
FIGHTS=[
('Anthony Hernandez','Gregory Rodrigues',-220,180,5,'UFC'),
('Serghei Spivac','Vitor Petrino',122,-155,3,'UFC'),
('Reinier de Ridder','Roman Dolidze',-400,290,3,'UFC'),
('MarQuel Mederos','Mason Jones',300,-380,3,'UFC'),
('Carli Judice','Jeisla Chaves',-575,425,3,'UFC'),
('Anthony Wint','Terrance Chatman',-1100,625,3,'UFC'),
('Jamall Emmers','Lerryan Douglas',300,-390,3,'UFC'),
('Kennedy Nzechukwu','Shamil Gaziev',102,-130,3,'UFC'),
('Chris Padilla','Nasrat Haqparast',-122,102,3,'UFC'),
('Marcio Barbosa','Ryan Kuse',-900,550,3,'UFC'),
('Gauge Young','Stan Dorsainvil',-190,150,3,'UFC'),
('Wes Schultz','Jackson McVey',145,-185,3,'UFC'),
('Shanelle Dyer','Elise Reed',-850,575,3,'DraftKings 2026-08-21'),
]
SELECTED=['reach_diff','recent_form_recent_avg_fight_time_diff','age_diff','ewm_sapm_diff','ewm_recent_sapm_diff','style_ko_finisher_score_diff','ewm_td_acc_diff','recent_finish_rate_diff','chin_risk_diff','recent_form_avg_opponent_elo_diff','recent_avg_fight_time_diff','aggression_index_diff','age_squared_diff','sapm_diff','ewm_kd_avg_diff','style_all_round_finisher_score_diff','recent_form_kd_absorbed_avg_diff','ewm_recent_splm_diff','elo_diff','ewm_elo_diff','ewm_recent_td_avg_diff','days_since_last_fight_diff','td_avg_diff','style_score_spread_diff','ko_dependency_diff','recent_form_avg_fight_time_diff','wrestling_mismatch_diff','win_pct_diff','recent_form_ko_rate_diff','recent_form_worst_loss_elo_diff','age_x_career_ko_losses_diff','ewm_str_def_diff','losses_diff','ewm_recent_win_pct_diff','avg_opponent_elo_diff','ewm_td_avg_diff','avg_fight_time_diff','ewm_days_since_last_fight_diff','pressure_striking_adv_diff','weight_diff','ctrl_against_per_min_diff','ewm_finish_loss_rate_diff','ewm_win_pct_diff','victory_concentration_index_diff','recent_form_td_acc_diff','sub_avg_diff','recent_form_best_win_elo_diff','ewm_best_win_elo_diff','style_primary_score_diff','recent_form_recent_finish_rate_diff']

def norm(s): return ''.join(ch for ch in unicodedata.normalize('NFKD',str(s)) if not unicodedata.combining(ch)).replace('’',"'").lower().strip()
def imp(ao): return 100/(ao+100) if ao>0 else (-ao)/((-ao)+100)
def clip_p(p): return np.clip(np.asarray(p,float),1e-6,1-1e-6)
def logit(p):
 p=clip_p(p); return np.log(p/(1-p))
def sigmoid(z):
 z=np.clip(np.asarray(z,float),-30,30); return 1/(1+np.exp(-z))

def add_engineered(livefv):
 def _num(c): return pd.to_numeric(livefv[c],errors='coerce') if c in livefv.columns else pd.Series(np.nan,index=livefv.index)
 def _rate(c):
  s=_num(c); return pd.Series(np.where(s>1,s/100.0,s),index=livefv.index)
 livefv['chin_risk_diff']=_num('r_pre_sapm')*(1-_rate('r_pre_str_def'))-_num('b_pre_sapm')*(1-_rate('b_pre_str_def'))
 livefv['aggression_index_diff']=(_num('r_pre_splm')+_num('r_pre_td_avg'))-(_num('b_pre_splm')+_num('b_pre_td_avg'))
 livefv['age_squared_diff']=_num('r_pre_age')**2-_num('b_pre_age')**2
 livefv['wrestling_mismatch_diff']=_num('r_pre_td_avg')*(1-_rate('b_pre_td_def'))-_num('b_pre_td_avg')*(1-_rate('r_pre_td_def'))
 livefv['pressure_striking_adv_diff']=_num('r_pre_splm')*(1-_rate('b_pre_str_def'))-_num('b_pre_splm')*(1-_rate('r_pre_str_def'))
 livefv['age_x_career_ko_losses_diff']=_num('r_pre_age')*_num('r_pre_career_ko_losses')-_num('b_pre_age')*_num('b_pre_career_ko_losses')
 return livefv

def fit_model():
 OUT.mkdir(parents=True,exist_ok=True)
 for rp,op in [('data/market/historical_market_outcomes.parquet','/tmp/v5_market.parquet'),('data/features/moneyline_feature_view.parquet','/tmp/v5_fv.parquet')]:
  with open(op,'wb') as f: subprocess.run(['git','show',f'{SNAP}:{rp}'],stdout=f,check=True)
 market=pd.read_parquet('/tmp/v5_market.parquet').copy(); market=market[(market.bookmaker=='legacy_consensus')&(market.result_status=='graded')&market.won.notna()].copy(); market['date']=pd.to_datetime(market.date,errors='coerce'); market['won']=market.won.astype(bool).astype(int); market['implied_probability']=pd.to_numeric(market.implied_probability,errors='coerce'); market=market.dropna(subset=['date','implied_probability'])
 ml=market[market.market_key=='moneyline'].copy(); good=ml.groupby('fight_id').size(); ml=ml[ml.fight_id.isin(good[good==2].index)].copy(); ml['market_overround']=ml.groupby('fight_id').implied_probability.transform('sum'); ml['fair_market_p']=ml.implied_probability/ml.market_overround; red=ml[ml.outcome_side.astype(str).eq('red')].copy(); fv=pd.read_parquet('/tmp/v5_fv.parquet')
 feature_cols=SELECTED+['market_overround']; df=red.merge(fv[['fight_id']+[c for c in SELECTED if c in fv.columns]],on='fight_id',how='inner').sort_values(['date','fight_id']); xraw=df[feature_cols].replace([np.inf,-np.inf],np.nan); tr=df.date<='2024-12-31'; valid=[c for c in feature_cols if xraw.loc[tr,c].notna().any()]; med=xraw.loc[tr,valid].median(numeric_only=True); xtr=xraw.loc[tr,valid].fillna(med).fillna(0.0); ytr=df.loc[tr,'won'].astype(int).to_numpy(); mtr=logit(df.loc[tr,'fair_market_p'])
 params={'max_depth':1,'eta':0.03,'subsample':0.8,'colsample_bytree':0.7,'min_child_weight':10,'lambda':8.0,'alpha':1.0,'objective':'binary:logistic','eval_metric':'logloss','seed':42,'nthread':2}; model=xgb.train(params,xgb.DMatrix(xtr,label=ytr,base_margin=mtr,feature_names=valid),num_boost_round=300,verbose_eval=False)
 return model,valid,med

def build_live():
 hist=pd.read_parquet('data/features/fighter_state_history.parquet').copy(); hist['date']=pd.to_datetime(hist['date'],errors='coerce'); hist['fighter_id']=hist.fighter_id.astype(str); hist['_norm_name']=hist.fighter_name.map(norm)
 prep=[]; states=[]; meta={}
 for i,(a,b,aoa,aob,rounds,src) in enumerate(FIGHTS,1):
  fid=f'sacramento_v5_20260822_{i:02d}'; ids=[]; found=[]; dates=[]
  for nm in (a,b):
   hit=hist[(hist._norm_name.eq(norm(nm)))&(hist.date<FIGHT_DATE)].sort_values('date')
   if len(hit): rec=hit.iloc[-1].drop(labels=['_norm_name']).to_dict(); fighter_id=str(rec['fighter_id']); ok=True; sd=pd.Timestamp(rec['date']).strftime('%Y-%m-%d')
   else: rec={c:np.nan for c in hist.columns if c!='_norm_name'}; fighter_id=f'missing::{norm(nm)}'; rec['fighter_id']=fighter_id; rec['fighter_name']=nm; ok=False; sd=None
   rec['fight_id']=fid; states.append(rec); ids.append(fighter_id); found.append(ok); dates.append(sd)
  prep.append({'fight_id':fid,'r_id':ids[0],'b_id':ids[1],'r_name':a,'b_name':b,'date':FIGHT_DATE,'title_fight':False,'total_rounds':rounds}); meta[fid]={'r':(a,aoa,found[0],dates[0]),'b':(b,aob,found[1],dates[1]),'src':src}
 live=add_engineered(build_moneyline_feature_view(prepared_fights_df=pd.DataFrame(prep),fighter_state_history_df=pd.DataFrame(states)))
 return live,meta

def main():
 model,valid,med=fit_model(); live,meta=build_live(); rows=[]
 for _,r in live.iterrows():
  a,aoa,fa,da=meta[r.fight_id]['r']; b,aob,fb,db=meta[r.fight_id]['b']; ipa,ipb=imp(aoa),imp(aob); over=ipa+ipb; fpa,fpb=ipa/over,ipb/over; vals={c:r[c] for c in SELECTED}; vals['market_overround']=over; x=pd.DataFrame([{c:vals.get(c,np.nan) for c in valid}],columns=valid).replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0.0); ml=logit([fpa]); dm=xgb.DMatrix(x,base_margin=ml,feature_names=valid); full=model.predict(dm,output_margin=True); pa=float(sigmoid(full)[0]); pb=1-pa; cold=not(fa and fb); src=meta[r.fight_id]['src']
  for fighter,opp,side,ao,found,sd,fair,p in [(a,b,'red',aoa,fa,da,fpa,pa),(b,a,'blue',aob,fb,db,fpb,pb)]:
   mr=float(logit([fair])[0]); vr=float(logit([p])[0]); rows.append({'fight_id':r.fight_id,'fighter':fighter,'opponent':opp,'side':side,'american_odds':ao,'market_source':src,'fighter_state_found':found,'fight_cold_start':cold,'prefight_state_date':sd,'fair_market_p':fair,'v5_model_p':p,'edge':p-fair,'market_logit':mr,'v5_logit':vr,'logit_residual':vr-mr,'qualifies_logit_030':(vr-mr)>=0.30,'selected_feature_count':51})
 out=pd.DataFrame(rows).sort_values(['fight_id','side']); out.to_csv(OUTPUT,index=False)
 pref=out.sort_values(['fight_id','logit_residual'],ascending=[True,False]).groupby('fight_id',as_index=False).first(); pref.to_csv(BET_OUTPUT,index=False)
 print(out.to_string(index=False)); print('\nPREFERRED SIDES'); print(pref[['fighter','opponent','american_odds','fight_cold_start','fair_market_p','v5_model_p','logit_residual','qualifies_logit_030']].to_string(index=False)); print('QUALIFIERS=',int(pref.qualifies_logit_030.sum())); print('OUTPUT=',OUTPUT); print('BET_OUTPUT=',BET_OUTPUT)

if __name__=='__main__': main()
