from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pipeline.features.views.moneyline import build_moneyline_feature_view
from pipeline.market.providers.draftkings_public import DEFAULT_USER_AGENT, build_event_subcategory_markets_url, fetch_public_json
from pipeline.research import xgboost_method_market_offset as method
from pipeline.research import xgboost_method_hierarchical_v5_oof as hier
from ufc_feature_engineering import add_v5_engineered_features

OUT=Path('data/research/prop_mispricing'); ML_PATH=OUT/'ufc_paris_v5_market_offset_current_20260831.csv'; OUTPUT=OUT/'ufc_paris_hierarchical_v5_methods_20260831.csv'; RAW=OUT/'ufc_paris_draftkings_method_prices_20260831.csv'
LEAGUE_NAV='https://sportsbook-nash.draftkings.com/sites/US-KS-SB/api/sportscontent/navigation/dkusks/v2/nav/leagues/9034'; METHOD_SUBCATEGORY='18911'
FIGHTS=[('Dan Hooker','Salahdine Parnasse'),('Fares Ziam','Axel Sola'),('Michael Page','Nursulton Ruziboev'),('Daniil Donchenko','Punahele Soriano'),('Morgan Charriere','Felipe Lima'),('Losene Keita','Muhammad Naimov'),('Mario Pinto','Ryan Spann'),('Kurtis Campbell','Trevor Peek'),('Oumar Sy','Modestas Bukauskas'),('Nathaniel Wood','Mairon Santos'),('Michael Aljarouj','Fabia Sintes'),('Nora Cornolle','Klaudia Sygula'),('Matthieu Letho Duclos','Luis Felipe Dias'),('Delphine Benouaich','Sofia Montenegro')]

def norm(x): return ''.join(ch for ch in unicodedata.normalize('NFKD',str(x)) if not unicodedata.combining(ch)).replace('’',"'").lower().strip()
def last(x): return norm(x).split()[-1]
def imp(v):
    ao=float(str(v).replace('−','-').replace('+','').strip()); return 100/(ao+100) if ao>0 else (-ao)/((-ao)+100)

def fetch_events():
    h={'Accept':'*/*','User-Agent':DEFAULT_USER_AGENT,'Origin':'https://sportsbook.draftkings.com','Referer':'https://sportsbook.draftkings.com/','X-Client-Name':'web'}
    r=requests.get(LEAGUE_NAV,headers=h,timeout=30); r.raise_for_status(); return r.json().get('events',[])

def match_event(events,a,b):
    la,lb=last(a),last(b); hits=[]
    for ev in events:
        names=[norm(p.get('name','')) for p in ev.get('participants',[])]; en=norm(ev.get('name',''))
        if (any(la in n.split() for n in names) and any(lb in n.split() for n in names)) or (la in en and lb in en): hits.append(ev)
    if len(hits)!=1: raise RuntimeError(f'DK event match {a} vs {b}: {[(x.get("id"),x.get("name")) for x in hits]}')
    return hits[0]

def fetch_prices(events):
    raw=[]; fm={}
    for i,(a,b) in enumerate(FIGHTS,1):
        fid=f'paris_v5_20260905_{i:02d}'; ev=match_event(events,a,b); p=fetch_public_json(build_event_subcategory_markets_url(str(ev['id']),METHOD_SUBCATEGORY)); mn={str(m['id']):str(m.get('name','')) for m in p.get('markets',[])}; six={}; odds={}
        for s in p.get('selections',[]):
            m=mn.get(str(s.get('marketId')),'').lower(); meth='ko' if 'ko/tko' in m else ('sub' if 'submission' in m else ('dec' if 'decision' in m else None)); ot=str(s.get('outcomeType','')).lower(); side='red' if ot=='home' else ('blue' if ot=='away' else None); ao=s.get('displayOdds',{}).get('american')
            if meth and side and ao is not None:
                slug=f'{side}_{meth}'; six[slug]=imp(ao); odds[slug]=str(ao).replace('−','-'); raw.append({'fight_id':fid,'provider_event_id':str(ev['id']),'event_name':ev.get('name'),'red_fighter':a,'blue_fighter':b,'class_slug':slug,'american_odds':odds[slug],'implied_probability':six[slug]})
        miss=[s for s in method.SLUGS if s not in six]
        if miss: raise RuntimeError(f'Incomplete DK method market {a} vs {b}, event={ev.get("id")}, missing={miss}, markets={list(mn.values())}')
        total=sum(six.values()); fm[fid]={**{f'market_{s}':six[s]/total for s in method.SLUGS},**{f'odds_{s}':odds[s] for s in method.SLUGS},'method_overround':total,'provider_event_id':str(ev['id']),'dk_event_name':ev.get('name')}
    return pd.DataFrame(raw),fm

def live_features():
    latest=pd.read_parquet('data/features/latest_fighter_state.parquet').copy(); latest['fighter_id']=latest.fighter_id.astype(str); latest['_n']=latest.fighter_name.map(norm); prep=[]; states=[]
    for i,(a,b) in enumerate(FIGHTS,1):
        fid=f'paris_v5_20260905_{i:02d}'; ids=[]
        for nm in (a,b):
            hit=latest[latest._n.eq(norm(nm))]
            if len(hit): rec=hit.iloc[-1].drop(labels=['_n']).to_dict(); fighter_id=str(rec['fighter_id'])
            else: rec={c:np.nan for c in latest.columns if c!='_n'}; fighter_id=f'missing::{norm(nm)}'; rec['fighter_id']=fighter_id; rec['fighter_name']=nm
            rec['fight_id']=fid; states.append(rec); ids.append(fighter_id)
        prep.append({'fight_id':fid,'r_id':ids[0],'b_id':ids[1],'r_name':a,'b_name':b,'date':pd.Timestamp('2026-09-05'),'title_fight':False,'total_rounds':5 if i==1 else 3})
    return add_v5_engineered_features(build_moneyline_feature_view(prepared_fights_df=pd.DataFrame(prep),fighter_state_history_df=pd.DataFrame(states)))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if not ML_PATH.exists(): raise RuntimeError(f'Missing {ML_PATH}')
    raw,markets=fetch_prices(fetch_events()); raw.to_csv(RAW,index=False); live=live_features(); train,features,_=method._build_rows(True,True); missing=[c for c in features if c not in live.columns]
    if missing: raise RuntimeError(f'Live feature view missing frozen hierarchical method features: {missing}')
    score=live[['fight_id']+features].copy()
    for c in method.MARKET_COLS: score[c]=score.fight_id.map(lambda fid:markets[str(fid)][c])
    red_cond,_,rfc=hier._fit_conditional(train,score,features,'red'); blue_cond,_,bfc=hier._fit_conditional(train,score,features,'blue')
    ml=pd.read_csv(ML_PATH); mlr=ml[ml.side.astype(str).eq('red')].set_index('fight_id'); pr=score.fight_id.map(lambda fid:float(mlr.loc[str(fid),'v5_model_p'])).to_numpy(float); six=np.concatenate([pr[:,None]*red_cond,(1-pr)[:,None]*blue_cond],axis=1); six=six/six.sum(axis=1,keepdims=True)
    rows=[]
    for k,(a,b) in enumerate(FIGHTS):
        fid=str(score.iloc[k].fight_id); probs={s:float(six[k,j]) for j,s in enumerate(method.SLUGS)}; ws='red' if pr[k]>=.5 else 'blue'; winner=a if ws=='red' else b; opp=b if ws=='red' else a; cand=[f'{ws}_ko',f'{ws}_sub',f'{ws}_dec']; top=max(cand,key=lambda s:probs[s]); mk=markets[fid]
        rows.append({'fight_id':fid,'red_fighter':a,'blue_fighter':b,'v5_projected_winner':winner,'opponent':opp,'v5_winner_probability':float(pr[k] if ws=='red' else 1-pr[k]),'winner_side':ws,'hier_winner_ko':probs[f'{ws}_ko'],'hier_winner_sub':probs[f'{ws}_sub'],'hier_winner_dec':probs[f'{ws}_dec'],'selected_top_method':top.split('_',1)[1],'selected_method_probability':probs[top],'dk_selected_method_odds':mk[f'odds_{top}'],'market_sixway_fair_probability':mk[f'market_{top}'],'method_probability_edge':probs[top]-mk[f'market_{top}'],'method_overround':mk['method_overround'],'provider_event_id':mk['provider_event_id'],'dk_event_name':mk['dk_event_name'],**{f'hier_{s}':probs[s] for s in method.SLUGS},'red_feature_count':rfc,'blue_feature_count':bfc})
    out=pd.DataFrame(rows); out.to_csv(OUTPUT,index=False); print(out[['red_fighter','blue_fighter','v5_projected_winner','v5_winner_probability','hier_winner_ko','hier_winner_sub','hier_winner_dec','selected_top_method','selected_method_probability','dk_selected_method_odds']].to_string(index=False)); print(json.dumps({'rows':len(out),'output':str(OUTPUT),'raw_prices':str(RAW)},indent=2))
if __name__=='__main__': main()
