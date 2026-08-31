from pathlib import Path
import re, unicodedata
import numpy as np
import pandas as pd

from pipeline.research import xgboost_method_market_offset as method
from pipeline.research.xgboost_method_hierarchical_v5_oof import _fit_conditional
from pipeline.research.score_ufc_abudhabi_hierarchical_v5_reconstructed import (
    build_prefight_features, logit, decimal_from_american,
)

OUT=Path('data/research/prop_mispricing')
FULL=OUT/'ufc_abudhabi_hierarchical_v5_dk_history_20260725.csv'
BETS=OUT/'ufc_abudhabi_hierarchical_v5_dk_history_bets_logit030_20260725.csv'
EVENT='UFC Fight Night: Ankalaev vs. Guskov'
CUTOFF=pd.Timestamp('2026-07-25 13:00:00',tz='UTC')
KEYS={'win_by_ko_tko_dq':'ko','win_by_submission':'sub','win_by_decision':'dec'}
SLUGS=['red_ko','red_sub','red_dec','blue_ko','blue_sub','blue_dec']
THRESHOLD=.30

CARD=[
('Magomed Ankalaev','Bogdan Guskov',.815315,'red_ko'),
('Steve Erceg','Ramazan Temirov',.493466,'blue_ko'),
('Magomed Zaynukov','Damian Rzepecki',.717040,'red_dec'),
('Rizvan Kuniev','Tyrell Fortune',.765695,'red_ko'),
('Abubakar Vagaev','Saygid Izagakhmaev',.698073,'red_dec'),
('Ismael Bonfim','Axel Sola',.340293,'blue_sub'),
('Valter Walker','Thomas Petersen',.663077,'red_sub'),
('Santiago Ponzinibbio','Sam Patterson',.229243,'blue_ko'),
('Magomed Tuchalov','Brendson Ribeiro',.863915,'red_dec'),
('Nurullo Aliev','Mike Davis',.729535,'red_dec'),
]

def norm(x):
    s=''.join(c for c in unicodedata.normalize('NFKD',str(x)) if not unicodedata.combining(c))
    s=re.sub(r'[^a-z0-9]+','',s.lower())
    aliases={'ramazantemirov':'ramazonbektemirov'}
    return aliases.get(s,s)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    h=pd.read_parquet('data/market/market_intelligence_history.parquet').copy()
    h['refresh_timestamp']=pd.to_datetime(h['refresh_timestamp'],utc=True,errors='coerce')
    h=h[(h.bookmaker=='DraftKings')&(h.event_name==EVENT)&(h.refresh_timestamp<CUTOFF)&h.market_key.isin(KEYS)].copy()
    h=h[h.refresh_timestamp==h.refresh_timestamp.max()].copy()
    h['fighter_norm']=h.outcome_display.map(norm)
    h['meth']=h.market_key.map(KEYS)

    rows=[]
    for i,(red,blue,pred,target) in enumerate(CARD,1):
        pair=h[h.fight_display.astype(str).map(lambda z: norm(red) in norm(z) and norm(blue) in norm(z))].copy()
        if len(pair)!=6:
            raise RuntimeError(f'{red} vs {blue}: expected 6 final-snapshot rows, got {len(pair)}')
        rec={'fight_id':f'dkabu_{i:02d}','red_fighter':red,'blue_fighter':blue,'v5_p_red':pred,'target_slug':target,'snapshot_timestamp':pair.refresh_timestamp.iloc[0]}
        raw=[]
        odds=[]
        for fighter,side in [(red,'red'),(blue,'blue')]:
            for meth in ['ko','sub','dec']:
                z=pair[(pair.fighter_norm==norm(fighter))&(pair.meth==meth)]
                if len(z)!=1: raise RuntimeError(f'{fighter} {meth}: {len(z)} rows')
                rr=z.iloc[0]
                rec[f'odds_{side}_{meth}']=float(rr.american_odds)
                raw.append(float(rr.implied_probability)); odds.append(float(rr.american_odds))
        raw=np.asarray(raw,float); fair=raw/raw.sum()
        for j,slug in enumerate(SLUGS): rec[f'market_{slug}']=fair[j]
        rows.append(rec)
    card=pd.DataFrame(rows)

    train,features,_=method._build_rows(True,True)
    live,found=build_prefight_features(card)
    live=live.set_index('fight_id',drop=False)
    score=[]
    for _,r in card.iterrows():
        d=r.to_dict(); d['fight_cold_start']=not found[str(r.fight_id)]
        lr=live.loc[r.fight_id]
        for c in features: d[c]=lr[c] if c in lr.index else np.nan
        score.append(d)
    score=pd.DataFrame(score)
    rc,rn,rfc=_fit_conditional(train,score,features,'red')
    bc,bn,bfc=_fit_conditional(train,score,features,'blue')
    pr=np.clip(score.v5_p_red.to_numpy(float),1e-12,1-1e-12)
    hp=np.concatenate([pr[:,None]*rc,(1-pr)[:,None]*bc],axis=1)
    hp/=hp.sum(axis=1,keepdims=True)
    for j,s in enumerate(SLUGS):
        score[f'hier_{s}']=hp[:,j]
        score[f'residual_{s}']=[logit(hp[i,j])-logit(score.iloc[i][f'market_{s}']) for i in range(len(score))]

    br=[]
    for _,r in score.iterrows():
        ps='red' if r.v5_p_red>=.5 else 'blue'
        for meth in ['ko','sub','dec']:
            s=f'{ps}_{meth}'; resid=float(r[f'residual_{s}'])
            odds=float(r[f'odds_{s}'])
            dec=decimal_from_american(odds)
            won=int(r.target_slug==s)
            br.append({'fight_id':r.fight_id,'red_fighter':r.red_fighter,'blue_fighter':r.blue_fighter,'projected_side':ps,'bet_fighter':r[f'{ps}_fighter'],'bet_method':meth.upper(),'american_odds':odds,'model_probability':float(r[f'hier_{s}']),'normalized_market_probability':float(r[f'market_{s}']),'signed_logit_residual':resid,'qualifies_030':resid>=THRESHOLD,'actual_slug':r.target_slug,'won':won,'profit_if_flat1':(dec-1 if won else -1),'snapshot_timestamp':r.snapshot_timestamp})
    diag=pd.DataFrame(br).sort_values('signed_logit_residual',ascending=False)
    score.to_csv(FULL,index=False)
    q=diag[diag.qualifies_030].copy(); q.to_csv(BETS,index=False)
    print(f'SNAPSHOT={h.refresh_timestamp.max()} FIGHTS={len(score)} RED_TRAIN_N={rn} BLUE_TRAIN_N={bn} RED_FC={rfc} BLUE_FC={bfc}')
    print('\n=== ALL WINNER-SIDE METHOD CANDIDATES ===')
    print(diag[['bet_fighter','bet_method','american_odds','model_probability','normalized_market_probability','signed_logit_residual','qualifies_030','actual_slug','won','profit_if_flat1']].to_string(index=False))
    print('\n=== LOCKED +0.30 QUALIFIERS ===')
    if q.empty: print('NONE')
    else:
        print(q[['bet_fighter','bet_method','american_odds','signed_logit_residual','actual_slug','won','profit_if_flat1']].to_string(index=False))
        print(f"BETS={len(q)} WINS={int(q.won.sum())} LOSSES={len(q)-int(q.won.sum())} PROFIT={q.profit_if_flat1.sum():.6f} ROI={q.profit_if_flat1.sum()/len(q):.6f}")
    print('FULL=',FULL); print('BETS=',BETS)

if __name__=='__main__': main()
