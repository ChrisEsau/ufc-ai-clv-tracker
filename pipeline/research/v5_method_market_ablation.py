from __future__ import annotations

import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import xgboost as xgb
import pipeline.research.v5_method_market_feature_test as b

OUT=Path('data/research/prop_mispricing')
df=b.df; Xraw=b.Xraw
candidates={
    'v5': b.FEATURES,
    'plus_method_implied_red_win_p': b.FEATURES+['method_implied_red_win_p'],
    'plus_method_vs_ml_red_gap': b.FEATURES+['method_vs_ml_red_gap'],
    'plus_method_inside_distance_p': b.FEATURES+['method_inside_distance_p'],
    'plus_method_win_concentration_diff': b.FEATURES+['method_win_concentration_diff'],
    'plus_all_four': b.FEATURES+b.METHOD_FEATURES,
}
summary={'experiment':'v5_method_market_single_feature_ablation_v1','selection_objective':'2021-2024 chronological OOF log loss only; ROI not used','models':{}}
for name,cols in candidates.items():
    parts=[]; folds=[]
    for fn,te,vs,ve in b.FOLDS:
        tr=df.date<=te; va=(df.date>=vs)&(df.date<=ve)
        valid=[c for c in cols if Xraw.loc[tr,c].notna().any()]
        med=Xraw.loc[tr,valid].median(numeric_only=True)
        Xtr=Xraw.loc[tr,valid].fillna(med).fillna(0); Xva=Xraw.loc[va,valid].fillna(med).fillna(0)
        ytr=df.loc[tr,'won'].astype(int).to_numpy(); yva=df.loc[va,'won'].astype(int).to_numpy()
        mtr=b.logit(df.loc[tr,'fair_market_p']); mva=b.logit(df.loc[va,'fair_market_p'])
        dtr=xgb.DMatrix(Xtr,label=ytr,base_margin=mtr,feature_names=valid); dva=xgb.DMatrix(Xva,label=yva,base_margin=mva,feature_names=valid)
        model=xgb.train(b.PARAMS,dtr,num_boost_round=300,verbose_eval=False)
        p=b.sigmoid(model.predict(dva,output_margin=True)); mx=b.met(yva,p)
        folds.append({'fold':fn,'model':mx})
        parts.append(pd.DataFrame({'won':yva,'model_p':p}))
    o=pd.concat(parts,ignore_index=True)
    summary['models'][name]={'feature_count':len(cols),'folds':folds,'oof':b.met(o.won,o.model_p)}
base=summary['models']['v5']['oof']['log_loss']
ranking=[]
for name,v in summary['models'].items():
    v['delta_log_loss_vs_v5']=float(v['oof']['log_loss']-base)
    ranking.append({'model':name,'log_loss':v['oof']['log_loss'],'delta_vs_v5':v['delta_log_loss_vs_v5'],'brier':v['oof']['brier'],'auc':v['oof']['auc']})
summary['ranking']=sorted(ranking,key=lambda x:x['log_loss'])
json.dump(summary,open(OUT/'v5_method_market_ablation_summary.json','w'),indent=2)
pd.DataFrame(summary['ranking']).to_csv(OUT/'v5_method_market_ablation_ranking.csv',index=False)
print(json.dumps(summary,indent=2))
