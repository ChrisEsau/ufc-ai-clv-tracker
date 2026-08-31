from __future__ import annotations

import json
import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.research.xgboost_method_market_offset import (
    ROOT, PARAMS, EPS, CLASS_ORDER, MARKET_COLS,
    _build_rows, _metrics, _calibration,
)

FEATURE_LIST_PATH = ROOT / "xgboost_method_market_offset__feature_list.json"
FROZEN_SIXWAY_PRED = ROOT / "xgboost_method_market_offset__oof_predictions.csv"
OUT_PRED = ROOT / "xgboost_method_market_offset__family_specific_specialists_nested_oof_predictions.csv"
OUT_SUMMARY = ROOT / "xgboost_method_market_offset__family_specific_specialists_nested_oof_summary.json"
OUT_SELECTIONS = ROOT / "xgboost_method_market_offset__family_specific_specialists_nested_selections.csv"

FAMILIES = {"KO_TKO":[0,3], "SUB":[1,4], "DEC":[2,5]}
FEATURE_COUNTS = [140,75,35]
CAPACITY = [
    {"name":"d1_r150","max_depth":1,"rounds":150},
    {"name":"d1_r300","max_depth":1,"rounds":300},
    {"name":"d2_r150","max_depth":2,"rounds":150},
    {"name":"d2_r300","max_depth":2,"rounds":300},
]
TEST_YEARS=[2022,2023,2024]

def logit(p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS); return np.log(p/(1-p))

def softmax(z):
    z=np.asarray(z,float); z=z-z.max(axis=1,keepdims=True); e=np.exp(z); return e/e.sum(axis=1,keepdims=True)

def load_features():
    obj=json.loads(FEATURE_LIST_PATH.read_text()); return list(obj["features"] if isinstance(obj,dict) else obj)

def prep(train,val,features):
    a=train[features].replace([np.inf,-np.inf],np.nan); b=val[features].replace([np.inf,-np.inf],np.nan)
    valid=[c for c in features if a[c].notna().any()]; med=a[valid].median(numeric_only=True)
    return a[valid].fillna(med).fillna(0.0),b[valid].fillna(med).fillna(0.0),valid

def family_target(frame,fam): return frame["target"].astype(int).isin(FAMILIES[fam]).astype(int).to_numpy()
def family_market(frame,fam): return frame[[MARKET_COLS[j] for j in FAMILIES[fam]]].sum(axis=1).to_numpy(float)

def rank_features(df,features,fam):
    train=df[df.date<=pd.Timestamp("2020-12-31")].copy(); x,_,valid=prep(train,train,features)
    d=xgb.DMatrix(x,label=family_target(train,fam),feature_names=list(x.columns)); d.set_base_margin(logit(family_market(train,fam)))
    p=dict(PARAMS); p.update({"objective":"binary:logistic","eval_metric":"logloss","max_depth":1}); p.pop("num_class",None)
    b=xgb.train(p,d,num_boost_round=150,verbose_eval=False); gain=b.get_score(importance_type="gain")
    return sorted(valid,key=lambda c:(-float(gain.get(c,0.0)),c))

def fit(train,val,fam,features,depth,rounds):
    xtr,xva,_=prep(train,val,features); y=family_target(train,fam); mtr=family_market(train,fam); mva=family_market(val,fam)
    dtr=xgb.DMatrix(xtr,label=y,feature_names=list(xtr.columns)); dva=xgb.DMatrix(xva,feature_names=list(xva.columns))
    dtr.set_base_margin(logit(mtr)); dva.set_base_margin(logit(mva))
    p=dict(PARAMS); p.update({"objective":"binary:logistic","eval_metric":"logloss","max_depth":depth}); p.pop("num_class",None)
    b=xgb.train(p,dtr,num_boost_round=rounds,verbose_eval=False); return np.asarray(b.predict(dva),float)

def bll(y,p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS); y=np.asarray(y,float); return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))

def select(selection_years,df,ranked):
    out={}
    for fam in FAMILIES:
        rows=[]
        for nf in FEATURE_COUNTS:
            feats=ranked[fam][:min(nf,len(ranked[fam]))]
            for cap in CAPACITY:
                ys=[];ps=[];ms=[]
                for yr in selection_years:
                    tr=df[df.date<=pd.Timestamp(f"{yr-1}-12-31")].copy(); va=df[(df.date>=pd.Timestamp(f"{yr}-01-01"))&(df.date<=pd.Timestamp(f"{yr}-12-31"))].copy()
                    ys.append(family_target(va,fam)); ps.append(fit(tr,va,fam,feats,cap["max_depth"],cap["rounds"])); ms.append(family_market(va,fam))
                y=np.concatenate(ys); p=np.concatenate(ps); m=np.concatenate(ms)
                rows.append({"family":fam,"feature_count_requested":nf,"max_depth":cap["max_depth"],"rounds":cap["rounds"],"binary_log_loss":bll(y,p),"market_binary_log_loss":bll(y,m),"candidate":f"{fam}_top{nf}_{cap['name']}"})
        out[fam]=sorted(rows,key=lambda r:(r["binary_log_loss"],r["feature_count_requested"],r["max_depth"],r["rounds"]))[0]
    return out

def combine(market,fprob):
    scores=np.log(np.clip(market,EPS,1.0))
    for fam,idx in FAMILIES.items():
        resid=logit(fprob[fam])-logit(market[:,idx].sum(axis=1)); scores[:,idx]+=resid[:,None]
    return softmax(scores)

def main():
    features=load_features(); df,_,_=_build_rows(True,True,forced_features=features); ranked={f:rank_features(df,features,f) for f in FAMILIES}
    rows=[]; sels=[]; yearly=[]
    for tyr in TEST_YEARS:
        syears=list(range(2021,tyr)); chosen=select(syears,df,ranked)
        for fam,cfg in chosen.items(): sels.append({"test_year":tyr,"selection_years":','.join(map(str,syears)),**cfg})
        tr=df[df.date<=pd.Timestamp(f"{tyr-1}-12-31")].copy(); va=df[(df.date>=pd.Timestamp(f"{tyr}-01-01"))&(df.date<=pd.Timestamp(f"{tyr}-12-31"))].copy()
        fp={}
        for fam,cfg in chosen.items():
            feats=ranked[fam][:min(int(cfg["feature_count_requested"]),len(ranked[fam]))]; fp[fam]=fit(tr,va,fam,feats,int(cfg["max_depth"]),int(cfg["rounds"]))
        market=va[MARKET_COLS].to_numpy(float); nested=combine(market,fp); y=va.target.to_numpy(int)
        yearly.append({"year":tyr,"n":len(va),"market":_metrics(y,market),"nested":_metrics(y,nested)})
        for i,(_,r) in enumerate(va.iterrows()):
            o={"fight_id":r.fight_id,"date":r.date.date().isoformat(),"event_name":r.event_name,"red_fighter":r.red_fighter,"blue_fighter":r.blue_fighter,"target":int(r.target),"actual_class":CLASS_ORDER[int(r.target)],"test_year":tyr}
            for j,c in enumerate(CLASS_ORDER): o[f"market_{c.lower()}"]=float(market[i,j]); o[f"nested_{c.lower()}"]=float(nested[i,j])
            rows.append(o)
    pred=pd.DataFrame(rows).sort_values(["date","fight_id"]).reset_index(drop=True); y=pred.target.to_numpy(int)
    market=pred[[f"market_{c.lower()}" for c in CLASS_ORDER]].to_numpy(float); nested=pred[[f"nested_{c.lower()}" for c in CLASS_ORDER]].to_numpy(float)
    frozen=pd.read_csv(FROZEN_SIXWAY_PRED); frozen=frozen[frozen.fold.astype(int).isin(TEST_YEARS)].sort_values(["date","fight_id"]).reset_index(drop=True)
    if list(frozen.fight_id.astype(str))!=list(pred.fight_id.astype(str)): raise RuntimeError("frozen comparison rows do not align")
    frozen_p=frozen[["model_red_ko","model_red_sub","model_red_dec","model_blue_ko","model_blue_sub","model_blue_dec"]].to_numpy(float)
    mm,nm,fm=_metrics(y,market),_metrics(y,nested),_metrics(y,frozen_p)
    summary={"experiment":"six_way_method_family_specific_specialists_nested_oof_v1","period":"nested chronological 2022-2024 OOF","reads_2025_plus":False,"uses_roi":False,"selection_rule":"each test year selects KO/SUB/DEC configs only from earlier OOF years","pooled_n":int(len(pred)),"pooled_market":mm,"pooled_nested":nm,"pooled_frozen_sixway_same_rows":fm,"delta_nested_minus_market_log_loss":float(nm["log_loss"]-mm["log_loss"]),"delta_nested_minus_frozen_sixway_log_loss":float(nm["log_loss"]-fm["log_loss"]),"beats_frozen_sixway_same_rows":bool(nm["log_loss"]<fm["log_loss"]),"yearly":yearly,"calibration":_calibration(y,nested)}
    pd.DataFrame(sels).to_csv(OUT_SELECTIONS,index=False); pred.to_csv(OUT_PRED,index=False); OUT_SUMMARY.write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
