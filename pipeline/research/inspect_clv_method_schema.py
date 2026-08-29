#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

FILES=[
 Path('data/clv/ufc_model_candidate_clv.parquet'),
 Path('data/clv/ufc_model_candidate_tracker.parquet'),
 Path('data/market/canonical_market_catalog.parquet'),
 Path('data/market/draftkings_market_catalog.parquet'),
 Path('data/market/fanduel_market_catalog.parquet'),
]
for p in FILES:
    print('\n###',p)
    if not p.exists():
        print('MISSING'); continue
    df=pd.read_parquet(p)
    print('shape',df.shape)
    print('columns',list(df.columns))
    for c in df.columns:
        lc=c.lower()
        if any(k in lc for k in ['market','method','outcome','price','odds','close','timestamp','fighter','event','bout']):
            vals=df[c].dropna().astype(str).unique()[:20]
            print(c, vals.tolist())
    print(df.head(5).to_string())
