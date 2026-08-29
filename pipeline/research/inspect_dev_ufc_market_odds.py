#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

p=Path('/tmp/ufc_market_odds.parquet')
df=pd.read_parquet(p)
print('shape', df.shape)
print('columns', list(df.columns))
for c in df.columns:
    s=df[c]
    name=c.lower()
    if any(k in name for k in ['date','time','market','method','outcome','odds','book','fighter','close','price']):
        try:
            vals=s.dropna().astype(str).drop_duplicates().head(30).tolist()
            print(c, vals)
        except Exception as e:
            print(c, 'ERR', e)
print('\nHEAD')
print(df.head(12).to_string())
