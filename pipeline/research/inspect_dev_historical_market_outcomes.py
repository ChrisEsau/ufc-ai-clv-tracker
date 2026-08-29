from pathlib import Path
import pandas as pd

p = Path('/tmp/historical_market_outcomes.parquet')
df = pd.read_parquet(p)
print('shape', df.shape)
print('columns', list(df.columns))
for c in df.columns:
    lc = c.lower()
    if any(k in lc for k in ['date','time','fighter','method','market','odds','price','prob','outcome','book','close','event','fight']):
        vals = df[c].dropna().astype(str).unique()[:30]
        print(f'{c}:', vals.tolist())
print('\nHEAD')
print(df.head(20).to_string())
