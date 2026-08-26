from pathlib import Path
import pandas as pd

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH

OUT = Path('data/diagnostics/leavitt_brito_fsr')
OUT.mkdir(parents=True, exist_ok=True)

TARGET_DATE = pd.Timestamp('2026-06-06')
TARGETS = ['Jordan Leavitt', 'Joanderson Brito']

df = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
df['event_date'] = pd.to_datetime(df['event_date'])
rows = df[(df['fighter_name'].isin(TARGETS)) & (df['event_date'] == TARGET_DATE)].copy()
if set(rows['fighter_name']) != set(TARGETS):
    raise RuntimeError(f'Expected exact prefight rows for {TARGETS} on {TARGET_DATE.date()}, got {rows[["fighter_name","event_date"]].to_dict("records")}')

# Keep every FSR/runtime-relevant field so nothing is hidden.
rows = rows.sort_values('fighter_name')
rows.to_csv(OUT / 'full_prefight_rows.csv', index=False)

keys = [c for c in rows.columns if any(k in c.lower() for k in [
    'standing','takedown','ground','power','knockdown','escape','submission','durability','stamina','age'
])]
meta = [c for c in ['event_date','fight_id','fighter_id','fighter_name','opponent_name'] if c in rows.columns]
rows[meta + [c for c in keys if c not in meta]].to_csv(OUT / 'trait_comparison.csv', index=False)
print(rows[meta + [c for c in keys if c not in meta]].to_string(index=False))
