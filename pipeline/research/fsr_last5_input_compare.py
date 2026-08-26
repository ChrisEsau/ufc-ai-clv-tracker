from pathlib import Path
import pandas as pd

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
import pipeline.research.fsr_recency_cohort_shadow as rec

OUT = Path('data/diagnostics/fsr_last5_input_compare')
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = ['Santiago Ponzinibbio','Sam Patterson','Jan Blachowicz','Navajo Stirling']

canonical = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH)

# Build research-only recency snapshots. No simulator run.
rec.WINDOW = 3
last3 = rec.build_variant(canonical, 'last3')
rec.WINDOW = 5
last5 = rec.build_variant(canonical, 'last3')

# Use each target fighter's latest available prefight row in 2026 up to the target bouts.
# For these four fighters that corresponds to Ponzinibbio-Patterson and Blachowicz-Stirling.
def pick(df, variant):
    x = df[df['fighter_name'].isin(TARGETS)].copy()
    x['event_date'] = pd.to_datetime(x['event_date'])
    # Keep 2026 target-era prefight rows and select latest row per fighter.
    x = x[x['event_date'] <= pd.Timestamp('2026-08-26')]
    x = x.sort_values(['fighter_name','event_date']).groupby('fighter_name', as_index=False).tail(1)
    x['variant'] = variant
    return x

rows = pd.concat([pick(canonical,'canonical'), pick(last3,'last3'), pick(last5,'last5')], ignore_index=True)

meta = ['variant','event_date','fight_id','fighter_id','fighter_name','opponent_name']
trait_cols = [c for c in rows.columns if c not in meta and pd.api.types.is_numeric_dtype(rows[c])]
# Keep the V3-native rating-like fields plus runtime-relevant physical/damage inputs when present.
keep = [c for c in trait_cols if any(k in c.lower() for k in [
    'standing','takedown','ground','power','knockdown','escape'
])]
cols = [c for c in meta if c in rows.columns] + keep
rows[cols].to_csv(OUT/'comparison.csv', index=False)

# Long-form deltas for easier review.
base = rows[rows.variant=='canonical'].set_index('fighter_name')
long=[]
for _,r in rows[rows.variant!='canonical'].iterrows():
    name=r['fighter_name']
    if name not in base.index: continue
    b=base.loc[name]
    for c in keep:
        try:
            long.append({'fighter_name':name,'variant':r['variant'],'trait':c,'canonical':float(b[c]),'value':float(r[c]),'delta':float(r[c])-float(b[c])})
        except Exception:
            pass
pd.DataFrame(long).to_csv(OUT/'deltas.csv', index=False)
print(rows[cols].to_string(index=False))
