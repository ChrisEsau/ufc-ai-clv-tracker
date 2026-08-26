from pathlib import Path
import pandas as pd

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
import pipeline.research.fsr_recency_cohort_shadow as rec

OUT = Path('data/diagnostics/fsr_last5_input_compare')
OUT.mkdir(parents=True, exist_ok=True)

TARGET_DATES = {
    'Santiago Ponzinibbio': pd.Timestamp('2026-07-25'),
    'Sam Patterson': pd.Timestamp('2026-07-25'),
    'Jan Blachowicz': pd.Timestamp('2026-08-01'),
    'Navajo Stirling': pd.Timestamp('2026-08-01'),
}
TARGETS = list(TARGET_DATES)

canonical = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH)

# Build research-only recency snapshots. No simulator run.
rec.WINDOW = 3
last3 = rec.build_variant(canonical, 'last3')
rec.WINDOW = 5
last5 = rec.build_variant(canonical, 'last3')


def pick(df, variant):
    x = df[df['fighter_name'].isin(TARGETS)].copy()
    x['event_date'] = pd.to_datetime(x['event_date'])
    selected = []
    for name, target_date in TARGET_DATES.items():
        f = x[(x['fighter_name'] == name) & (x['event_date'] <= target_date)].copy()
        if f.empty:
            raise RuntimeError(f'No prefight row for {name} on/before {target_date.date()}')
        row = f.sort_values('event_date').tail(1)
        if row.iloc[0]['event_date'] != target_date:
            raise RuntimeError(
                f'Expected target prefight row for {name} on {target_date.date()}, '
                f"got {row.iloc[0]['event_date'].date()}"
            )
        selected.append(row)
    out = pd.concat(selected, ignore_index=True)
    out['variant'] = variant
    return out


rows = pd.concat([
    pick(canonical, 'canonical'),
    pick(last3, 'last3'),
    pick(last5, 'last5'),
], ignore_index=True)

meta = ['variant', 'event_date', 'fight_id', 'fighter_id', 'fighter_name', 'opponent_name']
trait_cols = [c for c in rows.columns if c not in meta and pd.api.types.is_numeric_dtype(rows[c])]
keep = [c for c in trait_cols if any(k in c.lower() for k in [
    'standing', 'takedown', 'ground', 'power', 'knockdown', 'escape'
])]
cols = [c for c in meta if c in rows.columns] + keep
rows[cols].to_csv(OUT / 'comparison.csv', index=False)

base = rows[rows.variant == 'canonical'].set_index('fighter_name')
long = []
for _, r in rows[rows.variant != 'canonical'].iterrows():
    name = r['fighter_name']
    if name not in base.index:
        continue
    b = base.loc[name]
    for c in keep:
        try:
            long.append({
                'fighter_name': name,
                'variant': r['variant'],
                'trait': c,
                'canonical': float(b[c]),
                'value': float(r[c]),
                'delta': float(r[c]) - float(b[c]),
            })
        except Exception:
            pass
pd.DataFrame(long).to_csv(OUT / 'deltas.csv', index=False)
print(rows[cols].to_string(index=False))

# Trigger the existing FSR Recency Event Compare workflow via PR path filter.
