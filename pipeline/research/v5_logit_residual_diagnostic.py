from pathlib import Path
import json
import numpy as np
import pandas as pd

SRC=Path('data/research/prop_mispricing/v5_depth1_vs_depth2_oof.csv')
OUT=Path('data/research/prop_mispricing')
df=pd.read_csv(SRC)

def logit(p):
    p=np.clip(np.asarray(p,float),1e-9,1-1e-9)
    return np.log(p/(1-p))

df['prob_edge']=df.depth1_p-df.market_p
df['logit_residual']=logit(df.depth1_p)-logit(df.market_p)
df['abs_prob_edge']=df.prob_edge.abs()
df['abs_logit_residual']=df.logit_residual.abs()
df['direction']=np.sign(df.logit_residual)
df['direction_correct']=np.where(df.direction>0,df.won,np.where(df.direction<0,1-df.won,np.nan))
# Directional realized gap: outcome minus market expectation, oriented in model correction direction.
df['realized_directional_gap']=df.direction*(df.won-df.market_p)

# Equal-count deciles avoid arbitrary unit thresholds and allow direct monotonicity comparison.
def bucket_table(metric,label):
    x=df.copy()
    x['bucket']=pd.qcut(x[metric].rank(method='first'),10,labels=False)+1
    rows=[]
    for b,g in x.groupby('bucket'):
        rows.append({
            'metric':label,'decile':int(b),'n':len(g),
            'metric_min':float(g[metric].min()),'metric_mean':float(g[metric].mean()),'metric_max':float(g[metric].max()),
            'mean_abs_prob_edge':float(g.abs_prob_edge.mean()),
            'mean_abs_logit_residual':float(g.abs_logit_residual.mean()),
            'direction_accuracy':float(g.direction_correct.mean()),
            'realized_directional_gap':float(g.realized_directional_gap.mean()),
        })
    return pd.DataFrame(rows)

prob=bucket_table('abs_prob_edge','absolute_probability_edge')
logr=bucket_table('abs_logit_residual','absolute_logit_residual')
combined=pd.concat([prob,logr],ignore_index=True)
combined.to_csv(OUT/'v5_probability_vs_logit_residual_deciles.csv',index=False)

def monotonic_stats(t):
    # Correlation of bucket order with realized outcomes; diagnostic only, not selection.
    return {
        'spearman_decile_vs_direction_accuracy':float(t.decile.corr(t.direction_accuracy,method='spearman')),
        'spearman_decile_vs_realized_directional_gap':float(t.decile.corr(t.realized_directional_gap,method='spearman')),
        'top_decile_direction_accuracy':float(t.iloc[-1].direction_accuracy),
        'top_decile_realized_directional_gap':float(t.iloc[-1].realized_directional_gap),
        'bottom_decile_direction_accuracy':float(t.iloc[0].direction_accuracy),
        'bottom_decile_realized_directional_gap':float(t.iloc[0].realized_directional_gap),
    }
summary={
    'experiment':'v5_probability_edge_vs_logit_residual_diagnostic_v1',
    'source':str(SRC),
    'n':len(df),
    'note':'Measurement only. No model fitting, no threshold selection, no ROI tuning.',
    'probability_edge':monotonic_stats(prob),
    'logit_residual':monotonic_stats(logr),
}
json.dump(summary,open(OUT/'v5_probability_vs_logit_residual_summary.json','w'),indent=2)
print(json.dumps(summary,indent=2))
print('\nPROBABILITY EDGE DECILES\n',prob.to_string(index=False))
print('\nLOGIT RESIDUAL DECILES\n',logr.to_string(index=False))
