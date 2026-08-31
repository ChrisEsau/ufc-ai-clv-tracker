from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path('data/research/prop_mispricing')
SNAP_MARKET = Path('data/market/historical_market_outcomes.parquet')
SNAP_FV = Path('data/features/moneyline_feature_view.parquet')
CUR_MARKET = Path('/tmp/current_historical_market_outcomes.parquet')
CUR_FV = Path('/tmp/current_moneyline_feature_view.parquet')
EXACT_PRED = ROOT / 'xgboost_v5_exact_reproduction_test_predictions.csv'
EXACT_SUMMARY = ROOT / 'xgboost_v5_exact_reproduction_summary.json'
PRIOR_LEDGER = ROOT / 'v5_frozen_logit020_later_ledger.csv'
THRESH = 0.20

PARAMS = {
    'max_depth': 1,
    'eta': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 10,
    'lambda': 8.0,
    'alpha': 1.0,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 42,
    'nthread': 2,
}
ROUNDS = 300


def clip_p(p):
    return np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)


def logit(p):
    p = clip_p(p)
    return np.log(p / (1 - p))


def sigmoid(z):
    z = np.clip(np.asarray(z, float), -30, 30)
    return 1 / (1 + np.exp(-z))


def prep_market(path: Path) -> pd.DataFrame:
    m = pd.read_parquet(path).copy()
    m = m[(m['bookmaker'] == 'legacy_consensus') & (m['result_status'] == 'graded') & m['won'].notna()].copy()
    m['date'] = pd.to_datetime(m['date'], errors='coerce')
    m['won'] = m['won'].astype(bool).astype(int)
    m['implied_probability'] = pd.to_numeric(m['implied_probability'], errors='coerce')
    m['profit_per_100'] = pd.to_numeric(m['profit_per_100'], errors='coerce')
    m = m.dropna(subset=['date', 'implied_probability', 'profit_per_100']).copy()
    ml = m[m['market_key'] == 'moneyline'].copy()
    good = ml.groupby('fight_id').size()
    good = good[good == 2].index
    ml = ml[ml['fight_id'].isin(good)].copy()
    ml['market_overround'] = ml.groupby('fight_id')['implied_probability'].transform('sum')
    ml['fair_market_p'] = ml['implied_probability'] / ml['market_overround']
    return ml


def build_red(ml: pd.DataFrame, fv_path: Path, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fv = pd.read_parquet(fv_path).copy()
    missing = [c for c in features if c not in fv.columns]
    if missing:
        raise RuntimeError(f'missing frozen V5 features: {missing}')
    red = ml[ml['outcome_side'].astype(str).eq('red')].copy()
    df = red.merge(fv[['fight_id'] + features], on='fight_id', how='inner').sort_values(['date', 'fight_id']).copy()
    xraw = df[features + ['market_overround']].replace([np.inf, -np.inf], np.nan)
    return df, xraw


def fit_frozen(snapshot_ml: pd.DataFrame, features: list[str]):
    df, xraw = build_red(snapshot_ml, SNAP_FV, features)
    tr = df['date'] <= '2024-12-31'
    cols = features + ['market_overround']
    valid = [c for c in cols if xraw.loc[tr, c].notna().any()]
    if valid != cols:
        raise RuntimeError('frozen feature availability/order changed in snapshot')
    med = xraw.loc[tr, valid].median(numeric_only=True)
    xtr = xraw.loc[tr, valid].fillna(med).fillna(0.0)
    ytr = df.loc[tr, 'won'].astype(int).to_numpy()
    mtr = logit(df.loc[tr, 'fair_market_p'])
    dtr = xgb.DMatrix(xtr, label=ytr, base_margin=mtr, feature_names=valid)
    model = xgb.train(PARAMS, dtr, num_boost_round=ROUNDS, verbose_eval=False)
    return model, med, valid, df, xraw


def score_side(model, med, valid, ml: pd.DataFrame, fv_path: Path, features: list[str]) -> pd.DataFrame:
    df, xraw = build_red(ml, fv_path, features)
    x = xraw[valid].fillna(med).fillna(0.0)
    m = logit(df['fair_market_p'])
    d = xgb.DMatrix(x, base_margin=m, feature_names=valid)
    full_margin = model.predict(d, output_margin=True)
    p_red = sigmoid(full_margin)
    redp = df[['fight_id', 'date']].copy()
    redp['model_p_red'] = p_red
    side = ml.merge(redp[['fight_id', 'model_p_red']], on='fight_id', how='inner')
    side['model_p'] = np.where(side['outcome_side'].astype(str).eq('red'), side['model_p_red'], 1 - side['model_p_red'])
    side['edge'] = side['model_p'] - side['fair_market_p']
    side['signed_logit_residual'] = logit(side['model_p']) - logit(side['fair_market_p'])
    side['abs_logit_residual'] = np.abs(side['signed_logit_residual'])
    return side.sort_values(['date', 'fight_id', 'outcome_side']).reset_index(drop=True)


def summarize(g: pd.DataFrame) -> dict:
    n = len(g)
    profit = float(g['profit_units'].sum()) if n else 0.0
    if n:
        c = g['profit_units'].cumsum().to_numpy()
        peak = np.maximum.accumulate(np.r_[0.0, c])[1:]
        maxdd = float((peak - c).max())
    else:
        maxdd = 0.0
    return {
        'bets': int(n),
        'wins': int(g['won'].sum()) if n else 0,
        'losses': int(n - g['won'].sum()) if n else 0,
        'profit_units': profit,
        'roi': profit / n if n else None,
        'max_drawdown_units': maxdd,
    }


def main():
    exact_summary = json.loads(EXACT_SUMMARY.read_text())
    assert exact_summary['selected_candidate'] == 'top_50_pre2021_gain'
    features = list(exact_summary['selected_features'])
    assert len(features) == 50

    snap_ml = prep_market(SNAP_MARKET)
    model, med, valid, _, _ = fit_frozen(snap_ml, features)

    # Gate 1: recreated frozen model must reproduce authoritative exact side predictions.
    snap_side = score_side(model, med, valid, snap_ml, SNAP_FV, features)
    exact = pd.read_csv(EXACT_PRED)
    exact['date'] = pd.to_datetime(exact['date'])
    chk = snap_side.merge(exact[['fight_id', 'outcome_side', 'model_p']], on=['fight_id', 'outcome_side'], suffixes=('_new', '_exact'))
    chk = chk[chk['date'] >= '2025-01-01'].copy()
    if len(chk) != len(exact):
        raise RuntimeError(f'exact reproduction row mismatch: recreated={len(chk)} exact={len(exact)}')
    exact_max_err = float(np.max(np.abs(chk['model_p_new'] - chk['model_p_exact']))) if len(chk) else None
    if exact_max_err is None or exact_max_err > 1e-12:
        raise RuntimeError(f'exact V5 prediction reproduction failed: max_abs_error={exact_max_err}')

    cur_ml = prep_market(CUR_MARKET)
    cur_side = score_side(model, med, valid, cur_ml, CUR_FV, features)

    # Gate 2: current appended data must preserve the already-authoritative overlap.
    overlap = cur_side.merge(exact[['fight_id', 'outcome_side', 'model_p', 'fair_market_p']], on=['fight_id', 'outcome_side'], suffixes=('_current', '_exact'))
    overlap = overlap[overlap['date'] >= '2025-01-01'].copy()
    overlap_model_err = float(np.max(np.abs(overlap['model_p_current'] - overlap['model_p_exact']))) if len(overlap) else None
    overlap_market_err = float(np.max(np.abs(overlap['fair_market_p_current'] - overlap['fair_market_p_exact']))) if len(overlap) else None
    if len(overlap) != len(exact) or overlap_model_err is None or overlap_model_err > 1e-12 or overlap_market_err > 1e-12:
        raise RuntimeError(
            f'current-data overlap gate failed rows={len(overlap)}/{len(exact)} '
            f'model_err={overlap_model_err} market_err={overlap_market_err}'
        )

    cutoff = exact['date'].max()
    post = cur_side[cur_side['date'] > cutoff].copy()
    bets = post[(post['edge'] > 0) & (post['abs_logit_residual'] >= THRESH)].copy()
    bets = bets.sort_values(['date', 'fight_id']).reset_index(drop=True)
    bets['profit_units'] = np.where(bets['won'].astype(int).eq(1), bets['profit_per_100'].astype(float) / 100.0, -1.0)
    bets['market_role'] = np.where(bets['fair_market_p'] >= 0.5, 'favorite', 'underdog')
    bets['year'] = bets['date'].dt.year
    bets['cum_profit_units'] = bets['profit_units'].cumsum()

    prior = pd.read_csv(PRIOR_LEDGER)
    prior['date'] = pd.to_datetime(prior['date'])
    combined = pd.concat([prior, bets], ignore_index=True, sort=False).sort_values(['date', 'fight_id']).reset_index(drop=True)
    combined['cum_profit_units'] = combined['profit_units'].cumsum()
    combined['running_peak_units'] = np.maximum.accumulate(np.r_[0.0, combined['cum_profit_units'].to_numpy()])[1:]
    combined['drawdown_units'] = combined['running_peak_units'] - combined['cum_profit_units']

    by_card = []
    for (d, event), g in bets.groupby(['date', 'event_name'], sort=True):
        x = summarize(g)
        x['date'] = d.date().isoformat()
        x['event_name'] = event
        by_card.append(x)

    summary = {
        'experiment': 'frozen_v5_logit020_no_cold_start_current_continuation_v1',
        'rule': {'abs_logit_residual_min': THRESH, 'stake_units': 1.0, 'cold_start_filter': False},
        'frozen_snapshot': '7df1b61126be1f4e036b256d1c774c531b8a281f',
        'prior_cutoff': cutoff.date().isoformat(),
        'current_market_date_max': cur_ml['date'].max().date().isoformat() if len(cur_ml) else None,
        'current_feature_date_max': pd.to_datetime(pd.read_parquet(CUR_FV)['date'], errors='coerce').max().date().isoformat(),
        'post_cutoff_scored_fights': int(post['fight_id'].nunique()),
        'reproduction_gates': {
            'exact_snapshot_side_rows': int(len(chk)),
            'exact_snapshot_max_abs_model_p_error': exact_max_err,
            'current_overlap_side_rows': int(len(overlap)),
            'current_overlap_expected_rows': int(len(exact)),
            'current_overlap_max_abs_model_p_error': overlap_model_err,
            'current_overlap_max_abs_fair_market_p_error': overlap_market_err,
            'passed': True,
        },
        'continuation': summarize(bets),
        'combined': summarize(combined),
        'by_card': by_card,
        'note': 'No policy changes. Cold-start remains off. Threshold remains frozen at |logit residual| >= 0.20. Continuation is retrospective confirmation, not pristine prospective evidence.',
    }

    post.to_csv(ROOT / 'v5_frozen_logit020_2026_continuation_predictions.csv', index=False)
    bets.to_csv(ROOT / 'v5_frozen_logit020_2026_continuation_ledger.csv', index=False)
    combined.to_csv(ROOT / 'v5_frozen_logit020_combined_ledger.csv', index=False)
    pd.DataFrame(by_card).to_csv(ROOT / 'v5_frozen_logit020_2026_continuation_by_card.csv', index=False)
    (ROOT / 'v5_frozen_logit020_2026_continuation_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if len(bets):
        print('\nCONTINUATION BETS\n' + bets[['date','event_name','outcome_label','american_odds','fair_market_p','model_p','abs_logit_residual','won','profit_units']].to_string(index=False))


if __name__ == '__main__':
    main()
