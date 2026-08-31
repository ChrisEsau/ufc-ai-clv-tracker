import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

OOF_PATH = Path('data/research/prop_mispricing/v5_oof_2021_2024_positive_edge_fights.csv')
TEST_PATH = Path('data/research/prop_mispricing/v5_historical_2025_to_20260328_positive_edge_fights.csv')
OUT_DIR = Path('data/research/prop_mispricing')

FEATURES = ['fair_market_p', 'model_p', 'edge']
PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 1,
    'eta': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 1.0,
    'min_child_weight': 20,
    'lambda': 10.0,
    'alpha': 1.0,
    'seed': 42,
    'nthread': 2,
}
NUM_BOOST_ROUND = 300


def prep(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    for c in FEATURES + ['profit_units', 'bet_win']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=FEATURES + ['profit_units', 'bet_win']).sort_values(['date', 'fight_id']).reset_index(drop=True)


def fit_predict(train, test):
    dtrain = xgb.DMatrix(train[FEATURES], label=train['profit_units'], feature_names=FEATURES)
    dtest = xgb.DMatrix(test[FEATURES], feature_names=FEATURES)
    booster = xgb.train(PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
    return booster, booster.predict(dtest)


def summarize(df, mask, label):
    q = df.loc[mask].copy()
    n = len(q)
    wins = int(q['bet_win'].sum()) if n else 0
    profit = float(q['profit_units'].sum()) if n else 0.0
    return {
        'strategy': label,
        'bets': n,
        'wins': wins,
        'losses': n - wins,
        'win_rate': wins / n if n else None,
        'profit_units': profit,
        'roi': profit / n if n else None,
        'avg_v5_edge': float(q['edge'].mean()) if n else None,
        'avg_market_p': float(q['fair_market_p'].mean()) if n else None,
        'avg_predicted_return': float(q['predicted_return'].mean()) if n and 'predicted_return' in q else None,
    }


oof = prep(OOF_PATH)
test = prep(TEST_PATH)

# Strict expanding-year meta OOF. 2021 seeds the ROI model; predictions begin in 2022.
parts = []
for year in [2022, 2023, 2024]:
    train = oof[oof['date'].dt.year < year].copy()
    valid = oof[oof['date'].dt.year == year].copy()
    booster, pred = fit_predict(train, valid)
    valid['predicted_return'] = pred
    valid['meta_train_end_year'] = year - 1
    parts.append(valid)
meta_oof = pd.concat(parts, ignore_index=True).sort_values(['date', 'fight_id']).reset_index(drop=True)

# Natural zero-EV decision rule. No threshold tuning.
oof_rows = [
    summarize(meta_oof, meta_oof['predicted_return'] > 0, 'roi_meta_predicted_return_gt_0'),
    summarize(meta_oof, meta_oof['edge'] >= 0.075, 'v5_edge_ge_7_5pct'),
    summarize(meta_oof, np.ones(len(meta_oof), dtype=bool), 'all_v5_liked_sides'),
]

# Freeze the fixed specification after development and train it on all 2021-2024 OOF rows.
final_booster, test_pred = fit_predict(oof, test)
test['predicted_return'] = test_pred

test_rows = [
    summarize(test, test['predicted_return'] > 0, 'roi_meta_predicted_return_gt_0'),
    summarize(test, test['edge'] >= 0.075, 'v5_edge_ge_7_5pct'),
    summarize(test, np.ones(len(test), dtype=bool), 'all_v5_liked_sides'),
]

summary = {
    'experiment': 'frozen_v5_roi_meta_xgboost_v1',
    'target': 'realized 1-unit return of the V5-liked side',
    'features': FEATURES,
    'decision_rule': 'bet iff predicted_return > 0; fixed a priori, not tuned',
    'params': PARAMS,
    'num_boost_round': NUM_BOOST_ROUND,
    'development_protocol': 'expanding chronological year OOF: train 2021->test 2022; train 2021-22->test 2023; train 2021-23->test 2024',
    'development_fights_available': int(len(oof)),
    'development_meta_oof_fights': int(len(meta_oof)),
    'test_period': '2025-01-01 through 2026-03-28',
    'test_fights': int(len(test)),
    'development_results': oof_rows,
    'untouched_test_results': test_rows,
    'feature_importance_gain': final_booster.get_score(importance_type='gain'),
}

meta_oof.to_csv(OUT_DIR / 'v5_roi_meta_xgboost_oof_predictions.csv', index=False)
test.to_csv(OUT_DIR / 'v5_roi_meta_xgboost_2025_to_20260328_predictions.csv', index=False)
pd.DataFrame([dict(period='2022_2024_meta_oof', **r) for r in oof_rows] + [dict(period='2025_to_20260328', **r) for r in test_rows]).to_csv(OUT_DIR / 'v5_roi_meta_xgboost_comparison.csv', index=False)
with open(OUT_DIR / 'v5_roi_meta_xgboost_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
