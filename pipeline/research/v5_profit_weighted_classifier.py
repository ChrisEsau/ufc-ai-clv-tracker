import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

OUT = Path('data/research/prop_mispricing')
DEV_PATH = OUT / 'v5_oof_2021_2024_positive_edge_fights.csv'
TEST_PATH = OUT / 'v5_historical_2025_to_20260328_positive_edge_fights.csv'
SUMMARY_PATH = OUT / 'v5_profit_weighted_classifier_summary.json'
COMPARISON_PATH = OUT / 'v5_profit_weighted_classifier_comparison.csv'
OOF_PATH = OUT / 'v5_profit_weighted_classifier_meta_oof.csv'
TEST_PRED_PATH = OUT / 'v5_profit_weighted_classifier_2025_to_20260328.csv'

FEATURES = ['fair_market_p', 'model_p', 'edge']
PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
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


def load_dev():
    df = pd.read_csv(DEV_PATH)
    df['date'] = pd.to_datetime(df['date'])
    for c in FEATURES + ['bet_win', 'profit_per_100', 'profit_units']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=FEATURES + ['bet_win', 'profit_per_100', 'profit_units']).sort_values(['date', 'fight_id']).reset_index(drop=True)


def load_test():
    df = pd.read_csv(TEST_PATH)
    df['date'] = pd.to_datetime(df['date'])
    for c in FEATURES + ['bet_win', 'profit_per_100', 'profit_units']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2026-03-28')]
    return df.dropna(subset=FEATURES + ['bet_win', 'profit_per_100', 'profit_units']).sort_values(['date', 'fight_id']).reset_index(drop=True)


def fit_predict(train, test):
    # Economic swing of a 1u wager: loss outcome=-1; win outcome=+payout.
    # Weight is outcome-independent and known pre-fight: 1 + potential payout units.
    w = 1.0 + train['profit_per_100'].to_numpy(float) / 100.0
    dtrain = xgb.DMatrix(train[FEATURES], label=train['bet_win'].astype(int), weight=w, feature_names=FEATURES)
    dtest = xgb.DMatrix(test[FEATURES], feature_names=FEATURES)
    booster = xgb.train(PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
    return booster, booster.predict(dtest)


def strategy_row(period, name, df, mask, score_col='weighted_win_score'):
    q = df.loc[mask].copy()
    n = len(q)
    wins = int(q['bet_win'].sum()) if n else 0
    profit = float(q['profit_units'].sum()) if n else 0.0
    return {
        'period': period,
        'strategy': name,
        'bets': n,
        'wins': wins,
        'losses': n - wins,
        'win_rate': wins / n if n else None,
        'profit_units': profit,
        'roi': profit / n if n else None,
        'avg_v5_edge': float(q['edge'].mean()) if n else None,
        'avg_market_p': float(q['fair_market_p'].mean()) if n else None,
        'avg_weighted_win_score': float(q[score_col].mean()) if n and score_col in q else None,
    }


dev = load_dev()
test = load_test()

# Expanding chronological meta-OOF. 2021 is warm-up only.
oof_parts = []
for test_year in [2022, 2023, 2024]:
    tr = dev[dev['date'].dt.year < test_year].copy()
    te = dev[dev['date'].dt.year == test_year].copy()
    booster, pred = fit_predict(tr, te)
    te['weighted_win_score'] = pred
    te['meta_train_through'] = test_year - 1
    oof_parts.append(te)
meta_oof = pd.concat(oof_parts, ignore_index=True).sort_values(['date', 'fight_id']).reset_index(drop=True)

# Freeze the same specification on all 2021-2024 and evaluate subsequent 2025-Mar 28 2026.
final_booster, test_score = fit_predict(dev, test)
test = test.copy()
test['weighted_win_score'] = test_score

rows = []
for period, frame in [('2022_2024_meta_oof', meta_oof), ('2025_to_20260328', test)]:
    rows.append(strategy_row(period, 'profit_weighted_classifier_score_ge_0_5', frame, frame['weighted_win_score'] >= 0.5))
    rows.append(strategy_row(period, 'v5_edge_ge_7_5pct', frame, frame['edge'] >= 0.075))
    rows.append(strategy_row(period, 'all_v5_liked_sides', frame, np.ones(len(frame), dtype=bool)))

comparison = pd.DataFrame(rows)
comparison.to_csv(COMPARISON_PATH, index=False)
meta_oof.to_csv(OOF_PATH, index=False)
test.to_csv(TEST_PRED_PATH, index=False)

importance = final_booster.get_score(importance_type='gain')
summary = {
    'experiment': 'frozen_v5_profit_weighted_classifier_v1',
    'target': 'binary win/loss of the V5-liked side',
    'sample_weight': '1 + potential_win_units; outcome-independent economic swing of a 1u wager',
    'features': FEATURES,
    'decision_rule': 'bet iff profit-weighted classifier score >= 0.5; fixed a priori, not threshold-tuned',
    'params': PARAMS,
    'num_boost_round': NUM_BOOST_ROUND,
    'development_protocol': 'expanding chronological year OOF: train 2021->test 2022; train 2021-22->test 2023; train 2021-23->test 2024',
    'development_fights_available': len(dev),
    'development_meta_oof_fights': len(meta_oof),
    'evaluation_period': '2025-01-01 through 2026-03-28; previously examined in prior diagnostics, so not described as untouched here',
    'evaluation_fights': len(test),
    'results': rows,
    'feature_importance_gain': importance,
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

print(json.dumps(summary, indent=2))
print('\nCOMPARISON')
print(comparison.to_string(index=False))
