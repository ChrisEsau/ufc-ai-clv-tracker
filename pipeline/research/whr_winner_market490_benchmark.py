#!/usr/bin/env python3
"""Leakage-safe canonical Whole-History Rating winner benchmark.

Research only. Market data is used only to select/evaluate the exact historical
complete-six-way cohort; it is never an input to WHR.

WHR specification:
  r_i(t) is latent fighter strength on natural-log-odds scale
  P(i beats j) = sigmoid(r_i-r_j)
  binary fight likelihood
  temporal random walk: r_next-r_prev ~ N(0, w^2 * dt_years)
  initial state anchor: N(0, 1)

The recovered prior research drift is w=2.75. Predictions are strictly causal:
for each UFC event date, WHR is fit using fights strictly before that date, all
fights on that date are predicted, then they become eligible for later dates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from pipeline.research.prefight_strength_elo import build_bouts

W = 2.75
ANCHOR_SD = 1.0
EPS = 1e-12
METHOD_KEYS = {'win_by_ko_tko_dq', 'win_by_submission', 'win_by_decision'}


def bin_metrics(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), EPS, 1-EPS)
    return {
        'n': int(len(y)),
        'accuracy': float(np.mean((p > .5) == (y > .5))),
        'brier': float(np.mean((p-y)**2)),
        'log_loss': float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p))),
        'mean_probability_actual_winner': float(np.mean(np.where(y == 1, p, 1-p))),
    }


def build_problem(hist: pd.DataFrame):
    """One latent state per fighter appearance; return objective structures."""
    n = len(hist)
    # states are 2 per bout: red=2k, blue=2k+1
    red_idx = 2*np.arange(n, dtype=np.int64)
    blue_idx = red_idx + 1
    y = (hist['winner'].to_numpy() == hist['red_fighter'].to_numpy()).astype(float)

    fighter_states = {}
    dates = pd.to_datetime(hist['date']).to_numpy(dtype='datetime64[D]')
    reds = hist['red_fighter'].astype(str).to_numpy()
    blues = hist['blue_fighter'].astype(str).to_numpy()
    for k in range(n):
        fighter_states.setdefault(reds[k], []).append((red_idx[k], dates[k]))
        fighter_states.setdefault(blues[k], []).append((blue_idx[k], dates[k]))

    first_idx = []
    prev_idx = []
    next_idx = []
    inv_var = []
    latest = {}
    for f, seq in fighter_states.items():
        seq.sort(key=lambda z: z[1])
        first_idx.append(seq[0][0])
        latest[f] = seq[-1][0]
        for (a, da), (b, db) in zip(seq[:-1], seq[1:]):
            days = max(1.0, float((db-da)/np.timedelta64(1, 'D')))
            dt_years = days / 365.25
            prev_idx.append(a); next_idx.append(b)
            inv_var.append(1.0 / (W*W*dt_years))

    return (
        red_idx, blue_idx, y,
        np.asarray(first_idx, dtype=np.int64),
        np.asarray(prev_idx, dtype=np.int64),
        np.asarray(next_idx, dtype=np.int64),
        np.asarray(inv_var, dtype=float),
        latest,
    )


def fit_whr(hist: pd.DataFrame, warm_by_fighter=None):
    red, blue, y, first, prev, nxt, inv_var, latest = build_problem(hist)
    size = 2*len(hist)
    x0 = np.zeros(size, dtype=float)
    if warm_by_fighter:
        # Initializing each fighter's entire path at its latest previous estimate
        # only accelerates optimization; it does not alter the objective.
        for f, idx in latest.items():
            if f in warm_by_fighter:
                # all appearances for this fighter can start at same warm value
                mask_r = hist['red_fighter'].astype(str).to_numpy() == f
                mask_b = hist['blue_fighter'].astype(str).to_numpy() == f
                x0[red[mask_r]] = warm_by_fighter[f]
                x0[blue[mask_b]] = warm_by_fighter[f]

    anchor_prec = 1.0/(ANCHOR_SD*ANCHOR_SD)

    def fg(x):
        z = x[red] - x[blue]
        p = expit(z)
        # negative binary log likelihood, stable form
        val = float(np.sum(np.logaddexp(0.0, z) - y*z))
        grad = np.zeros_like(x)
        err = p-y
        np.add.at(grad, red, err)
        np.add.at(grad, blue, -err)

        if first.size:
            xf = x[first]
            val += .5*anchor_prec*float(np.dot(xf, xf))
            np.add.at(grad, first, anchor_prec*xf)

        if prev.size:
            d = x[nxt]-x[prev]
            val += .5*float(np.dot(inv_var, d*d))
            g = inv_var*d
            np.add.at(grad, nxt, g)
            np.add.at(grad, prev, -g)
        return val, grad

    res = minimize(
        lambda x: fg(x), x0, method='L-BFGS-B', jac=True,
        options={'maxiter': 250, 'ftol': 1e-10, 'gtol': 1e-6, 'maxcor': 20},
    )
    x = res.x
    latest_rating = {f: float(x[idx]) for f, idx in latest.items()}
    return latest_rating, {'success': bool(res.success), 'nit': int(res.nit), 'fun': float(res.fun)}


def causal_predictions(bouts: pd.DataFrame, holdout_from='2025-01-01'):
    bouts = bouts[bouts.winner.notna()].copy().sort_values(['date','bout_id']).reset_index(drop=True)
    bouts['date'] = pd.to_datetime(bouts['date'])
    cutoff = pd.Timestamp(holdout_from)
    out = []
    warm = None

    # Fit only when a holdout event date needs predictions. History is strictly < date.
    for dt in bouts.loc[bouts.date >= cutoff, 'date'].drop_duplicates().sort_values():
        hist = bouts[bouts.date < dt]
        ratings, opt = fit_whr(hist, warm_by_fighter=warm)
        warm = ratings
        day = bouts[bouts.date == dt]
        for b in day.itertuples(index=False):
            rr = ratings.get(str(b.red_fighter), 0.0)
            rb = ratings.get(str(b.blue_fighter), 0.0)
            out.append({
                'date': b.date, 'fight_id': b.bout_id,
                'red_fighter': b.red_fighter, 'blue_fighter': b.blue_fighter,
                'winner': b.winner, 'p_red': float(expit(rr-rb)),
                'whr_red': rr, 'whr_blue': rb,
                'optimizer_success': opt['success'], 'optimizer_nit': opt['nit'],
            })
    return pd.DataFrame(out)


def main():
    bouts = build_bouts(pd.read_parquet('data/master/ufc_master.parquet'))
    pred = causal_predictions(bouts)

    m = pd.read_parquet('/tmp/historical_market_outcomes.parquet')
    m = m[m.market_key.isin(METHOD_KEYS) & m.outcome_side.isin(['red','blue'])].copy()
    counts = (m.dropna(subset=['fight_id']).groupby('fight_id')
              .apply(lambda x: x[['market_key','outcome_side']].drop_duplicates().shape[0], include_groups=False))
    complete = set(counts[counts == 6].index)
    d = pred[pred.fight_id.isin(complete)].copy()

    mm = m[m.fight_id.isin(set(d.fight_id))].copy()
    mm['ip'] = pd.to_numeric(mm.implied_probability, errors='coerce')
    mm = (mm.dropna(subset=['ip'])
          .sort_values(['fight_id','market_key','outcome_side'])
          .drop_duplicates(['fight_id','market_key','outcome_side'], keep='last'))
    side = mm.groupby(['fight_id','outcome_side']).ip.sum().unstack().dropna(subset=['red','blue'])
    side['market_p_red'] = side.red/(side.red+side.blue)
    d = d.merge(side[['market_p_red']], left_on='fight_id', right_index=True, how='inner')

    y = (d.winner == d.red_fighter).astype(float).to_numpy()
    summary = {
        'cohort': 'exact complete-six-way historical market overlap, 2025+ holdout',
        'whr': {
            'spec': 'canonical binary-logistic WHR, causal event-date refit',
            'temporal_drift_w': W,
            'delta_t_units': 'years',
            'initial_anchor_sd': ANCHOR_SD,
        },
        'matched_fights': int(len(d)),
        'whr_winner': bin_metrics(y, d.p_red.to_numpy()),
        'market_winner_from_six_way_no_vig': bin_metrics(y, d.market_p_red.to_numpy()),
        'optimizer_fail_rows': int((~d.optimizer_success.astype(bool)).sum()),
    }
    out = Path('data/diagnostics/whr_winner_market490'); out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out/'matched_winner_predictions.csv', index=False)
    pd.DataFrame([
        {'model':'Canonical WHR', **summary['whr_winner']},
        {'model':'Market no-vig winner from six-way', **summary['market_winner_from_six_way_no_vig']},
    ]).to_csv(out/'comparison.csv', index=False)
    with open(out/'summary.json','w') as f: json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
