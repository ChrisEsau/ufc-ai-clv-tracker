"""Leakage-safe Stage-2 recency attribution for bantamweight FSR V3.

Keeps the validated current priors and cumulative population baselines fixed,
but changes only each fighter's historical evidence retention:

- current cumulative FSR V3 publication;
- recent 5 prior fighter observations;
- recent 3 prior fighter observations;
- exponential time decay with 365-day half-life.

The likelihood families remain the validated V3 NB2 / Beta-Binomial models.
Same-event updates remain delayed. Market probability is evaluation-only and is
never used to fit a state. No production FSR publication is modified.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.fsr_v3.replay import rate_families as rf
from pipeline.fsr_v3.replay import paired_effectiveness as pe
from pipeline.fsr_v3.replay.math import (
    nb2_log_likelihood,
    beta_binomial_log_likelihood,
    normalize_log_weights,
    weighted_mean_sd,
)
from pipeline.simulation.event_clock_mc_v2.diagnostics.bantamweight_fsr_shrinkage_attribution import (
    KEYS,
    DIVISION,
    actual_by_fight,
    evaluate_variant,
)
from pipeline.simulation.event_clock_mc_v2.diagnostics.fsr_market_residual_audit import (
    build_two_way_market,
    MARKET_PATH,
)

OUT = Path('data/diagnostics/event_clock_mc_v2/bantamweight_fsr_recency_attribution')


def _history_state(history, event_date, mode):
    """Sum strictly-prior observation log-likelihoods under one retention rule."""
    if not history:
        return None
    if mode == 'recent3':
        selected = history[-3:]
        weights = [1.0] * len(selected)
    elif mode == 'recent5':
        selected = history[-5:]
        weights = [1.0] * len(selected)
    elif mode == 'decay365':
        selected = history
        now = pd.Timestamp(event_date)
        weights = [0.5 ** (max((now - pd.Timestamp(d)).days, 0) / 365.0) for d, _ in selected]
    else:
        raise ValueError(mode)
    out = np.zeros_like(selected[0][1], dtype=float)
    for w, (_, ll) in zip(weights, selected):
        out += float(w) * ll
    out -= np.max(out)
    return out


def _stable_ll(ll):
    ll = np.asarray(ll, dtype=float)
    return ll - np.max(ll)


def replay_tendency_recency(fights, spec, mode):
    grid = np.linspace(spec.tendency_grid_min, spec.tendency_grid_max, spec.tendency_grid_points)
    histories = {}
    pop_y, pop_e, params, rows = [], [], None, []
    for event_date, batch in fights.groupby('event_date', sort=True):
        params = rf._fit_tendency_population(pop_y, pop_e, params, spec)
        q_pop, alpha = params
        prior_shape = max(q_pop * spec.tendency_prior_seconds / 900.0, 1e-9)
        prior_lp = rf._log_gamma_prior(grid, q_pop, prior_shape)
        pending = []
        for rec in batch.to_dict('records'):
            fighter = str(rec['fighter_id'])
            y = float(rec['numerator']); exposure = float(rec['exposure_seconds'])
            lp = prior_lp.copy()
            state = _history_state(histories.get(fighter, []), event_date, mode)
            if state is not None:
                lp += state
            w = normalize_log_weights(lp)
            pre_mean, pre_sd = weighted_mean_sd(grid, w)
            obs_ll = None
            if exposure > 0:
                obs_ll = nb2_log_likelihood(y, exposure / 900.0 * grid, alpha)
            rows.append({
                **rec,
                'trait': spec.tendency_trait,
                'pre_rating': pre_mean,
                'pre_posterior_sd': pre_sd,
                'numerator': y,
                'denominator': exposure,
                'population_rate_15m': q_pop,
                'observation_alpha': alpha,
            })
            pending.append((fighter, y, exposure, obs_ll))
        for fighter, y, exposure, obs_ll in pending:
            if exposure <= 0 or obs_ll is None:
                continue
            histories.setdefault(fighter, []).append((event_date, _stable_ll(obs_ll)))
            pop_y.append(y); pop_e.append(exposure)
    return pd.DataFrame(rows).sort_values(['event_date','fight_id','fighter_id']).reset_index(drop=True)


def replay_suppression_recency(tendency_history, spec, mode):
    grid = np.linspace(spec.suppression_grid_min, spec.suppression_grid_max, spec.suppression_grid_points)
    source = tendency_history.copy()
    source['expected_attempts'] = source['denominator'].astype(float) / 900.0 * source['pre_rating'].astype(float)
    histories = {}; pop_y = []; pop_expected = []; params = None; rows = []
    for event_date, batch in source.groupby('event_date', sort=True):
        params = rf._fit_suppression_population(pop_y, pop_expected, params, spec)
        s_pop, alpha = params
        prior_lp = rf._log_gamma_prior(grid, s_pop, spec.suppression_prior_shape)
        pending = []
        for rec in batch.to_dict('records'):
            defender = str(rec['opponent_id'])
            y = float(rec['numerator']); expected = float(rec['expected_attempts'])
            lp = prior_lp.copy()
            state = _history_state(histories.get(defender, []), event_date, mode)
            if state is not None:
                lp += state
            w = normalize_log_weights(lp)
            pre_mean, pre_sd = weighted_mean_sd(grid, w)
            obs_ll = None
            if expected > 0:
                obs_ll = nb2_log_likelihood(y, expected * grid, alpha)
            rows.append({
                'event_date': rec['event_date'], 'fight_id': rec['fight_id'],
                'fighter_id': defender, 'fighter_name': rec['opponent_name'],
                'opponent_id': str(rec['fighter_id']), 'opponent_name': rec['fighter_name'],
                'trait': spec.suppression_trait, 'pre_rating': pre_mean,
                'pre_posterior_sd': pre_sd,
            })
            pending.append((defender, y, expected, obs_ll))
        for defender, y, expected, obs_ll in pending:
            if expected <= 0 or obs_ll is None:
                continue
            histories.setdefault(defender, []).append((event_date, _stable_ll(obs_ll)))
            pop_y.append(y); pop_expected.append(expected)
    return pd.DataFrame(rows).sort_values(['event_date','fight_id','fighter_id']).reset_index(drop=True)


def replay_effectiveness_recency(fights, spec, mode):
    grid = np.linspace(spec.grid_min, spec.grid_max, spec.grid_points)
    off_prior = pe._normal_prior(grid, spec.sigma_offense)
    def_prior = pe._normal_prior(grid, spec.sigma_defense)
    off_hist = {}; def_hist = {}; pop_y = []; pop_n = []; beta = None; rows = []
    for event_date, batch in fights.groupby('event_date', sort=True):
        beta = pe._fit_population_beta(pop_y, pop_n, spec.rho, beta)
        pending = []
        for rec in batch.to_dict('records'):
            attacker = str(rec['fighter_id']); defender = str(rec['opponent_id'])
            y = float(rec['landed']); n = float(rec['attempted'])
            off_lp = off_prior.copy(); def_lp = def_prior.copy()
            s = _history_state(off_hist.get(attacker, []), event_date, mode)
            if s is not None: off_lp += s
            s = _history_state(def_hist.get(defender, []), event_date, mode)
            if s is not None: def_lp += s
            off_mean, off_sd = weighted_mean_sd(grid, normalize_log_weights(off_lp))
            def_mean, def_sd = weighted_mean_sd(grid, normalize_log_weights(def_lp))
            off_ll = def_ll = None
            if n > 0:
                off_ll = beta_binomial_log_likelihood(y, n, expit(beta + grid - def_mean), spec.rho)
                def_ll = beta_binomial_log_likelihood(y, n, expit(beta + off_mean - grid), spec.rho)
            rows.append({
                'event_date': rec['event_date'], 'fight_id': rec['fight_id'],
                'fighter_id': attacker, 'fighter_name': rec['fighter_name'],
                'opponent_id': defender, 'opponent_name': rec['opponent_name'],
                'trait': spec.offense_trait, 'pre_rating': off_mean, 'pre_posterior_sd': off_sd,
            })
            rows.append({
                'event_date': rec['event_date'], 'fight_id': rec['fight_id'],
                'fighter_id': defender, 'fighter_name': rec['opponent_name'],
                'opponent_id': attacker, 'opponent_name': rec['fighter_name'],
                'trait': spec.defense_trait, 'pre_rating': def_mean, 'pre_posterior_sd': def_sd,
            })
            pending.append((attacker, defender, y, n, off_ll, def_ll))
        for attacker, defender, y, n, off_ll, def_ll in pending:
            if n <= 0 or off_ll is None or def_ll is None:
                continue
            off_hist.setdefault(attacker, []).append((event_date, _stable_ll(off_ll)))
            def_hist.setdefault(defender, []).append((event_date, _stable_ll(def_ll)))
            pop_y.append(y); pop_n.append(n)
    hist = pd.DataFrame(rows)
    return hist.sort_values(['event_date','fight_id','fighter_id','trait']).reset_index(drop=True)


def _trait(frame, trait):
    return frame[frame['trait'].eq(trait)][KEYS+['pre_rating']].rename(columns={'pre_rating': trait})


def build_recency_variant(sources, cfg, mode):
    ss = rf.standing_spec(cfg); ts = rf.takedown_spec(cfg)
    sth = replay_tendency_recency(sources['standing_rate'], ss, mode)
    ssh = replay_suppression_recency(sth, ss, mode)
    tth = replay_tendency_recency(sources['td_rate'], ts, mode)
    tsh = replay_suppression_recency(tth, ts, mode)
    seh = replay_effectiveness_recency(sources['standing_eff'], pe.standing_effectiveness_spec(cfg), mode)
    teh = replay_effectiveness_recency(sources['td_eff'], pe.takedown_effectiveness_spec(cfg), mode)
    pieces = [
        sth[KEYS+['pre_rating']].rename(columns={'pre_rating':'standing_striking_tendency'}),
        ssh[KEYS+['pre_rating']].rename(columns={'pre_rating':'standing_striking_suppression'}),
        _trait(seh,'standing_striking_offense'), _trait(seh,'standing_striking_defense'),
        tth[KEYS+['pre_rating']].rename(columns={'pre_rating':'takedown_tendency'}),
        tsh[KEYS+['pre_rating']].rename(columns={'pre_rating':'takedown_suppression'}),
        _trait(teh,'takedown_offense'), _trait(teh,'takedown_defense'),
    ]
    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p,on=KEYS,how='outer',validate='one_to_one')
    return out


def main():
    cfg = FSRV3Config()
    master = pd.read_parquet(MASTER_PATH).drop_duplicates('fight_id').copy()
    master['fight_id'] = master['fight_id'].astype(str)
    master['event_date'] = pd.to_datetime(master['date'],errors='coerce').dt.normalize()
    cohort = master[master['division'].astype(str).str.strip().str.lower().eq(DIVISION)].copy()

    base = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    base['fight_id'] = base['fight_id'].astype(str); base['fighter_id'] = base['fighter_id'].astype(str)
    base['event_date'] = pd.to_datetime(base['event_date']).dt.normalize()
    valid = set(base.groupby('fight_id').size().loc[lambda s:s==2].index.astype(str))
    cohort = cohort[cohort['fight_id'].isin(valid) & cohort['match_time_sec'].notna()].copy()
    actual = actual_by_fight(cohort)
    market = build_two_way_market(MARKET_PATH).copy(); market['fight_id'] = market['fight_id'].astype(str)
    market = market[market['fight_id'].isin(set(cohort['fight_id']))]

    paired = build_paired_rounds()
    sources = {
        'standing_rate': rf.build_rate_fighter_fights(rf.standing_spec(cfg), paired_rounds=paired),
        'td_rate': rf.build_rate_fighter_fights(rf.takedown_spec(cfg), paired_rounds=paired),
        'standing_eff': pe.build_effectiveness_fighter_fights(pe.standing_effectiveness_spec(cfg), paired_rounds=paired),
        'td_eff': pe.build_effectiveness_fighter_fights(pe.takedown_effectiveness_spec(cfg), paired_rounds=paired),
    }

    current_cols = KEYS + [
        'standing_striking_tendency','standing_striking_suppression',
        'standing_striking_offense','standing_striking_defense',
        'takedown_tendency','takedown_suppression','takedown_offense','takedown_defense',
    ]
    variants = [('current_cumulative', base[current_cols].copy())]
    for mode in ('recent5','recent3','decay365'):
        print(f'building {mode}...')
        variants.append((mode, build_recency_variant(sources,cfg,mode)))

    all_detail=[]; all_summary=[]; all_fights=[]; market_rows=[]
    for name,variant in variants:
        d,s,f,m = evaluate_variant(name,variant,cohort,actual,base,market)
        all_detail.append(d); all_summary.append(s); all_fights.append(f); market_rows.append(m)
    detail=pd.concat(all_detail,ignore_index=True); summary=pd.concat(all_summary,ignore_index=True)
    fights=pd.concat(all_fights,ignore_index=True); msum=pd.DataFrame(market_rows)

    OUT.mkdir(parents=True,exist_ok=True)
    detail.to_csv(OUT/'runtime_output_fit_detail.csv',index=False)
    summary.to_csv(OUT/'runtime_output_fit_summary.csv',index=False)
    fights.to_csv(OUT/'fight_runtime_separation.csv',index=False)
    msum.to_csv(OUT/'market_separation_summary.csv',index=False)
    print('\nBANTAMWEIGHT FSR RECENCY ATTRIBUTION — LEAKAGE SAFE')
    print(f'fights={cohort.fight_id.nunique()} priced={market.fight_id.nunique()}')
    print('\nRUNTIME OUTPUT FIT')
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nMARKET SEPARATION')
    print(msum.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nInterpretation: recency is useful only if it improves next-fight output fit and market-strength separation without simply inflating runtime variance.')

if __name__ == '__main__':
    main()
