"""Population audit of bantamweight FSR mean residuals and market probability compression.

Measurement only. No production FSR or Event Clock mechanics are changed.

For each eligible bantamweight fight:
- compare prefight FSR means with concrete offense/tendency means implied by realized outputs;
- exposure-normalize attempt rates to 15 minutes so early finishes do not masquerade as low pace;
- keep defender suppression/defense and attacker baselines fixed when solving the inverse;
- for fights with a two-way historical offered/legacy-consensus moneyline, compare market fair
  favorite probability, current prefight MC favorite probability, and fixed realized-output replay
  probability (with actual submission attempts and with submissions zeroed as a sensitivity).

The actual-output replay does not force historical knockdowns; KD/KO consequences remain frozen.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.diagnostics import canonical_c_validation as canonical
from pipeline.simulation.event_clock_mc_v2.diagnostics import kd_finishing_sequence_screen as seq
from pipeline.simulation.event_clock_mc_v2.diagnostics import weight_class_audit as wc_audit
from pipeline.simulation.event_clock_mc_v2.diagnostics.fsr_market_residual_audit import build_two_way_market, MARKET_PATH
from pipeline.simulation.event_clock_mc_v2.run_event_or_fight_frozen import (
    DETAILED_PATH_SEED_OFFSET, _submission_inputs, load_frozen_context,
)
from pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle import DEFAULT_BUNDLE_PATH as V2_BUNDLE_PATH
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import load_prefight_snapshots, historical_fighter_rows
from pipeline.simulation.event_clock_mc_v2.canonical_c import (
    load_kd_resistance_history, historical_kd_resistance_row, fight_with_kd_resistance,
)
from pipeline.simulation.event_clock_mc_v2.inference import predict_target_v3
from pipeline.simulation.event_mc_v1.diagnostics.population_validation import _fight

ROUND_STATS = Path('data/fight_details/ufc_round_stats.parquet')
DIVISION = 'bantamweight'
PER_15M = 900.0
OUT = Path('data/diagnostics/event_clock_mc_v2/bantamweight_fsr_population_inverse_market_audit')


def install_i10_b0() -> None:
    seq.INTERCEPT = 10.0; seq.DENOMINATOR = 12.0; seq.LOWER_CAP = -40.0; seq.UPPER_CAP = 10.0
    seq.ARMS = {'i10_b0': None}; seq._MODE = 'i10_b0'
    canonical.simulate_detailed_path = seq.sequence_simulate_detailed_path
    canonical.__dict__['simulate_detailed_path'] = seq.sequence_simulate_detailed_path


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-9, 1 - 1e-9))
    return math.log(p / (1 - p))


def pick_col(df: pd.DataFrame, *names: str, required: bool = True):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n in df.columns: return n
        if n.lower() in lower: return lower[n.lower()]
    if required: raise RuntimeError(f'missing columns {names}; available={list(df.columns)}')
    return None


def numsum(g: pd.DataFrame, *aliases: str) -> float:
    c = pick_col(g, *aliases, required=False)
    if c is None: return 0.0
    return float(pd.to_numeric(g[c], errors='coerce').fillna(0).sum())


def round_totals(g: pd.DataFrame) -> dict[str, float]:
    sig_a = numsum(g, 'sig_str_attempted','sig_str_att','significant_strikes_attempted','sig_attempted')
    sig_l = numsum(g, 'sig_str_landed','sig_str_land','significant_strikes_landed','sig_landed')
    grd_a = numsum(g, 'ground_attempted','ground_att','ground_sig_str_attempted','ground_sig_att')
    grd_l = numsum(g, 'ground_landed','ground_land','ground_sig_str_landed','ground_sig_land')
    td_a = numsum(g, 'td_attempted','td_att','takedowns_attempted')
    td_l = numsum(g, 'td_landed','td_land','takedowns_landed')
    sub = numsum(g, 'sub_attempts','sub_att','submission_attempts')
    return {
        'sig_att': sig_a, 'sig_land': sig_l,
        'standing_att': max(sig_a - grd_a, 0.0), 'standing_land': max(sig_l - grd_l, 0.0),
        'ground_att': grd_a, 'ground_land': grd_l,
        'td_att': td_a, 'td_land': td_l, 'sub_att': sub,
    }


def fighter_rows(fr: pd.DataFrame, mr: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    fighter = pick_col(fr, 'fighter_id')
    r = fr[fr[fighter].astype(str).eq(str(mr['r_id']))]
    b = fr[fr[fighter].astype(str).eq(str(mr['b_id']))]
    if not r.empty and not b.empty: return r, b
    corner = pick_col(fr, 'corner')
    return fr[fr[corner].astype(str).str.lower().eq('red')], fr[fr[corner].astype(str).str.lower().eq('blue')]


def inverse_side(att: dict, deff: dict, realized: dict, exposure: float) -> dict[str, float]:
    factor = PER_15M / max(float(exposure), 1.0)
    standing_rate = realized['standing_att'] * factor
    td_rate = realized['td_att'] * factor
    out = {
        'standing_striking_tendency_needed': standing_rate / max(float(deff['standing_striking_suppression']), 1e-9),
        'takedown_tendency_needed': td_rate / max(float(deff['takedown_suppression']), 1e-9),
    }
    if realized['standing_att'] > 0:
        acc = realized['standing_land'] / realized['standing_att']
        out['standing_striking_offense_needed'] = (
            logit(acc) - logit(float(att['standing_accuracy_baseline'])) + float(deff['standing_striking_defense'])
        )
    else:
        out['standing_striking_offense_needed'] = np.nan
    if realized['td_att'] > 0:
        comp = realized['td_land'] / realized['td_att']
        out['takedown_offense_needed'] = (
            logit(comp) - logit(float(att['takedown_completion_baseline'])) + float(deff['takedown_defense'])
        )
    else:
        out['takedown_offense_needed'] = np.nan
    return out


def market_favorite_p(summary_row: pd.Series, mkt: pd.Series) -> float:
    fav_side = str(mkt['favorite_side']) if 'favorite_side' in mkt else ('red' if str(mkt['favorite_id']) == str(summary_row['red_id']) else 'blue')
    p_red = float(summary_row['p_red_win'])
    return p_red if fav_side == 'red' else 1.0 - p_red


def actual_replay(mr: pd.Series, actual: dict[str, dict[str,float]], hist_summary: pd.Series,
                  context: dict, fsr: pd.DataFrame, kd_hist: pd.DataFrame,
                  paths: int, seed0: int, zero_subs: bool) -> tuple[float,float]:
    fid = str(mr['fight_id']); event_date = pd.Timestamp(mr['event_date']).normalize()
    rr, br = historical_fighter_rows(fsr, event_date=event_date, fight_id=fid,
                                     fighter_ids=(str(mr['r_id']), str(mr['b_id'])))
    target = pd.DataFrame([mr])
    inferred, _ = predict_target_v3(target, fsr, context['inference_models'], context['submission_scale'], context['conversion_offset'])
    _, conversion = _submission_inputs(inferred)
    rkd = historical_kd_resistance_row(kd_hist,event_date=event_date,fight_id=fid,fighter_id=str(mr['r_id']))
    bkd = historical_kd_resistance_row(kd_hist,event_date=event_date,fight_id=fid,fighter_id=str(mr['b_id']))
    fight = fight_with_kd_resistance(_fight(mr,context['fsr_all']),
        red_native_resistance=float(rkd['pre_rating']), blue_native_resistance=float(bkd['pre_rating']))
    budgets = {}
    for side in ('red','blue'):
        a = actual[side]
        budgets.update({
            f'{side}_standing_attempted': a['standing_att'], f'{side}_standing_landed': a['standing_land'],
            f'{side}_ground_attempted': a['ground_att'], f'{side}_ground_landed': a['ground_land'],
            f'{side}_td_attempted': a['td_att'], f'{side}_td_landed': a['td_land'],
            f'{side}_control': float(hist_summary[f'hist_{side}_control_seconds']),
        })
    exposure = max(float(mr['match_time_sec']), 1.0)
    sub_rates = {s: (0.0 if zero_subs else actual[s]['sub_att'] / exposure) for s in ('red','blue')}
    rw = 0
    for p in range(paths):
        r = canonical.simulate_detailed_path(fight, dict(budgets), sub_rates, conversion,
            context['judge_model'], context['judge_features'], seed0+p+DETAILED_PATH_SEED_OFFSET)
        rw += int(str(r['winner']) == 'red')
    return rw/paths, float(conversion)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--current-paths', type=int, default=100)
    ap.add_argument('--actual-paths', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20260823)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    args = ap.parse_args(); install_i10_b0()

    cohort, eligible = wc_audit.select_cohort(DIVISION, 100)
    cohort = cohort.copy(); cohort['fight_id'] = cohort['fight_id'].astype(str)
    print(f'BANTAMWEIGHT POPULATION INVERSE + MARKET AUDIT | cohort={len(cohort)} eligible={eligible}')
    current = canonical._simulate_c(cohort, args.current_paths, args.seed).copy()
    current['fight_id'] = current['fight_id'].astype(str)
    current_lookup = current.set_index('fight_id')

    rs = pd.read_parquet(ROUND_STATS).copy(); fcol = pick_col(rs,'fight_id','bout_id'); rs[fcol]=rs[fcol].astype(str)
    fsr = load_prefight_snapshots(canonical.FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    context = load_frozen_context(V2_BUNDLE_PATH); kd_hist = load_kd_resistance_history()
    market = build_two_way_market(MARKET_PATH).copy(); market['fight_id']=market['fight_id'].astype(str)
    market = market[market['fight_id'].isin(set(cohort['fight_id']))].set_index('fight_id')

    inv_rows=[]; market_rows=[]
    for i,mr in cohort.iterrows():
        fid=str(mr['fight_id']); fr=rs[rs[fcol].eq(fid)]; rfr,bfr=fighter_rows(fr,mr)
        if rfr.empty or bfr.empty: continue
        actual={'red':round_totals(rfr),'blue':round_totals(bfr)}
        exposure=float(mr['match_time_sec'])
        rr,br=historical_fighter_rows(fsr,event_date=mr['event_date'],fight_id=fid,
                                     fighter_ids=(str(mr['r_id']),str(mr['b_id'])))
        curr={'red':rr.to_dict(),'blue':br.to_dict()}
        need={'red':inverse_side(curr['red'],curr['blue'],actual['red'],exposure),
              'blue':inverse_side(curr['blue'],curr['red'],actual['blue'],exposure)}
        for side,name in (('red',mr['r_name']),('blue',mr['b_name'])):
            rec={'fight_id':fid,'event_date':mr['event_date'],'method':mr['method'],'is_decision':str(mr['method']).lower().startswith('decision'),
                 'side':side,'fighter':name,'exposure_sec':exposure}
            for trait in ('standing_striking_tendency','standing_striking_offense','takedown_tendency','takedown_offense'):
                rec[f'{trait}_prefight']=float(curr[side][trait])
                rec[f'{trait}_needed']=float(need[side][f'{trait}_needed']) if np.isfinite(need[side][f'{trait}_needed']) else np.nan
                rec[f'{trait}_delta']=rec[f'{trait}_needed']-rec[f'{trait}_prefight'] if np.isfinite(rec[f'{trait}_needed']) else np.nan
            rec.update({f'actual_{k}':v for k,v in actual[side].items()})
            inv_rows.append(rec)

        if fid in market.index and fid in current_lookup.index:
            mkt=market.loc[fid]; hs=current_lookup.loc[fid]
            fav_side='red' if str(mkt['favorite_id'])==str(mr['r_id']) else 'blue'
            current_fav=float(hs['p_red_win']) if fav_side=='red' else 1-float(hs['p_red_win'])
            p_red_actual, conv=actual_replay(mr,actual,hs,context,fsr,kd_hist,args.actual_paths,args.seed+i*1_000_000,False)
            p_red_zero,_=actual_replay(mr,actual,hs,context,fsr,kd_hist,args.actual_paths,args.seed+i*1_000_000,True)
            market_rows.append({'fight_id':fid,'event_date':mr['event_date'],'red':mr['r_name'],'blue':mr['b_name'],
                'favorite_side':fav_side,'market_favorite_fair_p':float(mkt['market_favorite_fair_p']),
                'favorite_won':float(mkt['favorite_won']),'current_prefight_mc_favorite_p':current_fav,
                'actual_output_mc_favorite_p':p_red_actual if fav_side=='red' else 1-p_red_actual,
                'actual_output_zero_sub_mc_favorite_p':p_red_zero if fav_side=='red' else 1-p_red_zero,
                'submission_conversion':conv})
        if (i+1)%10==0: print(f'processed {i+1}/{len(cohort)}')

    inv=pd.DataFrame(inv_rows); mk=pd.DataFrame(market_rows)
    summaries=[]
    for label,g in [('all',inv),('decisions',inv[inv['is_decision']]),('finishes',inv[~inv['is_decision']])]:
        for t in ('standing_striking_tendency','standing_striking_offense','takedown_tendency','takedown_offense'):
            x=pd.to_numeric(g[f'{t}_delta'],errors='coerce').dropna()
            summaries.append({'group':label,'trait':t,'n':len(x),'mean_delta':x.mean() if len(x) else np.nan,
                              'median_delta':x.median() if len(x) else np.nan,'mae_delta':x.abs().mean() if len(x) else np.nan,
                              'corr_prefight_needed':g[[f'{t}_prefight',f'{t}_needed']].corr().iloc[0,1] if g[f'{t}_needed'].notna().sum()>2 else np.nan})
    inv_summary=pd.DataFrame(summaries)

    market_summary=pd.DataFrame()
    if not mk.empty:
        rows=[]
        for c in ('current_prefight_mc_favorite_p','actual_output_mc_favorite_p','actual_output_zero_sub_mc_favorite_p'):
            d=mk[c]-mk['market_favorite_fair_p']
            rows.append({'arm':c,'n':len(mk),'mean_p':mk[c].mean(),'market_mean_p':mk['market_favorite_fair_p'].mean(),
                         'mean_error_pp':100*d.mean(),'mae_pp':100*d.abs().mean(),
                         'within_5pp':float((d.abs()<=.05).mean()),'within_10pp':float((d.abs()<=.10).mean()),
                         'corr_market':mk[[c,'market_favorite_fair_p']].corr().iloc[0,1]})
        market_summary=pd.DataFrame(rows)
        mk['current_gap_pp']=100*(mk['current_prefight_mc_favorite_p']-mk['market_favorite_fair_p'])
        mk['actual_gap_pp']=100*(mk['actual_output_mc_favorite_p']-mk['market_favorite_fair_p'])
        mk['actual_zero_sub_gap_pp']=100*(mk['actual_output_zero_sub_mc_favorite_p']-mk['market_favorite_fair_p'])

    args.out_dir.mkdir(parents=True,exist_ok=True)
    inv.to_csv(args.out_dir/'fighter_fight_inverse_residuals.csv',index=False)
    inv_summary.to_csv(args.out_dir/'inverse_summary.csv',index=False)
    mk.to_csv(args.out_dir/'market_fight_comparison.csv',index=False)
    market_summary.to_csv(args.out_dir/'market_summary.csv',index=False)
    current.to_csv(args.out_dir/'current_mc_summary.csv',index=False)
    print('\nINVERSE SUMMARY'); print(inv_summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nMARKET / CURRENT / ACTUAL-OUTPUT SUMMARY'); print(market_summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    if not mk.empty:
        print('\nPRICED FIGHTS'); print(mk[['red','blue','market_favorite_fair_p','current_prefight_mc_favorite_p','actual_output_mc_favorite_p','actual_output_zero_sub_mc_favorite_p']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))

if __name__=='__main__': main()
