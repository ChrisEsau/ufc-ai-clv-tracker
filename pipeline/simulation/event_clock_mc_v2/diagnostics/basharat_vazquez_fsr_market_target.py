"""Measurement-only FSR-to-market target audit for Basharat vs Vazquez.

Starting from the exact prefight FSR V3 state, interpolate only the four traits
previously inverted from realized outputs toward those inverse-needed values:
standing tendency, standing offense, TD tendency, and TD offense.  Frozen Event
Clock mechanics, all other FSR traits, submissions, damage, judging, and market
are untouched.  The market is used only as the target probability after the
simulation; it is never used to fit FSR.

Question: what fraction of the observed-output FSR correction is required for
the frozen MC to reach the historical offered/legacy-consensus fair favorite
probability for Javid Basharat vs Gianni Vazquez?
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_UNCERTAINTY_PATH
from pipeline.simulation.event_clock_mc_v2.diagnostics import canonical_c_validation as canonical
from pipeline.simulation.event_clock_mc_v2.diagnostics import kd_finishing_sequence_screen as seq
from pipeline.simulation.event_clock_mc_v2.diagnostics import weight_class_audit as wc_audit
from pipeline.simulation.event_clock_mc_v2.diagnostics.fsr_market_residual_audit import build_two_way_market, MARKET_PATH
from pipeline.simulation.event_clock_mc_v2.feature_builder import build_sampled_fight_feature_rows_v3
from pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle import DEFAULT_BUNDLE_PATH as V2_BUNDLE_PATH
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    historical_fighter_rows, historical_uncertainty_rows, initialize_path_matchup,
    load_prefight_snapshots,
)
from pipeline.simulation.event_clock_mc_v2.inference import load_submission_baseline_v3, predict_feature_frame_v3
from pipeline.simulation.event_clock_mc_v2.canonical_c import (
    fight_with_kd_resistance, historical_kd_resistance_row,
    load_kd_resistance_history, sample_kd_resistance_latent,
)
from pipeline.simulation.event_clock_mc_v2.run_event_or_fight_frozen import (
    DETAILED_PATH_SEED_OFFSET, EPISTEMIC_SEED_OFFSET, _draw_budgets,
    _submission_inputs, load_frozen_context,
)
from pipeline.simulation.event_mc_v1.diagnostics.population_validation import _fight

FIGHT_ID = '86df0e75f41784d8'
OUT = Path('data/diagnostics/event_clock_mc_v2/basharat_vazquez_fsr_market_target')
SEED = 20260824
PATHS = 1500

# Exact inverse convention from the prior population inverse audit.
INVERSE = {
    'Javid Basharat': {
        'standing_striking_tendency': 78.260177,
        'standing_striking_offense': 0.658153,
        'takedown_tendency': 8.540082,
        'takedown_offense': 0.010714,
    },
    'Gianni Vazquez': {
        'standing_striking_tendency': 92.669708,
        'standing_striking_offense': -0.592432,
        'takedown_tendency': 0.0,
        # Zero attempts means TD offense is not identified; leave it at prefight.
    },
}
TRAITS = ('standing_striking_tendency','standing_striking_offense','takedown_tendency','takedown_offense')


def install_i10_b0() -> None:
    seq.INTERCEPT=10.0; seq.DENOMINATOR=12.0; seq.LOWER_CAP=-40.0; seq.UPPER_CAP=10.0
    seq.ARMS={'i10_b0':None}; seq._MODE='i10_b0'
    canonical.simulate_detailed_path = seq.sequence_simulate_detailed_path


def patched_row(row: pd.Series, fighter_name: str, alpha: float, active_traits: set[str]) -> pd.Series:
    out=row.copy()
    target=INVERSE[fighter_name]
    for trait in active_traits:
        if trait not in target:
            continue
        current=float(row[trait]); needed=float(target[trait])
        out[trait]=current + float(alpha)*(needed-current)
    return out


def patched_uncertainty(unc: pd.DataFrame, row: pd.Series, fighter_name: str, alpha: float, active_traits: set[str]) -> pd.DataFrame:
    out=unc.copy()
    target=INVERSE[fighter_name]
    # Only tendency traits are sampled epistemically. Match their posterior mean
    # to the altered FSR mean while retaining the frozen posterior SD/multiplier.
    # A zero endpoint is a valid deterministic trait value but cannot be Gamma-
    # projected, so disable epistemic sampling for that trait exactly at zero.
    for trait in ('standing_striking_tendency','takedown_tendency'):
        if trait not in active_traits or trait not in target:
            continue
        current=float(row[trait]); needed=float(target[trait])
        mean=current + float(alpha)*(needed-current)
        mask=out['trait'].astype(str).eq(trait)
        out.loc[mask,'posterior_mean']=mean
        if mean <= 0.0:
            out.loc[mask,'sampling_enabled']=False
            out.loc[mask,'variance_multiplier']=0.0
    return out


def simulate_arm(master_row: pd.Series, red_row: pd.Series, blue_row: pd.Series,
                 red_unc: pd.DataFrame, blue_unc: pd.DataFrame, context: dict,
                 submission_baseline: pd.DataFrame, red_kd: pd.Series, blue_kd: pd.Series,
                 paths: int, seed0: int) -> dict:
    base_fight=_fight(master_row,context['fsr_all'])
    wins={'red':0,'blue':0}; methods={}
    for path in range(paths):
        seed=seed0+path
        erng=np.random.default_rng(seed+EPISTEMIC_SEED_OFFSET)
        matchup=initialize_path_matchup(red_row,blue_row,red_unc,blue_unc,rng=erng,sample_epistemic=True)
        path_fight=fight_with_kd_resistance(
            base_fight,
            red_native_resistance=sample_kd_resistance_latent(red_kd,erng),
            blue_native_resistance=sample_kd_resistance_latent(blue_kd,erng),
        )
        features=build_sampled_fight_feature_rows_v3(
            master_row, red_record=red_row.to_dict(), blue_record=blue_row.to_dict(),
            red_traits=matchup.red, blue_traits=matchup.blue,
        )
        pair,control=predict_feature_frame_v3(
            features,context['inference_models'],context['submission_scale'],context['conversion_offset'],
            submission_baseline=submission_baseline,
        )
        subs,conv=_submission_inputs(pair)
        budgets=_draw_budgets(pair,control.iloc[0],context,np.random.default_rng(seed))
        res=canonical.simulate_detailed_path(
            path_fight,budgets,subs,conv,context['judge_model'],context['judge_features'],
            seed+DETAILED_PATH_SEED_OFFSET,
        )
        winner=str(res['winner']); method=str(res['method'])
        wins[winner]=wins.get(winner,0)+1
        methods[(winner,method)]=methods.get((winner,method),0)+1
    return {
        'p_red_win':wins.get('red',0)/paths,
        'p_blue_win':wins.get('blue',0)/paths,
        **{f'p_{w}_{m}':n/paths for (w,m),n in methods.items()},
    }


def main() -> None:
    install_i10_b0()
    cohort,_=wc_audit.select_cohort('bantamweight',100)
    cohort=cohort.copy(); cohort['fight_id']=cohort['fight_id'].astype(str)
    hit=cohort[cohort['fight_id'].eq(FIGHT_ID)]
    if len(hit)!=1:
        raise RuntimeError(f'fight {FIGHT_ID} not uniquely found in bantam cohort: {len(hit)}')
    mr=hit.iloc[0].copy(); mr['event_date']=pd.Timestamp(mr['event_date']).normalize()

    context=load_frozen_context(V2_BUNDLE_PATH)
    fsr=load_prefight_snapshots(canonical.FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    uncertainty=pd.read_parquet(FSR_V3_PREFIGHT_UNCERTAINTY_PATH).copy()
    uncertainty['event_date']=pd.to_datetime(uncertainty['event_date']).dt.normalize()
    uncertainty['fight_id']=uncertainty['fight_id'].astype(str); uncertainty['fighter_id']=uncertainty['fighter_id'].astype(str)
    uncertainty=uncertainty[~uncertainty['trait'].eq('knockdown_resistance_v3')].copy()
    submission_baseline=load_submission_baseline_v3()
    kd_hist=load_kd_resistance_history()

    red0,blue0=historical_fighter_rows(
        fsr,event_date=mr['event_date'],fight_id=FIGHT_ID,
        fighter_ids=(str(mr['r_id']),str(mr['b_id']))
    )
    red_unc0=historical_uncertainty_rows(uncertainty,event_date=mr['event_date'],fight_id=FIGHT_ID,fighter_id=str(mr['r_id']))
    blue_unc0=historical_uncertainty_rows(uncertainty,event_date=mr['event_date'],fight_id=FIGHT_ID,fighter_id=str(mr['b_id']))
    red_kd=historical_kd_resistance_row(kd_hist,event_date=mr['event_date'],fight_id=FIGHT_ID,fighter_id=str(mr['r_id']))
    blue_kd=historical_kd_resistance_row(kd_hist,event_date=mr['event_date'],fight_id=FIGHT_ID,fighter_id=str(mr['b_id']))

    market=build_two_way_market(MARKET_PATH)
    mm=market[market['fight_id'].astype(str).eq(FIGHT_ID)]
    if len(mm)!=1:
        raise RuntimeError(f'market row count for {FIGHT_ID}: {len(mm)}')
    market_row=mm.iloc[0]
    favorite_side='red' if str(market_row['favorite_id'])==str(mr['r_id']) else 'blue'
    market_p=float(market_row['market_favorite_fair_p'])

    print('BASHARAT–VAZQUEZ FSR -> MARKET TARGET')
    print(f"fight={mr['r_name']} vs {mr['b_name']} market_favorite={favorite_side} market_p={market_p:.5f} paths/arm={PATHS}")

    # Main path: move all identifiable inverted traits together using common RNG seeds.
    rows=[]
    all_traits=set(TRAITS)
    for alpha in np.linspace(0.0,1.0,21):
        rr=patched_row(red0,str(mr['r_name']),alpha,all_traits)
        bb=patched_row(blue0,str(mr['b_name']),alpha,all_traits)
        ru=patched_uncertainty(red_unc0,red0,str(mr['r_name']),alpha,all_traits)
        bu=patched_uncertainty(blue_unc0,blue0,str(mr['b_name']),alpha,all_traits)
        sim=simulate_arm(mr,rr,bb,ru,bu,context,submission_baseline,red_kd,blue_kd,PATHS,SEED)
        fav_p=sim['p_red_win'] if favorite_side=='red' else sim['p_blue_win']
        rec={'arm':'all_inverse_direction','alpha':float(alpha),'favorite_p':fav_p,'market_p':market_p,'gap_pp':100*(fav_p-market_p)}
        for side,name,row in [('red',str(mr['r_name']),rr),('blue',str(mr['b_name']),bb)]:
            for t in TRAITS:
                rec[f'{side}_{t}']=float(row[t])
        rows.append(rec)
        print(f'all alpha={alpha:.2f} favorite_p={fav_p:.4f} gap={100*(fav_p-market_p):+.2f}pp')

    # Endpoint attribution: which trait family can move the probability by itself?
    families={
        'standing_offense_only':{'standing_striking_offense'},
        'standing_tendency_only':{'standing_striking_tendency'},
        'td_tendency_only':{'takedown_tendency'},
        'td_offense_only':{'takedown_offense'},
        'standing_offense_plus_td_tendency':{'standing_striking_offense','takedown_tendency'},
    }
    family_rows=[]
    for label,active in families.items():
        rr=patched_row(red0,str(mr['r_name']),1.0,active); bb=patched_row(blue0,str(mr['b_name']),1.0,active)
        ru=patched_uncertainty(red_unc0,red0,str(mr['r_name']),1.0,active); bu=patched_uncertainty(blue_unc0,blue0,str(mr['b_name']),1.0,active)
        sim=simulate_arm(mr,rr,bb,ru,bu,context,submission_baseline,red_kd,blue_kd,PATHS,SEED)
        fav_p=sim['p_red_win'] if favorite_side=='red' else sim['p_blue_win']
        family_rows.append({'arm':label,'favorite_p':fav_p,'market_p':market_p,'gap_pp':100*(fav_p-market_p)})
        print(f'{label}: favorite_p={fav_p:.4f} gap={100*(fav_p-market_p):+.2f}pp')

    grid=pd.DataFrame(rows); fam=pd.DataFrame(family_rows)
    # Nearest simulated alpha is the robust reported target; also bracket the first crossing.
    grid['abs_gap_pp']=grid['gap_pp'].abs()
    nearest=grid.loc[grid['abs_gap_pp'].idxmin()].copy()
    above=grid[grid['favorite_p']>=market_p].sort_values('alpha')
    below=grid[grid['favorite_p']<market_p].sort_values('alpha')
    first_cross=float(above.iloc[0]['alpha']) if not above.empty else np.nan
    prev_alpha=float(below[below['alpha']<first_cross].iloc[-1]['alpha']) if np.isfinite(first_cross) and not below[below['alpha']<first_cross].empty else np.nan

    target_rows=[]
    for side,name,row0 in [('red',str(mr['r_name']),red0),('blue',str(mr['b_name']),blue0)]:
        target=INVERSE[name]
        for t in TRAITS:
            cur=float(row0[t]); inv=float(target[t]) if t in target else cur
            at_nearest=cur+float(nearest['alpha'])*(inv-cur)
            target_rows.append({'side':side,'fighter':name,'trait':t,'prefight_fsr':cur,'inverse_needed':inv,
                                'nearest_market_alpha':float(nearest['alpha']),'market_target_value':at_nearest,
                                'delta_from_prefight':at_nearest-cur})
    target_df=pd.DataFrame(target_rows)
    summary=pd.DataFrame([{
        'fight_id':FIGHT_ID,'red':mr['r_name'],'blue':mr['b_name'],'favorite_side':favorite_side,
        'market_p':market_p,'baseline_mc_p':float(grid.loc[grid['alpha'].eq(0.0),'favorite_p'].iloc[0]),
        'nearest_alpha':float(nearest['alpha']),'nearest_mc_p':float(nearest['favorite_p']),
        'nearest_gap_pp':float(nearest['gap_pp']),'first_cross_alpha':first_cross,'previous_alpha':prev_alpha,
        'full_inverse_mc_p':float(grid.loc[grid['alpha'].eq(1.0),'favorite_p'].iloc[0]),'paths_per_arm':PATHS,
    }])

    OUT.mkdir(parents=True,exist_ok=True)
    grid.to_csv(OUT/'alpha_grid.csv',index=False)
    fam.to_csv(OUT/'trait_family_endpoints.csv',index=False)
    target_df.to_csv(OUT/'market_target_fsr_values.csv',index=False)
    summary.to_csv(OUT/'summary.csv',index=False)

    print('\nTARGET SUMMARY')
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nFSR VALUES AT NEAREST MARKET ALPHA')
    print(target_df.to_string(index=False,float_format=lambda x:f'{x:.6f}'))

if __name__=='__main__':
    main()
