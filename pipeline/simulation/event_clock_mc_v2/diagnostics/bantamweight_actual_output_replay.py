"""Replay a real ~80% bantamweight favorite using realized UFCStats outputs as MC means.

Measurement only. FSR and simulator code are untouched.

Question: if the upstream model had predicted the realized fight-flow outputs exactly,
what moneyline probability would frozen Event Clock MC V2 produce?

Default case is Javid Basharat vs Gianni Vazquez (fight_id 86df0e75f41784d8),
market fair favorite probability ~80.8% in the historical offered/legacy-consensus file.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.diagnostics import canonical_c_validation as canonical
from pipeline.simulation.event_clock_mc_v2.diagnostics import kd_finishing_sequence_screen as seq
from pipeline.simulation.event_clock_mc_v2.run_event_or_fight_frozen import (
    load_frozen_context,
    DETAILED_PATH_SEED_OFFSET,
    _submission_inputs,
)
from pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle import DEFAULT_BUNDLE_PATH as V2_BUNDLE_PATH
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import load_prefight_snapshots, historical_fighter_rows
from pipeline.simulation.event_clock_mc_v2.canonical_c import load_kd_resistance_history, historical_kd_resistance_row, fight_with_kd_resistance
from pipeline.simulation.event_clock_mc_v2.inference import predict_target_v3
from pipeline.simulation.event_mc_v1.diagnostics.population_validation import _fight

ROUND_STATS = Path('data/fight_details/ufc_round_stats.parquet')
MASTER = Path('data/master/ufc_master.parquet')
MARKET = Path('data/market/historical_market_outcomes.parquet')
DEFAULT_FIGHT_ID = '86df0e75f41784d8'


def install_i10_b0() -> None:
    seq.INTERCEPT = 10.0
    seq.DENOMINATOR = 12.0
    seq.LOWER_CAP = -40.0
    seq.UPPER_CAP = 10.0
    seq.ARMS = {'i10_b0': None}
    seq._MODE = 'i10_b0'
    canonical.simulate_detailed_path = seq.sequence_simulate_detailed_path


def pick_col(df: pd.DataFrame, *names: str, required: bool = True):
    by_lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        if n.lower() in by_lower:
            return by_lower[n.lower()]
    if required:
        raise RuntimeError(f'missing any of columns {names}; available={list(df.columns)}')
    return None


def numeric_sum(g: pd.DataFrame, *aliases: str) -> float:
    c = pick_col(g, *aliases, required=False)
    if c is None:
        return 0.0
    return float(pd.to_numeric(g[c], errors='coerce').fillna(0).sum())


def actual_side_totals(g: pd.DataFrame) -> dict[str, float]:
    sig_att = numeric_sum(g, 'sig_str_attempted', 'sig_str_att', 'significant_strikes_attempted', 'sig_attempted')
    sig_land = numeric_sum(g, 'sig_str_landed', 'sig_str_land', 'significant_strikes_landed', 'sig_landed')
    ground_att = numeric_sum(g, 'ground_attempted', 'ground_att', 'ground_sig_str_attempted', 'ground_sig_att')
    ground_land = numeric_sum(g, 'ground_landed', 'ground_land', 'ground_sig_str_landed', 'ground_sig_land')
    td_att = numeric_sum(g, 'td_attempted', 'td_att', 'takedowns_attempted')
    td_land = numeric_sum(g, 'td_landed', 'td_land', 'takedowns_landed')
    control = numeric_sum(g, 'control_seconds', 'control_time_sec', 'ctrl_seconds', 'control')
    sub_att = numeric_sum(g, 'sub_attempts', 'sub_att', 'submission_attempts')
    kd = numeric_sum(g, 'knockdowns', 'kd')
    standing_att = max(0.0, sig_att - ground_att)
    standing_land = max(0.0, sig_land - ground_land)
    return {
        'sig_attempted': sig_att,
        'sig_landed': sig_land,
        'standing_attempted': standing_att,
        'standing_landed': standing_land,
        'ground_attempted': ground_att,
        'ground_landed': ground_land,
        'td_attempted': td_att,
        'td_landed': td_land,
        'control': control,
        'sub_attempts': sub_att,
        'knockdowns_actual': kd,
    }


def elapsed_seconds(master_row: pd.Series) -> float:
    rounds = float(master_row.get('total_rounds', 3) or 3)
    for c in ('actual_elapsed_seconds', 'fight_elapsed_seconds', 'elapsed_seconds'):
        if c in master_row and pd.notna(master_row[c]):
            return float(master_row[c])
    return rounds * 300.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fight-id', default=DEFAULT_FIGHT_ID)
    ap.add_argument('--paths', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=20260823)
    ap.add_argument('--zero-subs', action='store_true')
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()

    install_i10_b0()
    master = pd.read_parquet(MASTER).drop_duplicates('fight_id').copy()
    master['fight_id'] = master['fight_id'].astype(str)
    mr = master.loc[master['fight_id'].eq(str(args.fight_id))]
    if len(mr) != 1:
        raise RuntimeError(f'fight {args.fight_id} not unique in master: {len(mr)}')
    mr = mr.iloc[0].copy()
    event_date = pd.to_datetime(mr.get('date', mr.get('event_date'))).normalize()
    mr['event_date'] = event_date

    rs = pd.read_parquet(ROUND_STATS).copy()
    fid_col = pick_col(rs, 'fight_id', 'bout_id')
    rs[fid_col] = rs[fid_col].astype(str)
    fr = rs[rs[fid_col].eq(str(args.fight_id))].copy()
    if fr.empty:
        raise RuntimeError(f'fight {args.fight_id} absent from round stats')

    fighter_col = pick_col(fr, 'fighter_id')
    red_id, blue_id = str(mr['r_id']), str(mr['b_id'])
    red_rows = fr[fr[fighter_col].astype(str).eq(red_id)]
    blue_rows = fr[fr[fighter_col].astype(str).eq(blue_id)]
    if red_rows.empty or blue_rows.empty:
        corner_col = pick_col(fr, 'corner', required=False)
        if corner_col is None:
            raise RuntimeError('could not map round rows to red/blue')
        red_rows = fr[fr[corner_col].astype(str).str.lower().eq('red')]
        blue_rows = fr[fr[corner_col].astype(str).str.lower().eq('blue')]

    actual = {'red': actual_side_totals(red_rows), 'blue': actual_side_totals(blue_rows)}
    horizon = elapsed_seconds(mr)

    budgets = {}
    for side in ('red', 'blue'):
        a = actual[side]
        budgets.update({
            f'{side}_standing_attempted': a['standing_attempted'],
            f'{side}_standing_landed': a['standing_landed'],
            f'{side}_ground_attempted': a['ground_attempted'],
            f'{side}_ground_landed': a['ground_landed'],
            f'{side}_td_attempted': a['td_attempted'],
            f'{side}_td_landed': a['td_landed'],
            f'{side}_control': a['control'],
        })

    if args.zero_subs:
        sub_rates = {'red': 0.0, 'blue': 0.0}
    else:
        sub_rates = {side: actual[side]['sub_attempts'] / max(horizon, 1.0) for side in ('red', 'blue')}

    context = load_frozen_context(V2_BUNDLE_PATH)
    fsr = load_prefight_snapshots(canonical.FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    red_fsr, blue_fsr = historical_fighter_rows(
        fsr, event_date=event_date, fight_id=str(args.fight_id), fighter_ids=(red_id, blue_id)
    )

    target = pd.DataFrame([mr])
    inferred_pair, _ = predict_target_v3(
        target,
        fsr,
        context['inference_models'],
        context['submission_scale'],
        context['conversion_offset'],
    )
    _, conversion = _submission_inputs(inferred_pair)

    kd_hist = load_kd_resistance_history()
    red_kd = historical_kd_resistance_row(kd_hist, event_date=event_date, fight_id=str(args.fight_id), fighter_id=red_id)
    blue_kd = historical_kd_resistance_row(kd_hist, event_date=event_date, fight_id=str(args.fight_id), fighter_id=blue_id)
    fight = fight_with_kd_resistance(
        _fight(mr, context['fsr_all']),
        red_native_resistance=float(red_kd['pre_rating']),
        blue_native_resistance=float(blue_kd['pre_rating']),
    )

    rows = []
    for p in range(args.paths):
        result = canonical.simulate_detailed_path(
            fight, dict(budgets), sub_rates, conversion,
            context['judge_model'], context['judge_features'],
            args.seed + p + DETAILED_PATH_SEED_OFFSET,
        )
        rows.append(result)
    paths = pd.DataFrame(rows)

    red_win = float((paths['winner'].astype(str) == 'red').mean())
    blue_win = float((paths['winner'].astype(str) == 'blue').mean())
    favorite_side = 'red'
    market = pd.read_parquet(MARKET)
    mm = market[(market['market_key'].eq('moneyline')) & market['fight_id'].astype(str).eq(str(args.fight_id))].copy()
    fair_p = np.nan
    if len(mm) >= 2:
        probs = pd.to_numeric(mm['implied_probability'], errors='coerce').dropna().to_numpy(float)
        if len(probs) >= 2 and probs.sum() > 0:
            fair = probs / probs.sum()
            favorite_side = str(mm.iloc[int(np.argmax(fair))].get('outcome_side', 'red'))
            fair_p = float(np.max(fair))
    favorite_p = red_win if favorite_side == 'red' else blue_win

    methods = paths.groupby(['winner','method']).size().reset_index(name='n')
    methods['p'] = methods['n'] / len(paths)

    summary = pd.DataFrame([{
        'fight_id': str(args.fight_id), 'red': mr['r_name'], 'blue': mr['b_name'],
        'market_favorite_fair_p': fair_p, 'favorite_side': favorite_side,
        'actual_winner': mr.get('winner_name', mr.get('winner', '')),
        'actual_method': mr.get('method', ''), 'paths': len(paths),
        'zero_subs': bool(args.zero_subs),
        'replay_favorite_p': favorite_p, 'replay_red_p': red_win, 'replay_blue_p': blue_win,
        'submission_conversion_used': conversion,
    }])
    actual_df = pd.DataFrame([{'side': side, **actual[side]} for side in ('red','blue')])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir/'summary.csv', index=False)
    actual_df.to_csv(args.out_dir/'actual_outputs.csv', index=False)
    methods.to_csv(args.out_dir/'method_paths.csv', index=False)

    print('BANTAMWEIGHT ACTUAL-OUTPUT REPLAY')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nACTUAL REALIZED OUTPUTS USED AS FIXED BUDGETS')
    print(actual_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print('\nREPLAY METHOD/WINNER MASS')
    print(methods.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

if __name__ == '__main__':
    main()
