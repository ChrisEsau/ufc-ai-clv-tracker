"""Solve a concrete FSR mean target for Basharat-Vazquez realized standing/TD outputs.

Convention: keep defensive traits and baselines fixed at prefight V3 values; solve
attacker tendency/offense means required to reproduce realized standing attempts,
standing accuracy, and takedown attempts/completion over a 15-minute fight.
Ground/control is not solved here because the prior replay control parser returned zero.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import load_prefight_snapshots, historical_fighter_rows, initialize_path_matchup
from pipeline.simulation.event_clock_mc_v2.diagnostics import canonical_c_validation as canonical

FIGHT_ID='86df0e75f41784d8'
MASTER=Path('data/master/ufc_master.parquet')
TRAITS=[
 'standing_striking_tendency','standing_striking_suppression','standing_striking_offense','standing_striking_defense','standing_accuracy_baseline',
 'takedown_tendency','takedown_suppression','takedown_offense','takedown_defense','takedown_completion_baseline',
 'ground_striking_tendency','ground_striking_suppression','ground_striking_offense','ground_accuracy_baseline','ground_striking_burst_baseline']

def logit(p):
    p=float(np.clip(p,1e-9,1-1e-9)); return math.log(p/(1-p))

def main():
    master=pd.read_parquet(MASTER); master['fight_id']=master['fight_id'].astype(str)
    mr=master.loc[master['fight_id'].eq(FIGHT_ID)].iloc[0]
    date=pd.to_datetime(mr.get('event_date',mr.get('date'))).normalize()
    fsr=load_prefight_snapshots(canonical.FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    rr,br=historical_fighter_rows(fsr,event_date=date,fight_id=FIGHT_ID,fighter_ids=(str(mr['r_id']),str(mr['b_id'])))
    current={'red':rr.to_dict(),'blue':br.to_dict()}
    needed={s:{t:float(current[s][t]) for t in TRAITS} for s in ('red','blue')}

    # Realized standing outputs from replay: red 80 att/45 land, blue 100/24.
    # Realized TD outputs: red 8 att/3 land, blue 0/0. Fight horizon is 15m.
    needed['red']['standing_striking_tendency']=80.0/float(current['blue']['standing_striking_suppression'])
    needed['blue']['standing_striking_tendency']=100.0/float(current['red']['standing_striking_suppression'])
    needed['red']['standing_striking_offense']=logit(45/80)-logit(current['red']['standing_accuracy_baseline'])+float(current['blue']['standing_striking_defense'])
    needed['blue']['standing_striking_offense']=logit(24/100)-logit(current['blue']['standing_accuracy_baseline'])+float(current['red']['standing_striking_defense'])
    needed['red']['takedown_tendency']=8.0/float(current['blue']['takedown_suppression'])
    needed['blue']['takedown_tendency']=0.0
    needed['red']['takedown_offense']=logit(3/8)-logit(current['red']['takedown_completion_baseline'])+float(current['blue']['takedown_defense'])
    # blue TD offense is not identified when target attempts are zero; leave at current.

    rows=[]
    for trait in TRAITS:
        rows.append({'trait':trait,
                     'basharat_prefight':float(current['red'][trait]),'basharat_needed':float(needed['red'][trait]),'basharat_delta':float(needed['red'][trait]-current['red'][trait]),
                     'vazquez_prefight':float(current['blue'][trait]),'vazquez_needed':float(needed['blue'][trait]),'vazquez_delta':float(needed['blue'][trait]-current['blue'][trait]),
                     'solved':trait in {'standing_striking_tendency','standing_striking_offense','takedown_tendency','takedown_offense'}})
    out=pd.DataFrame(rows)
    print('BASHARAT-VAZQUEZ PREFIGHT FSR MEANS VS CONCRETE NEEDED MEANS')
    print(out.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('\nNOTES: defensive/baseline/ground traits are held fixed; blue TD offense is not identified at zero target TD attempts. Ground/control target not solved due prior control parse issue.')
    Path('data/diagnostics/event_clock_mc_v2/bantamweight_fsr_inverse_target').mkdir(parents=True,exist_ok=True)
    out.to_csv('data/diagnostics/event_clock_mc_v2/bantamweight_fsr_inverse_target/fsr_means_side_by_side.csv',index=False)

if __name__=='__main__': main()
