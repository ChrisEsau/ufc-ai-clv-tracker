from pathlib import Path
import pandas as pd
import pipeline.research.score_ufc330_v5_ml as base

base.OUTPUT = Path('data/research/prop_mispricing/ufc_abudhabi_v5_moneyline_20260725.csv')
base.BET_OUTPUT = Path('data/research/prop_mispricing/ufc_abudhabi_v5_betting_logit030_no_cold_exclusion_20260725.csv')
base.FIGHT_DATE = pd.Timestamp('2026-07-25')
base.FIGHTS = [
    ('Magomed Ankalaev','Bogdan Guskov',-535,400,5,False),
    ('Steve Erceg','Ramazan Temirov',-115,-105,3,False),
    ('Magomed Zaynukov','Damian Rzepecki',-270,220,3,False),
    ('Rizvan Kuniev','Tyrell Fortune',-350,255,3,False),
    ('Abubakar Vagaev','Saygid Izagakhmaev',-260,195,3,False),
    ('Ismael Bonfim','Axel Sola',165,-200,3,False),
    ('Valter Walker','Thomas Petersen',-185,145,3,False),
    ('Dustin Jacoby','Muhammad Saidov',-185,145,3,False),
    ('Santiago Ponzinibbio','Sam Patterson',375,-500,3,False),
    ('Magomed Tuchalov','Brendson Ribeiro',-900,550,3,False),
    ('Nurullo Aliev','Mike Davis',-220,170,3,False),
    ('Cody Gibson','Abdul Hussein',400,-600,3,False),
]

if __name__ == '__main__':
    base.main()
    for p in (base.OUTPUT, base.BET_OUTPUT):
        df = pd.read_csv(p)
        if 'market_source' in df.columns:
            df['market_source'] = 'UFC official fight-day odds 2026-07-25'
        df.to_csv(p, index=False)
