from pathlib import Path
import pandas as pd
import pipeline.research.score_ufc330_v5_ml as base

base.OUTPUT = Path('data/research/prop_mispricing/ufc_belgrade_v5_moneyline_20260801.csv')
base.BET_OUTPUT = Path('data/research/prop_mispricing/ufc_belgrade_v5_betting_logit030_no_cold_exclusion_20260801.csv')
base.FIGHT_DATE = pd.Timestamp('2026-08-01')
base.FIGHTS = [
    ('Uroš Medić','Daniel Rodriguez',-380,300,5,False),
    ('Jan Błachowicz','Navajo Stirling',260,-325,3,False),
    ('Aleksandar Rakić','Marcin Tybura',-380,300,3,False),
    ('Duško Todorović','Robert Valentin',142,-170,3,False),
    ('Vlasto Čepo','Gilbert Urbina',-355,280,3,False),
    ('Miloš Janičić','Noah Gugnon',-110,-110,3,False),
    ('Ľudovít Klein','Tofiq Musayev',-270,220,3,False),
    ('Oban Elliott','Michael Oliveira',285,-360,3,False),
    ('Mark Vologdin','Borislav Nikolić',170,-205,3,False),
    ('Dennis Buzukja','Bogdan Grad',160,-192,3,False),
    ('Mateusz Rębecki','Kyle Prepolec',-700,500,3,False),
    ('Nina Milošević','Hailey Cowan',-535,400,3,False),
    ('Jovan Leka','Alexander Poppeck',-258,210,3,False),
    ('Marina Spasić','Stephanie Luciano',260,-325,3,False),
]

if __name__ == '__main__':
    base.main()
    for p in (base.OUTPUT, base.BET_OUTPUT):
        df = pd.read_csv(p)
        if 'market_source' in df.columns:
            df['market_source'] = 'DraftKings fight-day 2026-08-01'
        df.to_csv(p, index=False)
