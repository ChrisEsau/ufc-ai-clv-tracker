from pathlib import Path
import pandas as pd
import pipeline.research.score_ufc330_v5_ml as base

base.OUTPUT = Path('data/research/prop_mispricing/ufc_vegas120_v5_moneyline_20260808.csv')
base.BET_OUTPUT = Path('data/research/prop_mispricing/ufc_vegas120_v5_betting_logit030_no_cold_exclusion_20260808.csv')
base.FIGHT_DATE = pd.Timestamp('2026-08-08')
base.FIGHTS = [
    ('Mateusz Gamrot','Quillan Salkilld',114,-135,5,False),
    ('Diego Ferreira','Billy Quarantillo',-185,154,3,False),
    ('Darren Elkins','Yadier del Valle',575,-850,3,False),
    ('Amanda Lemos','Alexia Thainara',210,-258,3,False),
    ('Ty Miller','Billy Ray Goff',-410,320,3,False),
    ('Diyar Nurgozhay','Bruno Lopes',-162,136,3,False),
    ('Louie Sutherland','José Montanha',-218,180,3,False),
    ('Steven Asplund','Guilherme Pat',-310,250,3,False),
    ('Manoel Sousa','Richie Miranda',-310,250,3,False),
    ('Miles Johns','Gianni Vazquez',-185,154,3,False),
    ('Juliana Miller','Ravena Oliveira',-310,250,3,False),
    ('Gigi Canuto','Carol Foro',195,-238,3,False),
]

if __name__ == '__main__':
    base.main()
    for p in (base.OUTPUT, base.BET_OUTPUT):
        df = pd.read_csv(p)
        if 'market_source' in df.columns:
            df['market_source'] = 'DraftKings fight-day 2026-08-08'
        df.to_csv(p, index=False)
