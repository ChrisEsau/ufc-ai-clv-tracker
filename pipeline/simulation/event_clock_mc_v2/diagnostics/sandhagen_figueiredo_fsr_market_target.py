"""Measurement-only mature-FSR market target audit for Sandhagen vs Figueiredo.

Reuses the generic interpolation/simulation machinery from the Basharat-Vazquez
market-target audit, but replaces the fight and inverse endpoints with the exact
population-inverse values for Cory Sandhagen vs Deiveson Figueiredo.

Both fighters had established UFC histories before this bout, so this is the
clean mature-FSR diagnostic for whether current prefight FSR means are too
compressed relative to the state required by the frozen MC to match market.
"""
from pathlib import Path

from pipeline.simulation.event_clock_mc_v2.diagnostics import basharat_vazquez_fsr_market_target as base

base.FIGHT_ID = '9b3bade95e1dd484'
base.OUT = Path('data/diagnostics/event_clock_mc_v2/sandhagen_figueiredo_fsr_market_target')
base.PATHS = 1500
base.INVERSE = {
    'Cory Sandhagen': {
        'standing_striking_tendency': 86.159825,
        'standing_striking_offense': 0.441636,
        'takedown_tendency': 3.051284,
        'takedown_offense': 21.422840,
    },
    'Deiveson Figueiredo': {
        'standing_striking_tendency': 14.448097,
        'standing_striking_offense': 1.252426,
        'takedown_tendency': 7.811634,
        'takedown_offense': 0.021921,
    },
}

if __name__ == '__main__':
    base.main()
