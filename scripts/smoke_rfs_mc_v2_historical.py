"""First real historical RFS Monte Carlo plumbing smoke test."""

from pathlib import Path
import sys

# Support direct execution from the repository root:
#     python scripts/smoke_rfs_mc_v2_historical.py
REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from audit_rfs_mc_v2_finish_paths import (
    zero_finish_calibration,
    zero_phase_effect_calibration,
    zero_state_calibration,
    zero_transition_effect_calibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.historical_matchup_loader import (
    load_historical_matchup,
)
from pipeline.simulation.rfs_mc_v2_shared_state.monte_carlo_runner import (
    run_matchup_monte_carlo,
)
from pipeline.simulation.rfs_mc_v2_shared_state.rfs_parameter_resolver import (
    resolve_fighter_parameters,
)


HISTORY_PATH = (
    "data/simulation/rfs_mc_v2_shared_state/"
    "historical_fighter_state.parquet"
)
MASTER_PATH = "data/master/ufc_master.parquet"

FIGHT_ID = "f425b7a527dd53aa"


history = pd.read_parquet(HISTORY_PATH)
master = pd.read_parquet(MASTER_PATH)

matchup = load_historical_matchup(
    history,
    master,
    FIGHT_ID,
)

# Historical calibration population must precede the simulated fight.
population = history.loc[
    pd.to_datetime(history["date"]) < matchup.date
].copy()

population = population.loc[
    pd.to_numeric(
        population["rfs_traj_prior_fight_count"],
        errors="coerce",
    ) > 0
]

red = resolve_fighter_parameters(
    profile=matchup.red.features,
    prior_fight_count=matchup.red.prior_fight_count,
    population_history=population,
)

blue = resolve_fighter_parameters(
    profile=matchup.blue.features,
    prior_fight_count=matchup.blue.prior_fight_count,
    population_history=population,
)

# Normalize boundary metadata to the exact Python type required by
# the existing Monte Carlo engine contract.
scheduled_rounds = int(matchup.scheduled_rounds)

summary = run_matchup_monte_carlo(
    red.transition,
    blue.transition,
    red.phase,
    blue.phase,
    red.dynamic,
    blue.dynamic,
    dynamic_state_calibration=zero_state_calibration(),
    phase_effect_calibration=zero_phase_effect_calibration(),
    transition_effect_calibration=zero_transition_effect_calibration(),
    finish_probability_calibration=zero_finish_calibration(),
    simulation_count=100,
    seed_start=20260801,
    scheduled_rounds=scheduled_rounds,
)

print("=" * 78)
print("FIRST HISTORICAL RFS MONTE CARLO — PLUMBING SMOKE TEST")
print("=" * 78)

print(
    f"{matchup.red.fighter_name} vs "
    f"{matchup.blue.fighter_name}"
)
print("Fight ID:", matchup.fight_id)
print("Scheduled rounds:", scheduled_rounds)
print("Simulations:", summary.simulation_count)

print()
print("Red wins :", summary.red_win_count)
print("Blue wins:", summary.blue_win_count)
print("Draws    :", summary.draw_count)

print()
print("Finishes :", summary.finish_count)
print("Distance :", summary.scheduled_distance_count)

print()
print("Actual winner ID:", matchup.actual.winner_id)
print("Actual method   :", matchup.actual.method)

assert summary.simulation_count == 100
assert (
    summary.red_win_count
    + summary.blue_win_count
    + summary.draw_count
    == 100
)

# Zero-finish calibration is intentional for this structural test.
assert summary.finish_count == 0
assert summary.scheduled_distance_count == 100

print()
print("=" * 78)
print("HISTORICAL MONTE CARLO PIPELINE PASSED")
print("=" * 78)
print("NOTE: probabilities are not yet UFC-calibrated predictions.")
