# UFC Round Fight Simulator V0

## Status

**Shadow-only mechanics and training-data foundation. Not calibrated for wagering or production promotion.**

## Purpose

V0 establishes a typed, testable, round-level Monte Carlo simulation kernel and a leakage-safe parameter-training layer without changing the production moneyline pipeline, RFS builders, feature contracts, betting artifacts, or master schema.

The architecture deliberately separates the major problems:

```text
Historical round stats + fight results
        ↓
Leakage-safe fighter-round training table
        ↓
Component parameter models
        ↓
Validated parameter prediction contract
        ↓
Round-level Monte Carlo engine
        ↓
Shadow simulation summary
```

The simulator still does not guess how current RFS feature scales map to finish or activity probabilities. That mapping must be learned, calibrated, and walk-forward validated.

## Simulator Mechanics Scope

V0 simulates:

- a broad fight regime;
- distance, clinch, and ground phase shares;
- significant-strike attempts and lands;
- takedown attempts and completions;
- control time;
- knockdowns;
- competing KO/TKO and submission finish hazards;
- fatigue, damage, recovery, confidence, and tactical adjustment;
- round scoring and decisions;
- correlated market outcomes across complete simulated paths.

V0 summarizes:

- winner probabilities;
- decision, KO/TKO, and submission probabilities;
- goes-distance and inside-distance probabilities;
- round-reach and round-total probabilities;
- expected fight duration;
- expected fighter strike, takedown, control, and knockdown totals;
- selected joint probabilities for correlated props.

## Fighter-Round Training Table

The training builder creates:

```text
data/model_lab/simulation/training/fighter_round_parameter_training.parquet
```

Grain:

```text
one row per fighter per observed round
```

Inputs:

- `data/fight_details/ufc_round_stats.parquet`;
- `data/master/ufc_master.parquet`;
- available point-in-time RFS history artifacts for trajectory, suppression, wrestling, and defense.

Predictors include:

- fight context available before the round;
- scheduled rounds and title-fight context;
- cumulative and previous-round statistics from earlier rounds in the same fight;
- the opponent's equivalent prior-round context;
- fighter and opponent RFS state joined at the target-fight grain.

Current-round observations are copied only into `target_*` columns and then removed from the predictor namespace. Realized target-fight RFS observation columns containing `_fight_` are explicitly excluded.

The builder validates:

- unique `fight_id + fighter_id + round` grain;
- exactly two fighter rows for every observed fight-round;
- master/round-stat metadata agreement;
- no rounds after the recorded finish or scheduled duration;
- nonnegative round statistics;
- point-in-time RFS dates not later than the target fight;
- mutually exclusive KO/TKO and submission finish targets;
- no current-round raw statistics remaining as predictors.

Audit output:

```text
data/audits/simulation_fighter_round_training_audit.parquet
```

## Initial Parameter Model Registry

V0 defines model-agnostic contracts for eight component targets:

| Parameter | Statistical task | Historical target |
|---|---|---|
| Significant-strike attempts | Count | `target_sig_attempted` |
| Significant-strike accuracy | Binomial | landed conditional on attempts |
| Takedown attempts | Count | `target_td_attempted` |
| Takedown accuracy | Binomial | landed conditional on attempts |
| Control seconds | Zero-inflated continuous | `target_control_seconds` |
| Knockdowns | Count | `target_knockdowns` |
| KO/TKO finish hazard | Binary competing risk | `target_fighter_ko_tko_finish` |
| Submission finish hazard | Binary competing risk | `target_fighter_submission_finish` |

Every fitted model must emit the normalized long-form prediction schema defined in:

```text
pipeline/simulation/parameter_models.py
```

The prediction contract carries the expected mean, probability, dispersion, zero probability, model name, and model version as applicable. Complete fighter-round estimates are rejected if any required parameter is missing or if competing finish probabilities exceed one.

## Files

```text
pipeline/simulation/contracts.py
pipeline/simulation/engine.py
pipeline/simulation/io.py
pipeline/simulation/artifacts.py
pipeline/simulation/training_dataset.py
pipeline/simulation/parameter_models.py
pipeline/simulation/run_round_simulator.py
pipeline/simulation/run_build_training_dataset.py
configs/simulation/example_matchup.json
tests/test_round_simulator.py
tests/test_simulation_training_dataset.py
```

## Build Historical Training Data

From the repository root:

```bash
python -m pipeline.simulation.run_build_training_dataset
```

Build targets without RFS joins:

```bash
python -m pipeline.simulation.run_build_training_dataset --without-rfs
```

Require every registered RFS history artifact:

```bash
python -m pipeline.simulation.run_build_training_dataset --require-all-rfs
```

Missing RFS histories are optional by default because the state architecture is still being promoted incrementally. The target and prior-round context table can therefore be inspected before all state families are available.

## Run the Mechanics Simulator

```bash
python -m pipeline.simulation.run_round_simulator \
  --input configs/simulation/example_matchup.json \
  --simulations 25000 \
  --seed 7
```

Generated summary:

```text
data/model_lab/simulation/latest_round_simulation_summary.json
```

Generated parquet and JSON outputs should not be committed.

## Validate

```bash
python -m unittest discover -s tests -p 'test_round_simulator.py'
python -m unittest discover -s tests -p 'test_simulation_training_dataset.py'
python -m compileall pipeline/simulation
```

The fighter-round test suite covers:

- decision and stoppage target construction;
- elapsed fight-time repair;
- prior-round-only context;
- side-aware fighter/opponent RFS joins;
- realized fight-state exclusion;
- future-state rejection;
- duplicate-key rejection;
- parameter prediction bounds and completeness.

## Non-Goals

V0 does not:

- train or select a production algorithm;
- automatically map component predictions into the existing heuristic engine;
- change production predictions;
- publish bets or EV decisions;
- claim calibrated probabilities;
- simulate second-by-second event sequences;
- reproduce exact judge identities or scorecards.

## Promotion Gates

Before simulator probabilities can be used in the betting pipeline:

1. Run the historical builder against the real round-stat, master, and RFS artifacts.
2. Inspect target base rates, missingness, era coverage, and RFS availability.
3. Train one component model at a time with date-based walk-forward splits.
4. Calibrate count distributions, conversion probabilities, zero mass, and competing finish hazards.
5. Connect validated parameter estimates to a trained-parameter engine version while retaining the heuristic engine as a baseline.
6. Compare simulator-derived markets against direct prop models and closing prices.
7. Validate distribution coverage, Brier score, log loss, ROI, and CLV.
8. Validate joint correlations before any same-game parlay use.
9. Run live cards in shadow mode before production promotion.

## Next Build

Run and audit the real historical fighter-round table. The first trained component should be significant-strike attempts because it has a dense target, directly affects duration-dependent props, and provides an early test of whether pre-fight RFS plus prior-round context improves distributional forecasts over simple fighter-average baselines.
