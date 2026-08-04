# UFC Round Fight Simulator V0

## Status

**Shadow-only mechanics foundation. Not calibrated for wagering or production promotion.**

## Purpose

V0 establishes a typed, testable, round-level Monte Carlo simulation kernel without changing the production moneyline pipeline, RFS builders, feature contracts, betting artifacts, or master schema.

The architecture deliberately separates two problems:

```text
Leakage-safe fighter/RFS features
        ↓
Trained parameter estimation layer (future)
        ↓
Validated simulator parameter contract
        ↓
Round-level Monte Carlo engine (this build)
        ↓
Shadow simulation summary
```

The simulator does not guess how current RFS feature scales map to finish or activity probabilities. That mapping must be learned, calibrated, and validated in a later phase.

## V0 Scope

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

## Non-Goals

V0 does not:

- train parameter models;
- ingest live fighter-state artifacts automatically;
- change production predictions;
- publish bets or EV decisions;
- claim calibrated probabilities;
- simulate second-by-second event sequences;
- reproduce exact judge identities or scorecards.

## Files

```text
pipeline/simulation/contracts.py
pipeline/simulation/engine.py
pipeline/simulation/io.py
pipeline/simulation/run_round_simulator.py
configs/simulation/example_matchup.json
tests/test_round_simulator.py
```

Generated output:

```text
data/model_lab/simulation/latest_round_simulation_summary.json
```

The output is a generated artifact and should not be committed.

## Run

From the repository root:

```bash
python -m pipeline.simulation.run_round_simulator \
  --input configs/simulation/example_matchup.json \
  --simulations 25000 \
  --seed 7
```

## Validate

```bash
python -m unittest discover -s tests -p 'test_round_simulator.py'
python -m compileall pipeline/simulation
```

## Promotion Gates

Before simulator probabilities can be used in the betting pipeline, the following must be completed:

1. Build leakage-safe historical simulator targets at the fighter-round grain.
2. Train parameter models for activity, phase allocation, conversion, and finish hazards.
3. Calibrate component distributions and full-market probabilities.
4. Run walk-forward tests against direct prop models and closing prices.
5. Validate distribution coverage, Brier score, log loss, ROI, and CLV.
6. Validate joint correlations for same-game parlay use.
7. Run shadow mode on live cards before any production promotion.

## Next Build

The next isolated phase should create a historical training table and parameter-model interfaces. It should consume existing round-stat and RFS artifacts without adding simulator outputs to the production feature view.
