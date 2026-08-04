# UFC Round Fight Simulator V0

## Status

**Shadow-only mechanics, historical-training, and first-component research foundation. Not approved for wagering or production promotion.**

## Purpose

V0 establishes a typed, testable, round-level Monte Carlo simulation kernel and a leakage-safe parameter-training layer without changing the production moneyline pipeline, RFS builders, feature contracts, betting artifacts, or master schema.

The architecture separates the major problems:

```text
Historical round stats + fight results
        ↓
Leakage-safe fighter-round training table
        ↓
Walk-forward component models
        ↓
Sequential calibration and distribution contracts
        ↓
Validated round-parameter provider (next phase)
        ↓
Round-level Monte Carlo engine
        ↓
Historical replay and shadow simulation summaries
```

The mechanics engine remains heuristic except for the separately trained significant-strike pace research component. Trained outputs are not connected to live simulation yet because doing so without a provider/replay boundary could double-count pace, regime, and fatigue effects.

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

## Terminal-Round Accounting

Simulator version `round_simulator_v0_1` corrects an important V0 mechanics defect.

The original mechanics sampled a full five-minute performance line before sampling a finish. A fighter finishing at 1:00 could therefore receive five minutes of strikes, takedowns, control, and knockdowns.

The corrected engine:

- samples finish time;
- calculates terminal-round exposure;
- binomially thins landed and missed strikes separately;
- binomially thins landed and missed takedowns separately;
- proportionally scales control time;
- thins knockdowns;
- removes judge scoring for the unfinished terminal round.

This keeps `landed <= attempted`, preserves nonnegative statistics, and prevents terminal-round volume inflation.

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

The historical source boundary also removes:

- scraper URLs and source-only metadata;
- duplicate fight-level context in favor of the master dataset;
- raw target-round head, body, leg, distance, and clinch observations not yet registered as component targets;
- raw control-time aliases after canonicalizing `control_seconds`.

### Real historical audit

The GitHub Actions historical build completed successfully against the authoritative repository artifacts.

| Measure | Result |
|---|---:|
| Fighter-round rows | 32,236 |
| Eligible fights | 6,671 |
| Fighters | 2,180 |
| Columns | 500 |
| Date range | 2010-03-21 to 2026-08-01 |
| RFS feature columns | 382 |
| RFS state availability | 87.19% |
| Duplicate fighter-round keys | 0 |
| Paired fighter rows per fight-round | Pass |
| Raw current-round predictor columns | 0 |
| Source-only/URL modeling columns | 0 |
| Realized target-fight RFS columns | 0 |

The current simulator supports standard three- and five-round UFC bouts. Historical fights with missing or nonstandard scheduled-round values are excluded rather than coerced:

| Eligibility exclusion | Result |
|---|---:|
| Excluded fights | 24 |
| Excluded fighter-round rows | 110 |

These exclusions are written into the audit artifact.

Audit output:

```text
data/audits/simulation_fighter_round_training_audit.parquet
```

## Initial Parameter Model Registry

V0 defines model-agnostic contracts for eight component targets:

| Parameter | Statistical task | Historical target |
|---|---|---|
| Significant-strike attempts | Count/rate | `target_sig_attempted` |
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

The prediction contract carries expected mean, probability, dispersion, zero probability, model name, and model version as applicable. Complete fighter-round estimates are rejected if a required parameter is missing or competing finish probabilities exceed one.

## Significant-Strike Pace Component V0

The first trained component models exposure-adjusted significant-strike attempts per minute. Terminal rounds use their actual observed exposure during evaluation, preventing short-round counts from being treated as complete five-minute rounds.

### Leakage-safe fighter history

For each target fight, fighter pace history is calculated from completed prior fights only. Expanding, last-three, and EWM pace states are shifted one complete fight before joining the target fight.

### Walk-forward design

Expanding yearly folds test 2022 through 2026. Four models are compared:

1. round-specific historical mean;
2. shrinkage-adjusted fighter-history pace;
3. XGBoost with fight context and prior-round context;
4. XGBoost with context plus point-in-time RFS features.

### Aggregate walk-forward results

| Model | Poisson deviance | Count MAE | Count RMSE | Improvement vs fighter history |
|---|---:|---:|---:|---:|
| XGBoost context + RFS | 11.5276 | 15.345 | 20.584 | 20.46% |
| XGBoost context | 11.6600 | 15.384 | 20.638 | 19.55% |
| Fighter-history baseline | 14.4929 | — | — | 0.00% |
| Round-mean baseline | 15.3441 | — | — | -5.87% |

RFS improved aggregate Poisson deviance by approximately 1.14% over context-only XGBoost. It helped in the 2022–2025 folds but regressed in 2026, so RFS remains provisional and must continue to be evaluated as an ablation rather than being assumed beneficial.

Generated benchmark artifacts:

```text
data/model_lab/simulation/models/sig_attempt_pace_v0/
```

## Sequential Mean and Distribution Calibration

The log-rate models underpredicted arithmetic mean counts. Calibration is performed sequentially: the correction applied to a test year uses only completed earlier walk-forward years.

The calibrated statistical contract is gamma-Poisson:

```text
variance = mean + alpha * mean^2
```

### Sequential calibration results

| Model | Raw Poisson deviance | Calibrated Poisson deviance | Improvement | Raw bias | Calibrated bias |
|---|---:|---:|---:|---:|---:|
| XGBoost context | 11.6600 | 11.3133 | 2.97% | -4.09 | -0.92 |
| XGBoost context + RFS | 11.5276 | 11.1231 | 3.51% | -4.38 | -1.06 |

For context + RFS, the sequential mean factors after the 2022 cold start were stable:

```text
2023: 1.1305
2024: 1.1310
2025: 1.1287
2026: 1.1244
```

Final out-of-fold research parameters:

| Model | Mean calibration factor | Gamma-Poisson alpha | Calibration rows | Fights |
|---|---:|---:|---:|---:|
| XGBoost context | 1.1196 | 0.2101 | 9,652 | 1,980 |
| XGBoost context + RFS | 1.1289 | 0.2076 | 9,652 | 1,980 |

These parameters are research outputs, not live constants. They must be re-estimated under the final training and replay protocol.

Generated calibration artifacts:

```text
data/model_lab/simulation/models/sig_attempt_pace_v0/calibration/
```

## Files

```text
pipeline/simulation/contracts.py
pipeline/simulation/engine.py
pipeline/simulation/terminal_round.py
pipeline/simulation/io.py
pipeline/simulation/artifacts.py
pipeline/simulation/training_dataset.py
pipeline/simulation/historical_training.py
pipeline/simulation/parameter_models.py
pipeline/simulation/sig_attempt_model.py
pipeline/simulation/sig_attempt_calibration.py
pipeline/simulation/run_round_simulator.py
pipeline/simulation/run_build_training_dataset.py
pipeline/simulation/run_train_sig_attempt_model.py
pipeline/simulation/run_calibrate_sig_attempt_model.py
configs/simulation/example_matchup.json
tests/test_round_simulator.py
tests/test_simulation_training_dataset.py
tests/test_simulation_historical_training.py
tests/test_simulation_sig_attempt_model.py
tests/test_simulation_sig_attempt_calibration.py
tests/test_simulation_terminal_round.py
```

## Commands

Build historical training data:

```bash
python -m pipeline.simulation.run_build_training_dataset
```

Build without RFS joins:

```bash
python -m pipeline.simulation.run_build_training_dataset --without-rfs
```

Require all registered RFS histories:

```bash
python -m pipeline.simulation.run_build_training_dataset --require-all-rfs
```

Run strike-pace benchmark:

```bash
python -m pipeline.simulation.run_train_sig_attempt_model
```

Run sequential calibration:

```bash
python -m pipeline.simulation.run_calibrate_sig_attempt_model
```

Run mechanics simulator:

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

Generated parquet, model, JSON, and report outputs should not be committed.

## Validation

Relevant checks include:

```bash
python -m unittest discover -s tests -p 'test_round_simulator.py'
python -m unittest discover -s tests -p 'test_simulation_*.py'
python -m compileall pipeline/simulation
```

The suites cover:

- decision and stoppage target construction;
- elapsed-time repair;
- prior-round-only context;
- side-aware fighter/opponent RFS joins;
- realized fight-state exclusion;
- future-state rejection;
- duplicate-key rejection;
- historical source-boundary cleanup;
- no-current-year calibration leakage;
- pandas 3 cold-start compatibility;
- parameter prediction bounds and completeness;
- partial terminal-round exposure and scoring.

## Non-Goals

V0 does not:

- promote a production simulator component;
- automatically map component predictions into the heuristic engine;
- change production predictions;
- publish bets or EV decisions;
- claim calibrated market probabilities;
- reproduce second-by-second event sequences;
- reproduce exact judge identities or scorecards.

## Promotion Gates

Before simulator probabilities can be used in the betting pipeline:

1. Define and validate the trained round-parameter provider boundary.
2. Replay historical fights using out-of-fold component parameters.
3. Verify simulated strike-count coverage, quantiles, mean, variance, and fighter/opponent correlation.
4. Train and calibrate the remaining activity, conversion, control, and competing-risk components.
5. Compare simulator-derived markets against direct prop models and closing prices.
6. Validate distribution coverage, Brier score, log loss, ROI, and CLV.
7. Validate joint correlations before any same-game parlay use.
8. Run live cards in shadow mode before production promotion.

## Next Build

Build a trained round-parameter provider and historical replay harness. The provider must consume point-in-time feature rows and emit calibrated significant-strike pace plus dispersion without reapplying the heuristic engine's pace, regime, or fatigue multipliers. Historical replay should prove the generated distribution before any live-card integration.