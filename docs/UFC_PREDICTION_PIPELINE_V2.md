# UFC Prediction Pipeline V2

## Purpose

This document defines the approved target architecture for the next prediction, market, betting-board, dashboard, and Model Lab pipeline.

The core decision is:

```text
All canonical prediction, odds, and betting-board artifacts should be outcome-based, not red/blue-column based.
```

This design replaces the long-term assumption that the betting stack should be centered around columns such as `red_model_prob`, `blue_model_prob`, `red_american_odds`, and `blue_american_odds`.

Those columns may still exist temporarily in legacy compatibility code, but they are not the future canonical contract.

---

## Current Implementation Status

This is a target architecture document.

As of this decision:

- Moneyline V5 training is modular and config-driven.
- The active training config is `configs/models/moneyline_xgboost_v5.yaml`.
- The live prediction runner, market update runner, betting decision runner, and dashboard betting board are still partly legacy and side-based.
- The V2 prediction/betting architecture has been approved conceptually but not yet implemented in code.

Do not assume the files described here already exist until they are verified in the repository.

---

## High-Level Flow

```text
Dashboard selected event/model
        ↓
UFCStats upcoming event refresh
        ↓
Fight-level live card
        ↓
Fight-level live feature rows
        ↓
Model registry
        ↓
Model config YAML
        ↓
Generic model loader
        ↓
Generic prediction adapter
        ↓
Algorithm prediction plug-in
        ↓
Generic output formatter
        ↓
Outcome-level model predictions
        ↓
Outcome-level market odds
        ↓
Outcome-level betting board
        ↓
Dashboard / Model Lab / CLV / Bankroll
```

The important transition is:

```text
fight-level features → outcome-level predictions → outcome-level odds → outcome-level betting decisions
```

---

## Design Principles

### 1. Outcome Rows Are Canonical

Every bettable option should be represented as a row.

For moneyline, one fight produces two outcome rows:

```text
market_key = moneyline
outcome_label = Fighter A
model_probability = 0.64
```

```text
market_key = moneyline
outcome_label = Fighter B
model_probability = 0.36
```

For goes-distance, one fight produces two outcome rows:

```text
market_key = goes_distance
outcome_label = goes_distance
model_probability = 0.72
```

```text
market_key = goes_distance
outcome_label = inside_distance
model_probability = 0.28
```

For method, one fight may produce three outcome rows:

```text
market_key = method
outcome_label = ko_tko
model_probability = 0.42
```

```text
market_key = method
outcome_label = submission
model_probability = 0.18
```

```text
market_key = method
outcome_label = decision
model_probability = 0.40
```

### 2. No Long-Term Dual Output Contract

The future architecture should not maintain both:

```text
side-based market outputs
outcome-based market outputs
```

as equal first-class contracts.

The long-term canonical contract is outcome-based. Legacy side-based files can be retained temporarily only during migration.

### 3. The Adapter Is Generic

There should not be a separate adapter for every model family.

The generic adapter owns:

- loading the model bundle
- aligning feature columns
- coercing numeric values
- validating missing features
- calling an algorithm prediction plug-in
- passing probabilities to a formatter

### 4. Algorithms Use Plug-ins

Prediction should mirror the training framework.

Training pattern:

```text
pipeline/training/model_training.py
        ↓
pipeline/training/algorithms/xgboost_trainer.py
```

Prediction pattern:

```text
pipeline/modeling/prediction_adapter.py
        ↓
pipeline/modeling/probability.py
        ↓
pipeline/modeling/algorithms/xgboost_predictor.py
```

Algorithm plug-ins should be narrow. For XGBoost, the plug-in only needs to expose positive-class or class-probability prediction from the fitted model.

### 5. The Formatter Is Config-Driven

The formatter should be generic. The model YAML decides how probabilities become outcome rows.

Formatter types may include:

- `binary_matchup`
- `binary_prop`
- `multiclass`
- future `multi_output`

The formatter should not become a mini programming language. Keep logic in Python and choices in config.

### 6. Model Selection Is Runtime-Configurable

Dashboard model selection should flow through workflow inputs and runtime environment variables.

Selection precedence:

```text
CLI argument
    ↓
Environment variable
    ↓
Registry active model
```

Recommended environment variables:

```text
UFC_MODEL_ID
UFC_EVENT_ID
```

The model registry remains the source of available/selectable models. The environment variable is only the runtime override.

---

## Proposed Repository Structure

```text
configs/models/
  model_registry.yaml
  moneyline_xgboost_v5.yaml
  goes_distance_xgboost_v1.yaml

pipeline/modeling/
  model_registry.py
  model_config.py
  model_loader.py
  prediction_adapter.py
  prediction_formatter.py
  probability.py
  algorithms/
    xgboost_predictor.py

pipeline/prediction/
  live_feature_builder.py
  run_model_predictions_v2.py

pipeline/market/
  run_market_update_v2.py
  odds_normalizer.py
  outcome_matcher.py

pipeline/betting/
  run_betting_decision_v2.py
  outcome_ev.py
  staking.py
```

The exact names can change during implementation, but the separation of responsibilities should remain.

---

## Model Registry

The registry answers:

```text
Which model should be used?
```

Example:

```yaml
active_models:
  moneyline:
    primary: moneyline_xgboost_v5

models:
  moneyline_xgboost_v5:
    display_name: Moneyline XGBoost V5
    model_family: moneyline
    config_path: configs/models/moneyline_xgboost_v5.yaml
    status: production
    dashboard_selectable: true

  goes_distance_xgboost_v1:
    display_name: Goes Distance XGBoost V1
    model_family: goes_distance
    config_path: configs/models/goes_distance_xgboost_v1.yaml
    status: candidate
    dashboard_selectable: true
```

The registry should not duplicate the full model contract. It should point to the model config.

---

## Model Config Contract

The model config YAML remains the model contract.

The prediction layer should read the same config used by training where possible.

A moneyline config should include a prediction section similar to:

```yaml
prediction:
  format: binary_matchup
  market_key: moneyline
  threshold:
    source: fixed
    value: 0.50
  probability:
    positive_class: red
    clip_low: 0.02
    clip_high: 0.98
  output:
    path: data/predictions/model_outcomes.parquet
```

A binary prop config should include:

```yaml
prediction:
  format: binary_prop
  market_key: goes_distance
  positive_label: goes_distance
  negative_label: inside_distance
  threshold:
    source: fixed
    value: 0.50
```

A multiclass method config should include:

```yaml
prediction:
  format: multiclass
  market_key: method
  class_labels:
    - ko_tko
    - submission
    - decision
```

Market requirements can also live in the model config:

```yaml
market:
  source: the_odds_api
  sport: mma_mixed_martial_arts
  regions: us
  bookmakers:
    - draftkings
  markets:
    - h2h
```

For props:

```yaml
market:
  source: the_odds_api
  sport: mma_mixed_martial_arts
  regions: us
  bookmakers:
    - draftkings
  markets:
    - totals
    - method_of_victory
```

---

## Prediction Adapter

The generic adapter should:

1. load the selected model config
2. load the model bundle
3. load the configured feature columns
4. align live features to the feature contract
5. coerce model inputs to numeric
6. fill missing numeric values consistently with training assumptions
7. call the algorithm prediction dispatcher
8. clip probabilities if configured
9. pass probability output to the formatter
10. return outcome-level prediction rows

The adapter should not contain moneyline-specific, prop-specific, or dashboard-specific branching except through formatter configuration.

---

## Algorithm Prediction Plug-ins

Algorithm-specific logic belongs in narrow plug-ins.

Example XGBoost plug-in responsibilities:

```text
input: fitted XGBoost classifier + numeric feature matrix
output: probability array
```

For binary models:

```python
model.predict_proba(X)[:, 1]
```

For multiclass models:

```python
model.predict_proba(X)
```

Future algorithms such as CatBoost, LightGBM, sklearn ensembles, or neural networks can add plug-ins without changing the adapter contract.

---

## Formatter Architecture

The formatter converts model probabilities into canonical outcome rows.

### Binary Matchup Formatter

Moneyline fight-level probability:

```text
positive_probability = red wins
```

Outcome rows:

```text
outcome_label = red_fighter
model_probability = positive_probability
```

```text
outcome_label = blue_fighter
model_probability = 1 - positive_probability
```

### Binary Prop Formatter

Binary prop probability:

```text
positive_probability = configured positive label
```

Outcome rows:

```text
outcome_label = positive_label
model_probability = positive_probability
```

```text
outcome_label = negative_label
model_probability = 1 - positive_probability
```

### Multiclass Formatter

Multiclass class probabilities:

```text
class_labels = [ko_tko, submission, decision]
```

Outcome rows:

```text
outcome_label = class_label
model_probability = class_probability
```

The formatter should mark the highest probability outcome as the model pick.

---

## Canonical Prediction Outcome Artifact

Recommended path:

```text
data/predictions/model_outcomes.parquet
```

For Model Lab comparison, also write model-scoped outputs:

```text
data/predictions/by_model/{model_id}/model_outcomes.parquet
```

Recommended schema:

```text
prediction_run_id
prediction_timestamp

model_id
model_family
algorithm
prediction_type

model_artifact_dir
model_config_path

selected_event_id
selected_model_id

event_id
event_name
commence_time
fight_id

red_fighter
blue_fighter
red_fighter_id
blue_fighter_id

market_key
outcome_label
outcome_side
model_probability

is_model_pick
model_pick
model_confidence

passes_model_data_quality
passes_feature_validation
nonzero_feature_count
zero_feature_pct

feature_match_type
red_feature_match
blue_feature_match
```

Notes:

- `outcome_side` can be `red`, `blue`, `positive`, `negative`, or a class label depending on formatter type.
- `model_confidence` should be the model probability of the model pick for that fight/market.
- `is_model_pick` identifies the top outcome row within a fight/market/model.

---

## Market Odds Architecture

The market layer should also normalize to outcome rows.

Recommended artifact:

```text
data/market/market_outcomes.parquet
```

Recommended schema:

```text
snapshot_run_id
snapshot_timestamp

source
bookmaker
sport
region
market_key

event_id
event_name
commence_time
fight_id

red_fighter
blue_fighter

outcome_label
american_odds
decimal_odds
implied_probability

odds_match_type
odds_match_score
odds_min_single_score
matched_market_name
matched_outcome_name
```

Moneyline odds become outcome rows by fighter name.

Prop odds become outcome rows by market outcome label.

The odds normalizer should map provider-specific labels to canonical labels used by the prediction formatter.

---

## Betting Engine Architecture

The betting engine joins:

```text
model_outcomes
+
market_outcomes
```

on:

```text
fight_id
market_key
outcome_label
```

Then calculates:

```text
edge = model_probability - implied_probability
EV = expected value from model_probability and odds
stake = staking rule output
```

Recommended output:

```text
data/predictions/betting_outcomes.parquet
```

Recommended schema:

```text
decision_run_id
decision_timestamp

model_id
model_family
market_key
outcome_label

event_id
event_name
fight_id
red_fighter
blue_fighter

model_probability
implied_probability
edge
ev

american_odds
decimal_odds
bookmaker

is_model_pick
model_confidence

recommended_stake
bet_status
bet_reason
failed_filters
failed_filter_count

passes_model_quality_filter
passes_feature_validation_filter
passes_odds_match_filter
passes_edge_filter
passes_confidence_filter
passes_odds_range_filter
passes_positive_ev_filter
passes_all_bet_filters
```

The betting board becomes one row per bettable outcome, not one row per fight.

---

## Dashboard Integration

The future dashboard should read `betting_outcomes.parquet` and display rows by:

```text
event
fight
market
outcome
model probability
odds
edge
EV
stake
status
```

Example table:

| Fight | Market | Outcome | Model | Odds | Edge | EV | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Fighter A vs Fighter B | Moneyline | Fighter A | 64% | -150 | +4.0% | +$8 | Watchlist |
| Fighter A vs Fighter B | Goes Distance | Yes | 72% | +110 | +24.4% | +$51 | Strong Bet |

The dashboard should not assume red/blue columns as the canonical shape.

A display layer may pivot outcomes back into side-by-side cards, but storage and computation should remain outcome-based.

---

## Model Lab Integration

Model Lab should use model-scoped artifacts:

```text
data/predictions/by_model/{model_id}/model_outcomes.parquet
data/predictions/by_model/{model_id}/betting_outcomes.parquet
```

This allows comparison across:

- production moneyline model
- candidate moneyline model
- prop models
- method models
- future ensembles

Dashboard-selected model runs should pass model selection through:

```text
workflow input → environment variable → prediction runner
```

Runtime selection precedence:

```text
--model-id CLI argument
UFC_MODEL_ID environment variable
registry active model
```

---

## CLV Integration

CLV should also become outcome-based.

Recommended join keys:

```text
fight_id
market_key
outcome_label
bookmaker
```

This supports CLV for both moneyline and props.

Do not design CLV only around red/blue moneyline fields.

---

## Migration Plan

### Phase 1: Documentation and Contracts

- Add this document.
- Update older architecture docs to reference this as the target prediction architecture.
- Do not change code yet.

### Phase 2: Registry and Loader

- Add `configs/models/model_registry.yaml`.
- Add model registry loader.
- Add model config loader.
- Keep training config as source of truth.

### Phase 3: Generic Prediction Adapter

- Add generic model loader.
- Add prediction adapter.
- Add algorithm prediction plug-in for XGBoost.
- Add binary-matchup formatter.

### Phase 4: Outcome Prediction Artifact

- Add V2 runner that writes outcome-level predictions.
- Keep legacy prediction runner until dashboard/betting pipeline migration begins.

### Phase 5: Outcome Market Layer

- Rewrite market update to normalize all odds into outcome rows.
- Do not maintain a long-term dual output contract.

### Phase 6: Outcome Betting Engine

- Rewrite EV, filters, staking, watchlist, and official bet logic around outcome rows.

### Phase 7: Dashboard Rewrite

- Rewrite Betting Board around fight/market/outcome rows.
- Add Model Lab model selector and comparison views.

---

## Superseded Assumptions

The following assumptions are superseded for long-term architecture:

```text
Moneyline-specific red/blue columns are the canonical prediction/market/betting schema.
```

```text
Each model family needs its own adapter implementation.
```

```text
The dashboard betting board should be designed around one row per fight.
```

The approved target is:

```text
one row per fight/outcome/model/market
```

for prediction, market odds, betting decisions, CLV, and Model Lab comparison.
