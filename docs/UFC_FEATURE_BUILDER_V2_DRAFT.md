# UFC Feature Builder V2 Draft

## Status

Draft architecture for later implementation.

This document captures the agreed direction for refactoring feature generation after the first Prediction V2 moneyline path is working end-to-end.

Do not treat this as implemented code.

Primary near-term priority remains:

```text
moneyline_xgboost_v5
        ↓
Prediction V2 runner
        ↓
model_outcomes.parquet
```

After that milestone, feature generation refactor becomes a high-priority item before expanding aggressively into prop models, Model Lab comparisons, and market-aware modeling.

---

## Core Principles

### 1. Fighter IDs Are Canonical

Feature generation, training, live prediction, settlement, and model outputs should use fighter IDs as the identity backbone.

Names are display fields only.

Canonical fighter identity fields:

```text
r_id
b_id
winner_id
red_fighter_id
blue_fighter_id
fighter_id
```

Do not rely on fighter names for feature lookup or target generation.

---

### 2. Training Needs Historical Results

Training starts from completed historical fights because it needs both:

```text
prefight state
postfight result
```

Training flow:

```text
completed historical fights
        ↓
sort by date
        ↓
for each fight:
    get red fighter state before fight
    get blue fighter state before fight
    build requested features
    create target from result using IDs
    update fighter states after fight
        ↓
training matrix
```

Target example:

```text
target = 1 if winner_id == r_id else 0
```

---

### 3. Live Prediction Does Not Need Upcoming Results

Live prediction uses the same historical completed fight data to build fighter states, but it does not need the result of the upcoming fight.

Live flow:

```text
completed historical fights before event date
        ↓
build latest fighter states by fighter_id
        ↓
upcoming card with red_fighter_id / blue_fighter_id
        ↓
build requested features
        ↓
prediction
```

---

### 4. Avoid Permanent Training/Live Feature Duplication

Training and live prediction should not have separate formula implementations.

Target pattern:

```text
shared feature engine
        ↓
training mode
        ↓
live mode
```

Not:

```text
training feature formulas
live feature formulas copied separately
```

---

## Proposed Architecture

```text
ufc_master.parquet
        ↓
Feature Builder V2
        ↓
Feature Plugins
        ↓
Model-ready feature dataframe
        ↓
Training or prediction adapter
```

The feature builder should support two modes:

```text
historical_training
live_prediction
```

Both modes should use the same feature registry and plugins.

---

## Feature System Layers

### Layer 1: Historical Fight Results

Authoritative source:

```text
data/master/ufc_master.parquet
```

Required training fields:

```text
fight_id
event_id
date
r_id
b_id
winner_id
method
finish_round
match_time_sec
```

This layer stores facts/results.

---

### Layer 2: Fighter State Timeline

Potential future derived asset:

```text
data/features/fighter_state_history.parquet
```

Grain:

```text
one row per fighter_id / as_of_date / state_version
```

Example columns:

```text
fighter_id
as_of_date
state_version
fights
wins
losses
elo
win_pct
splm
sapm
td_avg
td_acc
td_def
sub_avg
finish_rate
ko_rate
avg_fight_time
```

This is not the model feature contract. It is reusable point-in-time state.

---

### Layer 3: Feature Plugins

Plugins convert fighter state, event context, market context, and matchup context into model-ready columns.

Potential plugins:

```text
pipeline/features/plugins/
  fighter_state.py
  side_features.py
  diff_features.py
  ratio_features.py
  interaction_features.py
  ewm_features.py
  recent_form_features.py
  engineered_v5.py
  event_context.py
  market_context.py
```

Plugins should declare what they produce and what they require.

Example:

```text
plugin: ewm_features
requires: fighter_id history + base metric
produces: ewm_elo_diff, ewm_finish_rate_diff, ...
```

---

### Layer 4: Feature Registry

Potential config path:

```text
configs/features/ufc_feature_registry_v2.yaml
```

The registry defines feature metadata, not necessarily all formula code.

Example:

```yaml
features:
  ewm_elo_diff:
    plugin: ewm_features
    base_metric: elo
    output_column: ewm_elo_diff
    point_in_time_safe: true
    live_safe: true

  event_altitude_ft:
    plugin: event_context
    output_column: event_altitude_ft
    point_in_time_safe: true
    live_safe: true

  finish_interaction:
    plugin: interaction_features
    required_metrics:
      - finish_rate
    output_column: finish_interaction
    point_in_time_safe: true
    live_safe: true
```

---

### Layer 5: Model YAML Feature Selection

Model configs should continue to define the final model feature contract.

Example:

```yaml
features:
  selection_mode: explicit
  feature_columns:
    - ewm_elo_diff
    - red_finish_rate
    - blue_finish_rate
    - finish_interaction
    - event_altitude_ft
```

Important distinction:

```text
Feature builder may use hidden internal inputs.
Model YAML controls final output columns.
```

Example:

```text
ewm_elo_diff requires Elo internally,
but the model does not need elo_diff unless the YAML requests it.
```

---

## Feature Types To Support

The current V5 moneyline model is mostly differential features.

Future models, especially props, need more flexibility.

Feature Builder V2 should support:

```text
side features
  red_finish_rate
  blue_finish_rate

differential features
  finish_rate_diff
  elo_diff

ratio features
  finish_rate_ratio

interaction features
  finish_interaction
  altitude_finish_pressure

EWM features
  ewm_elo_diff
  ewm_finish_rate_diff

recent-form features
  recent_form_finish_rate_diff

event context features
  event_altitude_ft
  high_altitude_flag

market context features
  open_implied_probability
  current_implied_probability
  line_movement
```

---

## Prop Modeling Example With Altitude

Model:

```text
goes_distance_xgboost_v2
```

Feature contract:

```yaml
features:
  feature_columns:
    - red_finish_rate
    - blue_finish_rate
    - avg_fight_time_diff
    - finish_volatility
    - event_altitude_ft
    - altitude_finish_pressure
```

Builder behavior:

```text
red_finish_rate / blue_finish_rate
        from fighter state plugin

avg_fight_time_diff
        from diff plugin

finish_volatility
        from engineered finish plugin

event_altitude_ft
        from event context plugin

altitude_finish_pressure
        from event-context × finish-rate interaction plugin
```

Training then runs:

```text
X = feature_df[model_config.features.feature_columns]
y = prop target
```

---

## Training Mode

Training mode needs historical completed fight results.

For each completed fight:

```text
1. use r_id and b_id
2. get point-in-time fighter states before fight date
3. build requested model features
4. create target from completed result
5. after feature row is captured, update fighter states with the fight result
```

This preserves point-in-time safety.

---

## Live Prediction Mode

Live mode uses an upcoming card with IDs.

Required live card fields:

```text
event_id
event_name
commence_time
fight_id
red_fighter_id
blue_fighter_id
red_fighter
blue_fighter
```

Flow:

```text
1. load completed historical fights before event date
2. build latest fighter states by fighter_id
3. join upcoming fighters by red_fighter_id / blue_fighter_id
4. build requested model features
5. return model-ready rows to the prediction adapter
```

---

## Wide Warehouse vs On-Demand vs Hybrid

### Wide Warehouse

Build all features and store all columns.

Pros:

```text
fast training
simple model experiments
```

Cons:

```text
large artifacts
many unused columns
feature drift risk if live duplicates logic
```

### On-Demand Features

Build only requested features per model run.

Pros:

```text
clean model-specific feature contracts
minimal unused outputs
```

Cons:

```text
slower training
more builder complexity
```

### Recommended Long-Term Direction

Hybrid:

```text
store reusable point-in-time fighter state history
compute requested model features via plugins
```

This avoids a giant warehouse while keeping training/live logic unified.

---

## Deferred Implementation Notes

This is not required before the first Prediction V2 milestone.

Recommended order:

```text
1. Finish moneyline Prediction V2 runner
2. Produce model_outcomes.parquet
3. Refactor Feature Builder V2
4. Then expand market outcomes, betting outcomes, props, and Model Lab
```

---

## Open Design Questions

These should be revisited before implementation:

1. Should `fighter_state_history.parquet` be persisted, or should state be recomputed per run?
2. Should feature plugins declare dependencies in code, YAML, or both?
3. How should feature versions be recorded in model artifacts?
4. Should market features be allowed in production models, or kept separate from pure fight-skill models?
5. How should point-in-time odds snapshots be joined for historical training?
6. Should Feature Builder V2 create model-scoped feature artifacts for reproducibility?

---

## Locked Decision For Now

Feature-generation refactoring is high priority, but it should not block completing the first Prediction V2 moneyline artifact.

The current V2 prediction implementation may use the existing V5 feature artifact path until Feature Builder V2 is implemented.
