# UFC AI CLV Tracker – V2 Production Architecture 1

Status: Active
Version: 1
Last Updated: June 2026

This document is the authoritative production architecture reference after the fighter-state plugin migration, transform registry work, generic feature builder work, production cutover, parity validation, and V2 training framework implementation.

`V2 PRODUCTION ARCHITECTURE.md` is retained as a historical architecture snapshot.

---

## 1. Data Ingestion Layer

### Inputs

* UFCStats event pages
* UFCStats fight detail pages
* Fighter profile pages

### Outputs

* `data/master/ufc_master.parquet`
* `data/staging/*`
* `data/audits/*`
* `data/status/*`

### URL Boundary

URLs are permitted only in:

* scraper layers
* staging layers
* audit layers

URLs are removed at the mapper boundary.

### Modeling Boundary

Only IDs are retained in:

* `ufc_master.parquet`
* feature stores
* model artifacts
* prediction artifacts

---

## 2. Fighter State Platform

### Production Builder

```text
pipeline/features/state/plugin_history_builder.py
```

### Registry

```text
configs/features/raw_fighter_feature_registry.yaml
```

### Feature Families

#### Record State

* fights
* wins
* losses
* win_pct
* streak metrics
* days_since_last_fight

#### ELO State

* elo
* avg_opponent_elo
* best_win_elo
* worst_loss_elo

#### Striking Rates

* kd_avg
* kd_absorbed_avg
* splm
* sapm
* str_acc
* str_def

#### Grappling Rates

* td_avg
* td_acc
* td_def
* sub_avg
* ctrl_per_min
* ctrl_against_per_min

#### Finish Profile

* finish_rate
* ko_rate
* sub_win_rate
* decision_win_rate
* finish_loss_rate
* decision_loss_rate
* avg_fight_time

#### Recent Form

* recent_win_pct
* recent_splm
* recent_sapm
* recent_td_avg
* recent_finish_rate
* recent_avg_fight_time

#### EWM State

Exponentially weighted fighter metrics generated from historical fighter-state history.

### Outputs

```text
data/features/fighter_state_history.parquet
data/features/latest_fighter_state.parquet
```

### Validation Status

Validated against production:

* 72 fighter-feature parity checks
* 111 builder parity checks

Result:

```text
PASS = 100%
```

---

## 3. Feature Platform

### Transform Registry

```text
configs/features/transform_registry.yaml
```

### Built-In Transform Plugins

* red_minus_blue
* blue_minus_red
* ratio
* absolute_gap

### Generic Feature Graph

Feature graphs define:

* bundles
* transforms
* passthrough features

Example:

```text
configs/feature_views/moneyline_base.yaml
```

Feature views are generated from configuration rather than hardcoded feature creation.

---

## 4. Historical Feature Engineering

### Current State

Historical feature generation remains partially legacy.

Current warehouse:

```text
data/features/UFC_enhanced_rolling_features_EWM.parquet
```

Current shape:

```text
8574 rows
483 columns
```

Future goal:

Historical training datasets generated entirely from the feature graph platform.

---

## 5. Model Training

### Entrypoint

```bash
python -m pipeline.training.run_train_model \
    --config configs/models/moneyline_xgboost_v5.yaml
```

### Current Production Model

```text
moneyline_xgboost_v5
```

### Feature Contract

```text
124 features
```

Features are explicitly defined in the model configuration.

### Temporal Split

```text
Train       <= 2022
Calibration = 2023
Test        >= 2024
```

### Symmetry

```text
8574 rows
→
17148 rows
```

### Calibration

Method:

```text
isotonic
```

### Current Results

```text
Accuracy    : 0.8075
ROC-AUC     : 0.8948
Log Loss    : 0.3993
Brier Score : 0.1324
```

---

## 6. Live Feature Engineering

### Entrypoint

```bash
python -m pipeline.prediction.run_live_features \
    --model-id moneyline_xgboost_v5
```

### Inputs

* `data/predictions/ufc_live_card.parquet`
* `data/features/latest_fighter_state.parquet`
* model feature contract

### Outputs

* `data/predictions/live_model_features.parquet`
* `data/audits/ufc_live_feature_audit.parquet`

---

## 7. Prediction Platform

### Entrypoint

```bash
python -m pipeline.modeling.run_prediction \
    --model-id moneyline_xgboost_v5
```

### Outputs

* `data/predictions/model_outcomes.parquet`
* `data/predictions/by_model/*`

---

## 8. Market Platform

### Entrypoint

```bash
python -m pipeline.market.run_market_update_v2
```

### Required Secret

```text
ODDS_API_KEY
```

### Outputs

* `data/market/market_outcomes.parquet`
* `data/market/market_outcome_snapshots.parquet`

---

## 9. Betting Platform

### Entrypoint

```bash
python -m pipeline.betting.run_betting_outcomes_v2
```

### Outputs

* `data/predictions/betting_outcomes.parquet`
* betting audits

---

## 10. Dashboard Platform

### Entry Point

```text
dashboard.py
```

### Tabs

* Betting Board
* Line Movement
* Bankroll
* Model Lab
* Data Maintenance

---

## 11. Production Pipeline Flow

```text
ufc_master.parquet
        ↓
plugin_history_builder
        ↓
fighter_state_history.parquet
        ↓
latest_fighter_state.parquet
        ↓
ufc_live_card.parquet
        ↓
live_model_features.parquet
        ↓
model_outcomes.parquet
        ↓
market_outcomes.parquet
        ↓
betting_outcomes.parquet
        ↓
Streamlit Dashboard
```

---

## 12. Current Roadmap

### Remaining Work

* Historical feature graph migration
* Training warehouse generation from feature graphs
* Expanded model registry
* Additional algorithms
* CatBoost support
* Neural-network support
* Ensemble models
* Expanded feature-family audits
* Prop market development

### Guiding Principles

* Feature generation is independent from model training.
* Models select features rather than generate them.
* Fighter IDs are primary join keys.
* Confidence is distinct from probability.
* Feature families are independently auditable.
* Prediction, Market, and Betting layers remain generic and reusable.
