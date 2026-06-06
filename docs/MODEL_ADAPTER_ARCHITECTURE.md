# UFC Model Adapter Architecture

## Purpose

This document defines the future architecture for supporting multiple predictive models across multiple UFC betting market families.

Supported future model types may include:

- XGBoost
- Random Forest
- Neural Networks
- Ensembles
- Market-specific classifiers

Supported market families may include:

- Moneyline
- KO/TKO props
- Submission props
- Decision props
- Goes Distance / Does Not Go Distance
- Round totals
- Round-specific props

Core principles:

- Market families own their feature-builder layer.
- Models own their required feature contract.
- The pipeline owns standardized prediction output contracts.
- Downstream systems should remain model-agnostic and market-aware.

This allows models and markets to expand without rewriting the Betting Board, CLV Tracker, Bankroll Tracker, or Dashboard.

---

# High-Level Architecture

```text
Raw UFC Data
    ↓
Base Rolling Feature Store
    ↓
Market Family Feature Layer
    ↓
Model Adapter Layer
    ↓
Standard Prediction Output
    ↓
Betting Board / CLV / Bankroll / Dashboard
```

For example:

```text
Base Rolling Features
    ↓
Moneyline Feature Builder
    ↓
Moneyline Model Adapter
    ↓
Moneyline Prediction Output
```

and:

```text
Base Rolling Features
    ↓
Prop Feature Builder
    ↓
Prop Model Adapter
    ↓
Prop Prediction Output
```

---

# Repository Structure

Expected long-term structure:

```text
pipeline/
├── features/
│   ├── base/
│   ├── moneyline/
│   └── props/
│
├── training/
│   ├── moneyline/
│   └── props/
│
├── backtesting/
│   ├── moneyline/
│   └── props/
│
├── modeling/
│   ├── adapters/
│   │   ├── base_adapter.py
│   │   ├── xgboost_adapter.py
│   │   ├── random_forest_adapter.py
│   │   ├── ensemble_adapter.py
│   │   └── prop_adapter.py
│   ├── model_registry.py
│   ├── model_loader.py
│   └── feature_contracts.py
│
├── prediction/
├── clv/
└── bankroll/
```

Model bundles should be grouped first by market family:

```text
models/
├── moneyline/
│   ├── UFC_Model_V5_XGBoost/
│   ├── UFC_Model_V5_RF/
│   └── UFC_Model_V5_Ensemble/
│
└── props/
    ├── UFC_PROP_KO_TKO_V1/
    ├── UFC_PROP_SUB_V1/
    ├── UFC_PROP_DEC_V1/
    ├── UFC_PROP_GOES_DISTANCE_V1/
    └── UFC_PROP_ROUNDS_V1/
```

Configs should follow the same market-family structure:

```text
configs/
└── models/
    ├── moneyline/
    │   ├── xgboost_v5.yaml
    │   ├── random_forest_v1.yaml
    │   └── ensemble_v1.yaml
    │
    └── props/
        ├── ko_tko_v1.yaml
        ├── submission_v1.yaml
        ├── decision_v1.yaml
        ├── goes_distance_v1.yaml
        └── rounds_v1.yaml
```

---

# Model Bundle Contract

Every production model should contain:

```text
model.pkl
feature_columns.pkl
production_config.json
confidence_buckets.parquet
training_metrics.json
```

Optional artifacts:

```text
calibration_model.pkl
feature_importance.csv
backtest_predictions.parquet
model_card.md
```

Example:

```text
models/
└── moneyline/
    └── UFC_Model_V5_XGBoost/
        ├── model.pkl
        ├── feature_columns.pkl
        ├── confidence_buckets.parquet
        ├── training_metrics.json
        └── production_config.json
```

Example prop model:

```text
models/
└── props/
    └── UFC_PROP_SUB_V1/
        ├── model.pkl
        ├── feature_columns.pkl
        ├── confidence_buckets.parquet
        ├── training_metrics.json
        └── production_config.json
```

---

# Feature Ownership

Different market families require different feature builders.

Moneyline examples:

- Elo differential
- Rolling EWM differential stats
- Reach differential
- Age differential
- Recent form
- Wrestling mismatch
- Striking edge

KO/TKO prop examples:

- KO win rate
- Knockdown rate
- Striking pace
- Opponent knockdowns absorbed
- Chin risk
- Finish loss rate
- Defensive striking weakness

Submission prop examples:

- Submission win rate
- Submission attempt rate
- Takedown rate
- Takedown defense
- Control time
- Submission mismatch
- Grappling edge

Decision / Goes Distance examples:

- Average fight time
- Decision win rate
- Finish rate
- Finish loss rate
- Pace
- Durability
- Total rounds
- Title fight flag

Round prop examples:

- Early finish rate
- Late finish rate
- Round-specific finish distribution
- Cardio / fight-time history
- Pace decay

Models declare their own required feature set through:

- `feature_columns.pkl`
- `production_config.json`
- optional YAML config under `configs/models/...`

The prediction pipeline must build or load the features required by the selected model and market family.

---

# Adapter Layer

Prediction code should never directly call a specific model implementation.

Avoid:

```python
xgb.predict_proba(X)
```

Use:

```python
adapter = load_model(model_name)
predictions = adapter.predict(df)
```

Adapters are responsible for:

- Loading artifacts
- Reading model metadata
- Validating market family
- Aligning feature columns
- Validating missing features
- Calling the underlying model
- Applying calibration when needed
- Looking up confidence buckets
- Returning standardized output

---

# Standard Prediction Outputs

## Moneyline Prediction Output

Required fields:

```text
fight_id
event_name
red_name
blue_name
red_model_prob
blue_model_prob
model_pick
model_version
model_type
market_family
feature_set
confidence_bucket
bucket_reliability_score
prediction_timestamp
```

## Prop Prediction Output

Required fields:

```text
fight_id
event_name
market_family
market_type
selection
fighter_name
model_prob
market_implied_prob
edge
ev
model_name
model_version
model_type
feature_set
confidence_bucket
bucket_reliability_score
prediction_timestamp
```

Examples of `market_type`:

```text
KO_TKO
SUBMISSION
DECISION
GOES_DISTANCE
DOES_NOT_GO_DISTANCE
ROUND_TOTAL_OVER
ROUND_TOTAL_UNDER
ROUND_1_FINISH
```

Downstream systems should consume these stable schemas and should not care whether the source model is XGBoost, Random Forest, Neural Network, or an Ensemble.

---

# Confidence Buckets

Confidence buckets are model-specific and market-specific artifacts.

Generated during backtesting.

Suggested location:

```text
models/<market_family>/<model_name>/confidence_buckets.parquet
```

Example columns:

```text
market_family
market_type
bucket_min
bucket_max
sample_size
avg_model_prob
actual_win_rate
calibration_error
bucket_reliability_score
```

Moneyline confidence buckets measure how well model probability bands predicted fight winners.

Prop confidence buckets measure how well model probability bands predicted a specific prop outcome.

---

# Data Flow by Market Family

## Moneyline

```text
Raw UFC Data
    ↓
Base Rolling Feature Store
    ↓
Moneyline Feature Builder
    ↓
Moneyline Model
    ↓
Moneyline Predictions
    ↓
Betting Board
```

## Props

```text
Raw UFC Data
    ↓
Base Rolling Feature Store
    ↓
Prop Label Builder
    ↓
Prop Feature Builder
    ↓
Prop Model
    ↓
Prop Predictions
    ↓
Betting Board
```

Prop models need both specialized labels and specialized features. For example, a submission model requires a target column for whether a fight ended by submission, not simply whether red won.

---

# Migration Plan

Phase 1

- Convert the rolling feature notebook into Python modules.
- Preserve existing V5 moneyline feature output.

Phase 2

- Convert the XGBoost training notebook into `pipeline/training/moneyline/` modules.
- Preserve current XGBoost V5 behavior.

Phase 3

- Convert the backtest notebook into `pipeline/backtesting/moneyline/` modules.
- Save confidence buckets as required model artifacts.

Phase 4

- Standardize model bundle format.
- Standardize moneyline prediction outputs.

Phase 5

- Build model adapter layer and model registry.

Phase 6

- Add Random Forest and ensemble support for moneyline.

Phase 7

- Add prop feature builders, prop labels, prop backtests, and prop model adapters.

---

# Current Recommendation

Do not build the adapter layer first.

Immediate priority:

1. Convert the rolling feature notebook into reusable feature-store modules.
2. Convert the training notebook into moneyline training modules.
3. Convert the backtest notebook into moneyline backtesting modules.
4. Verify identical outputs.
5. Then introduce model bundles and adapters.
6. Then add prop models as a separate market family.

This minimizes risk while preserving the current production moneyline pipeline.