# UFC Model Adapter Architecture

## Purpose

This document defines the future architecture for supporting multiple predictive models (XGBoost, Random Forest, Neural Networks, Ensembles, etc.) within the UFC betting pipeline.

Core principle:

- Models own their required features.
- The pipeline owns the prediction output contract.

This allows models to be swapped without changing the Betting Board, CLV Tracker, Bankroll Tracker, or Dashboard.

---

# High-Level Architecture

Feature Builders
    ↓
Model Adapter Layer
    ↓
Standard Prediction Output
    ↓
Betting Board / CLV / Dashboard

---

# Repository Structure

pipeline/
├── training/
├── backtesting/
├── modeling/
│   ├── adapters/
│   ├── model_registry.py
│   ├── model_loader.py
│   └── feature_contracts.py
├── prediction/
└── feature_engineering/

models/
├── UFC_Model_V5_XGBoost/
├── UFC_Model_V5_RF/
└── active_model.json

configs/
└── models/

---

# Model Bundle Contract

Every production model should contain:

- model.pkl
- feature_columns.pkl
- production_config.json
- confidence_buckets.parquet
- training_metrics.json

Example:

models/
└── UFC_Model_V5_XGBoost/
    ├── model.pkl
    ├── feature_columns.pkl
    ├── confidence_buckets.parquet
    ├── training_metrics.json
    └── production_config.json

---

# Feature Ownership

Different models may require different features.

Example:

XGBoost:
- Rolling EWM features
- Elo
- Reach differential
- Age differential
- Activity metrics

Random Forest:
- Elo differential
- Wrestling edge
- Reach differential
- Recent form

Models declare their own required feature set through:

- feature_columns.pkl
- production_config.json

The prediction pipeline must build only the features required by the active model.

---

# Adapter Layer

Prediction code should never directly call a specific model implementation.

Avoid:

xgb.predict_proba()

Use:

adapter = load_model(model_name)
predictions = adapter.predict(df)

Adapters are responsible for:

- Loading artifacts
- Feature alignment
- Missing feature validation
- Probability generation
- Confidence bucket lookup
- Standardized output creation

---

# Standard Prediction Output

Every model must return the same schema.

Required fields:

- fight_id
- red_name
- blue_name
- red_model_prob
- blue_model_prob
- model_pick
- model_version
- model_type
- confidence_bucket
- bucket_reliability_score
- prediction_timestamp

This allows downstream systems to remain model-agnostic.

---

# Confidence Buckets

Confidence buckets are model-specific artifacts.

Generated during backtesting.

Suggested location:

models/<model_name>/confidence_buckets.parquet

Example columns:

- bucket_min
- bucket_max
- sample_size
- avg_model_prob
- actual_win_rate
- calibration_error
- bucket_reliability_score

---

# Migration Plan

Phase 1
- Convert notebooks to Python modules.
- Preserve current XGBoost functionality.

Phase 2
- Standardize model bundle format.
- Standardize prediction outputs.

Phase 3
- Build adapter layer.
- Build model registry.

Phase 4
- Add Random Forest support.
- Add ensemble support.

Phase 5
- Enable model selection through configuration.

---

# Current Recommendation

Do not build the adapter layer yet.

Immediate priority:

1. Move training notebook logic into pipeline/training.
2. Move backtesting notebook logic into pipeline/backtesting.
3. Verify identical outputs.
4. Then introduce model bundles and adapters.

This minimizes risk while preserving the current production pipeline.