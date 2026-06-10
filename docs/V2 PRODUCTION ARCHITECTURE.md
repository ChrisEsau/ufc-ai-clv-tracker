UFC AI CLV Tracker – V2 Production Architecture

Overview

This document describes the current production architecture for the UFC AI CLV Tracker platform, including data ingestion, feature engineering, model training, prediction generation, market ingestion, betting intelligence, dashboard consumption, and GitHub Actions orchestration.

1. Data Ingestion Layer

Inputs

* UFCStats event pages
* UFCStats fight detail pages
* Fighter profile data

Outputs

* data/master/ufc_master.parquet
* data/staging/*
* data/audits/*
* data/status/*

Architecture Rules

URL Boundary

URLs are allowed only in:

* scraper layers
* staging layers
* audit layers

URLs are removed at the mapper boundary.

Modeling Boundary

Only IDs are retained in:

* ufc_master.parquet
* feature stores
* model artifacts
* prediction artifacts

2. Historical Feature Engineering

Current State

Historical feature engineering remains partially legacy and notebook-driven.

Known historical artifacts include:

* UFC_enhanced_rolling_features_EWM.parquet
* ufc_rolling_features_EWM.parquet

Current Feature Families

Base Features

Career and fight-level statistics.

ELO Features

Skill ratings and differential metrics.

EWM Features

Exponentially weighted fighter metrics.

Recent Form Features

Recent performance windows.

Fighter Profile Features

Physical and demographic information.

Engineered V5 Features

Matchup-derived features including:

* striking_edge
* grappling_edge
* finish_volatility
* chin_risk_diff
* experience_ratio_diff
* aggression_index_diff
* wrestling_mismatch_diff
* submission_mismatch_diff

Future Refactor Target

Target architecture:

* data/features/base_features.parquet
* data/features/elo_features.parquet
* data/features/ewm_features.parquet
* data/features/recent_form_features.parquet
* data/features/market_features.parquet
* data/features/fighter_state.parquet

Feature generation should be independent from model training.

3. Model Training

Inputs

* Historical feature dataset
* Model config
* Feature selection list
* Outcome labels
* Validation configuration

Configs

* configs/models/model_registry.yaml
* configs/models/moneyline_xgboost_v5.yaml

Outputs

* raw_model.joblib
* calibrated_model.joblib
* feature_columns.joblib

Model artifacts reside under:

models/moneyline/xgboost_v5/

Additional Outputs

* Calibration buckets
* Confidence buckets
* Validation metrics
* Feature registry
* Training reports

4. Live Feature Engineering V2

Workflow

.github/workflows/run-live-features-v2.yml

Command

python -m pipeline.prediction.run_live_features --model-id moneyline_xgboost_v5

Inputs

* data/predictions/ufc_live_card.parquet
* Current fighter feature store
* Model feature list
* Model registry

Core Files

* pipeline/prediction/run_live_features.py
* pipeline/prediction/live_feature_builder.py
* ufc_feature_engineering.py
* pipeline/common/paths.py

Outputs

* data/predictions/live_model_features.parquet
* data/audits/ufc_live_feature_audit.parquet

5. Prediction V2

Workflow

.github/workflows/run-prediction-v2.yml

Command

python -m pipeline.modeling.run_prediction --model-id moneyline_xgboost_v5

Inputs

* Model registry
* Model config
* Trained model artifacts
* Live card
* Current fighter features

Core Files

* pipeline/modeling/run_prediction.py
* pipeline/modeling/model_registry.py
* pipeline/modeling/model_config.py
* pipeline/modeling/model_loader.py
* pipeline/modeling/prediction_adapter.py
* pipeline/prediction/live_feature_builder.py

Outputs

* data/predictions/live_model_features.parquet
* data/predictions/model_outcomes.parquet
* data/predictions/by_model/moneyline_xgboost_v5/model_outcomes.parquet
* data/audits/ufc_live_feature_audit.parquet

6. Market V2

Workflow

.github/workflows/run-market-v2.yml

Command

python -m pipeline.market.run_market_update_v2

Required Secret

ODDS_API_KEY

Inputs

* The Odds API
* data/predictions/ufc_live_card.parquet

Outputs

* data/market/market_outcomes.parquet
* data/market/market_outcome_snapshots.parquet
* data/audits/ufc_market_outcome_audit.parquet
* data/audits/ufc_market_match_audit_v2.parquet

7. Betting Outcomes V2

Workflow

.github/workflows/run-betting-outcomes-v2.yml

Command

python -m pipeline.betting.run_betting_outcomes_v2

Inputs

* data/predictions/model_outcomes.parquet
* data/market/market_outcomes.parquet
* Risk settings

Outputs

* data/predictions/betting_outcomes.parquet
* data/audits/ufc_betting_outcomes_audit.parquet

8. Dashboard Architecture

Entry Point

dashboard.py

Primary Tabs

* Betting Board
* Line Movement
* Bankroll
* Model Lab
* Data Maintenance

9. Betting Board V2

Source of Truth

data/predictions/betting_outcomes.parquet

10. Bankroll / Risk Settings

Controls

* Bankroll
* Kelly fraction
* Max stake %
* Minimum edge
* Minimum confidence
* Minimum odds
* Maximum odds

11. GitHub Actions Production Flow

Current workflow order:

1. Run Live Features V2
2. Run Prediction V2
3. Run Market V2
4. Run Betting Outcomes V2

12. Production Artifact Chain

ufc_master.parquet
↓
historical/current fighter feature store
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
Streamlit Betting Board

13. Current Architectural Gaps

1. Training runner not fully standardized.
2. Historical feature engineering not fully split by feature family.
3. Fighter-state store should become a first-class artifact.
4. Feature registry should become the authoritative source of truth.
5. Feature-family audits should be expanded.
6. Betting Board event filter should likely source from betting_outcomes only.
7. Prop market development is currently paused.

Guiding Principles

* Feature generation should be independent of model training.
* Models select features rather than generate them.
* Fighter IDs are primary join keys.
* Confidence is distinct from probability.
* Feature families should be independently auditable.
* Prediction, Market, and Betting layers should remain generic and reusable across future market types.