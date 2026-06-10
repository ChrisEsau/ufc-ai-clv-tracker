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

Current State

The V2 training framework is now operational and no longer relies exclusively on notebook execution.

Training is launched through a dedicated training runner.

Entrypoint

Command:

python -m pipeline.training.run_train_model `
    --config configs/models/moneyline_xgboost_v5.yaml

Primary runner:

pipeline/training/run_train_model.py

⸻

Inputs

Training consumes:

* Historical feature warehouse
* Model configuration
* Explicit feature contract
* Outcome labels
* Temporal split configuration

Current feature warehouse:

data/features/UFC_enhanced_rolling_features_EWM.parquet

Current warehouse shape:

8574 rows
483 columns

⸻

Current Training Architecture

Training modules:

pipeline/training/
feature_selection.py
symmetry.py
temporal_split.py
model_training.py
calibration.py
metrics.py
run_train_model.py
algorithms/
    xgboost_trainer.py

Execution flow:

Load Model Config
        ↓
Load Feature Warehouse
        ↓
Resolve Explicit Feature Contract
        ↓
Apply Symmetry
        ↓
Temporal Train/Calibration/Test Split
        ↓
Train Model
        ↓
Probability Calibration
        ↓
Metrics / Threshold Analysis
        ↓
Confidence Bucket Generation
        ↓
Artifact Persistence

⸻

Feature Contract

Current production model:

moneyline_xgboost_v5

Configuration:

configs/models/moneyline_xgboost_v5.yaml

Current feature count:

124

Features are explicitly declared inside the model configuration.

Training does not infer production feature lists from naming conventions.

Models select features from feature warehouses rather than generating features.

⸻

Temporal Split

Current split architecture:

Train       <= 2022
Calibration = 2023
Test        >= 2024

Purpose:

* Prevent future leakage
* Support calibration without contamination
* Preserve realistic deployment conditions

⸻

Symmetry

Current training pipeline applies fighter-side symmetry augmentation.

Purpose:

* Remove red/blue corner bias
* Improve model robustness
* Double training observations

Observed transformation:

8574 rows
→
17148 rows

⸻

Calibration

Calibration is a first-class training stage.

Current method:

isotonic

Calibration is fit using calibration rows only and evaluated against an independent test set.

Outputs include:

* Raw probabilities
* Calibrated probabilities
* Confidence buckets
* Calibration diagnostics

⸻

Current Outputs

Model artifacts are written to:

models/moneyline/xgboost_v5/

Current outputs include:

* raw_model.joblib
* calibrated_model.joblib
* feature_columns.joblib
* feature_columns.json
* metrics.json
* threshold_sweep.parquet
* confidence_buckets.parquet
* model_card.yaml

⸻

Current Successful Framework Run

Observed results:

Feature dataframe shape: (8574, 483)
Resolved feature count: 124
Model dataframe shape: (17148, 483)
Train rows        : 13874
Calibration rows  : 1040
Test rows         : 2234
Best threshold    : 0.47
Accuracy          : 0.8075
ROC-AUC           : 0.8948
Log loss          : 0.3993
Brier score       : 0.1324

⸻

Future Training Roadmap

Planned enhancements:

* Model registry
* Additional algorithm adapters
* CatBoost support
* Neural-network support
* Ensemble support
* Expanded calibration reporting
* Automated model comparison
* Model Lab integration

Guiding principle:

Feature generation is independent from model training.
Model configs define:
- algorithm
- feature contract
- split strategy
- calibration strategy
- hyperparameters
- artifact locations
Training code remains model-agnostic.

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