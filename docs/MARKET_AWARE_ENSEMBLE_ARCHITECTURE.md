# Market-Aware Ensemble Architecture

## Purpose

This document defines the production architecture for expandable market-aware UFC moneyline ensembles.

The goal is not to productionize a single hardcoded V13 Top60 model. The goal is to create a reusable ensemble framework that can support:

- Different feature counts
- Different feature sets per ensemble member
- Favorite/dog perspective modeling
- Market-aware and non-market-aware child models
- Any number of child models
- Simple average, weighted average, and future meta-model combiners
- Production-compatible prediction outputs

V13 Top60 is the first implementation target, not the final architecture.

---

## Current State

The current production prediction runner supports single-model moneyline models:

    live card
    + live fighter/RFS features
    -> one calibrated model
    -> two outcome rows per fight

The current experimental V13 full ensemble requires more:

    live card
    + live fighter/RFS features
    + market moneyline odds
    + favorite/dog perspective transform
    + multiple child models
    + combiner
    + decision rule
    -> ensemble outcomes

The experimental V13 scorer proved this can run locally by:

- Training four child XGBoost models from a historical favorite/dog market-aware dataset
- Joining live model features with data/market/market_outcomes.parquet
- Creating favorite/dog perspective features
- Scoring the upcoming card
- Applying the saved production rule

---

## Design Principle

Do not build a hardcoded V13 four-model scorer.

Build a generic ensemble engine.

V13 should be represented as configuration:

    ensemble framework
    + V13 feature set
    + V13 member definitions
    + V13 combiner
    + V13 decision rule

Future versions should be able to add or remove members without rewriting core training or prediction code.

---

## Required Capabilities

### 1. Expandable Feature Counts

The ensemble must support more than 60 features.

Supported selection modes should include:

- top_n_from_ranking
- explicit_feature_list
- feature_family_filter
- all_available

Example model variants:

- moneyline_xgboost_v13_ensemble_top60
- moneyline_xgboost_v13_ensemble_top90
- moneyline_xgboost_v13_ensemble_top125
- moneyline_xgboost_v13_ensemble_top175
- moneyline_xgboost_v13_ensemble_top250
- moneyline_xgboost_v13_ensemble_top347
- moneyline_xgboost_v14_ensemble_custom_rfs

No code should assume top60.

---

### 2. Different Feature Sets Per Child Model

Each ensemble member must be able to define its own feature set.

Examples:

    feature_only_favorite:
      favorite perspective Top60, no market features

    market_aware_favorite:
      favorite perspective Top125 + market features

    dog_rfs_model:
      dog perspective RFS-only features

    dog_market_model:
      dog perspective market + recent form features

    volatility_filter:
      market movement / disagreement features only

Feature selection belongs inside each member config, not only at the parent ensemble level.

---

### 3. Any Number of Child Models

The framework must support 3, 4, 5, or more child models.

The trainer should loop over:

    ensemble_members[]

The predictor should load and score:

    members/<member_id>/model.joblib

The combiner should reference whichever members are assigned to each output group.

---

### 4. Flexible Combiners

The current V13 experiment uses simple average:

    ensemble_favorite_probability =
      average(feature_only_favorite_probability, market_aware_favorite_probability)

    ensemble_dog_probability =
      average(feature_only_dog_probability, market_aware_dog_probability)

Production must also support weighted combinations:

    ensemble_favorite_probability =
      0.35 * feature_only_favorite_probability
    + 0.65 * market_aware_favorite_probability

Combiner types should be designed for future expansion:

- simple_average
- weighted_average
- rule_weighted_average
- stacked_meta_model
- gated_ensemble

Initial implementation can support grouped_weighted_average with equal weights as the default.

---

## Architecture Layers

The system should be separated into four layers.

### Layer 1: Wide Feature View

Build a reusable historical favorite/dog market-aware training view.

Target path:

    data/features/moneyline_favdog_market_feature_view.parquet

This view should be wide and model-agnostic. It should contain all eligible features, not only Top60.

Required column families:

    fight_id
    date
    event_name

    favorite_fighter_id
    dog_fighter_id
    favorite_fighter_name
    dog_fighter_name

    favorite_won
    dog_won

    favpersp__<clean_feature>
    dogpersp__<clean_feature>

    favorite_odds
    dog_odds
    favorite_implied_probability
    dog_implied_probability
    market_prob_gap
    abs_market_prob_gap
    favorite_odds_abs
    dog_odds_abs
    market_price_width

This replaces the research dependency on:

    data/model_lab/dual_regular_dog_model_2018_2025_market_aware/dual_regular_dog_market_aware_dataset.parquet

The research dataset may remain useful for validation, but production training should eventually depend on data/features.

---

### Layer 2: Ensemble Member Models

Each member is trained independently.

Each member defines:

- member_id
- algorithm
- target
- perspective
- feature_set
- uses_market_features
- hyperparameters
- calibration mode

Example members:

- feature_only_favorite
- feature_only_dog
- market_aware_favorite
- market_aware_dog

Future examples:

- favorite_rfs_only
- dog_rfs_only
- market_pressure_model
- line_movement_model
- favorite_recent_form_model
- dog_longshot_model

---

### Layer 3: Combiner

The combiner consumes child model probabilities and emits named ensemble outputs.

Initial required outputs:

- ensemble_favorite_probability
- ensemble_dog_probability

Combiner config should define groups:

    favorite_probability group:
      members:
        - feature_only_favorite
        - market_aware_favorite
      weights:
        feature_only_favorite: 0.50
        market_aware_favorite: 0.50

    dog_probability group:
      members:
        - feature_only_dog
        - market_aware_dog
      weights:
        feature_only_dog: 0.50
        market_aware_dog: 0.50

---

### Layer 4: Decision Rule

Decision rules must remain separate from child models and combiner logic.

Initial V13 rule:

    favorite bet:
      ensemble_favorite_probability >= threshold
      ensemble_dog_probability <= max dog probability
      ensemble_favorite_edge >= min edge

    dog bet:
      ensemble_favorite_probability <= max favorite probability
      ensemble_dog_probability >= threshold
      ensemble_dog_edge >= min edge
      dog_odds <= max dog odds

The rule should output:

- bet_side
- pick_fighter_id
- pick_fighter_name
- pick_probability
- pick_edge
- pick_odds
- rule_passed
- rule_reason

---

## Proposed Config Shape

Example config:

    model_id: moneyline_xgboost_v13_ensemble_top60
    model_family: moneyline
    market_key: moneyline
    algorithm: ensemble
    status: experimental

    architecture:
      type: ensemble
      name: configurable_moneyline_ensemble
      requires_market: true
      perspective_mode: favorite_dog

    data:
      training_view_path: data/features/moneyline_favdog_market_feature_view.parquet
      live_features_path: data/predictions/live_model_features.parquet
      market_outcomes_path: data/market/market_outcomes.parquet

    artifacts:
      output_dir: models/moneyline/moneyline_xgboost_v13_ensemble_top60

    market:
      preferred_bookmaker: DraftKings
      market_key: moneyline

    ensemble_members:
      - member_id: feature_only_favorite
        algorithm: xgboost
        target: favorite_won
        perspective: favorite
        feature_set:
          mode: top_n_from_ranking
          top_n: 60
          ranking_path: models/moneyline/moneyline_xgboost_v13_ensemble_top60/v13_stable_feature_ranking.csv
          prefix: favpersp__
          include_market_features: false

      - member_id: feature_only_dog
        algorithm: xgboost
        target: dog_won
        perspective: dog
        feature_set:
          mode: top_n_from_ranking
          top_n: 60
          ranking_path: models/moneyline/moneyline_xgboost_v13_ensemble_top60/v13_stable_feature_ranking.csv
          prefix: dogpersp__
          include_market_features: false

      - member_id: market_aware_favorite
        algorithm: xgboost
        target: favorite_won
        perspective: favorite
        feature_set:
          mode: top_n_from_ranking
          top_n: 60
          ranking_path: models/moneyline/moneyline_xgboost_v13_ensemble_top60/v13_stable_feature_ranking.csv
          prefix: favpersp__
          include_market_features: true

      - member_id: market_aware_dog
        algorithm: xgboost
        target: dog_won
        perspective: dog
        feature_set:
          mode: top_n_from_ranking
          top_n: 60
          ranking_path: models/moneyline/moneyline_xgboost_v13_ensemble_top60/v13_stable_feature_ranking.csv
          prefix: dogpersp__
          include_market_features: true

    market_features:
      - favorite_odds
      - dog_odds
      - favorite_implied_probability
      - dog_implied_probability
      - market_prob_gap
      - abs_market_prob_gap
      - favorite_odds_abs
      - dog_odds_abs
      - market_price_width

    combiner:
      type: grouped_weighted_average
      groups:
        ensemble_favorite_probability:
          members:
            feature_only_favorite: 0.50
            market_aware_favorite: 0.50
        ensemble_dog_probability:
          members:
            feature_only_dog: 0.50
            market_aware_dog: 0.50

    decision_rule:
      type: threshold_rule
      favorite_rule:
        ensemble_favorite_probability_min: 0.68
        ensemble_dog_probability_max: 0.45
        ensemble_favorite_edge_min: 0.05
      dog_rule:
        ensemble_favorite_probability_max: 0.52
        ensemble_dog_probability_min: 0.40
        ensemble_dog_edge_min: 0.05
        dog_odds_max: 400

---

## Artifact Layout

Target artifact layout:

    models/moneyline/<model_id>/
      ensemble_manifest.json
      model_config.yaml
      training_config_snapshot.yaml
      feature_set_resolved.json
      production_rule.json

      members/
        feature_only_favorite/
          model.joblib
          feature_columns.json
          metrics.json

        feature_only_dog/
          model.joblib
          feature_columns.json
          metrics.json

        market_aware_favorite/
          model.joblib
          feature_columns.json
          metrics.json

        market_aware_dog/
          model.joblib
          feature_columns.json
          metrics.json

Future member count should not require folder structure changes.

---

## Ensemble Manifest

The manifest should be the runtime source of truth.

Example:

    {
      "model_id": "moneyline_xgboost_v13_ensemble_top60",
      "architecture_type": "ensemble",
      "requires_market": true,
      "market_key": "moneyline",
      "preferred_bookmaker": "DraftKings",
      "members": [
        {
          "member_id": "feature_only_favorite",
          "target": "favorite_won",
          "perspective": "favorite",
          "uses_market": false,
          "feature_count": 60,
          "artifact_path": "members/feature_only_favorite/model.joblib",
          "feature_columns_path": "members/feature_only_favorite/feature_columns.json"
        },
        {
          "member_id": "market_aware_favorite",
          "target": "favorite_won",
          "perspective": "favorite",
          "uses_market": true,
          "feature_count": 69,
          "artifact_path": "members/market_aware_favorite/model.joblib",
          "feature_columns_path": "members/market_aware_favorite/feature_columns.json"
        }
      ],
      "combiner": {
        "type": "grouped_weighted_average",
        "groups": {
          "ensemble_favorite_probability": {
            "feature_only_favorite": 0.5,
            "market_aware_favorite": 0.5
          },
          "ensemble_dog_probability": {
            "feature_only_dog": 0.5,
            "market_aware_dog": 0.5
          }
        }
      }
    }

---

## Live Prediction Flow

For ensemble models:

1. Load model config and ensemble manifest
2. Load live_model_features.parquet
3. Load market_outcomes.parquet
4. Filter moneyline rows
5. Select preferred bookmaker
6. Build favorite/dog market frame
7. Convert live red-minus-blue features into perspective features
8. Score each ensemble member
9. Apply combiner
10. Compute edge vs implied probability
11. Apply decision rule
12. Emit model-scoped outcomes

---

## Perspective Transform

Current live features are red-minus-blue diff columns.

For any clean signed-diff feature:

    if favorite is red:
      favpersp__feature = live_feature
      dogpersp__feature = -live_feature

    if favorite is blue:
      favpersp__feature = -live_feature
      dogpersp__feature = live_feature

This is valid for the current V13 feature list because the historical favorite/dog feature relationship check showed all checked features are opposites.

Future feature sets must either:

1. use signed diff features that obey this transform, or
2. declare feature transform rules explicitly.

---

## Output Contract

The ensemble prediction adapter should still write model-scoped outcomes:

    data/predictions/by_model/<model_id>/model_outcomes.parquet

It should also write an ensemble detail artifact:

    data/predictions/by_model/<model_id>/ensemble_details.parquet

The standard model outcome rows should remain dashboard-compatible:

- fight_id
- event_name
- red_fighter
- blue_fighter
- outcome_fighter_id
- outcome_fighter_name
- model_id
- model_family
- market_key
- model_probability
- model_pick_probability
- model_pick
- model_edge
- american_odds
- implied_probability

Additional ensemble detail columns should include:

- favorite_fighter_id
- dog_fighter_id
- favorite_odds
- dog_odds
- favorite_implied_probability
- dog_implied_probability
- ensemble_favorite_probability
- ensemble_dog_probability
- ensemble_favorite_edge
- ensemble_dog_edge
- bet_side
- rule_passed
- rule_reason

Member probability columns should also be preserved:

- member_probability__feature_only_favorite
- member_probability__feature_only_dog
- member_probability__market_aware_favorite
- member_probability__market_aware_dog

---

## Production Safety Rules

1. Ensemble models must be registered as experimental first.
2. Ensemble models must not become primary automatically.
3. Prediction must fail clearly if required market data is missing.
4. Missing market rows should exclude that fight from ensemble scoring, not silently fabricate odds.
5. Missing selected features should fail model training.
6. Missing live features should fail prediction unless explicitly allowed by config.
7. Model artifacts should not be retrained during live prediction.
8. Live prediction must only load persisted child models.
9. Market-aware feature creation must use already-matched market artifacts, not scrape odds directly.
10. Decision rules must remain separate from model training.

---

## Implementation Phases

### Phase 1: Documentation and Contracts

- Add this architecture document.
- Add draft config schema example.
- Do not modify production behavior.

### Phase 2: Wide Favorite/Dog Market Feature View

Create:

    pipeline/features/run_build_moneyline_favdog_market_view.py

Output:

    data/features/moneyline_favdog_market_feature_view.parquet

This should be wide and reusable.

### Phase 3: Ensemble Config and Feature Resolver

Create reusable utilities:

    pipeline/ensemble/config.py
    pipeline/ensemble/feature_sets.py

Responsibilities:

- Load ensemble config
- Resolve feature sets per member
- Apply prefixes
- Append market features if requested
- Validate required columns

### Phase 4: Ensemble Trainer

Create:

    pipeline/ensemble/train_ensemble.py

Responsibilities:

- Train all configured members
- Save member artifacts
- Save manifest
- Save metrics
- Save config snapshot

Then route from:

    pipeline.training.run_train_model

when:

    architecture.type == ensemble

### Phase 5: Ensemble Predictor

Create:

    pipeline/ensemble/predict_ensemble.py

Responsibilities:

- Load manifest
- Build live market frame
- Build live perspective features
- Score child members
- Apply combiner
- Apply decision rule
- Write model-scoped outputs

Then route from:

    pipeline.modeling.run_prediction

when:

    architecture.type == ensemble

### Phase 6: Register First Experimental Model

Register:

    moneyline_xgboost_v13_ensemble_top60

Status:

    experimental
    dashboard_selectable: true
    primary: false

### Phase 7: Expand Variants

Add additional configs without code changes:

    moneyline_xgboost_v13_ensemble_top90
    moneyline_xgboost_v13_ensemble_top125
    moneyline_xgboost_v13_ensemble_top175

---

## Non-Goals

This work should not:

- Replace the current primary moneyline model
- Remove the single-model prediction path
- Hardcode V13-only logic into run_prediction.py
- Scrape odds inside the model scorer
- Require the dashboard to understand each child model
- Force every ensemble to use exactly four members
- Force every ensemble to use simple averaging
- Force every child model to share the same features

---

## First Production Target

The first production target is:

    moneyline_xgboost_v13_ensemble_top60

But the code must be generic enough that the next targets require only config changes:

    moneyline_xgboost_v13_ensemble_top90
    moneyline_xgboost_v13_ensemble_top125
