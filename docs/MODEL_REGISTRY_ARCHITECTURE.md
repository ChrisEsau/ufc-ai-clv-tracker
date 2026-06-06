# UFC Model Registry Architecture

## Purpose

This document defines how model bundles should be named, stored, selected, and loaded as the UFC project moves from a single V5 moneyline model to a multi-model architecture.

The registry must support:

- Multiple market families
- Multiple model algorithms
- Multiple feature sets per algorithm
- Multiple calibration strategies
- Market-aware models
- Prop models
- Ensemble models
- Clear production versus experimental selection

Core principle:

```text
The pipeline should load a model by model_id, not by hard-coded folder names like UFC_Model_v5_Experiment.
```

---

## Why This Matters

The old naming style:

```text
UFC_Model_v5_Experiment
current_moneyline_v5_features.yaml
```

is too version-centric.

It does not scale well when the project includes:

```text
moneyline_xgboost_core_ewm_v1
moneyline_xgboost_grappling_heavy_v1
moneyline_xgboost_market_aware_v1
moneyline_random_forest_core_ewm_v1
props_submission_xgboost_v1
```

The new naming should describe what the model is, not only which experiment number it came from.

---

## Model ID Pattern

Recommended model ID pattern:

```text
<market_family>_<model_type>_<feature_set>_<training_variant>_<version>
```

Optional expanded pattern:

```text
<market_family>_<model_type>_<feature_set>_<training_variant>_<calibration>_<version>
```

Examples:

```text
moneyline_xgboost_core_ewm_v1
moneyline_xgboost_grappling_heavy_v1
moneyline_xgboost_market_aware_v1
moneyline_random_forest_core_ewm_v1
moneyline_ensemble_core_ewm_v1
props_ko_tko_xgboost_method_v1
props_submission_xgboost_grappling_v1
props_goes_distance_xgboost_duration_v1
market_edge_xgboost_overlay_v1
```

---

## Model ID Components

### market_family

Examples:

```text
moneyline
props
market
ensemble
```

### model_type

Examples:

```text
xgboost
random_forest
lightgbm
neural_net
ensemble
```

### feature_set

Examples:

```text
core
core_ewm
grappling_heavy
striking_heavy
recent_form
market_aware
duration
grappling
method
```

### training_variant

Examples:

```text
baseline
balanced
recent_era
calibrated
market_overlay
```

### version

Examples:

```text
v1
v2
v3
```

Version should represent a meaningful model-bundle revision, not every small local experiment.

---

## Recommended Folder Structure

```text
models/
├── moneyline/
│   ├── xgboost/
│   │   ├── core_ewm_v1/
│   │   ├── grappling_heavy_v1/
│   │   └── market_aware_v1/
│   ├── random_forest/
│   │   └── core_ewm_v1/
│   └── ensemble/
│       └── core_ewm_v1/
│
├── props/
│   ├── ko_tko/
│   │   └── xgboost_method_v1/
│   ├── submission/
│   │   └── xgboost_grappling_v1/
│   ├── decision/
│   │   └── xgboost_duration_v1/
│   └── goes_distance/
│       └── xgboost_duration_v1/
│
└── market/
    ├── edge_model/
    │   └── xgboost_overlay_v1/
    └── clv_model/
        └── xgboost_line_movement_v1/
```

Folder names may be shorter than model IDs, but each model bundle must declare its full `model_id` in metadata.

---

## Required Model Bundle Files

Each model bundle should contain:

```text
model.pkl
feature_contract.yaml
training_config.yaml
metrics.json
confidence_buckets.parquet
model_card.md
```

Optional files:

```text
calibration_model.pkl
feature_importance.csv
backtest_predictions.parquet
calibration_report.parquet
shap_importance.csv
```

---

## feature_contract.yaml

Each model owns its own feature contract.

Example:

```yaml
model_id: moneyline_xgboost_core_ewm_v1
market_family: moneyline
model_type: xgboost
feature_set: core_ewm
expected_feature_count: 124
source_feature_registry: configs/features/feature_registry.yaml
source_inventory: configs/features/full_rolling_feature_inventory.yaml
features:
  - elo_diff
  - win_pct_diff
  - ewm_elo_diff
  - recent_form_win_pct_diff
validation_rules:
  - all_features_must_exist
  - feature_order_must_match_training
  - no_raw_red_blue_corner_columns
```

The prediction pipeline should use this file to align live prediction inputs.

---

## training_config.yaml

Each model should also declare its training settings.

Example:

```yaml
model_id: moneyline_xgboost_core_ewm_v1
algorithm: xgboost
training_dataset: data/features/ufc_moneyline_features.parquet
target_column: target
temporal_split_date: 2022-12-31
calibration: isotonic
probability_clipping:
  enabled: true
```

---

## metrics.json

Each model should preserve training/backtest quality information.

Example:

```json
{
  "model_id": "moneyline_xgboost_core_ewm_v1",
  "accuracy": 0.61,
  "roc_auc": 0.64,
  "log_loss": 0.80,
  "brier_score": 0.23,
  "calibration_method": "isotonic"
}
```

---

## Production Selection

Production selection should not be hard-coded into path constants.

Use a registry file such as:

```text
configs/models/model_registry.yaml
```

Example:

```yaml
active_models:
  moneyline:
    primary: moneyline_xgboost_core_ewm_v1
    candidates:
      - moneyline_xgboost_market_aware_v1
      - moneyline_random_forest_core_ewm_v1

  props:
    ko_tko: props_ko_tko_xgboost_method_v1
    submission: props_submission_xgboost_grappling_v1
    goes_distance: props_goes_distance_xgboost_duration_v1

  market:
    edge_overlay: market_edge_xgboost_overlay_v1
```

The prediction pipeline should ask the registry which model is active.

---

## Model Status Values

Recommended statuses:

```text
development
candidate
shadow
production
retired
archived
```

Definitions:

- `development`: being trained or tested locally
- `candidate`: passed basic validation and ready for comparison
- `shadow`: runs beside production but does not drive betting decisions
- `production`: active model used by the pipeline
- `retired`: no longer used but kept for reference
- `archived`: historical artifact only

---

## Market-Aware Models

Market-aware models should be treated separately from pure fighter-skill models.

Pure fighter model:

```text
fighter data → win probability
```

Market-aware model:

```text
fighter data + market data → pricing/edge quality
```

Potential market-aware model IDs:

```text
moneyline_xgboost_market_aware_v1
market_edge_xgboost_overlay_v1
market_clv_xgboost_line_movement_v1
```

Market-aware features may include:

```text
opening_implied_prob
closing_implied_prob
current_implied_prob
line_movement_pct
market_disagreement
sportsbook_consensus_prob
steam_move_flag
model_market_edge
historical_clv_signal
```

The current V5 moneyline model should remain a pure fighter-performance model until a market-aware model is intentionally trained and validated.

---

## Multiple XGBoost Models

Multiple XGBoost models are allowed and expected.

They should be distinguished by feature set and training variant, not just by algorithm.

Examples:

```text
moneyline_xgboost_core_ewm_v1
moneyline_xgboost_grappling_heavy_v1
moneyline_xgboost_striking_heavy_v1
moneyline_xgboost_market_aware_v1
```

Do not refer to "the XGBoost model" in production code.

Instead, refer to the explicit `model_id` selected by the registry.

---

## Compatibility With Current V5 Model

The current V5 model should be mapped into the new system as a legacy-compatible model bundle.

Recommended future ID:

```text
moneyline_xgboost_core_ewm_v1
```

This should preserve:

```text
current 124-feature contract
isotonic calibration
current feature order
current prediction behavior
```

The old names may remain as compatibility aliases during migration.

---

## Migration Plan

Phase 1

- Document model registry architecture.
- Preserve current V5 model artifacts.
- Preserve current V5 124-feature contract.

Phase 2

- Create `configs/models/model_registry.yaml`.
- Add compatibility alias for the current V5 model.

Phase 3

- Convert training notebook outputs into a model bundle matching this contract.

Phase 4

- Build model loader and adapter layer.

Phase 5

- Add additional XGBoost variants, Random Forest, and ensembles.

Phase 6

- Add market-aware models and prop models.

---

## Related Files

```text
docs/MODEL_ADAPTER_ARCHITECTURE.md
docs/UFC_FEATURE_LAYER_ARCHITECTURE.md
docs/UFC_PROP_MODEL_ARCHITECTURE.md
configs/features/feature_registry.yaml
configs/features/current_moneyline_v5_features.yaml
configs/features/full_rolling_feature_inventory.yaml
```
