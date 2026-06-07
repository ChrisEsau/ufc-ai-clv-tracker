# UFC Prediction V2 Implementation Plan

## Purpose

This document converts the approved V2 prediction architecture into an ordered implementation plan.

Primary references:

```text
docs/UFC_PREDICTION_PIPELINE_V2.md
docs/UFC_OUTCOME_SCHEMA_SPEC.md
```

This plan is implementation guidance only. The schemas and architecture docs remain the source of truth for contracts.

---

## Current Status

Completed / approved:

- Moneyline V5 training framework is modular and config-driven.
- Moneyline target generation has been corrected to use fighter IDs.
- Outcome-based prediction architecture is approved.
- Canonical outcome schemas are documented.

Not yet implemented:

- Model registry runtime loader.
- Generic prediction adapter.
- Algorithm prediction plug-ins.
- Config-driven formatter.
- Outcome-level prediction artifact.
- Outcome-level market artifact.
- Outcome-level betting board.

---

## Implementation Principles

1. Implement in small commits.
2. Preserve current legacy runners until V2 artifacts are validated.
3. Do not mutate training contracts while building prediction V2.
4. Do not introduce new long-term red/blue side-based contracts.
5. All new V2 prediction, odds, betting, CLV, and Model Lab artifacts should use outcome rows.
6. New runtime selection should follow this precedence:

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

---

## Phase 0: Pre-Implementation Review

Before coding, verify:

- latest `dev` is pulled locally
- `docs/UFC_PREDICTION_PIPELINE_V2.md` exists
- `docs/UFC_OUTCOME_SCHEMA_SPEC.md` exists
- current moneyline model artifacts exist under `models/moneyline/xgboost_v5/`
- current feature warehouse exists under `data/features/UFC_enhanced_rolling_features_EWM.parquet`
- current live card generation still works

No code changes should be made during this phase.

---

## Phase 1: Model Registry

### Goal

Add a registry that answers:

```text
Which model config should this run use?
```

### New file

```text
configs/models/model_registry.yaml
```

### Initial content should include

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
```

### Acceptance Criteria

- Registry lists the active production moneyline model.
- Registry supports future candidate models.
- Registry does not duplicate full model contracts.
- Registry points to model config YAML files.

---

## Phase 2: Model Registry Loader

### Goal

Create a small runtime utility for loading registry entries and resolving active/selected models.

### New file

```text
pipeline/modeling/model_registry.py
```

### Responsibilities

- load `configs/models/model_registry.yaml`
- list registered models
- list dashboard-selectable models
- resolve a model by `model_id`
- resolve active model by `model_family`
- apply selection precedence:

```text
explicit model_id
UFC_MODEL_ID environment variable
active model from registry
```

### Suggested public functions

```python
load_model_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict
get_model_entry(model_id: str, registry: dict) -> dict
get_active_model_id(model_family: str, registry: dict) -> str
resolve_selected_model_id(model_family: str, model_id: str | None = None) -> str
```

### Acceptance Criteria

- Raises readable errors for unknown model IDs.
- Raises readable errors for missing active model families.
- Does not load model artifacts.
- Does not contain prediction logic.

---

## Phase 3: Model Config Loader

### Goal

Centralize model config loading for both training and prediction.

### New file

```text
pipeline/modeling/model_config.py
```

### Responsibilities

- load a model YAML config
- validate required top-level fields
- expose prediction section
- expose artifact directory
- expose feature columns

### Required fields for prediction

```text
model_id
model_family
algorithm
features.feature_columns
artifacts.output_dir
prediction.format
prediction.market_key
```

### Acceptance Criteria

- Loader can read `configs/models/moneyline_xgboost_v5.yaml`.
- Loader provides clear validation errors.
- Loader does not train or predict.

---

## Phase 4: Model Loader

### Goal

Load trained artifacts from the model config's artifact directory.

### New file

```text
pipeline/modeling/model_loader.py
```

### Responsibilities

- read `artifacts.output_dir`
- load `calibrated_model.joblib` when available
- fall back to `raw_model.joblib` if configured/needed
- load `feature_columns.json` or `feature_columns.joblib`
- load optional `metrics.json`
- load optional `model_card.yaml`
- return a model bundle object

### Suggested bundle fields

```text
model
model_id
model_family
algorithm
artifact_dir
feature_columns
metrics
model_card
uses_calibrated_model
```

### Acceptance Criteria

- Loads current moneyline V5 artifacts.
- Fails clearly if no model artifact exists.
- Fails clearly if feature columns are missing.
- Does not know about moneyline or prop markets.

---

## Phase 5: Algorithm Prediction Plug-in

### Goal

Mirror the training framework's algorithm plug-in design.

### New files

```text
pipeline/modeling/algorithms/__init__.py
pipeline/modeling/algorithms/xgboost_predictor.py
pipeline/modeling/probability.py
```

### Responsibilities

`xgboost_predictor.py`:

- binary probability prediction
- multiclass probability prediction if needed later

`probability.py`:

- dispatch by algorithm name
- call the correct algorithm plug-in
- validate probability shapes

### Suggested public functions

```python
predict_binary_probability(model, X, algorithm: str) -> np.ndarray
predict_class_probabilities(model, X, algorithm: str) -> np.ndarray
```

### Acceptance Criteria

- XGBoost binary models return `predict_proba(X)[:, 1]`.
- Unsupported algorithms fail clearly.
- No formatter or dashboard logic appears here.

---

## Phase 6: Prediction Formatter

### Goal

Convert fight-level model probabilities into canonical outcome rows.

### New file

```text
pipeline/modeling/prediction_formatter.py
```

### Initial formatter support

```text
binary_matchup
```

### Later formatter support

```text
binary_prop
multiclass
```

### binary_matchup behavior

Input:

```text
positive_probability = red fighter wins
```

Output rows:

```text
outcome_label = red_fighter
outcome_side = red
model_probability = positive_probability
```

```text
outcome_label = blue_fighter
outcome_side = blue
model_probability = 1 - positive_probability
```

### Required output schema

Must follow:

```text
docs/UFC_OUTCOME_SCHEMA_SPEC.md
```

### Acceptance Criteria

- One moneyline fight creates exactly two outcome rows.
- One and only one row per fight/market/model is marked `is_model_pick = true`.
- `model_confidence` equals the probability of the model pick.
- Output includes required prediction outcome columns.

---

## Phase 7: Generic Prediction Adapter

### Goal

Create the orchestration layer that aligns features, predicts, and formats outcomes.

### New file

```text
pipeline/modeling/prediction_adapter.py
```

### Responsibilities

- receive model bundle
- receive model config
- receive live feature dataframe
- align features to feature contract
- coerce feature matrix to numeric
- fill missing numeric values consistently with training
- call probability dispatcher
- clip probabilities if configured
- call prediction formatter
- return outcome rows

### Acceptance Criteria

- Does not hardcode moneyline output columns such as `red_model_prob` / `blue_model_prob` as canonical fields.
- Produces `model_outcomes`-compatible rows.
- Captures model quality and feature validation fields.

---

## Phase 8: Live Feature Builder Extraction

### Goal

Move live feature-building logic out of legacy root-level prediction scripts into a reusable module.

### New file

```text
pipeline/prediction/live_feature_builder.py
```

### Responsibilities

- read live card rows
- match fighters to current fighter features
- build model-ready fight-level feature rows
- create feature/match audit rows
- remain model-family aware through config, not hardcoded model paths

### Acceptance Criteria

- Produces fight-level rows compatible with the current moneyline V5 feature contract.
- Produces audit outputs.
- Does not run model prediction.
- Does not write betting-board artifacts.

---

## Phase 9: Prediction Runner V2

### Goal

Create the first end-to-end V2 prediction runner.

### New file

```text
pipeline/prediction/run_model_predictions_v2.py
```

### Responsibilities

1. parse CLI args
2. read selected model from CLI/env/registry
3. load model config
4. load model bundle
5. build live features
6. run generic adapter
7. write outcome prediction artifacts
8. write feature/match audits

### Suggested CLI

```powershell
python -m pipeline.prediction.run_model_predictions_v2 `
  --model-id moneyline_xgboost_v5 `
  --event-id <event_id>
```

### Outputs

```text
data/predictions/model_outcomes.parquet
data/predictions/by_model/{model_id}/model_outcomes.parquet
```

### Acceptance Criteria

- Produces outcome rows for the current moneyline model.
- Does not break legacy prediction runner.
- Does not require dashboard changes yet.

---

## Phase 10: Market Outcomes V2

### Goal

Normalize odds into outcome rows.

### New files may include

```text
pipeline/market/run_market_update_v2.py
pipeline/market/odds_normalizer.py
pipeline/market/outcome_matcher.py
```

### Outputs

```text
data/market/market_outcomes.parquet
data/market/market_outcome_snapshots.parquet
```

### Acceptance Criteria

- Moneyline odds are represented as outcome rows by fighter name.
- Future prop odds can use the same artifact shape.
- Join keys match prediction outcomes:

```text
fight_id
market_key
outcome_label
```

---

## Phase 11: Betting Outcomes V2

### Goal

Calculate edge, EV, stake, and status from joined outcome rows.

### New files may include

```text
pipeline/betting/run_betting_decision_v2.py
pipeline/betting/outcome_ev.py
pipeline/betting/staking.py
```

### Outputs

```text
data/predictions/betting_outcomes.parquet
data/predictions/by_model/{model_id}/betting_outcomes.parquet
```

### Acceptance Criteria

- Joins predictions and odds on `fight_id`, `market_key`, and `outcome_label`.
- Computes `edge = model_probability - implied_probability`.
- Computes `ev` with consistent ROI-style semantics.
- Applies filters and staking rules to outcome rows.

---

## Phase 12: Dashboard Rewrite

### Goal

Rewrite Betting Board and Model Lab around outcome rows.

### Dashboard reads

```text
data/predictions/betting_outcomes.parquet
```

### Model Lab reads

```text
data/predictions/by_model/{model_id}/model_outcomes.parquet
data/predictions/by_model/{model_id}/betting_outcomes.parquet
```

### Acceptance Criteria

- Betting Board displays fight/market/outcome rows.
- Dashboard no longer requires red/blue canonical probability or odds columns.
- Model Lab can compare selected models by reading model-scoped artifacts.

---

## Phase 13: CLV / Bankroll Migration

### Goal

Migrate CLV and bankroll tracking to outcome-level records.

### Join keys

```text
fight_id
market_key
outcome_label
bookmaker
```

### Acceptance Criteria

- CLV works for moneyline and prop outcomes.
- Official bet ledger references outcome rows.
- Bankroll exposure is market/outcome aware.

---

## Recommended Build Order Summary

```text
1. configs/models/model_registry.yaml
2. pipeline/modeling/model_registry.py
3. pipeline/modeling/model_config.py
4. pipeline/modeling/model_loader.py
5. pipeline/modeling/algorithms/xgboost_predictor.py
6. pipeline/modeling/probability.py
7. pipeline/modeling/prediction_formatter.py
8. pipeline/modeling/prediction_adapter.py
9. pipeline/prediction/live_feature_builder.py
10. pipeline/prediction/run_model_predictions_v2.py
11. pipeline/market/run_market_update_v2.py
12. pipeline/betting/run_betting_decision_v2.py
13. dashboard Betting Board rewrite
14. Model Lab comparison integration
15. CLV / bankroll outcome migration
```

---

## Testing Checklist

### Unit-Level Checks

- Registry resolves active model.
- Registry resolves explicit model ID.
- Environment variable override works.
- Model config validates required prediction fields.
- Model loader loads current moneyline artifacts.
- XGBoost predictor returns probabilities between 0 and 1.
- Formatter produces correct number of outcome rows.
- Outcome schema validation passes.

### Integration Checks

- V2 prediction runner creates moneyline outcome rows.
- Moneyline fight creates exactly two rows.
- Probabilities sum to approximately 1.0 per fight/market/model.
- One model pick exists per fight/market/model.
- Feature quality fields are populated.
- Model-scoped artifacts are written.

### Migration Checks

- Legacy runners still work until explicitly replaced.
- V2 artifacts do not overwrite legacy artifacts unless intentionally configured.
- Dashboard migration does not begin until `betting_outcomes.parquet` is validated.

---

## Do Not Do Yet

Do not start with:

- neural networks
- ensembles
- prop model training
- dashboard redesign
- CLV rewrite
- bankroll rewrite
- market-aware training features

until the current moneyline model can run through the V2 prediction adapter and produce validated `model_outcomes.parquet`.

---

## First Implementation Milestone

The first successful milestone is:

```text
moneyline_xgboost_v5
        ↓
run_model_predictions_v2.py
        ↓
data/predictions/model_outcomes.parquet
        ↓
two outcome rows per fight
```

No betting-board rewrite is required for this milestone.
