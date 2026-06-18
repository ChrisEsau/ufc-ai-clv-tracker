# Model Lab Refactor Implementation Plan

## Status

Locked implementation plan.

This document defines the full scope of work for the Model Lab refactor.

No code implementation should begin until this plan is approved.

---

# 1. Goals

The Model Lab refactor should convert the current Model Lab from a mixed workflow/config/debug area into a clean model management workspace.

Primary goals:

* Modularize Model Lab code.
* Streamline the Model Lab UI.
* Move workflow execution out of Model Lab.
* Preserve existing model configuration functionality.
* Add model promotion/demotion through Compare.
* Add better model versioning.
* Centralize Model Lab CSS.
* Make Model Setup the primary model configuration editor.
* Keep Overview informational only.
* Keep Features as global Feature Studio.
* Keep Operations Center responsible for workflow execution.

---

# 2. Current Problems

The current implementation has several issues:

* Model Lab logic is spread across too many files.
* UI code, config logic, YAML mutation, workflow dispatch, CSS, and GitHub writes are mixed together.
* There are overlapping Model Lab implementations.
* Configuration behavior is partially legacy.
* Model Lab sidebar navigation is too cluttered.
* Model Lab exposes workflow/debug controls that belong in Operations Center.
* Promotion/lifecycle controls are separated from model comparison.
* Feature registry management and model-specific feature selection are mixed together.
* Styling is scattered between global theme files, Model Lab workflow files, section files, and feature-selection utilities.
* Save/create flows are confusing.
* Display Name duplicates Model ID.
* Model ID is manually editable instead of derived from config identity.

---

# 3. Final Model Lab Workspace Structure

Model Lab should contain exactly these internal workspaces:

```text
Overview
Model Setup
Features
Performance
Compare
```

Internal Model Lab navigation should appear as a horizontal workspace strip inside the Model Lab page.

The app sidebar should not contain Model Lab sub-workspaces.

---

# 4. Top-Level Sidebar Navigation

The sidebar should only contain major application workspaces:

```text
Betting Board
Line Movement / CLV
Model Lab
Operations Center
Data Maintenance
Bankroll
```

Remove Model Lab sub-workspaces from the sidebar.

Remove legacy sidebar patching where possible.

---

# 5. Model Lab Internal Navigation

Inside Model Lab, use a top navigation strip:

```text
Overview | Model Setup | Features | Performance | Compare
```

The active workspace should be visually highlighted.

This navigation replaces the current sidebar-based Model Lab navigation.

---

# 6. Overview Workspace

## Purpose

Overview is a portfolio dashboard only.

It should not contain workflow buttons or configuration actions.

## Should Show

### Production Models

* Model family
* Current champion
* Status
* Key performance snapshot
* Last trained or last updated date

### Draft Models

* Model ID
* Family
* Market
* Algorithm
* Version
* Last updated
* Status

### Archived Models

* Model ID
* Family
* Market
* Algorithm
* Version
* Archived status

### Portfolio Metrics

* Total models
* Production models
* Draft models
* Archived models

## Should Not Include

* New model button
* Edit model button
* Clone model button
* Train button
* Backtest button
* Prediction button
* Promotion button
* Configuration forms

---

# 7. Model Setup Workspace

## Purpose

Model Setup is the primary model configuration editor.

It replaces the old Configuration workspace.

It must support:

* Selecting existing models
* Auto-populating the form from model YAML
* Editing existing model configuration
* Saving changes to the selected model
* Creating a new model version from the current form state
* Model-specific feature bundle selection

---

## 7.1 Model Setup Header

The header should include:

1. Model information display card
2. Model selector dropdown

The model information display should match the approved dark card style.

Example:

```text
moneyline_xgboost_v7   [DRAFT]

Family: moneyline · Market: moneyline · Algorithm: XGBoost
```

Rules:

* Remove "Current:" label.
* Remove Artifact information.
* Remove Next Version Preview.
* Include DRAFT or PRODUCTION status badge.
* Include Algorithm.
* Model selector should sit to the right of the model display card.
* Selecting a model loads its YAML and populates the full form.

---

## 7.2 Model Selector

The model selector should contain registered models.

Example:

```text
Model
[moneyline_xgboost_v7 ▼]
```

Behavior:

* Selecting a model loads the associated config YAML.
* All form fields populate from the loaded YAML.
* The selected model becomes the active model context for Save/New.

---

## 7.3 Section 1 — Model Identity

Fields:

* Family
* Market Key
* Algorithm
* Generated Model ID
* Description

Removed:

* Editable Model ID
* Display Name
* Manual Version Selector

Rules:

* Model ID is read-only.
* Model ID is generated automatically.
* Display Name should not be used.
* Version should be inferred from the selected model or from next-version generation.

Model ID format:

```text
family_market_key_algorithm_version
```

Duplicate family/market names should collapse.

Examples:

```text
moneyline + moneyline + xgboost + v7
→ moneyline_xgboost_v7

prop + ko + xgboost + v1
→ prop_ko_xgboost_v1
```

---

## 7.4 Section 2 — Training Setup

Fields:

* Train Start Date
* Train End Date
* Calibration End Date
* Split Mode
* Target Column
* Date Column
* Symmetry Enabled
* Symmetry Mode

Required YAML support:

```yaml
split:
  mode: train_calibration_test
  train_start_date: "YYYY-MM-DD"
  train_end_date: "YYYY-MM-DD"
  calibration_end_date: "YYYY-MM-DD"

symmetry:
  enabled: true
  mode: flip_all
```

Notes:

* Train Start Date is required for reproducibility.
* Training window should be considered part of model identity and evaluation context.
* Symmetry must be exposed as a first-class setting.

---

## 7.5 Section 3 — Model Behavior

Fields:

* Calibration Enabled
* Calibration Method
* Probability Clip Low
* Probability Clip High
* Prediction Threshold Source
* Prediction Threshold Value
* Threshold Sweep Enabled
* Threshold Min
* Threshold Max
* Threshold Step
* Confidence Bucket Edges

Relevant YAML areas:

```yaml
calibration:
  enabled: true
  method: isotonic

prediction:
  threshold:
    source: best_sweep
    value: 0.5
  probability:
    clip_low: 0.02
    clip_high: 0.98

metrics:
  threshold_min: 0.4
  threshold_max: 0.6
  threshold_step: 0.01
  confidence_bucket_edges:
    - 0.5
    - 0.55
    - 0.6
    - 0.65
    - 0.7
    - 0.75
    - 0.8
    - 0.85
    - 0.9
    - 0.95
    - 1.0
```

---

## 7.6 Section 4 — Hyperparameters

Initial XGBoost fields:

* n_estimators
* max_depth
* learning_rate
* subsample
* colsample_bytree
* random_state
* eval_metric

Relevant YAML:

```yaml
params:
  n_estimators: 500
  max_depth: 4
  learning_rate: 0.03
  subsample: 0.8
  colsample_bytree: 0.8
  random_state: 42
  eval_metric: logloss
```

Future algorithms should render algorithm-specific parameters dynamically.

---

## 7.7 Section 5 — Feature Selection

Purpose:

Configure features for the selected model.

This is model-specific feature selection, not global feature registry management.

Fields:

* Selected Bundles
* Include Overrides
* Exclude Overrides
* Resolved Feature Count
* Expected Feature Count
* Allow Unsafe Features

Relevant YAML:

```yaml
features:
  selection_mode: explicit
  expected_feature_count: 130
  allow_unsafe_features: false
  selected_bundles:
    - core_state
    - ewm_state
    - recent_form
  include_features: []
  exclude_features: []
  feature_columns: []
```

Rules:

* Model-specific bundle selection remains in Model Setup.
* Global feature registry editing remains in Features workspace.
* Resolved features should update based on selected bundles and overrides.
* Saving should write selected_bundles, include_features, exclude_features, feature_columns, and expected_feature_count.

---

## 7.8 Save Actions

Only two action buttons should exist in Model Setup:

```text
Save
New
```

Location:

* Bottom-right corner of the workspace.

No duplicate save buttons.

No sticky footer.

No top save actions.

No Save Draft.

No Save & View Summary.

No Create New Model Version label.

Button behavior:

### Save

Overwrite the currently selected model configuration.

Example:

```text
configs/models/moneyline_xgboost_v7.yaml
```

is updated in place.

### New

Create the next available model version using the current form state.

Example:

Selected model:

```text
moneyline_xgboost_v7
```

Current form still points to moneyline/moneyline/xgboost.

Clicking New creates:

```text
moneyline_xgboost_v8
```

Cross-family behavior:

Selected model:

```text
moneyline_xgboost_v7
```

User changes:

```text
Family = prop
Market Key = ko
Algorithm = xgboost
```

Clicking New checks:

```text
prop_ko_xgboost_v*
```

If none exist, create:

```text
prop_ko_xgboost_v1
```

---

# 8. Features Workspace

## Purpose

Features is the global Feature Studio.

It should preserve all current feature registry functionality.

## Must Preserve

* Feature registry summary
* Feature registry validation
* Feature library table
* Bundle library table
* Feature editor
* Bundle editor
* Registry YAML preview
* Save feature registry to GitHub
* Save bundle registry to GitHub
* Stage/discard registry changes
* Formula validation
* Transform selector
* Archive/delete feature
* Delete bundle with protection

## Should Not Own

* Model-specific feature bundle selection

That belongs in Model Setup.

---

# 9. Performance Workspace

## Purpose

Single-model evaluation and diagnostics.

## Should Show

Core metrics:

* Accuracy
* ROC AUC
* Log Loss
* Brier Score
* Best Threshold

Training context:

* Train Start Date
* Train End Date
* Calibration End Date
* Training fight count if available

Tabs:

```text
Overview
Calibration
Thresholds
SHAP
```

Thresholds tab should show threshold sweep diagnostics.

Calibration tab should show calibration diagnostics.

SHAP tab should show feature importance / SHAP outputs when available.

---

# 10. Compare Workspace

## Purpose

Champion vs Challenger comparison and deployment decisions.

Promotion is not a separate workspace.

## Should Show

Champion model:

* Current production model for selected family/market

Challenger model:

* Selected draft or candidate model

Comparison metrics:

* Accuracy
* ROC AUC
* Log Loss
* Brier Score
* ROI
* CLV
* Threshold
* Training window
* Feature count

## Deployment Actions

Buttons:

```text
Promote Challenger
Archive Model
```

Rules:

* Promote Challenger should update registry production status.
* Promoting a challenger should demote the previous champion.
* Archive should mark a model as archived.
* Primary production model should not be accidentally demoted without replacement.

---

# 11. Operations Center

Workflow execution belongs in Operations Center, not Model Lab.

## Primary Actions

```text
Train Model
Run Backtest
Run Predictions
```

## Advanced / Debug Actions

```text
Build Fighter State
Build Feature View
```

Model Lab should not contain:

* Build Fighter State
* Build Feature View
* Run Predictions
* Run Betting Outcomes
* Train workflow buttons
* Backtest workflow buttons

Workflow execution should be separated from model management.

---

# 12. Code Refactor Target Structure

Create a cleaner Model Lab module structure.

## Target Files

```text
tabs/
  model_lab.py
  model_lab_sections/
    overview.py
    model_setup.py
    features.py
    performance.py
    compare.py

utils/
  model_lab/
    __init__.py
    registry.py
    config.py
    versioning.py
    features.py
    metrics.py
    deployment.py
    workflows.py

utils/ui/
  model_lab_theme.py
```

---

# 13. Responsibilities

## tabs/model_lab.py

Router only.

Responsibilities:

* Render Model Lab workspace strip
* Store active workspace state
* Route to selected workspace
* Pass model context when needed

Should not:

* Save YAML
* Write registry
* Dispatch workflows
* Own CSS
* Parse version strings
* Perform promotion logic

---

## tabs/model_lab_sections/model_setup.py

UI only.

Responsibilities:

* Render Model Setup form
* Render Model Setup header
* Render Save/New buttons
* Display validation messages
* Call service functions

Should not:

* Manually mutate registry dictionaries
* Write GitHub files directly
* Parse model IDs
* Infer next version directly

---

## utils/model_lab/registry.py

Responsibilities:

* Load model registry
* Save model registry
* List models
* Get model entry
* Update model entry
* Validate registry consistency
* Sync config status with registry status

---

## utils/model_lab/config.py

Responsibilities:

* Load model config
* Save model config
* Normalize model config
* Convert YAML config to form state
* Convert form state to YAML config
* Apply defaults safely

---

## utils/model_lab/versioning.py

Responsibilities:

* Generate model ID
* Collapse duplicate family/market
* Parse model version
* Find next available version
* Create new model version ID
* Prevent version collisions

---

## utils/model_lab/features.py

Responsibilities:

* Resolve selected bundles
* Resolve include overrides
* Resolve exclude overrides
* Calculate resolved feature list
* Calculate expected feature count
* Validate feature availability
* Bridge model-specific feature selection to global feature registry

---

## utils/model_lab/metrics.py

Responsibilities:

* Load metrics
* Load model card
* Load threshold sweep
* Load confidence buckets
* Load SHAP outputs
* Summarize model performance

---

## utils/model_lab/deployment.py

Responsibilities:

* Promote challenger
* Demote existing production model
* Archive model
* Enforce active primary model rules
* Sync deployment status between registry and config

---

## utils/model_lab/workflows.py

Responsibilities:

* Store workflow dispatch metadata
* Support Operations Center model workflow buttons

Should not be used for Model Lab UI workflow execution.

---

## utils/ui/model_lab_theme.py

Responsibilities:

* Centralize all Model Lab-specific CSS
* Own `.mlab-*` classes
* Provide one function to inject Model Lab styles

Global app CSS remains in:

```text
utils/ui/theme.py
```

---

# 14. Files To Refactor Or Retire

## Refactor

```text
tabs/model_lab.py
tabs/model_lab_refactored.py
utils/model_lab_workflows.py
utils/model_lab_feature_selection.py
tabs/model_lab_sections/features.py
tabs/model_lab_sections/overview.py
tabs/model_lab_sections/performance.py
tabs/model_lab_sections/backtest.py
tabs/model_lab_sections/backtest_enhanced.py
tabs/model_lab_sections/actions.py
tabs/model_lab_sections/lifecycle.py
utils/sidebar.py
utils/sidebar_refactored.py
utils/ui/theme.py
tabs/model_lab_sections/styles.py
```

## Retire Eventually

```text
tabs/model_lab_refactored.py
tabs/model_lab_sections/actions.py
tabs/model_lab_sections/lifecycle.py
tabs/model_lab_sections/backtest.py
tabs/model_lab_sections/backtest_enhanced.py
```

Only delete after equivalent functionality is migrated or intentionally removed.

---

# 15. Styling Rules

## Global Theme

Keep global dashboard theme in:

```text
utils/ui/theme.py
```

## Model Lab Theme

Move Model Lab-specific CSS to:

```text
utils/ui/model_lab_theme.py
```

Rules:

* No scattered Model Lab CSS.
* No CSS inside workflow utility modules.
* No duplicate number-input CSS in multiple files.
* Reuse global color variables where possible.
* Preserve approved dark-theme look.

---

# 16. Removed UI Elements

Remove from Model Lab:

* Display Name
* View in Registry button
* Selected Model top-right control
* Next Version Preview card
* Artifact information in header
* Promotion workspace
* Actions workspace
* Backtest workspace
* Configuration workspace naming
* Clone workflow
* Edit/New mode selector
* Save Draft button
* Save & View Summary button
* Duplicate save controls
* Build Fighter State button
* Build Feature View button
* Run Predictions button
* Run Betting Outcomes button

---

# 17. New YAML Requirements

Model configs should support:

```yaml
split:
  train_start_date: "YYYY-MM-DD"
```

Existing fields must remain supported:

```yaml
split:
  mode:
  train_end_date:
  calibration_end_date:

symmetry:
  enabled:
  mode:

calibration:
  enabled:
  method:

metrics:
  threshold_min:
  threshold_max:
  threshold_step:
  confidence_bucket_edges:

prediction:
  threshold:
    source:
    value:
  probability:
    clip_low:
    clip_high:
```

Backward compatibility:

* Existing configs without `train_start_date` should load safely.
* UI should show a default or blank train start date if missing.
* Saving should add `train_start_date`.

---

# 18. Save/New Logic

## Save Logic

Input:

* Selected model ID
* Current form state

Behavior:

* Build config from form
* Preserve necessary existing fields
* Write to selected model config path
* Update model registry metadata
* Clear Streamlit cache if needed
* Reload selected model

## New Logic

Input:

* Current form state

Behavior:

* Derive family, market key, algorithm from form
* Determine next available version for that family/market/algorithm
* Generate model ID
* Generate config path
* Generate artifact output dir
* Write new config YAML
* Add model registry entry
* Select new model after save

---

# 19. Promotion Logic

Promotion belongs in Compare.

Promote Challenger should:

* Mark challenger as production
* Set active primary model for the model family/market
* Demote previous production primary to draft or non-primary status
* Sync config status with registry status
* Prevent promotion if required artifacts are missing unless explicitly allowed

Archive Model should:

* Mark model as archived
* Remove from active primary if applicable only with replacement
* Preserve config and artifacts

---

# 20. Implementation Phases

## Phase 1 — Sidebar And Navigation

* Remove Model Lab sub-workspaces from sidebar.
* Add Model Lab internal workspace strip.
* Remove sidebar mutation wrapper pattern if possible.
* Keep Operations Center as top-level workspace.

Files:

```text
utils/sidebar.py
utils/sidebar_refactored.py
tabs/model_lab.py
```

---

## Phase 2 — Model Lab Theme

* Create `utils/ui/model_lab_theme.py`.
* Move `.mlab-*` CSS there.
* Remove Model Lab CSS from workflow/section utilities.
* Keep global CSS in `utils/ui/theme.py`.

---

## Phase 3 — Service Layer

Create:

```text
utils/model_lab/
```

Add modules:

```text
registry.py
config.py
versioning.py
features.py
metrics.py
deployment.py
workflows.py
```

Move non-UI logic out of UI files.

---

## Phase 4 — Model Setup Workspace

Create:

```text
tabs/model_lab_sections/model_setup.py
```

Implement:

* Header model display
* Model selector
* Five configuration sections
* Save/New bottom-right
* Form hydration from YAML
* Save existing config
* New version creation

---

## Phase 5 — Features Workspace

Reorganize current Features workspace while preserving behavior.

Keep:

* Feature library
* Bundle library
* Feature editor
* Bundle editor
* Registry validation
* YAML preview
* Save/stage/discard behavior

---

## Phase 6 — Performance Workspace

Implement:

* Metric cards
* Training window display
* Thresholds tab
* Calibration tab
* SHAP tab

---

## Phase 7 — Compare Workspace

Implement:

* Champion/challenger selector
* Metric deltas
* Training window comparison
* Threshold comparison
* Promote Challenger
* Archive Model

Remove separate lifecycle/promote workspace.

---

## Phase 8 — Operations Center Workflow Cleanup

Move workflow controls to Operations Center:

Primary:

```text
Train Model
Run Backtest
Run Predictions
```

Advanced:

```text
Build Fighter State
Build Feature View
```

Remove workflow controls from Model Lab.

---

## Phase 9 — Legacy Cleanup

After new flow works:

* Delete retired workspaces.
* Remove dead helpers.
* Remove duplicate CSS.
* Remove old router.
* Remove sidebar mutation wrapper if no longer needed.

---

# 21. Validation Checklist

Before merge, verify:

* Sidebar only shows top-level workspaces.
* Model Lab shows internal workspace strip.
* Overview has no action buttons.
* Model Setup loads selected model YAML.
* Save overwrites selected model config.
* New creates next available model ID.
* New supports cross-family/cross-market creation.
* Generated model ID collapses duplicate family/market.
* Display Name is gone.
* View in Registry is gone.
* Artifact info is removed from header.
* DRAFT/PRODUCTION badge appears in model display.
* Training Start Date loads and saves.
* Symmetry settings load and save.
* Threshold sweep settings load and save.
* Feature bundles load and save per model.
* Features workspace still edits global registries.
* Performance shows threshold and training window.
* Compare contains promotion actions.
* Operations Center owns workflow execution.
* No duplicate Save/New controls.
* Model Lab CSS is centralized.

---

# 22. Implementation Rule

No UI module should directly:

* Write GitHub files
* Parse model versions
* Generate model IDs
* Mutate registry internals
* Dispatch workflows
* Inject scattered CSS
* Own promotion rules

UI modules should render and call service-layer functions.

---

# 23. Approved UI Baseline

The approved Model Setup baseline is the dark-theme layout with:

* Left top-level sidebar
* Model Lab internal workspace strip
* Model display card with status badge
* Model selector to the right
* Five main setup sections
* Save/New buttons in bottom-right only
* No artifact header display
* No next version preview card
* No Display Name
* No duplicate save actions

This is the implementation target.
