# UFC V2 Production Architecture V3

_Last updated: 2026-06-09_

## Purpose

This document is the current working handoff for the UFC AI CLV Tracker V2 architecture after the recent architecture overhaul and repo review.

It is intended for future chats, Codex sessions, and manual repo cleanup work. It explains:

- the intended V2 architecture,
- what is already implemented,
- where the code still has architecture drift,
- the correct artifact contracts,
- known issues and code errors,
- recommended next fixes.

This document should be treated as a **working architecture snapshot**, not a claim that every listed target path is fully wired and validated.

---

## Executive Summary

The V2 architecture is now centered around **outcome-level artifacts** and a new **feature-view architecture**.

The desired end-to-end production flow is:

```text
data/master/ufc_master.parquet
→ pipeline.features.run_build_fighter_state
→ data/features/fighter_state_history.parquet
→ data/features/latest_fighter_state.parquet
→ pipeline.features.run_build_feature_view
→ data/features/moneyline_feature_view.parquet
→ pipeline.training.run_train_model
→ models/moneyline/xgboost_v5/*
→ pipeline.prediction.run_build_live_card
→ data/predictions/ufc_live_card.parquet
→ pipeline.modeling.run_prediction
→ data/predictions/model_outcomes.parquet
→ pipeline.market.run_market_update_v2
→ data/market/market_outcomes.parquet
→ pipeline.betting.run_betting_outcomes_v2
→ data/predictions/betting_outcomes.parquet
→ dashboard.py
```

The main architecture decision is:

```text
Canonical prediction / market / betting grain:
one row per fight + market + outcome + model/bookmaker context
```

For moneyline, that means two rows per fight:

```text
fight_id | market_key | outcome_label | outcome_fighter_id | model_probability | odds | edge | ev
```

The **canonical join key** for V2 model, market, and betting artifacts is:

```text
fight_id + market_key + outcome_fighter_id
```

`outcome_label` is display/context only and should not be the primary join key.

---

## Current Architectural Status

### Implemented / Mostly Valid

These areas exist and are broadly aligned with V2:

- model registry
- model config architecture
- model loader
- prediction adapter
- outcome formatter
- Prediction V2 runner
- Market V2 runner
- Betting Outcomes V2 runner
- core V2 workflows
- fighter-state builder
- feature-view builder
- moneyline feature-view config
- Betting Board V2 dashboard tab
- Data Maintenance dashboard tab

### Partially Migrated / Bridge State

These areas are functional but still bridge old and new architecture:

- live feature generation
- selected-event workflow
- training config input path
- dashboard sidebar and compatibility loaders
- bankroll artifacts
- CLV / line movement tab
- Model Lab tab

### Legacy / Migration Candidates

These areas still depend heavily on pre-V2 or side-based artifacts:

- root scripts such as `run_model_predictions.py`, `run_market_update.py`, `run_betting_decision.py`
- legacy betting board artifact `data/predictions/ufc_betting_board.parquet`
- legacy action board/watchlist artifacts
- old CLV and market snapshot artifact readers
- Model Lab artifact readers tied to `UFC_Model_v5_Experiment`

---

## Source-of-Truth Concepts

### 1. IDs Over Names

Names are allowed for display and fuzzy matching, but model/market/betting joins should be ID-based.

Canonical IDs:

```text
fight_id
event_id
red_fighter_id
blue_fighter_id
outcome_fighter_id
```

Names:

```text
red_fighter
blue_fighter
outcome_label
outcome_display
```

Names should not be relied on as durable joins.

### 2. Outcome Rows Over Red/Blue Canonical Outputs

Old style:

```text
fight_id | red_model_prob | blue_model_prob | red_odds | blue_odds
```

V2 style:

```text
fight_id | market_key | outcome_fighter_id | outcome_label | model_probability | american_odds
```

Dashboard presentation may pivot outcome rows back into red/blue cards, but persisted V2 artifacts should stay outcome-level.

### 3. Feature Views Over Old Rolling Warehouse

The new target feature architecture is:

```text
fighter_state_history.parquet
+ ufc_master.parquet
→ moneyline_feature_view.parquet
```

The old training feature artifact:

```text
data/features/UFC_enhanced_rolling_features_EWM.parquet
```

should be treated as a legacy/parity artifact until the new feature-view path is fully validated.

---

## End-to-End Pipeline Detail

## A. Data Maintenance / Ingestion Layer

### Purpose

Maintain the historical master dataset.

### Main artifact

```text
data/master/ufc_master.parquet
```

### Dashboard area

```text
tabs/data_maintenance.py
```

### Current status

Data Maintenance is mostly coherent and separate from Prediction V2. It handles event discovery, scrape/staging status, validation, audit review, and gated append.

### Notes

This layer is upstream of all modeling. If master is stale or invalid, fighter-state and feature-view artifacts will also be stale or invalid.

---

## B. Fighter-State Layer

### Purpose

Build fighter-level point-in-time prefight state from the master dataset.

### Runner

```bash
python -m pipeline.features.run_build_fighter_state
```

### Inputs

```text
data/master/ufc_master.parquet
```

### Outputs

```text
data/features/fighter_state_history.parquet
data/features/latest_fighter_state.parquet
```

### Important modules

```text
pipeline/features/run_build_fighter_state.py
pipeline/features/state/plugin_history_builder.py
pipeline/features/state/history_builder.py
pipeline/features/state/ewm_state.py
pipeline/features/raw_fighter_features/*
```

### Current status

The runner exists and creates the intended V2 fighter-state artifacts.

### Important detail

`fighter_state_history.parquet` is historical point-in-time state. It is the correct source for training/feature-view generation.

`latest_fighter_state.parquet` is current fighter state. It is useful for live prediction, but live prediction should prefer shared feature-view logic where possible.

---

## C. Feature-View Layer

### Purpose

Build reusable, model-ready feature views from fighter-state artifacts.

This is the new feature engineering architecture.

### Runner

```bash
python -m pipeline.features.run_build_feature_view \
  --config configs/feature_views/moneyline_base.yaml
```

### Config

```text
configs/feature_views/moneyline_base.yaml
```

### Builder

```text
pipeline/features/views/moneyline.py
```

### Inputs

```text
data/master/ufc_master.parquet
data/features/fighter_state_history.parquet
```

### Output

```text
data/features/moneyline_feature_view.parquet
```

### Current behavior

The moneyline feature view:

1. prepares master rows with `prepare_master_for_rolling`,
2. joins red fighter state using `fight_id + r_id`,
3. joins blue fighter state using `fight_id + b_id`,
4. emits V5-compatible prefight columns such as `r_pre_elo`, `b_pre_elo`,
5. emits EWM columns such as `r_ewm_elo`, `b_ewm_elo`,
6. emits recent-form columns such as `r_recent_form_elo`, `b_recent_form_elo`,
7. creates red-minus-blue diffs such as `elo_diff`, `ewm_elo_diff`, `recent_form_elo_diff`,
8. applies existing V5 engineered features from `ufc_feature_engineering.py`.

### Why this matters

This layer is the intended replacement for training directly from:

```text
data/features/UFC_enhanced_rolling_features_EWM.parquet
```

The model config and training pipeline should move to:

```text
data/features/moneyline_feature_view.parquet
```

once parity is validated.

---

## D. Training Layer

### Purpose

Train a model from a model config and feature view.

### Runner

```bash
python -m pipeline.training.run_train_model \
  --config configs/models/moneyline_xgboost_v5.yaml
```

### Main config

```text
configs/models/moneyline_xgboost_v5.yaml
```

### Model registry

```text
configs/models/model_registry.yaml
```

### Current model

```text
model_id: moneyline_xgboost_v5
model_family: moneyline
algorithm: xgboost
artifact_dir: models/moneyline/xgboost_v5
```

### Expected model feature count

```text
124
```

### Artifact outputs

```text
models/moneyline/xgboost_v5/raw_model.joblib
models/moneyline/xgboost_v5/calibrated_model.joblib
models/moneyline/xgboost_v5/feature_columns.joblib
models/moneyline/xgboost_v5/feature_columns.json
models/moneyline/xgboost_v5/metrics.json
models/moneyline/xgboost_v5/threshold_sweep.parquet
models/moneyline/xgboost_v5/confidence_buckets.parquet
models/moneyline/xgboost_v5/model_card.yaml
```

### Current status

Training runner is well structured:

- config-driven,
- explicit feature contract,
- unsafe raw red/blue feature blocker,
- train/calibration/test split,
- symmetry augmentation,
- calibration,
- metrics,
- artifact saving.

### Known issue

`configs/models/moneyline_xgboost_v5.yaml` still points to the old rolling feature artifact:

```text
data/features/UFC_enhanced_rolling_features_EWM.parquet
```

Target path should be:

```text
data/features/moneyline_feature_view.parquet
```

This should be changed only after confirming the feature view contains all 124 configured model features and parity with the old rolling artifact is acceptable.

---

## E. Upcoming Card / Live Card Layer

### Refresh upcoming events

```bash
python -m pipeline.prediction.run_refresh_upcoming_events
```

Outputs:

```text
data/cards/ufcstats_upcoming_events.parquet
data/cards/ufcstats_upcoming_fights.parquet
```

### Build selected live card

```bash
python -m pipeline.prediction.run_build_live_card --event-id <event_id>
```

or:

```bash
python -m pipeline.prediction.run_build_live_card --all-upcoming
```

Outputs:

```text
data/predictions/ufc_live_card.parquet
data/cards/ufc_selected_live_card_event.parquet
data/audits/ufc_live_card_build_audit.parquet
data/audits/ufc_live_card_rejected_rows.parquet
```

### Required live card fields

```text
event_id
event_name
fight_id
red_fighter
blue_fighter
red_fighter_id
blue_fighter_id
```

### Current status

Live-card builder is strict and ID-focused. It rejects placeholder rows and invalid fights.

### Known issue

The existing selected-event workflow still runs legacy scripts after building the live card. V2 needs a clean selected-event orchestration workflow.

---

## F. Live Feature / Prediction V2 Layer

### Core workflow

```text
.github/workflows/run-prediction-v2.yml
```

### Runner

```bash
python -m pipeline.modeling.run_prediction --model-id moneyline_xgboost_v5
```

### Main modules

```text
pipeline/modeling/run_prediction.py
pipeline/modeling/model_registry.py
pipeline/modeling/model_config.py
pipeline/modeling/model_loader.py
pipeline/modeling/prediction_adapter.py
pipeline/modeling/prediction_formatter.py
pipeline/modeling/probability.py
pipeline/modeling/algorithms/xgboost_predictor.py
pipeline/prediction/live_feature_builder.py
```

### Inputs

```text
configs/models/model_registry.yaml
configs/models/moneyline_xgboost_v5.yaml
models/moneyline/xgboost_v5/*
data/predictions/ufc_live_card.parquet
```

The live feature builder currently also expects:

```text
data/features/ufc_current_fighter_features.parquet
```

This is a legacy/bridge artifact and should not remain the V2 source of truth.

### Outputs

```text
data/predictions/live_model_features.parquet
data/audits/ufc_live_feature_audit.parquet
data/predictions/model_outcomes.parquet
data/predictions/by_model/<model_id>/model_outcomes.parquet
```

### Current status

The modeling layer is strong and generic:

- registry-based model selection,
- model bundle loading,
- generic prediction adapter,
- config-driven outcome formatting,
- moneyline emitted as outcome rows.

### Known issue

The weak point is upstream live feature construction. `live_feature_builder.py` is still a bridge implementation and can silently zero-fill missing model features.

Target direction:

```text
live prediction should use the same feature-view/state architecture as training
```

---

## G. Model Outcomes Artifact

### Path

```text
data/predictions/model_outcomes.parquet
```

### Grain

```text
one row per fight + market + outcome + model
```

### Required conceptual columns

```text
prediction_run_id
prediction_timestamp
model_id
model_family
algorithm
prediction_type
event_id
event_name
fight_id
red_fighter
blue_fighter
red_fighter_id
blue_fighter_id
market_key
outcome_label
outcome_fighter_id
outcome_side
model_probability
model_pick_probability
is_model_pick
model_pick
model_confidence
confidence_pct
```

### Join key used downstream

```text
fight_id + market_key + outcome_fighter_id
```

---

## H. Market V2 Layer

### Workflow

```text
.github/workflows/run-market-v2.yml
```

### Runner

```bash
python -m pipeline.market.run_market_update_v2
```

### Main modules

```text
pipeline/market/run_market_update_v2.py
pipeline/market/providers/the_odds_api.py
pipeline/market/outcome_matcher.py
pipeline/market/normalizers/moneyline.py
pipeline/market/market_validator.py
```

### Inputs

```text
ODDS_API_KEY
configs/market/market_registry.yaml
data/predictions/ufc_live_card.parquet
```

### Outputs

```text
data/market/market_outcomes.parquet
data/market/market_outcome_snapshots.parquet
data/audits/ufc_market_outcome_audit.parquet
data/audits/ufc_market_match_audit_v2.parquet
```

### Current status

Market V2 is operational for moneyline.

It:

1. fetches odds from The Odds API,
2. flattens h2h/moneyline markets,
3. matches provider fighter names to UFCStats live-card fights,
4. maps provider names to canonical red/blue fighter IDs,
5. emits two outcome rows per fight/bookmaker,
6. validates the market artifact,
7. appends to snapshot history.

### Important design point

Provider names are only a bridge. Downstream joins use IDs.

### Known issue

`pipeline/market/normalizers/moneyline.py` has stale docstring text that says the join key is:

```text
fight_id + market_key + outcome_label
```

Runtime output includes `outcome_fighter_id`, and Betting Outcomes V2 correctly joins on ID. The docstring should be corrected.

---

## I. Betting Outcomes V2 Layer

### Workflow

```text
.github/workflows/run-betting-outcomes-v2.yml
```

### Runner

```bash
python -m pipeline.betting.run_betting_outcomes_v2
```

### Inputs

```text
data/predictions/model_outcomes.parquet
data/market/market_outcomes.parquet
data/bankroll/ufc_bankroll_settings.parquet or default risk settings
```

### Output

```text
data/predictions/betting_outcomes.parquet
data/audits/ufc_betting_outcomes_audit.parquet
```

### Join key

```text
fight_id + market_key + outcome_fighter_id
```

### Calculations

For each joined model/market outcome row:

```text
edge = model_probability - implied_probability
edge_pct = edge * 100
EV = model_probability * (decimal_odds - 1) - (1 - model_probability)
ev_pct = EV * 100
ev_dollars_at_100 = EV * 100
full_kelly_fraction = Kelly stake fraction
fractional_kelly_fraction = full Kelly * configured Kelly fraction
recommended_stake = bankroll * fractional Kelly, capped by max stake
```

### Filters

Rows are marked as bet candidates only if they pass:

```text
market data filter
edge filter
confidence filter
odds range filter
```

### Current status

Betting Outcomes V2 is coherent and aligned with outcome rows.

### Known issue

The audit uses a strict validation rule:

```text
passes_validation = joined_rows == len(market_df) and missing_prediction_count == 0
```

This may fail legitimately if market outcomes include rows not covered by a model, multiple unsupported books/markets, or future prop markets before model support exists.

---

## J. Dashboard Layer

### Root dashboard

```text
dashboard.py
```

### Main tabs

```text
tabs/betting_board_v2.py
tabs/line_movement.py
tabs/bankroll.py
tabs/model_lab.py
tabs/data_maintenance.py
```

### Current status by tab

#### Betting Board V2

Mostly aligned with V2.

Consumes:

```text
data/predictions/betting_outcomes.parquet
```

It may pivot outcome rows into red/blue presentation cards. This is acceptable as display logic.

#### Data Maintenance

Mostly aligned with ingestion architecture.

#### Bankroll

Functional but still moneyline/legacy-ledger shaped.

#### Line Movement / CLV

Still legacy. It reads older market/CLV artifacts and has not fully migrated to outcome-level market snapshots.

#### Model Lab

Most stale dashboard area. It still references older model artifact patterns such as `UFC_Model_v5_Experiment` and should be migrated to model registry / model card / model bundle artifacts.

### Compatibility shims

Some utilities intentionally bridge V2 artifacts to old display contracts:

```text
utils/data_loader.py
utils/betting_outcomes_adapter.py
utils/betting_board_artifacts.py
```

These should remain until the dashboard is fully outcome-native, then be marked for archive/removal.

---

# Current V2 Workflows

## Core V2 workflows

```text
.github/workflows/run-live-features-v2.yml
.github/workflows/run-prediction-v2.yml
.github/workflows/run-market-v2.yml
.github/workflows/run-betting-outcomes-v2.yml
```

### Current core run order

Minimum useful V2 order:

```text
run-prediction-v2.yml
→ run-market-v2.yml
→ run-betting-outcomes-v2.yml
```

Optional diagnostic step:

```text
run-live-features-v2.yml
```

`run-prediction-v2.yml` already rebuilds live features, so the standalone live-features workflow is redundant unless used for QA.

## Upstream workflows

```text
.github/workflows/run-refresh-upcoming-events.yml
.github/workflows/run-betting-board-selected-event.yml
```

### Important issue

`run-betting-board-selected-event.yml` currently refreshes/builds live card, then runs legacy scripts:

```text
run_model_predictions.py
run_market_update.py
run_betting_decision.py
```

This is not a clean V2 selected-event workflow.

## Missing recommended V2 workflows

The architecture needs these or equivalent orchestration:

```text
run-build-fighter-state-v2.yml
run-build-feature-view-v2.yml
run-selected-event-v2.yml
```

A clean selected-event V2 workflow should run:

```bash
python -m pipeline.prediction.run_refresh_upcoming_events
python -m pipeline.prediction.run_build_live_card --event-id "$EVENT_ID"
python -m pipeline.features.run_build_fighter_state
python -m pipeline.features.run_build_feature_view --config configs/feature_views/moneyline_base.yaml
python -m pipeline.modeling.run_prediction --model-id moneyline_xgboost_v5
python -m pipeline.market.run_market_update_v2
python -m pipeline.betting.run_betting_outcomes_v2
```

---

# Known Issues / Gaps / Code Errors

## Critical

### 1. Training config still points to old rolling feature warehouse - RESOLVED

Current:

```text
data/features/UFC_enhanced_rolling_features_EWM.parquet
```

Target:

```text
data/features/moneyline_feature_view.parquet
```

File:

```text
configs/models/moneyline_xgboost_v5.yaml
```

Impact:

Training and V2 feature-view architecture are not yet fully connected.

Recommended fix:

Validate `moneyline_feature_view.parquet` has all 124 configured model features, then update the config.

---

### 2. Live feature builder still uses legacy current-fighter feature path - RESOLVED

Current live feature bridge expects:

```text
data/features/ufc_current_fighter_features.parquet
```

Target architecture uses:

```text
data/features/fighter_state_history.parquet
data/features/latest_fighter_state.parquet
data/features/moneyline_feature_view.parquet
```

File:

```text
pipeline/prediction/live_feature_builder.py
```

Impact:

Prediction V2 can fail or use stale/legacy features.

Recommended fix:

Refactor live feature generation to use shared feature-view/state logic rather than `ufc_current_fighter_features.parquet`.

---

### 3. Silent zero-fill hides missing live features

`live_feature_builder.py` currently creates any missing requested model features as `0.0`.

Impact:

The model can run even when whole feature families are missing. This likely explains prior observations where EWM features were all zero for some live fighters.

Recommended fix:

Replace silent zero-fill with explicit failure or severe audit failure for missing model-contract features. Accept zero values only when the feature was actually computed as zero.

---

### 4. Clean V2 orchestration is missing

Core V2 runners exist, but the full run order is not yet captured in one clean V2 workflow.

Missing or recommended workflows:

```text
run-build-fighter-state-v2.yml
run-build-feature-view-v2.yml
run-selected-event-v2.yml
```

Impact:

Operators may run a mix of legacy and V2 scripts.

---

### 5. Selected-event workflow still runs legacy scripts

Current workflow:

```text
run-betting-board-selected-event.yml
```

Runs:

```text
run_model_predictions.py
run_market_update.py
run_betting_decision.py
```

Impact:

Dashboard event selection may create/update legacy artifacts, not V2 outcome artifacts.

Recommended fix:

Create a new V2 selected-event workflow and either rename/archive the old one or label it clearly as legacy.

---

## High Priority

### 6. Moneyline market normalizer docstring has stale join key

File:

```text
pipeline/market/normalizers/moneyline.py
```

Docstring says:

```text
fight_id + market_key + outcome_label
```

Correct V2 join key:

```text
fight_id + market_key + outcome_fighter_id
```

Runtime output is mostly correct; documentation is stale.

---

### 7. Betting Outcomes audit validation may be too strict

Current validation requires:

```text
joined_rows == len(market_df)
```

This may fail once unsupported books/markets/props exist.

Recommended fix:

Adjust audit to validate expected supported join coverage rather than all market rows universally.

---

### 8. Market V2 is moneyline only

Market V2 intentionally supports moneyline operationally for now.

Impact:

Prop-market architecture exists conceptually, but provider flattening/normalizers/models are not implemented for method, goes distance, totals, or round props.

---

### 9. Training workflow not confirmed

The training runner exists, but a dedicated current GitHub workflow for training was not confirmed during review.

Recommended fix:

Add or document a training workflow after feature-view migration is validated.

---

## Medium Priority

### 10. Model Lab dashboard is stale

Files:

```text
tabs/model_lab.py
utils/model_lab_artifacts.py
```

It still references older production model artifact patterns.

Recommended fix:

Migrate Model Lab to:

```text
configs/models/model_registry.yaml
models/moneyline/xgboost_v5/model_card.yaml
models/moneyline/xgboost_v5/metrics.json
models/moneyline/xgboost_v5/confidence_buckets.parquet
models/moneyline/xgboost_v5/threshold_sweep.parquet
```

---

### 11. CLV / Line Movement dashboard remains legacy

Files:

```text
tabs/line_movement.py
```

Still reads legacy market/CLV artifacts.

Recommended fix:

Migrate to outcome-level artifacts:

```text
data/market/market_outcome_snapshots.parquet
data/predictions/betting_outcomes.parquet
future official bet ledger outcome keys
```

---

### 12. Bankroll / ledger is not fully outcome-native

Bankroll works for moneyline but still uses older terms like fighter/opponent and moneyline-style records.

Recommended fix:

Move ledger schema toward:

```text
fight_id
market_key
outcome_fighter_id
outcome_label
bookmaker
odds_taken
stake
bet_status
closing_odds
clv
result
profit_loss
```

---

### 13. Compatibility shims need lifecycle labels

Files:

```text
utils/data_loader.py
utils/betting_outcomes_adapter.py
utils/betting_board_artifacts.py
```

Recommended fix:

Mark as compatibility shims in comments/docs and remove only after dashboard consumers are fully V2-native.

---

## Low Priority / Cleanup

### 14. Duplicate/older docs should be organized

Existing docs include older V2 architecture docs and implementation plans. They are useful historically but can confuse future work.

Recommended fix:

Add a docs index that labels each architecture doc as:

```text
current
historical
target/spec
implementation plan
legacy/archive candidate
```

---

### 15. Root-level utility modules still hold important logic

Examples:

```text
ufc_feature_engineering.py
ufc_odds_utils.py
ufc_pipeline_utils.py
pipeline_config.py
```

Recommended fix:

Do not move these casually. First document which production modules import them. Later, migrate into package namespaces if desired.

---

# Correct Current Artifact Map

## Historical / Training Source

```text
data/master/ufc_master.parquet
```

## Fighter State

```text
data/features/fighter_state_history.parquet
data/features/latest_fighter_state.parquet
```

## Feature View

```text
data/features/moneyline_feature_view.parquet
```

## Legacy Rolling Feature Artifact

```text
data/features/UFC_enhanced_rolling_features_EWM.parquet
```

Status: legacy/parity until feature-view migration complete.

## Live Card

```text
data/predictions/ufc_live_card.parquet
```

## Live Model Features

```text
data/predictions/live_model_features.parquet
```

Status: should be produced from V2 feature architecture; currently bridge implementation.

## Model Outcomes

```text
data/predictions/model_outcomes.parquet
data/predictions/by_model/<model_id>/model_outcomes.parquet
```

## Market Outcomes

```text
data/market/market_outcomes.parquet
data/market/market_outcome_snapshots.parquet
```

## Betting Outcomes

```text
data/predictions/betting_outcomes.parquet
```

## Audits

```text
data/audits/ufc_live_feature_audit.parquet
data/audits/ufc_market_outcome_audit.parquet
data/audits/ufc_market_match_audit_v2.parquet
data/audits/ufc_betting_outcomes_audit.parquet
```

---

# Recommended Next Work Order

## Phase 1: Document and freeze current state

1. Keep this document as the working V2/V3 handoff.
2. Do not delete legacy artifacts yet.
3. Label docs and workflows as active/legacy/bridge.

## Phase 2: Validate feature view

1. Run fighter-state builder.
2. Run feature-view builder.
3. Confirm `moneyline_feature_view.parquet` exists.
4. Confirm all 124 model features from `moneyline_xgboost_v5.yaml` exist in the feature view.
5. Compare feature distributions against `UFC_enhanced_rolling_features_EWM.parquet`.
6. Specifically inspect EWM families:

```text
ewm_elo_diff
ewm_splm_diff
ewm_td_avg_diff
ewm_recent_win_pct_diff
recent_form_elo_diff
recent_form_splm_diff
```

## Phase 3: Move training to feature view

1. Update `configs/models/moneyline_xgboost_v5.yaml` to use:

```text
data/features/moneyline_feature_view.parquet
```

2. Retrain.
3. Compare metrics to current model card.
4. Confirm artifacts still load through Prediction V2.

## Phase 4: Fix live feature generation

1. Refactor `live_feature_builder.py` away from `ufc_current_fighter_features.parquet`.
2. Use shared feature-view/state logic.
3. Remove silent model-feature zero-fill.
4. Make missing model features fail loudly or fail audit.

## Phase 5: Clean V2 orchestration

Create a V2 selected-event workflow:

```bash
python -m pipeline.prediction.run_refresh_upcoming_events
python -m pipeline.prediction.run_build_live_card --event-id "$EVENT_ID"
python -m pipeline.features.run_build_fighter_state
python -m pipeline.features.run_build_feature_view --config configs/feature_views/moneyline_base.yaml
python -m pipeline.modeling.run_prediction --model-id moneyline_xgboost_v5
python -m pipeline.market.run_market_update_v2
python -m pipeline.betting.run_betting_outcomes_v2
```

## Phase 6: Dashboard migration

1. Keep Betting Board V2.
2. Migrate Model Lab to model registry/artifacts.
3. Migrate CLV/Line Movement to outcome-level snapshots.
4. Update Bankroll ledger to outcome-native schema.
5. Remove compatibility shims only after consumers are updated.

---

# Minimal Debug Checklist

When V2 predictions or betting board look wrong, check in this order:

1. Does `data/predictions/ufc_live_card.parquet` contain the intended event?
2. Does live card have valid `fight_id`, `red_fighter_id`, and `blue_fighter_id`?
3. Does `data/features/fighter_state_history.parquet` exist and contain recent rows?
4. Does `data/features/moneyline_feature_view.parquet` exist?
5. Do all 124 model features exist in the selected feature source?
6. Are EWM features nonzero for experienced fighters?
7. Does `data/predictions/model_outcomes.parquet` have two moneyline rows per fight?
8. Does `data/market/market_outcomes.parquet` have matching `fight_id + market_key + outcome_fighter_id` rows?
9. Does `data/audits/ufc_market_match_audit_v2.parquet` show matched provider rows?
10. Does `data/predictions/betting_outcomes.parquet` have joined rows?
11. If betting output is empty, compare join keys between model and market outcomes.
12. If EV seems wrong, inspect `model_probability`, `american_odds`, `decimal_odds`, and `implied_probability`.

---

# Future Chat Handoff Summary

A future chat should know:

1. V2 canonical outputs are outcome-level, not red/blue side-level.
2. The join key is `fight_id + market_key + outcome_fighter_id`.
3. The new feature architecture is `fighter_state_history → moneyline_feature_view`.
4. Training has not yet fully switched to the new feature view.
5. Live feature generation is still a bridge and is the highest-risk code path.
6. Core V2 prediction/market/betting runners exist and are mostly coherent.
7. Orchestration is incomplete; selected-event workflow still runs legacy scripts.
8. Dashboard is partially migrated: Betting Board V2 is current, Model Lab and CLV are legacy.
9. Do not delete legacy files until parity and consumers are validated.
10. The next safest engineering move is documentation + parity validation, not broad refactoring.
