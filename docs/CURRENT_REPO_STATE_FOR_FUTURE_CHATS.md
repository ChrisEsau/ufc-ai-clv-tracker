# Current Repo State For Future Chats

## Purpose

This is the authoritative current-state handoff for future ChatGPT, Codex, and development sessions.

Read this file first before interpreting older architecture docs. Several docs describe future target architecture, while the codebase currently contains a mix of modern modular components and legacy root-level runners.

## Branch and Repo Rules

- Active development branch: `dev`
- Stable branch: `main`
- Do not change model contracts, feature contracts, master schema, artifact paths, or ingestion gates without explicit approval.
- This chat/workstream is documentation-focused. Code changes should be reviewed separately.

## Current High-Level Architecture Status

```text
Modern / modular:
- Feature generation
- Training framework
- Data Maintenance ingestion/review
- CLV pipeline
- Bankroll pipeline
- Upcoming event refresh
- Live-card builder

Legacy / not fully migrated:
- Live model prediction runner
- Market update runner
- Betting decision runner
- Model Lab artifact loading
```

## Important Current-State Warning

The training framework has migrated to the new modular model artifact path:

```text
models/moneyline/xgboost_v5/
```

But live prediction and Model Lab are still wired through `pipeline/common/paths.py` legacy model constants:

```text
models/UFC_Model_v5_Experiment/
```

Do not delete or migrate either path casually. A planned model-loader / registry migration is required before live prediction can rely on the new `models/moneyline/xgboost_v5/` artifacts.

## Active Training Framework

Current training command from repo root:

```powershell
python -m pipeline.training.run_train_model `
  --config configs/models/moneyline_xgboost_v5.yaml
```

Current active config:

```text
configs/models/moneyline_xgboost_v5.yaml
```

Current training artifact output:

```text
models/moneyline/xgboost_v5/
```

Current successful model artifacts include:

```text
calibrated_model.joblib
raw_model.joblib
feature_columns.joblib
feature_columns.json
metrics.json
model_card.yaml
```

Current final metrics from `models/moneyline/xgboost_v5/metrics.json`:

```text
Accuracy: 0.8075201432408237
ROC-AUC: 0.894757828272911
Log Loss: 0.3992839553206625
Brier: 0.13238324312634117
Best threshold: 0.47
Test rows: 2234
```

## Feature Generation Status

Modern feature-builder runner:

```powershell
python -m pipeline.features.run_build_rolling_features
```

Current rolling feature warehouse:

```text
data/features/UFC_enhanced_rolling_features_EWM.parquet
```

Current expected shape:

```text
8574 rows x 483 columns
```

Current model feature contract:

```text
124 explicit features
```

Important: do not split the rolling feature warehouse into separate base/moneyline/prop stores until parity is proven and documented.

## Live Betting Board Workflow

Primary workflow:

```text
.github/workflows/run-betting-board-selected-event.yml
```

Current sequence:

```text
pipeline.prediction.run_refresh_upcoming_events
→ pipeline.prediction.run_build_live_card
→ run_model_predictions.py
→ run_market_update.py
→ run_betting_decision.py
→ commit artifacts
```

This workflow is mixed architecture:

- `pipeline.prediction.run_refresh_upcoming_events` is modular.
- `pipeline.prediction.run_build_live_card` is modular.
- `run_model_predictions.py` is legacy root-level.
- `run_market_update.py` is legacy root-level.
- `run_betting_decision.py` is legacy root-level.

## Live Prediction Status

Current runner:

```text
run_model_predictions.py
```

It loads model artifacts from legacy constants:

```text
MODEL_CALIBRATED_PATH
MODEL_FEATURE_COLUMNS_PATH
MODEL_BEST_THRESHOLD_PATH
MODEL_PRODUCTION_CONFIG_PKL_PATH
```

Those constants currently point through:

```text
pipeline/common/paths.py
```

to:

```text
models/UFC_Model_v5_Experiment/
```

Live prediction has not yet been migrated to the new training artifact bundle under `models/moneyline/xgboost_v5/`.

## Market Update Status

Current runner:

```text
run_market_update.py
```

Current workflow:

```text
.github/workflows/run-market-update.yml
```

The market update runner:

- reads model predictions
- pulls The Odds API using `ODDS_API_KEY`
- uses DraftKings as preferred bookmaker
- performs side-aware odds matching
- writes current market odds
- appends market snapshots
- writes market match audit

Market update is not yet under `pipeline/market/`.

## Betting Decision Status

Current runner:

```text
run_betting_decision.py
```

It merges:

```text
data/predictions/ufc_model_predictions.parquet
data/market/ufc_market_odds.parquet
```

and writes:

```text
data/predictions/ufc_betting_board.parquet
data/predictions/ufc_live_watchlist.parquet
data/predictions/ufc_official_bets.parquet
```

Risk settings are read from:

```text
pipeline.common.risk_settings
```

Default locked settings:

```text
starting_bankroll: 10000
kelly_fraction: 0.50
max_stake_pct: 0.03
max_event_exposure_pct: 0.10
min_edge: 0.05
min_confidence: 70.0
min_odds: -250
max_odds: 400
```

## CLV Status

Modern CLV runner:

```powershell
python -m pipeline.clv.run_clv_pipeline
```

Current workflow:

```text
.github/workflows/run-clv-tracker.yml
```

Schedule:

```text
Every 15 minutes
```

Important: CLV does not fetch fresh odds. It transforms existing market snapshots plus bankroll ledger bets.

CLV pipeline reads:

```text
data/market/ufc_market_snapshots.parquet
data/bankroll/ufc_bet_ledger.parquet
```

CLV pipeline writes:

```text
data/market/ufc_normalized_market_snapshots.parquet
data/audits/ufc_clv_market_normalization_audit.parquet
data/market/ufc_closing_lines.parquet
data/market/ufc_line_movement.parquet
data/market/ufc_clv_results.parquet
```

## Bankroll Status

Bankroll is modern/modular.

Source of truth:

```text
data/bankroll/ufc_bet_ledger.parquet
```

Derived artifacts:

```text
data/bankroll/ufc_open_bets.parquet
data/bankroll/ufc_bankroll_snapshots.parquet
data/bankroll/ufc_bankroll_settings.parquet
```

Bankroll CI runners:

```text
pipeline/bankroll/run_bankroll_status.py
pipeline/bankroll/run_append_manual_bet.py
pipeline/bankroll/run_settle_manual_bet.py
pipeline/bankroll/run_save_risk_settings.py
```

Manual bet changes and risk-setting changes should be persisted through GitHub Actions, not only local Streamlit writes.

## Data Maintenance Status

Data Maintenance is modern/modular and gated.

Dashboard tab:

```text
tabs/data_maintenance.py
```

Single-event ingest workflow:

```text
dm-ingest-single-event.yml
```

Append workflow:

```text
run-append-staged-to-master.yml
```

Append must remain separate from ingest. Append requires both:

```text
append_ready == true
final_review_pass == true
```

Master dataset remains locked at:

```text
data/master/ufc_master.parquet
128 columns
```

## Dashboard Status

Main dashboard router:

```text
dashboard.py
```

Dashboard tabs:

```text
Betting Board          -> tabs/betting_board.py
Line Movement / CLV    -> tabs/line_movement.py
Bankroll               -> tabs/bankroll.py
Model Lab              -> tabs/model_lab.py
Data Maintenance       -> tabs/data_maintenance.py
```

Model Lab is read-only diagnostics and still legacy-artifact-aware. It is not the active model-training interface.

## Docs Interpretation Rule

When docs conflict, interpret them in this order:

1. This file: `docs/CURRENT_REPO_STATE_FOR_FUTURE_CHATS.md`
2. `README.md`
3. `docs/CURRENT_PROJECT_HANDOFF.md`
4. Current code and workflow files
5. Architecture docs describing future target state

Architecture docs such as model registry, model adapter, prop model, and feature layer may describe future desired architecture. Do not assume those future paths already exist in production unless verified in code.

## Current Main Documentation Drift To Remember

The most important drift is model artifact loading:

```text
Training output path:
models/moneyline/xgboost_v5/

Live prediction / Model Lab path:
models/UFC_Model_v5_Experiment/
```

This is intentional current-state drift, not something to silently fix. It should be resolved only through a planned model registry / model loader migration.

## Recommended Next Documentation Tasks

1. Mark future-only docs clearly as `FUTURE TARGET ARCHITECTURE`.
2. Mark current docs clearly as `CURRENT IMPLEMENTATION`.
3. Update model registry and adapter docs only after deciding the final model bundle contract.
4. Keep README and this handoff synchronized after major repo changes.
