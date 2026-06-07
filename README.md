# UFC AI CLV Tracker

Private UFC betting intelligence platform for data maintenance, feature generation, model training, live fight prediction, betting-board review, line movement / CLV tracking, bankroll management, and model diagnostics.

## Current project state

As of the current development checkpoint, the project has a working modular moneyline training framework that replaces the old notebook-only training flow for the first production XGBoost model.

Validated end-to-end run:

```powershell
python -m pipeline.training.run_train_model `
  --config configs/models/moneyline_xgboost_v5.yaml
```

Last successful local run after cleaning 25 invalid-date master rows and rebuilding the rolling feature warehouse:

```text
Feature dataframe shape: (8574, 483)
Resolved feature count: 124
Model dataframe shape: (17148, 483)
Train rows        : 13874
Calibration rows  : 1040
Test rows         : 2234
Feature count     : 124
Best threshold    : 0.47
Accuracy          : 0.8075
ROC-AUC           : 0.8948
Log loss          : 0.3993
Brier score       : 0.1324
```

Primary trained artifact output:

```text
models/moneyline/xgboost_v5/
```

## Current branch strategy

- `dev` is the active development branch.
- `main` should be treated as the stable branch.
- New work should be stabilized on `dev` before merging back to `main`.
- Avoid starting large new features while `dev` is heavily diverged from `main`.

## Local setup

A virtual environment is recommended, but the project can also be run with system Python if dependencies are installed there.

Recommended:

```bash
git clone https://github.com/ChrisEsau/ufc-ai-clv-tracker.git
cd ufc-ai-clv-tracker
git checkout dev
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If running without a venv, install dependencies into the active Python environment:

```bash
python -m pip install pandas pyarrow numpy pyyaml scikit-learn xgboost joblib streamlit plotly
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

To leave a venv:

```bash
deactivate
```

## Environment variables

Create a local `.env` file from `.env.example` when running locally.

Required for odds ingestion:

```bash
ODDS_API_KEY=your_api_key_here
```

Do not commit real API keys.

## Run the dashboard

```bash
streamlit run dashboard.py
```

Main dashboard areas:

- Betting Board
- Line Movement / CLV
- Model Lab
- Data Maintenance
- Bankroll

## Canonical data layout

The project uses `pipeline/common/paths.py` as the path registry. Important artifact folders:

- `data/master/` — authoritative master fight dataset
- `data/staging/` — staged scrape and mapping outputs before append
- `data/audits/` — validation and QA artifacts
- `data/status/` — dataset and ingestion status artifacts
- `data/features/` — rolling and current fighter feature stores
- `data/predictions/` — live card, model predictions, watchlist, action board
- `data/market/` — odds snapshots, normalized market data, CLV outputs
- `data/bankroll/` — bankroll settings, bet ledger, open bets, snapshots
- `models/` — trained model artifacts
- `configs/` — model, feature-source, and feature-registry configuration files
- `docs/` — architecture, migration, handoff, and registry documents

## Current feature pipeline

The current canonical rolling feature warehouse is:

```text
data/features/UFC_enhanced_rolling_features_EWM.parquet
```

It is rebuilt from:

```text
data/master/ufc_master.parquet
```

with:

```powershell
python -m pipeline.features.run_build_rolling_features
```

Current expected shape after the invalid-date cleanup:

```text
Master rows  : 8574
Feature cols : 483
```

The current moneyline model uses an explicit 124-feature contract stored directly in:

```text
configs/models/moneyline_xgboost_v5.yaml
```

## Current training framework

Current training modules:

```text
pipeline/training/
  calibration.py
  feature_selection.py
  metrics.py
  model_training.py
  run_train_model.py
  symmetry.py
  temporal_split.py
  algorithms/
    xgboost_trainer.py
```

Current model config:

```text
configs/models/moneyline_xgboost_v5.yaml
```

Current feature-source registry:

```text
configs/features/feature_sources.yaml
```

Current model workflow:

```text
Load model config
Load rolling feature warehouse
Validate explicit 124-feature contract
Apply symmetry augmentation
Build train/calibration/test temporal split
Train XGBoost
Fit isotonic calibration on calibration split
Evaluate final test split
Save raw model, calibrated model, metrics, threshold sweep, confidence buckets, model card
```

## Data maintenance workflow

Recommended order for a completed event:

1. Refresh/discover UFCStats events.
2. Scrape fight rows for selected event.
3. Scrape fight details.
4. Map staged rows to master schema.
5. Transform derived stats.
6. Enrich fighter profiles.
7. Run master column validation.
8. Run append precheck validation.
9. Run final staged review.
10. Append staged rows to master only after review passes.
11. Rebuild rolling features.
12. Re-run training/backtest if the master changed meaningfully.

The single-event ingestion runner stages and reviews data but intentionally does not append automatically.

```bash
python -m pipeline.data_maintenance.run_ingest_single_event --event-id <UFCSTATS_EVENT_ID>
```

## Prediction / betting workflow

Typical live-card flow:

1. Refresh upcoming events.
2. Select/build live card.
3. Refresh market odds.
4. Run model predictions.
5. Build betting board / decision artifacts.
6. Review Betting Board filters, EV, edge, confidence, odds match quality, and stake sizing.
7. Manually append official bets only after review.
8. Track line movement and closing-line value.
9. Settle bets into bankroll artifacts.

## GitHub Actions

Workflows in `.github/workflows/` are used to run major pipeline stages and commit selected parquet outputs. This is convenient during development, but parquet artifact commits can grow git history quickly. Long term, consider moving large/generated artifacts to releases, object storage, or another artifact store.

## Development rules

- Keep paths centralized in `pipeline/common/paths.py` where practical.
- Prefer module runners under `pipeline/...` and keep root scripts as compatibility wrappers only.
- Do not commit API keys, local `.env` files, virtual environments, or cache folders.
- Treat append-to-master as a gated operation.
- Prefer auditable parquet outputs for validation steps.
- Make dashboard failures visible instead of silently hiding broken artifacts.
- Model configs are the single source of truth for model-specific features, algorithm, params, split, calibration, metrics, and artifacts.
- Do not infer production model feature lists from naming rules; use explicit model config feature lists.

## Useful docs

Key docs:

- `docs/CURRENT_PROJECT_HANDOFF.md`
- `docs/TRAINING_FRAMEWORK_ARCHITECTURE.md`
- `docs/MODEL_REGISTRY_ARCHITECTURE.md`
- `docs/ROLLING_NOTEBOOK_MIGRATION_PLAN.md`
- `docs/MODEL_ADAPTER_ARCHITECTURE.md`
- `docs/UFC_PROJECT_OVERVIEW.md`
- `docs/UFC_REPOSITORY_STRUCTURE.md`
- `docs/UFC_DATA_FLOW.md`
- `docs/UFC_INGESTION_PIPELINE_REGISTRY.md`
- `docs/UFC_MASTER_SCHEMA.md`
- `docs/UFC_GITHUB_WORKFLOW_REGISTRY.md`
- `docs/UFC_DM_DASHBOARD_ARCHITECTURE.md`
- `docs/UFC_BETTING_BOARD_ARCHITECTURE.md`
- `docs/UFC_LINE_MOVEMENT_CLV_ARCHITECTURE.md`
- `docs/UFC_BANKROLL_ARCHITECTURE.md`
- `docs/UFC_MODEL_LAB_ARCHITECTURE.md`
