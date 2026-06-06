# UFC AI CLV Tracker

Private UFC betting intelligence platform for data maintenance, live fight prediction, betting-board review, line movement / CLV tracking, bankroll management, and model diagnostics.

## Current branch strategy

- `dev` is the active development branch.
- `main` should be treated as the stable branch.
- New work should be stabilized on `dev` before merging back to `main`.
- Avoid starting large new features while `dev` is heavily diverged from `main`.

## Local setup

```bash
git clone https://github.com/ChrisEsau/ufc-ai-clv-tracker.git
cd ufc-ai-clv-tracker
git checkout dev
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
source .venv/bin/activate
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
- `docs/` — architecture and registry documents

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

- Keep paths centralized in `pipeline/common/paths.py`.
- Prefer module runners under `pipeline/...` and keep root scripts as compatibility wrappers only.
- Do not commit API keys, local `.env` files, virtual environments, or cache folders.
- Treat append-to-master as a gated operation.
- Prefer auditable parquet outputs for validation steps.
- Make dashboard failures visible instead of silently hiding broken artifacts.

## Useful docs

See `docs/` for architecture and registry files, including:

- `UFC_PROJECT_OVERVIEW.md`
- `UFC_REPOSITORY_STRUCTURE.md`
- `UFC_DATA_FLOW.md`
- `UFC_INGESTION_PIPELINE_REGISTRY.md`
- `UFC_MASTER_SCHEMA.md`
- `UFC_GITHUB_WORKFLOW_REGISTRY.md`
- `UFC_DM_DASHBOARD_ARCHITECTURE.md`
- `UFC_BETTING_BOARD_ARCHITECTURE.md`
- `UFC_LINE_MOVEMENT_CLV_ARCHITECTURE.md`
- `UFC_BANKROLL_ARCHITECTURE.md`
- `UFC_MODEL_LAB_ARCHITECTURE.md`
