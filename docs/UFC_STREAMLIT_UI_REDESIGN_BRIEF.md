# UFC Streamlit UI Redesign Brief for Codex

## Purpose

Redesign the UFC Betting Intelligence Platform dashboard to visually match the provided mockup screenshots while keeping the current Python/Streamlit backend architecture intact.

This is a UI/UX refactor only.

Do not rewrite the backend pipeline.

---

## Design References

Use the provided screenshots as visual references for:

* Betting Board tab
* Line Movement / CLV tab
* Model Lab tab
* Data Maintenance tab
* Bankroll tab

The screenshots represent the desired visual direction:

```text
Dark SaaS-style dashboard
Card-based layout
Left sidebar navigation
High-contrast KPI tiles
Modern status badges
Compact data tables
Plotly-style charts
Professional betting analytics interface
```

---

## Technology Requirements

Use:

```text
Python
Streamlit
Plotly
Pandas
Custom CSS via st.markdown(..., unsafe_allow_html=True)
```

Do not convert the project to:

```text
React
FastAPI
Dash
Flask
Django
```

The UI must remain Streamlit-based.

---

## Important Project Rules

### Do Not Change Backend Logic

Do not change:

* Data pipeline logic
* Model calculations
* EV calculations
* CLV calculations
* GitHub workflow dispatch logic
* Artifact paths
* Master schema
* Append logic

### Use Existing Paths

Always use:

```python
from pipeline.common.paths import ...
```

Do not hardcode artifact paths.

### Use Module Architecture

Keep the approved module structure:

```text
pipeline/common
pipeline/data_maintenance
pipeline/prediction
pipeline/features
pipeline/clv
```

Preferred execution format:

```bash
python -m pipeline.<workspace>.<runner>
```

---

## Desired UI System

Create reusable UI components under:

```text
utils/ui/
```

Recommended files:

```text
utils/ui/theme.py
utils/ui/cards.py
utils/ui/badges.py
utils/ui/sections.py
utils/ui/charts.py
utils/ui/tables.py
```

---

## Required Shared Components

### Metric Card

Reusable card for KPI values.

Examples:

```text
Current Bankroll
ROI
Positive EV Bets
Open Risk
Model Accuracy
Append Ready
```

Should support:

* Label
* Value
* Optional delta
* Optional status color
* Optional caption

---

### Status Badge

Reusable colored pill.

Statuses:

```text
success
warning
danger
neutral
info
```

Example labels:

```text
Strong Bet
Lean Bet
Watchlist
Pass
Append Ready
Blocked
Success
Failed
Running
```

---

### Section Container

Reusable dark card container for each dashboard section.

Should support:

* Title
* Subtitle/caption
* Optional icon
* Child content

---

### Chart Container

Reusable card wrapper for Plotly charts.

Should match the dark theme.

---

### Styled Data Table

Reusable display helper for tables.

Should support:

* Compact spacing
* Optional highlighted columns
* Optional status columns
* Wide layout

---

## Global Visual Style

Use a dark professional theme.

Suggested colors:

```text
Background: #07111f
Panel: #0d1727
Panel Alt: #101c2d
Border: #26364a
Text Primary: #f5f7fb
Text Muted: #9aa8bd
Green: #35d96b
Blue: #3b82f6
Yellow: #facc15
Red: #ef4444
Purple: #a855f7
```

General style:

```text
Rounded cards
Subtle borders
Soft shadows
Compact spacing
High readability
Professional analytics feel
```

---

## Sidebar

Keep the existing Streamlit sidebar architecture.

Improve visual presentation if possible, but do not break navigation.

Expected tabs:

```text
Betting Board
Line Movement / CLV
Model Lab
Data Maintenance
Bankroll
```

If Bankroll tab does not exist yet, add the architecture placeholder only if safe.

---

# Tab Requirements

## Betting Board Tab

Purpose:

Primary betting decision workspace.

Should display:

```text
Upcoming fights
Model probability
Sportsbook odds
Implied probability
Edge
EV
Kelly stake
Recommendation
Watchlist status
```

Recommended layout:

```text
Top KPI Cards
    - Positive EV Bets
    - Best Edge
    - Total Recommended Stake
    - Watchlist Count

Filters
    - Event
    - Confidence Tier
    - Odds Range
    - EV Threshold

Main Betting Table
    - Fight
    - Fighter
    - Model Probability
    - Odds
    - Implied Probability
    - Edge
    - EV
    - Kelly Stake
    - Recommendation Badge

Charts
    - EV distribution
    - Recommendation breakdown
    - Edge vs odds
```

Recommendation badges:

```text
Strong Bet
Lean Bet
Watchlist
Pass
```

---

## Line Movement / CLV Tab

Purpose:

Track market movement and closing line performance.

Should display:

```text
Opening odds
Current odds
Closing odds
Line movement
CLV
Beat closing line rate
Sportsbook movement
```

Recommended layout:

```text
Top KPI Cards
    - Beat Closing Line %
    - Average CLV
    - Positive CLV Bets
    - Market Moves Tracked

Line Movement Chart
CLV Results Table
Book Comparison Table
```

Core artifacts may include:

```text
ufc_market_odds.parquet
ufc_market_snapshots.parquet
ufc_line_movement.parquet
ufc_closing_lines.parquet
ufc_clv_results.parquet
```

---

## Model Lab Tab

Purpose:

Research and model evaluation workspace.

Should display:

```text
Model accuracy
Log loss
ROC-AUC
Calibration metrics
Backtest ROI
Feature importance
Model comparisons
```

Recommended layout:

```text
Top KPI Cards
    - Accuracy
    - ROC-AUC
    - Log Loss
    - Backtest ROI

Charts
    - Calibration curve
    - Confidence bucket performance
    - Feature importance
    - ROI by threshold

Tables
    - Model comparison
    - Recent validation results
```

Do not retrain models from the UI unless existing backend support already exists.

---

## Data Maintenance Tab

Purpose:

UFC ingestion control tower.

Current section order should remain:

```text
Dataset Health
Workflow Status
Event Discovery
Fight Scrape
Enrichment
Validation Gate
Audit History
Append Status
```

Recommended visual layout:

```text
Dataset Health KPI cards
Event Discovery table
Selected Event ingest control
Workflow status badges
Validation gate card
Append readiness card
Audit history table
```

Important rules:

```text
Append button must stay disabled unless append_ready == True.
Dashboard launches workflows.
Pipeline modules perform work.
```

---

## Bankroll Tab

Purpose:

Financial control center.

Should display:

```text
Starting bankroll
Current bankroll
Available bankroll
Open exposure
Total profit/loss
ROI
Official bet ledger
Risk settings
CLV performance by bet
```

Recommended layout:

```text
Top KPI Cards
    - Current Bankroll
    - Total Profit
    - ROI
    - Open Risk

Open Bets Table
Official Bet Ledger
Bankroll Curve
Profit by Event
CLV / Bet Quality
Risk Settings
```

Core future artifacts:

```text
data/bankroll/ufc_bet_ledger.parquet
data/bankroll/ufc_open_bets.parquet
data/bankroll/ufc_bankroll_snapshots.parquet
data/bankroll/ufc_bankroll_settings.parquet
```

If these artifacts do not exist yet, create placeholder UI that clearly states:

```text
Bankroll artifacts not found.
```

Do not fabricate real betting results.

---

## Error Handling

All dashboard tabs should handle missing artifacts gracefully.

Use messages like:

```text
Artifact not found. Run the relevant workflow first.
No data available yet.
```

The dashboard should not crash when a parquet file is missing.

---

## GitHub Actions Integration

Preserve existing workflow dispatch logic.

Existing utility:

```text
utils/github_actions.py
```

Do not remove it.

Data Maintenance workflows should continue using GitHub Actions where already implemented.

---

## Output Expectations for Codex

Implement changes in small, reviewable commits.

Suggested implementation order:

1. Create shared UI component system in `utils/ui/`
2. Apply theme globally
3. Refactor Betting Board tab
4. Refactor Line Movement / CLV tab
5. Refactor Model Lab tab
6. Refactor Data Maintenance tab
7. Add or scaffold Bankroll tab
8. Verify dashboard starts without errors
9. Do not alter backend pipeline behavior

---

## Validation Steps

After changes, run:

```bash
streamlit run dashboard.py
```

Also run:

```bash
python -m pipeline.data_maintenance.run_dataset_status
```

The app should load without crashing.

Data Maintenance should still show:

```text
Dataset Health
Event Discovery
Validation Gate
Append Status
```

---

## Non-Goals

Do not:

```text
Rewrite the app in React
Replace Streamlit
Change the data model
Change the master schema
Change append behavior
Hardcode paths
Delete audit files
Move pipeline logic into dashboard tabs
```

---

## Final Goal

The final UI should feel like a polished dark-mode betting analytics platform while preserving the existing UFC backend architecture.

The dashboard should look more professional, but the backend should remain stable, auditable, and pipeline-driven.
