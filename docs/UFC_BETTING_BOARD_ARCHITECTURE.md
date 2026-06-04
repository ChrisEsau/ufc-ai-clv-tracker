# UFC Betting Board Architecture

## Purpose

The Betting Board is the primary betting decision workspace.

It answers:

```text
What fights are available?
What does the model think?
What does the market think?
Is there positive EV?
Should we bet, watchlist, or pass?
```

---

## Core Responsibilities

* Display upcoming fight cards
* Show model win probabilities
* Show sportsbook odds
* Calculate implied probability
* Calculate EV
* Rank betting opportunities
* Display confidence tiers
* Display watchlist and near-miss plays
* Support bankroll-aware staking

---

## Primary Inputs

* data/master/ufc_master.parquet
* model prediction artifacts
* market odds artifacts
* fighter feature stores
* CLV snapshots
* bankroll settings

---

## Primary Outputs

* Betting recommendations
* Watchlist entries
* No-bet decisions
* EV rankings
* Suggested stake sizes

---

## Key Concepts

### Model Probability

The model's estimated probability that a fighter wins.

### Market Implied Probability

Probability derived from sportsbook odds.

### Edge

```text
Model Probability - Market Implied Probability
```

### Expected Value

Used to determine whether a bet is profitable in the long run.

### Confidence Tier

A label based on model confidence and data quality.

Example tiers:

* Strong Bet
* Lean Bet
* Watchlist
* Pass

---

## Default Betting Rules

Current locked strategy:

```text
EV Threshold: $50
Confidence Threshold: 70%
Odds Range: -250 to +400
Bet Sizing: Half Kelly
```

---

## Future Enhancements

* Prop bet recommendations
* Multi-book odds comparison
* Dynamic requalification from line movement
* Confidence-weighted staking
* Market-aware model features
* Bet ledger integration

## Selected Upcoming Event Flow

The Betting Board now treats the target card as a selectable upstream artifact instead of a hard-coded live-card parquet file.

1. `run-refresh-upcoming-events.yml` runs `python -m pipeline.prediction.run_refresh_upcoming_events` to scrape the UFCStats upcoming-events page and each upcoming event detail page.
2. The refresh runner writes `data/cards/ufcstats_upcoming_events.parquet` and `data/cards/ufcstats_upcoming_fights.parquet`.
3. The Betting Board tab reads those card artifacts and lets the operator select one UFCStats event id.
4. `run-betting-board-selected-event.yml` rebuilds `data/predictions/ufc_live_card.parquet` from the selected event, then runs model predictions, market update, and betting decision in sequence.
5. The workflow commits only generated runtime artifacts with `git add -f`; source branches should not manually commit those parquet outputs.

This keeps event selection auditable while avoiding a permanent hard-coded prediction input file.

## Active Betting Board Workflows

| Workflow | Purpose | Primary Outputs |
|---|---|---|
| `run-refresh-upcoming-events.yml` | Refresh UFCStats upcoming event/card choices. | `data/cards/ufcstats_upcoming_events.parquet`, `data/cards/ufcstats_upcoming_fights.parquet` |
| `run-betting-board-selected-event.yml` | Run the full selected-event prediction and betting-board sequence. | `data/predictions/ufc_live_card.parquet`, `data/predictions/ufc_model_predictions.parquet`, `data/market/*`, `data/predictions/ufc_betting_board.parquet` |
| `run-market-update.yml` | Refresh market odds for the current model prediction artifact. | `data/market/ufc_market_odds.parquet`, `data/market/ufc_market_snapshots.parquet`, `data/market/ufc_market_match_audit.parquet` |
| `run-clv-tracker.yml` | Update closing-line and CLV tracking artifacts. | `data/market/ufc_closing_lines.parquet`, `data/market/ufc_clv_results.parquet`, `data/market/ufc_line_movement.parquet` |

## Adjustable Dashboard Betting Rules

The selected-event workflow remains the production/default execution path: when an operator presses **Run Betting Predictions for Selected Event**, the workflow uses the default betting filters and staking settings and writes the official Betting Board artifact.

After that artifact exists, the Betting Board tab supports dashboard-only scenario controls for:

* minimum edge,
* minimum confidence,
* American odds range,
* positive-EV requirement,
* watchlist near-miss behavior,
* high-EV watchlist override,
* bankroll,
* Kelly fraction,
* maximum stake percentage,
* minimum stake,
* stake rounding.

Scenario controls recalculate displayed statuses and stakes in memory. They do not overwrite `data/predictions/ufc_betting_board.parquet`, do not change the selected-event workflow inputs, and do not commit generated artifacts. The tab should clearly show production-vs-scenario official bets and stake totals so an operator can compare default output against adjusted rules.
