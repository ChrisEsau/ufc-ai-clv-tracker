
# UFC AI Betting Platform — Final Product Vision

## Core Objective
Build a production-grade UFC betting intelligence platform that:
1. Predicts fight outcomes
2. Finds +EV betting opportunities
3. Sizes bets optimally
4. Tracks market movement and CLV
5. Continuously improves through retraining
6. Maximizes long-term ROI

## Dashboard Structure
1. Betting Board
2. Line Movement / CLV
3. Bet Ledger / Bankroll
4. Model Lab
5. Data Maintenance

## Betting Board
Displays:
- Current UFC card
- Official bets
- Watchlist bets
- Model probabilities
- Market implied probabilities
- Current EV
- Recommended stake

Fight statuses:
- NO BET
- WATCHLIST
- OFFICIAL BET
- STEAM AGAINST
- LINE MISSED
- BET PLACED
- RESULT GRADED

## Line Movement / CLV
Tracks:
- Opening/current/closing lines
- Steam movement
- Reverse line movement
- CLV performance
- Market timing

## Bet Ledger / Bankroll
Tracks:
- Actual bets
- Stakes
- Odds taken
- Profit/loss
- ROI
- Bankroll curve

## Model Lab
Features:
- Training controls
- Backtests
- Threshold optimization
- Model comparison
- Calibration metrics
- Model promotion / rollback

## Data Maintenance
Tracks:
- Latest completed event
- Missing results
- Dataset freshness
- Feature coverage
- Rolling feature rebuilds

## Current Production Architecture

EVENT SCRAPER (Colab)
↓
ufcstats_upcoming_events.parquet
ufcstats_upcoming_fights.parquet
data/predictions/ufc_live_card.parquet

LIVE PREDICTION ENGINE (GitHub Actions)
↓
data/predictions/ufc_live_action_board.parquet
data/predictions/ufc_live_watchlist.parquet
ufc_live_card_with_odds.parquet

CLV ENGINE (Scheduled GitHub Actions)
↓
ufc_market_snapshots.parquet
ufc_clv_results.parquet
ufc_line_movement.parquet

STREAMLIT DASHBOARD
↓
Visualization + workflow controls

## Major Future Feature
Dynamic Bet Requalification Engine:
- Continuously monitor odds movement
- Recalculate EV
- Promote watchlist fights into official bets when thresholds are crossed
- Alert user to newly qualified bets

## Long-Term Goal
A self-maintaining UFC betting intelligence platform
that continuously learns, adapts to the market,
and maximizes long-term ROI.
