# UFC V2 Production Architecture V4

_Last updated: 2026-06-12_

## Executive Summary

The UFC platform architecture is operational and validated end-to-end.

Current production flow:

ufc_master.parquet
→ fighter_state_history.parquet
→ latest_fighter_state.parquet
→ feature views
→ model training
→ model_outcomes.parquet
→ DraftKings market pipeline
→ market_outcomes.parquet
→ betting_outcomes.parquet
→ dashboard

## Locked Architectural Contracts

### Canonical Outcome Grain

One row per:

fight_id + market_key + outcome

### Universal Outcome Join Contract

All downstream joins use:

fight_id + market_key + outcome_join_key

Examples:

fighter:<fighter_id>
fight:goes_distance
fight:inside_distance
fighter:<fighter_id>:submission
fighter:<fighter_id>:ko_tko_dq
fighter:<fighter_id>:decision

No downstream fuzzy matching is permitted after market matching.

## DraftKings Market Pipeline

DraftKings Event Index
→ Event Card Matching
→ Matched Discovery
→ Market Diagnostic
→ Canonical Market Catalog
→ Market Matching
→ Market Outcomes

Artifacts:

- data/market/draftkings_event_index.parquet
- data/market/draftkings_event_card_matches.parquet
- data/market/draftkings_market_diagnostic.parquet
- data/market/canonical_market_catalog.parquet
- data/market/market_outcomes.parquet

### Historical Odds Policy

Historical odds accumulation is not used.

Market artifacts represent current sportsbook state and are overwritten by refresh runs.

## Market Pipeline Validation (2026-06-12)

Validated results:

- Canonical catalog rows: 516
- Market outcome rows: 304
- Matched events: 14

Supported markets:

- moneyline
- goes_distance
- point_spread
- total_rounds
- exact_method
- round_method
- win_by_ko_tko_dq
- win_by_submission
- win_by_decision

## Betting Outcomes V2 Validation

Validated results:

- Model rows: 288
- Market rows: 304
- Joined rows: 40
- Bet candidates: 20

Status: OPERATIONAL

## Registry Driven Matching

Location:

configs/market/providers/

Current implementation:

draftkings_ufc_registry.yaml

All sportsbook normalization flows through registries.

## Feature Architecture

Authoritative artifacts:

- fighter_state_history.parquet
- latest_fighter_state.parquet
- moneyline_feature_view.parquet

Training consumes feature views.
Prediction consumes fighter-state artifacts.

## Prop Model Architecture

Independent model families:

- moneyline
- goes_distance
- exact_method
- ko_tko_dq
- submission
- decision
- round_method

## Live Feature Requirements

Missing required features → FAIL

Automatic zero filling → PROHIBITED

Required fight context:

- title_fight
- total_rounds

## Production Run Order

1. Refresh events
2. Build live card
3. Build features
4. Run prediction
5. Run DraftKings market pipeline
6. Run market matching
7. Run betting outcomes V2
8. Open dashboard

## Current Project Status

V2 Architecture: OPERATIONAL

Validated:

- Feature architecture
- Training architecture
- Prediction architecture
- DraftKings market pipeline
- Market matching
- Betting outcomes

Current focus:

- Documentation cleanup
- Legacy cleanup
- Dashboard modernization
- Prop model expansion
