# UFC Market Pipeline V2 Architecture

## Purpose

The UFC Market Pipeline V2 is responsible for ingesting sportsbook odds, normalizing them into a canonical outcome schema, and producing market artifacts that can be consumed by:

- Prediction V2
- Betting Decision V2
- CLV Tracking
- Bankroll Management
- Dashboard Components
- Future Prop Betting Models

The architecture is intentionally modular and mirrors the design philosophy used by:

- Feature Engineering V2
- Training V2
- Prediction V2

The primary objective is to support both moneyline and prop markets through a common interface.

---

# Design Principles

## 1. Provider Agnostic

Market consumers should not depend on provider-specific implementations.

Consumers should only read canonical market artifacts.

Provider-specific logic belongs inside provider adapters.

## 2. Outcome-Based Architecture

All market data is normalized into outcome rows.

Examples:

Moneyline:
- Red Fighter
- Blue Fighter

Method Market:
- KO/TKO
- Submission
- Decision

Totals Market:
- Over 2.5
- Under 2.5

This mirrors the Prediction V2 outcome schema.

## 3. Prop Ready

Moneyline is only the first implementation.

The architecture must support:

- Moneyline
- Goes Distance
- Inside Distance
- KO/TKO
- Submission
- Decision
- Round Props
- Totals
- Future Prop Models

without redesign.

---

# Pipeline Flow

Provider API
→ Provider Adapter
→ Market Normalizer
→ Outcome Matcher
→ Market Validator
→ market_outcomes.parquet

---

# Repository Structure

pipeline/market/

- run_market_update_v2.py
- market_config.py
- provider_registry.py
- providers/
  - the_odds_api.py
- normalizers/
  - moneyline.py
  - goes_distance.py
  - method.py
  - rounds.py
- outcome_matcher.py
- market_validator.py

---

# Configuration Layer

configs/market/market_registry.yaml

Example:

provider: the_odds_api

bookmakers:
  - DraftKings

markets:
  - moneyline
  - goes_distance
  - method
  - totals

---

# Provider Layer

Responsibilities:

- API authentication
- Event retrieval
- Market retrieval
- Raw response preservation

No matching logic.
No normalization logic.

---

# Market Normalizers

Responsibilities:

Convert provider-specific structures into canonical market rows.

Examples:

## Moneyline Normalizer

Produces:
- moneyline / red_fighter
- moneyline / blue_fighter

## Method Normalizer

Produces:
- method / ko_tko
- method / submission
- method / decision

---

# Outcome Matcher

Responsibilities:

Map sportsbook market rows to UFC fight_id values using existing fight matching logic.

Outputs:

- fight_id
- event_name
- market_key
- outcome_label

---

# Market Validator

Responsibilities:

Verify:

- valid fight IDs
- duplicate rows
- odds integrity
- outcome completeness
- market completeness

Produces:

market_audit.parquet

---

# Canonical Market Artifact

Output:

data/market/market_outcomes.parquet

Grain:

One row per:

- snapshot
- bookmaker
- fight
- market
- outcome

Columns:

- snapshot_run_id
- snapshot_timestamp
- source
- bookmaker
- event_id
- event_name
- fight_id
- market_key
- outcome_label
- american_odds
- decimal_odds
- implied_probability
- odds_match_type

---

# Supported Market Keys

## Phase 1

- moneyline

## Phase 2

- goes_distance
- inside_distance
- ko_tko
- submission
- decision

## Phase 3

- over_under_1_5
- over_under_2_5
- round_finish_props
- special props

---

# Betting Decision V2 Integration

Market Outcomes + Prediction Outcomes

Join Keys:

- fight_id
- market_key
- outcome_label

Produces:

- edge
- expected_value
- stake
- status

---

# Dashboard Integration

The dashboard should never consume provider-specific data.

All dashboard views should read:

- market_outcomes.parquet
- betting_outcomes.parquet

---

# Future Expansion

The architecture is intentionally designed to support:

- Multiple sportsbooks
- Multiple providers
- Additional prop markets
- Ensemble betting models
- CLV tracking
- Market movement analytics

without requiring schema redesign.

---

# Current Implementation Plan

## Phase 10A

Market Outcomes V2 Foundation

- provider registry
- provider adapters
- moneyline normalizer
- outcome matcher
- market validator
- market_outcomes.parquet

## Phase 10B

Prop Market Expansion

- goes distance
- method props
- totals props
- round props

## Phase 10C

Betting Outcomes V2

- edge calculations
- EV calculations
- stake sizing
- betting board integration

## Phase 10D

Dashboard Migration

- Betting Board V2
- Market Analytics
- CLV Tracking
- Bankroll Management
