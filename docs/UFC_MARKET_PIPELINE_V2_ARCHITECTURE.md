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

## 4. Raw Discovery Before Normalization

New sportsbooks or new prop market families should first be captured through a raw discovery layer.

The raw discovery layer should:

- preserve complete provider payloads,
- flatten all visible markets/selections into diagnostics,
- classify supported and unsupported markets,
- avoid writing production `market_outcomes.parquet`,
- avoid EV, CLV, staking, or betting decisions.

This keeps production Market V2 stable while provider schemas and prop semantics are inspected.

---

# Pipeline Flow

Provider API
→ Provider Adapter
→ Market Normalizer
→ Outcome Matcher
→ Market Validator
→ market_outcomes.parquet

For raw sportsbook discovery:

Public sportsbook JSON
→ Provider Discovery Adapter
→ Raw JSON Snapshot
→ Diagnostic Market Table
→ Manual Review / Future Normalizer Mapping

---

# Repository Structure

pipeline/market/

- run_market_update_v2.py
- run_draftkings_discovery.py
- market_config.py
- provider_registry.py
- providers/
  - the_odds_api.py
  - draftkings_public.py
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

# DraftKings Market Discovery Layer

## Purpose

The DraftKings discovery layer is an isolated research/diagnostic intake path for capturing all visible DraftKings UFC market payloads before promoting any market family into canonical Market V2 outputs.

It is intentionally separate from:

- `pipeline.market.run_market_update_v2`
- `data/market/market_outcomes.parquet`
- `data/market/market_outcome_snapshots.parquet`
- `pipeline.betting.run_betting_outcomes_v2`
- CLV, ledger, and dashboard outputs

## Runner

```bash
python -m pipeline.market.run_draftkings_discovery --url "<public-json-url>"
```

or:

```bash
DRAFTKINGS_DISCOVERY_URL="<public-json-url>" \
python -m pipeline.market.run_draftkings_discovery
```

## Provider Module

```text
pipeline/market/providers/draftkings_public.py
```

Approved scope:

- one read-only public JSON request per run,
- save raw provider response,
- flatten discovered markets/selections,
- flag parlays, boosts, promos, and recognized market families,
- fail on HTTP errors instead of retry-spamming.

Out of scope:

- login automation,
- account access,
- CAPTCHA handling,
- proxy rotation,
- IP spoofing,
- ban evasion,
- normalization into production Market V2,
- EV or betting decisions.

## Raw Outputs

```text
data/market/raw/draftkings/<yyyy-mm-dd>/snapshot_<snapshot_run_id>.json
data/market/draftkings_market_diagnostic.parquet
data/market/draftkings_raw_index.parquet
```

## Diagnostic Grain

One row per discovered provider market selection:

```text
snapshot_run_id
snapshot_timestamp
source
bookmaker
provider_event_id
event_name
provider_market_id
raw_market_name
provider_selection_id
raw_selection_name
price_american
price_decimal
implied_probability
line
is_parlay
is_boost
is_promo
is_supported_market
supported_market_family
raw_payload_path
```

## Discovery Market Families

The discovery layer may classify rows into these families when text/structure allows:

- moneyline
- goes_distance
- over_under_rounds
- ko_tko
- submission
- decision
- exact_round
- fighter_round_win

Rows that cannot be confidently classified are still preserved for review.

## Promotion Rule

No DraftKings market family should be promoted into `market_outcomes.parquet` until:

1. real payload examples are inspected,
2. provider market IDs and selection IDs are understood,
3. a canonical `market_key` and `outcome_key` are defined,
4. matching behavior is validated against `ufc_live_card.parquet`,
5. unsupported/parlay/boost/promo rows are explicitly excluded or separately modeled.

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
- outcome_fighter_id

Produces:

- edge
- expected_value
- stake
- status

DraftKings discovery diagnostics do not participate in Betting Decision V2 until a market family is promoted into canonical `market_outcomes.parquet` and a matching model outcome exists.

---

# Dashboard Integration

The dashboard should never consume provider-specific data.

All production dashboard views should read:

- market_outcomes.parquet
- betting_outcomes.parquet

Raw discovery diagnostics may be reviewed manually or in a future diagnostic-only admin view, but should not power Betting Board decisions.

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

## Phase 0

DraftKings Market Discovery

- DraftKings public provider scaffold
- raw JSON snapshot persistence
- flattened diagnostic market table
- raw snapshot index
- classification flags for parlays, boosts, promos, and supported market families
- no production Market V2 integration
- no betting integration

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
