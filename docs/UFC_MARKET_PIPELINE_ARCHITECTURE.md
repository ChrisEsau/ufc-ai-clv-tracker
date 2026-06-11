# UFC Market Pipeline Architecture

## Purpose

The UFC Market Pipeline is designed to support sportsbook market ingestion, normalization, matching, EV calculation, CLV tracking, and betting decision generation in a sportsbook-agnostic manner.

## Core Architecture

```text
Raw Sportsbook Data
        ↓
Provider Discovery
        ↓
Provider Normalization
        ↓
Canonical Market Catalog
        ↓
Market Matching
        ↓
Market Outcomes
        ↓
Betting Outcomes V2
        ↓
Action Board / Dashboard
```

## Locked Architecture Decisions

### 1. Centralized Market Paths
All market artifact paths must be defined in `pipeline/common/paths.py`.

### 2. Historical Snapshot Storage
Create `data/market/snapshots/` and preserve timestamped market snapshots for CLV, line movement, and market reconstruction.

### 3. Preserve Provider Identifiers
Retain:
- provider_event_id
- provider_market_id
- provider_selection_id

through market normalization and matching.

### 4. Mandatory Provider Registry
All providers must register through a central registry. No sportsbook-specific branching outside provider adapters.

### 5. Canonical Outcome Join Contract
Production joins must use a universal outcome join key.

Permanent join contract:

```text
fight_id
market_key
outcome_join_key
```

Examples:

```text
moneyline:      fighter:<fighter_id>
goes_distance:  fight:goes_distance
inside_distance: fight:inside_distance
over_1_5:       fight:over_1_5
under_1_5:      fight:under_1_5
```

The legacy fields remain useful and should be preserved:

```text
outcome_label
outcome_fighter_id
outcome_key
side
line
```

But they should not be the only join contract for props.

### 6. Discovery vs Production Separation
Discovery artifacts never overwrite production artifacts.

Discovery:
- draftkings_market_diagnostic.parquet
- draftkings_raw_index.parquet

Production:
- canonical_market_catalog.parquet
- market_outcomes.parquet
- betting_outcomes.parquet

### 7. Future Sportsbook Expansion
New sportsbooks should require only:
- provider scraper
- provider registry
- provider normalizer

### 8. Betting Outcomes V2 Is the Decision Engine
The existing implementation is the authoritative EV / Kelly / betting decision layer:

```text
pipeline/betting/run_betting_outcomes_v2.py
```

It consumes:

```text
data/predictions/model_outcomes.parquet
data/market/market_outcomes.parquet
```

and writes:

```text
data/predictions/betting_outcomes.parquet
data/audits/ufc_betting_outcomes_audit.parquet
```

Do not create a competing `market_decision_engine.py` unless Betting Outcomes V2 is intentionally deprecated.

### 9. Betting Dashboard Prop Path
The dashboard entrypoint remains:

```text
tabs/betting_board_v2.py
```

Current dashboard behavior is moneyline-focused. Prop support should be added downstream by extending the dashboard renderer, not by creating a separate prop dashboard.

Required dashboard sections:

```text
Moneyline section
Props section
```

Recommended prop table columns:

```text
Fight
Market
Outcome
Model Probability
Odds
Implied Probability
Edge
EV
Confidence
Recommended Stake
Status
```

### 10. Registry-Driven Matching Strategy
Market matching behavior must be provider-registry driven.

No sportsbook-specific matching logic may be hardcoded inside:

```text
pipeline/market/market_matcher.py
```

The matcher consumes a strategy resolved from the provider registry. Supported strategy families include:

```text
fighter_name
matchup_name
event_name
provider_crosswalk
```

Default intent:

```text
fighter-specific markets -> fighter_name
fight-level markets      -> matchup_name
card/event markets       -> event_name
```

DraftKings examples:

```text
moneyline            -> fighter_name
win_by_ko_tko_dq     -> fighter_name
win_by_submission    -> fighter_name
win_by_decision      -> fighter_name
goes_distance        -> matchup_name
total_rounds         -> matchup_name
exact_method         -> matchup_name
round_method         -> matchup_name
```

If downstream matcher logic requires sportsbook-specific branching, the abstraction boundary has been violated.

### 11. Locked Development Order
1. run_normalize_provider_markets.py
2. canonical_market_catalog.parquet
3. market_matcher.py
4. outcome_join_key support across model_outcomes and market_outcomes
5. registry-driven matching strategy support
6. run_market_matching.py
7. market_outcomes.parquet
8. run_betting_outcomes_v2.py
9. betting_outcomes.parquet
10. betting_board_v2.py prop section
11. Action Board integration

## Existing DraftKings Components

- pipeline/market/run_draftkings_discovery.py
- pipeline/market/providers/draftkings_public.py
- pipeline/market/normalizers/draftkings.py
- pipeline/market/normalizers/canonical_market_schema.py
- configs/market/providers/draftkings_ufc_registry.yaml

## Existing Betting Components

- pipeline/betting/run_betting_outcomes_v2.py
- tabs/betting_board_v2.py
- utils/betting_outcomes_adapter.py

## Current Build Status

Created:

```text
pipeline/market/run_normalize_provider_markets.py
pipeline/market/market_matcher.py
pipeline/market/run_market_matching.py
pipeline/common/outcome_join.py
```

Next implementation step:

```text
Add registry-driven matching strategy support so fight-level props such as goes_distance match by matchup text instead of fighter outcome text.
```
