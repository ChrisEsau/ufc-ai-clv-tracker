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
Decision Engine
        ↓
Betting Outcomes
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

### 5. Canonical Match Contract
Permanent join contract:

```text
fight_id
market_key
outcome_label
```

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

### 8. Locked Development Order
1. run_normalize_provider_markets.py
2. canonical_market_catalog.parquet
3. market_matcher.py
4. run_market_matching.py
5. market_outcomes.parquet
6. market_decision_engine.py
7. betting_outcomes.parquet
8. Action Board integration

## Existing DraftKings Components

- pipeline/market/run_draftkings_discovery.py
- pipeline/market/providers/draftkings_public.py
- pipeline/market/normalizers/draftkings.py
- pipeline/market/normalizers/canonical_market_schema.py
- configs/market/providers/draftkings_ufc_registry.yaml

## Next Phase

Create:

```text
pipeline/market/run_normalize_provider_markets.py
```

Outputs:

```text
data/market/canonical_market_catalog.parquet
data/audits/canonical_market_catalog_audit.parquet
```
