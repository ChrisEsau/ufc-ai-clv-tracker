# UFC Prop Market Schema

## Purpose

This document defines canonical market keys and outcome labels for Market Pipeline V2, Prediction V2, Betting Outcomes V2, CLV, bankroll tracking, and future dashboard views.

The goal is to prevent provider-specific odds labels from leaking into persisted artifacts.

All provider labels must be normalized into the canonical labels defined here before joining predictions, odds, betting decisions, CLV records, or dashboard views.

---

## Core Rule

The canonical join grain is:

```text
fight_id
market_key
outcome_label
```

Optional downstream dimensions include:

```text
model_id
bookmaker
snapshot_run_id
```

Provider-specific labels should be retained only as audit/display fields, for example:

```text
provider_market_key
provider_outcome_label
matched_market_name
matched_outcome_name
```

---

## Market Families

Market Pipeline V2 should support market families incrementally:

1. Moneyline
2. Goes distance / inside distance
3. Method props
4. Totals
5. Round props
6. Future special props

Moneyline is the first operational implementation, but the schema must be prop-ready from the start.

---

## Canonical Markets

### Moneyline

```text
market_key: moneyline
```

Outcome labels:

```text
<red_fighter display name>
<blue_fighter display name>
```

Rules:

- Outcome labels must match Prediction V2 `outcome_label` exactly.
- For moneyline, Prediction V2 emits fighter display names as outcome labels.
- The market normalizer must map sportsbook fighter names back to the UFCStats fighter display names used in prediction output.

Example:

```text
fight_id: abc123
market_key: moneyline
outcome_label: Sean O'Malley
```

---

### Goes Distance

```text
market_key: goes_distance
```

Outcome labels:

```text
goes_distance
inside_distance
```

Provider labels should normalize as follows:

| Provider Label Example | Canonical outcome_label |
|---|---|
| Yes | goes_distance |
| Fight Goes Distance | goes_distance |
| Goes Distance | goes_distance |
| No | inside_distance |
| Fight Does Not Go Distance | inside_distance |
| Inside Distance | inside_distance |

Rules:

- Use `goes_distance` for the positive distance outcome.
- Use `inside_distance` for the negative distance outcome.
- Do not use `yes` / `no` as persisted canonical outcome labels.

---

### Inside Distance

```text
market_key: inside_distance
```

Outcome labels:

```text
inside_distance
goes_distance
```

Rules:

- This is the inverse presentation of `goes_distance`.
- Prefer `market_key: goes_distance` for binary distance models unless a provider exposes a distinct inside-distance market that must be preserved.
- The canonical labels remain `inside_distance` and `goes_distance`.

---

### Method Group

```text
market_key: method
```

Outcome labels:

```text
ko_tko
submission
decision
```

Provider labels should normalize as follows:

| Provider Label Example | Canonical outcome_label |
|---|---|
| KO/TKO | ko_tko |
| KO, TKO or DQ | ko_tko |
| Knockout | ko_tko |
| Submission | submission |
| SUB | submission |
| Decision | decision |
| Points | decision |

Rules:

- Use one row per method outcome.
- If provider separates DQ from KO/TKO, either map DQ into `ko_tko` for UFCStats compatibility or flag as unsupported until settlement logic supports DQ separately.
- Prediction models should emit the same canonical labels.

---

### Fighter-Specific Method

For fighter-specific method markets, use side-specific market keys only when needed.

Preferred future market keys:

```text
fighter_method
```

Outcome labels:

```text
<fighter_name>_ko_tko
<fighter_name>_submission
<fighter_name>_decision
```

Current status:

```text
not implemented
```

Rules:

- Do not implement fighter-specific method odds until the provider mapping and settlement keys are explicitly designed.
- Generic fight method and fighter-specific method should not share the same `market_key`.

---

### Totals

Canonical total markets should include the round line in the market key.

Examples:

```text
market_key: over_under_1_5
market_key: over_under_2_5
market_key: over_under_3_5
market_key: over_under_4_5
```

Outcome labels:

```text
over_1_5
under_1_5
over_2_5
under_2_5
over_3_5
under_3_5
over_4_5
under_4_5
```

Rules:

- Preserve the total line in both `market_key` and `outcome_label`.
- Do not store generic labels like `over` and `under` without the line.
- The provider line must be parsed into normalized underscore format.

---

### Round Finish Props

Preferred market keys:

```text
round_1_finish
round_2_finish
round_3_finish
round_4_finish
round_5_finish
```

Outcome labels:

```text
round_1_finish
not_round_1_finish
round_2_finish
not_round_2_finish
round_3_finish
not_round_3_finish
round_4_finish
not_round_4_finish
round_5_finish
not_round_5_finish
```

Rules:

- Binary round finish models should use positive/negative canonical labels.
- Do not use provider labels like `Yes` / `No` as persisted canonical labels.

---

### Exact Round / Round Group Markets

Current status:

```text
not implemented
```

Potential future market keys:

```text
finish_round
fight_ends_in_round
```

Potential outcome labels:

```text
round_1
round_2
round_3
round_4
round_5
decision
```

Rules:

- Do not implement until prediction targets and settlement logic are aligned.
- Keep distinct from binary round-finish props.

---

## Provider Normalization Rules

Provider adapters must preserve raw fields, but normalizers must emit canonical labels.

Required normalized market columns:

```text
snapshot_run_id
snapshot_timestamp
source
bookmaker
event_id
event_name
fight_id
market_key
outcome_label
american_odds
decimal_odds
implied_probability
odds_match_type
```

Recommended provider/audit columns:

```text
provider_event_id
provider_market_key
provider_outcome_label
matched_market_name
matched_outcome_name
odds_match_score
odds_min_single_score
```

---

## Supported Initial Build Order

### Phase 1

```text
moneyline
```

Reason:

- Existing odds utility already supports H2H.
- Prediction V2 moneyline outcomes already exist.
- This validates the outcome-row join pattern.

### Phase 2

```text
goes_distance
method
```

Reason:

- These align directly with likely prop model targets.
- They have simple canonical label sets.

### Phase 3

```text
over_under_1_5
over_under_2_5
round_1_finish
round_2_finish
round_3_finish
```

Reason:

- These require line parsing and more provider-specific normalization.

---

## Validation Rules

Market outcomes must validate:

- Required columns exist.
- `market_key` is non-empty.
- `outcome_label` is non-empty.
- `american_odds` is numeric when present.
- `decimal_odds` is numeric when present.
- `implied_probability` is between 0 and 1.
- Matched rows have a valid `fight_id`.
- There are no duplicate rows at:

```text
snapshot_run_id
bookmaker
fight_id
market_key
outcome_label
```

---

## Do Not Do Yet

Do not implement:

- fighter-specific method markets
- exact-round markets
- special props
- dashboard prop filters
- CLV prop settlement

until the base Market Outcomes V2 artifact is validated for moneyline and at least one binary prop market.
