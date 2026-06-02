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
