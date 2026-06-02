# UFC Line Movement / CLV Architecture

## Purpose

The Line Movement / CLV workspace tracks how UFC betting markets move over time and whether the model is beating the closing line.

It answers:

```text
Did we bet at a good number?
Did the market move in our favor?
Are our model edges real?
Which books move first?
```

---

## Core Responsibilities

* Track opening odds
* Track current odds
* Track closing odds
* Store line snapshots
* Calculate Closing Line Value
* Measure beat-closing-line rate
* Compare sportsbook movement
* Identify steam moves
* Support market-aware model features

---

## Primary Inputs

* Opening odds artifacts
* Current market odds artifacts
* Closing odds artifacts
* Bet recommendations
* Bet ledger
* Sportsbook metadata

---

## Primary Outputs

* CLV results
* Line movement charts
* Book-by-book movement
* Beat closing line metrics
* Market efficiency indicators

---

## Key Concepts

### Opening Line

First observed market price for a fight or bet.

### Current Line

Most recent available price.

### Closing Line

Final market price before fight start.

### Closing Line Value

The difference between the bettor's price and the closing market price.

### Beat Closing Line

A bet is considered to beat the closing line when the taken price is better than the final market price.

---

## Core Artifacts

Expected artifacts include:

```text
ufc_market_odds.parquet
ufc_market_snapshots.parquet
ufc_line_movement.parquet
ufc_closing_lines.parquet
ufc_clv_results.parquet
```

---

## Future Enhancements

* Multi-book CLV comparison
* Automatic closing line capture
* Market steam alerts
* Fighter-level market movement profiles
* CLV by model confidence bucket
* CLV by odds range
* CLV by sportsbook
