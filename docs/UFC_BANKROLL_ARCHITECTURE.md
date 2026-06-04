# UFC Bankroll Architecture

## Purpose

The Bankroll workspace is the financial control center of the UFC Betting Intelligence Platform.

It answers:

```text
How much money do we have?
How much money is at risk?
How are bets performing?
Are we following staking rules?
What is the current bankroll?
```

The Bankroll workspace is responsible for tracking all official wagers, bankroll performance, exposure, risk management, and staking discipline.

---

## Core Responsibilities

### Bankroll Tracking

Track:

* Starting bankroll
* Current bankroll
* Available bankroll
* Open exposure
* Total profit/loss
* ROI

---

### Official Bet Ledger

Maintain a permanent record of all official bets.

Each wager should contain:

* Event
* Fight
* Fighter
* Bet type
* Odds taken
* Stake amount
* Result
* Profit/Loss
* CLV metrics
* Model metrics

---

### Risk Management

Monitor:

* Maximum stake size
* Maximum event exposure
* Total open exposure
* Kelly sizing compliance
* Risk concentration

---

### Performance Analytics

Track:

* Profit by event
* Profit by month
* Profit by year
* Win rate
* ROI
* Closing Line Value performance

---

## Dashboard Layout

Preferred section order:

```text
Bankroll Summary
↓
Open Exposure
↓
Official Bet Ledger
↓
Performance Charts
↓
CLV & Bet Quality
↓
Risk Settings
```

---

## Section: Bankroll Summary

Display:

```text
Starting Bankroll
Current Bankroll
Available Bankroll
Open Risk
Total Profit
ROI
```

Example:

```text
Starting Bankroll: $10,000
Current Bankroll: $12,450
Profit: $2,450
ROI: 24.5%
Open Risk: $350
```

---

## Section: Open Exposure

Display currently unresolved wagers.

Metrics:

```text
Number of Open Bets
Total Stake Pending
Potential Profit
Exposure by Event
Exposure by Fighter
```

Purpose:

Prevent excessive risk concentration.

---

## Section: Official Bet Ledger

The ledger is the authoritative source of betting history.

Required fields:

| Field               | Description                     |
| ------------------- | ------------------------------- |
| bet_id              | Unique bet identifier           |
| event_name          | UFC event                       |
| fight_id            | Fight identifier                |
| fighter             | Bet selection                   |
| market_type         | Moneyline, KO, Submission, etc. |
| odds_taken          | Odds at placement               |
| stake               | Amount risked                   |
| result              | Win/Loss/Push/Open              |
| profit_loss         | Realized P/L                    |
| model_probability   | Model prediction                |
| implied_probability | Market probability              |
| EV                  | Expected value                  |
| CLV                 | Closing line value              |
| timestamp           | Bet timestamp                   |

---

## Section: Performance Charts

Recommended charts:

### Bankroll Curve

```text
Date
↓
Bankroll
```

Shows growth over time.

---

### Profit By Event

```text
Event
↓
Profit/Loss
```

Measures event-level performance.

---

### ROI By Odds Bucket

Buckets:

```text
Favorite
Small Underdog
Medium Underdog
Large Underdog
```

---

### Win Rate By Confidence Tier

Tiers:

```text
Strong Bet
Lean Bet
Watchlist
Pass
```

---

## Section: CLV & Bet Quality

Track:

### Beat Closing Line Rate

```text
Beats Closing Line
------------------
Total Bets
Percent
```

---

### Average CLV

```text
Average CLV
Median CLV
Positive CLV %
```

---

### CLV Profitability

Track:

```text
Profit when CLV > 0
Profit when CLV < 0
```

Purpose:

Validate market performance.

---

## Section: Risk Settings

Configurable settings:

```text
Starting Bankroll
Kelly Fraction
Max Stake %
Max Event Exposure %
EV Threshold
Confidence Threshold
Odds Range
```

Current production defaults:

```text
EV Threshold = $50
Confidence Threshold = 70%
Odds Range = -250 to +400
Kelly Fraction = 0.50
```

---

## Core Artifacts

### Bet Ledger

```text
data/bankroll/ufc_bet_ledger.parquet
```

Authoritative betting history.

---

### Open Bets

```text
data/bankroll/ufc_open_bets.parquet
```

Unresolved wagers.

---

### Bankroll Snapshots

```text
data/bankroll/ufc_bankroll_snapshots.parquet
```

Historical bankroll tracking.

---

### Settings

```text
data/bankroll/ufc_bankroll_settings.parquet
```

Risk management configuration.

---

## Data Flow

```text
Betting Board
      ↓
Official Bet
      ↓
Bet Ledger
      ↓
Open Bets
      ↓
Fight Results
      ↓
Profit/Loss
      ↓
Bankroll Update
      ↓
Performance Metrics
```

---

## Long-Term Vision

The Bankroll workspace should become the financial operating system of the UFC platform.

Responsibilities include:

* Bet tracking
* Performance monitoring
* Exposure management
* Risk management
* Staking discipline
* CLV validation
* ROI analysis

The Bankroll workspace should provide a complete historical record of all betting activity and serve as the primary source of truth for platform profitability.
