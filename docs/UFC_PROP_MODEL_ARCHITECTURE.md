# UFC Prop Model Architecture

## Purpose

This document defines the long-term architecture for UFC prop models.

Prop models should be treated as a separate market family from moneyline models because they require different labels, features, calibration, backtesting, confidence buckets, and betting rules.

Core decision:

```text
Use one model per prop market.
```

Do not start with one large multi-output prop model.

---

## Prop Market Families

Initial prop model candidates:

```text
KO/TKO
Submission
Decision
Goes Distance
Does Not Go Distance
Round Totals
Round-Specific Finish Props
```

Recommended model bundles:

```text
models/props/UFC_PROP_KO_TKO_V1/
models/props/UFC_PROP_SUB_V1/
models/props/UFC_PROP_DEC_V1/
models/props/UFC_PROP_GOES_DISTANCE_V1/
models/props/UFC_PROP_ROUNDS_V1/
```

---

## Why One Model Per Prop Market

Each prop market has a different target label.

Examples:

```text
KO/TKO model target: fight ended by KO/TKO for selected fighter
Submission model target: fight ended by submission for selected fighter
Decision model target: selected fighter won by decision
Goes Distance target: fight reached final bell
Round model target: fight ended in a specific round or over/under a round total
```

Each prop market also needs different feature emphasis.

A submission model should care heavily about grappling and submission mismatch.

A KO/TKO model should care heavily about striking power, pace, knockdowns, and chin risk.

A goes-distance model should care heavily about durability, finish rates, pace, and scheduled rounds.

Because labels, features, base rates, calibration, and backtests differ, separate models are easier to validate and safer to use in production.

---

## Repository Structure

Expected future structure:

```text
pipeline/
├── features/
│   └── props/
│       ├── build_prop_labels.py
│       ├── build_ko_tko_features.py
│       ├── build_submission_features.py
│       ├── build_decision_features.py
│       ├── build_goes_distance_features.py
│       └── build_round_features.py
│
├── training/
│   └── props/
│       ├── train_ko_tko.py
│       ├── train_submission.py
│       ├── train_decision.py
│       ├── train_goes_distance.py
│       └── train_rounds.py
│
├── backtesting/
│   └── props/
│       ├── backtest_ko_tko.py
│       ├── backtest_submission.py
│       ├── backtest_decision.py
│       ├── backtest_goes_distance.py
│       └── backtest_rounds.py
│
└── modeling/
    └── adapters/
        └── prop_adapter.py
```

---

## Model Bundle Contract

Each prop model should produce a bundle like:

```text
models/props/<model_name>/
├── model.pkl
├── feature_columns.pkl
├── production_config.json
├── confidence_buckets.parquet
├── training_metrics.json
└── model_card.md
```

Optional artifacts:

```text
calibration_model.pkl
backtest_predictions.parquet
feature_importance.csv
prop_market_metrics.json
```

---

## Prop Feature Groups

### KO/TKO Features

Candidate features:

```text
ko_rate_diff
kd_avg_diff
kd_absorbed_avg_diff
splm_diff
sapm_diff
striking_edge
chin_risk_diff
finish_loss_rate_diff
pressure_striking_adv_diff
recent_form_splm_diff
recent_form_sapm_diff
```

### Submission Features

Candidate features:

```text
sub_win_rate_diff
sub_avg_diff
td_avg_diff
td_acc_diff
td_def_diff
ctrl_per_min_diff
ctrl_against_per_min_diff
submission_mismatch_diff
wrestling_mismatch_diff
recent_form_td_avg_diff
```

### Decision Features

Candidate features:

```text
decision_win_rate_diff
decision_loss_rate_diff
avg_fight_time_diff
finish_rate_diff
finish_loss_rate_diff
pace metrics
experience_ratio_diff
recent_form_avg_fight_time_diff
```

### Goes Distance Features

Candidate features:

```text
avg_fight_time
finish_rate
finish_loss_rate
decision_win_rate
decision_loss_rate
scheduled_rounds
title_fight
pace
cardio / fight-time history
durability indicators
```

### Round Features

Candidate features:

```text
early_finish_rate
late_finish_rate
round_1_finish_rate
round_2_finish_rate
round_3_finish_rate
average_finish_round
pace_decay
scheduled_rounds
title_fight
```

---

## Prop Label Builder

Prop labels should be generated from historical fight results.

Examples:

```text
fighter_ko_tko_win
fighter_submission_win
fighter_decision_win
fight_goes_distance
fight_does_not_go_distance
fight_ends_round_1
fight_ends_round_2
fight_ends_round_3
fight_ends_under_1_5
fight_ends_over_2_5
```

Label generation must be point-in-time safe and should not leak future data into feature rows.

---

## Standard Prop Prediction Output

Every prop model should output the same schema:

```text
fight_id
event_name
market_family
market_type
selection
fighter_name
opponent_name
model_prob
market_implied_prob
edge
ev
model_name
model_version
model_type
feature_set
confidence_bucket
bucket_reliability_score
prediction_timestamp
```

Examples of `market_type`:

```text
KO_TKO
SUBMISSION
DECISION
GOES_DISTANCE
DOES_NOT_GO_DISTANCE
ROUND_TOTAL_OVER
ROUND_TOTAL_UNDER
ROUND_1_FINISH
```

---

## Confidence Buckets

Each prop model should have its own confidence buckets.

Suggested location:

```text
models/props/<model_name>/confidence_buckets.parquet
```

Suggested columns:

```text
market_family
market_type
bucket_min
bucket_max
sample_size
avg_model_prob
actual_hit_rate
calibration_error
bucket_reliability_score
```

Do not reuse moneyline confidence buckets for prop models.

Prop outcomes have different base rates and calibration behavior.

---

## Backtesting Requirements

Prop backtests should evaluate:

```text
hit rate
log loss
Brier score
calibration by bucket
market edge
EV
ROI
CLV
sample size by market
sample size by odds range
```

Each prop market should be backtested separately.

Combining all props into one aggregate ROI can hide weak markets.

---

## Betting Board Integration

The Betting Board should consume prop predictions as standardized artifacts.

Potential artifact:

```text
data/predictions/ufc_prop_predictions.parquet
```

The prop Betting Board view should show:

```text
event
fight
market_type
selection
model_prob
market_implied_prob
edge
ev
confidence_score
bucket_reliability_score
odds
sportsbook
recommended_action
```

The dashboard should be market-aware but model-agnostic.

---

## Migration Plan

Phase 1

- Complete moneyline feature/training/backtest modularization.
- Save moneyline confidence buckets as model artifacts.

Phase 2

- Build prop label generator.
- Start with one prop market only.

Recommended first prop market:

```text
Goes Distance / Does Not Go Distance
```

Reason: it is fight-level, does not require picking a fighter side, and is usually simpler than exact method-of-victory models.

Phase 3

- Add KO/TKO model.
- Add Submission model.
- Add Decision model.

Phase 4

- Add round totals and round-specific finish models.

Phase 5

- Add prop Betting Board and prop CLV tracking.

---

## Locked Decision

```text
Props should be modeled as one model per prop market.
```

This should remain the default unless explicitly revisited and approved.