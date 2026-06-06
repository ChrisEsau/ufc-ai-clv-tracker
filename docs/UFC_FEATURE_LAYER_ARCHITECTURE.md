# UFC Feature Layer Architecture

## Purpose

This document defines how UFC model features should be organized as the project moves from notebook-based modeling into modular Python pipelines.

The goal is to allow moneyline models and future prop models to share a stable point-in-time feature foundation without losing any currently used V5 moneyline features.

---

## Core Principle

```text
Base features = reusable point-in-time fighter and fight state.
Moneyline features = winner-prediction model inputs.
Prop features = market-specific outcome labels and inputs.
```

No existing V5 moneyline training feature may be removed during the refactor unless it is explicitly deprecated and preserved in a compatibility layer.

---

## Feature Layer Flow

```text
Raw UFC Master Data
    ↓
Base Rolling Feature Store
    ↓
Market-Specific Feature Views
    ↓
Model-Specific Feature Contracts
    ↓
Training / Backtesting / Live Prediction
```

---

## Base Feature Store

The base feature store should preserve reusable, point-in-time fighter state.

Expected future artifact:

```text
data/features/ufc_base_rolling_features.parquet
```

The base store should include:

- Fight identifiers
- Event identifiers
- Fighter identifiers
- Fight metadata
- Target/result columns where appropriate for historical training
- Pre-fight career state
- Elo state
- Striking state
- Grappling/wrestling state
- Control-time state
- Finish-rate state
- Durability/history state
- Recent 3-fight form
- EWM recent-form features
- Red/blue differential features

The base feature store should remain wide enough to preserve historical compatibility.

---

## Moneyline Feature Layer

The moneyline feature layer is a model-ready view for fight-winner prediction.

Expected future artifact:

```text
data/features/ufc_moneyline_features.parquet
```

The current V5 moneyline model uses symmetry-safe features only:

```text
all *_diff columns
+
registered engineered matchup features
```

The current V5 feature contract is documented in:

```text
configs/features/current_moneyline_v5_features.yaml
```

This contract must be preserved during notebook-to-Python conversion.

---

## Prop Feature Layer

The prop feature layer should be built from the base rolling feature store plus prop-specific labels and market-specific feature transformations.

Expected future artifact:

```text
data/features/ufc_prop_features.parquet
```

Expected future modules:

```text
pipeline/features/props/build_prop_labels.py
pipeline/features/props/build_ko_tko_features.py
pipeline/features/props/build_submission_features.py
pipeline/features/props/build_decision_features.py
pipeline/features/props/build_goes_distance_features.py
pipeline/features/props/build_round_features.py
```

Prop feature views should not destabilize the current moneyline feature contract.

---

## Compatibility Rule

Splitting the current rolling feature file is allowed only if every current feature is accounted for.

Each current rolling feature must be mapped to one of:

```text
base
moneyline
props
deprecated_but_preserved
```

No feature should silently disappear.

---

## Current V5 Moneyline Input Rule

The current training notebook builds `safe_cols` using this rule:

```python
safe_cols = []

for col in df.columns:
    if col.endswith("_diff"):
        safe_cols.append(col)
    elif col in registered_features:
        safe_cols.append(col)
```

Then it blocks raw red/blue fighter-specific columns and trains using:

```python
X_train = train_df[safe_cols]
X_test = test_df[safe_cols]
```

The notebook verifies:

```text
Final feature count: 124
X_train: 13626 rows x 124 columns
X_test: 2754 rows x 124 columns
```

---

## Recommended Refactor Order

1. Create a base rolling feature builder that reproduces the current rolling notebook output.
2. Create a moneyline feature view that reproduces the current 124 V5 model inputs exactly.
3. Add validation that compares generated moneyline columns to `configs/features/current_moneyline_v5_features.yaml`.
4. Only after parity is proven, split or reorganize feature artifacts.
5. Add prop labels and prop feature views later.

---

## Testing Requirements

Before replacing notebook outputs with Python module outputs, verify:

```text
row count matches expected historical rows
current V5 feature count equals 124
all current V5 features are present
no required feature is renamed
no required feature is dropped
training/backtest metrics are reproducible within acceptable tolerance
```

---

## Related Files

```text
configs/features/feature_registry.yaml
configs/features/current_moneyline_v5_features.yaml
docs/MODEL_ADAPTER_ARCHITECTURE.md
docs/UFC_PROP_MODEL_ARCHITECTURE.md
docs/UFC_REPOSITORY_STRUCTURE.md
```