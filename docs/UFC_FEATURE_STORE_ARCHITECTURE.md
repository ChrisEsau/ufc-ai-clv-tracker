# UFC Feature Store Architecture

## Historical Store

Artifact:

```text
data/features/UFC_enhanced_rolling_features_EWM.parquet
```

Purpose:

Point-in-time training features.

Used for:

* Model training
* Backtesting
* Historical validation

---

## Current Store

Artifact:

```text
data/features/ufc_current_fighter_features.parquet
```

Purpose:

Current fighter state.

Used for:

* Live predictions
* Upcoming fight analysis

---

## Rule

Historical store and current store must remain separate.

Training code should use historical features.

Live prediction code should use current fighter features.
