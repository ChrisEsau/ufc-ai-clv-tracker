# UFC Model Adapter Architecture

## Status

This document has been updated to align with the approved V2 prediction architecture.

Authoritative target architecture:

```text
docs/UFC_PREDICTION_PIPELINE_V2.md
```

If this document and `UFC_PREDICTION_PIPELINE_V2.md` conflict, the V2 document wins.

---

## Purpose

The model adapter layer exists to let the UFC project run many predictive models through one shared prediction interface.

Supported future models may include:

- XGBoost
- Random Forest
- LightGBM
- CatBoost
- Neural networks
- Ensembles
- Market-specific classifiers

Supported market families may include:

- Moneyline
- Goes Distance / Inside Distance
- KO/TKO
- Submission
- Decision
- Method of Victory
- Round totals
- Round-specific props

---

## Core Design Decision

The future adapter layer is generic.

Do **not** build one adapter per market family such as:

```text
MoneylineAdapter
PropAdapter
GoesDistanceAdapter
SubmissionAdapter
```

Instead, use:

```text
Generic Prediction Adapter
        ↓
Algorithm Prediction Plug-in
        ↓
Config-Driven Formatter
        ↓
Outcome-Level Prediction Rows
```

---

## High-Level Architecture

```text
Model Registry
    ↓
Model Config YAML
    ↓
Model Loader
    ↓
Generic Prediction Adapter
    ↓
Algorithm Prediction Plug-in
    ↓
Prediction Formatter
    ↓
Outcome-Based Prediction Artifact
```

The adapter receives fight-level model-ready features and returns outcome-level prediction rows.

---

## Adapter Responsibilities

The generic adapter owns:

1. loading the selected model config
2. loading the model bundle
3. loading the feature contract
4. aligning live features to the model feature list
5. coercing model inputs to numeric values
6. filling missing numeric values consistently with training behavior
7. calling the algorithm prediction dispatcher
8. applying probability clipping if configured
9. sending probability output to the formatter
10. returning canonical outcome rows

The adapter should not contain moneyline-specific or prop-specific business rules.

---

## Algorithm Plug-ins

Prediction should mirror the training framework.

Training pattern:

```text
pipeline/training/model_training.py
        ↓
pipeline/training/algorithms/xgboost_trainer.py
```

Prediction pattern:

```text
pipeline/modeling/prediction_adapter.py
        ↓
pipeline/modeling/probability.py
        ↓
pipeline/modeling/algorithms/xgboost_predictor.py
```

Algorithm plug-ins should be narrow.

For binary XGBoost models, the plug-in only needs to return:

```python
model.predict_proba(X)[:, 1]
```

For multiclass XGBoost models, it should return the full class-probability matrix.

---

## Formatter Responsibilities

The formatter converts model probabilities into outcome rows.

Formatter behavior is controlled by the model YAML, not hardcoded market-specific adapters.

Supported formatter types should include:

```text
binary_matchup
binary_prop
multiclass
```

### binary_matchup

Used for moneyline.

One fight becomes two outcome rows:

```text
market_key = moneyline
outcome_label = red fighter
model_probability = red win probability
```

```text
market_key = moneyline
outcome_label = blue fighter
model_probability = blue win probability
```

### binary_prop

Used for markets such as goes-distance, KO/TKO, submission, decision, over/under.

One fight becomes two outcome rows:

```text
outcome_label = positive_label
model_probability = positive probability
```

```text
outcome_label = negative_label
model_probability = 1 - positive probability
```

### multiclass

Used for method-style models.

One fight becomes one row per configured class label.

---

## Canonical Output

The long-term canonical adapter output is not:

```text
red_model_prob
blue_model_prob
```

The canonical output is:

```text
market_key
outcome_label
model_probability
```

with metadata such as:

```text
prediction_run_id
prediction_timestamp
model_id
model_family
algorithm
prediction_type
event_id
event_name
fight_id
red_fighter
blue_fighter
is_model_pick
model_confidence
passes_model_data_quality
passes_feature_validation
```

Side-based columns may exist temporarily in legacy runners but are not the target architecture.

---

## Expected Future Structure

```text
pipeline/modeling/
  model_registry.py
  model_config.py
  model_loader.py
  prediction_adapter.py
  prediction_formatter.py
  probability.py
  algorithms/
    xgboost_predictor.py
```

---

## Migration Notes

- Keep current legacy prediction runner until the V2 outcome pipeline is implemented.
- Do not add new long-term side-based prediction contracts.
- New model families should be designed around outcome rows from the start.
- Dashboard, Model Lab, CLV, and bankroll should eventually consume outcome-based betting artifacts.
