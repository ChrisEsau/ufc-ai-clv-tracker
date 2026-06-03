# UFC Model Lab Architecture

## Purpose

The Model Lab is the research and model-development workspace.

It answers:

```text
How accurate is the model?
Is the model calibrated?
Which features matter?
What changes improve ROI?
Is the model stable over recent UFC eras?
```

---

## Core Responsibilities

* Train UFC prediction models
* Validate model performance
* Monitor calibration
* Compare model versions
* Run backtests
* Analyze feature importance
* Evaluate ROI by threshold
* Support ensemble modeling

---

## Primary Inputs

* data/master/ufc_master.parquet
* historical feature stores
* training feature registry
* model artifacts under `models/UFC_Model_v5_Experiment/`
* market odds history
* bet result history

---

## Primary Outputs

* Trained model files
* Feature column registry
* Calibration reports
* Backtest results
* ROI summaries
* Model comparison tables

---

## Current Model Direction

The platform is moving toward V5 model development.

Priority areas:

```text
1. V5 model development
2. Prop bet engine
3. Advanced market analytics
4. Closing line value tracking
5. Recent-era validation
6. Confidence-weighted staking
7. Market-aware features
8. Ensemble modeling
```

---

## Core Validation Metrics

* Accuracy
* Log loss
* ROC-AUC
* Calibration error
* Brier score
* ROI
* Flat bet profit
* Kelly bet profit
* Beat closing line rate

---

## Feature Stores

Historical point-in-time feature store:

```text
data/features/UFC_enhanced_rolling_features_EWM.parquet
```

Live fighter-state feature store:

```text
data/features/ufc_current_fighter_features.parquet
```

---

## Future Enhancements

* Model comparison dashboard
* Calibration dashboard
* Automated retraining
* Feature drift monitoring
* Recent-era validation panel
* Prop model lab
* Ensemble model selector
