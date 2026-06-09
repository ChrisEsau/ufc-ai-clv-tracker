# Migration Validation Notes

This folder contains one-off validation scripts used during the fighter-state / feature-view refactor.

These scripts are intentionally archived and are not part of the active production pipeline.

## Current refactor finding

The new fighter-state architecture successfully reproduces modern fighter-state and EWM behavior for modern high-history fighters.

Validated example:

- Fighter: Max Holloway
- Source: `data/master/ufc_master.parquet`
- Compared artifacts:
  - `data/features/UFC_enhanced_rolling_features_EWM.parquet`
  - `data/features/fighter_state_history.parquet`
  - `data/features/moneyline_feature_view.parquet`

Max Holloway checks passed for:

- `fights`
- `wins`
- `losses`
- `win_pct`
- `pre_elo`
- `ewm_elo`
- `ewm_avg_opponent_elo`
- `ewm_avg_fight_time`
- `ewm_days_since_last_fight`
- `ewm_splm`
- `ewm_td_avg`
- `ewm_sub_avg`

## Known parity exception

The remaining EWM parity differences are concentrated in early UFC history, especially tournament-era events.

Observed impact:

- Approx. 79 to 106 affected rows depending on feature
- Approx. 0.94% to 1.26% of the 8,427-row completed-fight dataset
- First affected date: 1994-03-11
- Last affected date: 2010-02-06
- Common affected events/fighters include early UFC tournament-era records such as Royce Gracie, Dan Severn, Oleg Taktarov, Ken Shamrock, Marco Ruas, Paul Varelans, and related opponents.

Current interpretation:

- This does not appear to be a modern live-prediction issue.
- Modern fighter validation passed.
- The discrepancy is likely tied to early-era data quality and/or tournament-era sequencing ambiguity.
- Continue the refactor without blocking on exact parity for those historical edge cases.

## Decision

Do not spend additional refactor time trying to perfectly reproduce tournament-era EWM behavior unless future model validation shows it materially affects performance.

Prefer modern-era validation and retraining experiments over deep parity work on early historical edge cases.

## Backlog

### Feature-state validation

- Verify additional modern fighters with long histories.
- Confirm `pre_*` features are true prefight snapshots and do not include current-fight data.
- Validate same-event / tournament-era behavior separately only if needed.

### Modern-era model experiments

Run training experiments with cutoff dates:

- 2005-01-01
- 2010-01-01
- Optional: 2013-01-01

Compare against the full-history model using:

- Accuracy
- Log loss
- ROC-AUC
- Calibration
- ROI / EV backtest
- CLV if available

### Architecture cleanup

After parity and model checks:

- Move hardcoded feature views toward a generic YAML-driven feature-view engine.
- Keep one canonical `moneyline.py` file and patch it forward rather than creating duplicate `moneyline_v2.py` style modules.
- Review legacy parity-only EWM features for removal after retraining:
  - `fights`
  - `wins`
  - `losses`
  - `win_pct`
  - `win_streak`
  - `loss_streak`
  - `days_since_last_fight`
