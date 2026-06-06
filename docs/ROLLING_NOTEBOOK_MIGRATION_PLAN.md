# Rolling Feature Notebook Migration Plan

## Purpose

This document summarizes `UFC_rolling_dataset_V4_refactored.ipynb` section by section and maps each section to the future modular feature architecture.

The goal is to convert the notebook into production Python modules without losing any existing rolling features, while also replacing outdated Google Drive / CSV inputs with the current Data Maintenance master parquet.

---

## Critical Source-Data Correction

The notebook currently reads the historical source dataset from Google Drive:

```python
df = pd.read_csv("/content/drive/MyDrive/UFC_AI/UFC.csv")
```

This is outdated.

The current canonical fight dataset is maintained by the Data Maintenance tab and append pipeline:

```text
data/master/ufc_master.parquet
```

Path constant:

```python
from pipeline.common.paths import MASTER_PATH
```

Future production code should read:

```python
df = pd.read_parquet(MASTER_PATH)
```

not the old Drive CSV.

---

## Current Notebook Summary

Notebook:

```text
UFC_rolling_dataset_V4_refactored.ipynb
```

Current output artifact:

```text
data/features/UFC_enhanced_rolling_features_EWM.parquet
```

Current protected output schema:

```text
483 columns
```

Current protected moneyline dependency:

```text
124 current moneyline training features
```

Protected contracts:

```text
configs/features/full_rolling_feature_inventory.yaml
configs/features/rolling_feature_preservation_contract.yaml
configs/features/current_moneyline_v5_features.yaml
configs/features/feature_registry.yaml
```

---

## Section-by-Section Notebook Review

## 1. Colab / Google Drive Mount

### What it does

Mounts Google Drive so the notebook can read files from:

```text
/content/drive/MyDrive/UFC_AI
```

### Current input/output

Input:

```text
Google Drive filesystem
```

Output:

```text
Mounted Drive path
```

### Future module destination

Remove from production Python modules.

### Refactor notes

Production code should not depend on Colab or Google Drive. It should use repo-relative paths from:

```python
pipeline.common.paths
```

---

## 2. Path Setup and Shared Imports

### What it does

Imports pandas, numpy, defaultdict, deque, and project helper modules from the old Drive path.

It creates a `UFCPipelinePaths` object with:

```text
base_path=/content/drive/MyDrive/UFC_AI
model_version=UFC_Model_v5_Experiment
```

### Current input/output

Input:

```text
Google Drive helper files
legacy UFCPipelinePaths
```

Output:

```text
RAW_CSV_PATH
ROLLING_CSV_PATH
BASE_PATH
```

### Future module destination

```text
pipeline/features/base/build_rolling_features.py
pipeline/common/paths.py
```

### Refactor notes

Replace Drive path logic with:

```python
from pipeline.common.paths import MASTER_PATH, ROLLING_FEATURES_PATH
```

Do not hard-code `UFC_Model_v5_Experiment` in the feature builder.

---

## 3. Load Raw UFC Data

### What it does

Reads historical UFC fight rows, converts `date`, sorts chronologically, drops rows missing required IDs/results, and creates:

```python
target = 1 if winner_id == r_id else 0
```

### Current input/output

Current input:

```text
/content/drive/MyDrive/UFC_AI/UFC.csv
```

Current output:

```text
8190 rows x 129 columns
```

### Future module destination

```text
pipeline/features/base/load_master_fights.py
```

or inside:

```text
pipeline/features/base/build_rolling_features.py
```

### Refactor notes

Replace old CSV source with:

```text
data/master/ufc_master.parquet
```

via:

```python
MASTER_PATH
```

Important: the current master may contain more rows than the old notebook output. The feature builder should validate:

- required columns exist
- dates parse correctly
- rows are sorted chronologically
- missing `winner_id`, `r_id`, or `b_id` rows are handled deliberately
- future/unresolved fights are excluded from historical training features unless intentionally staged

---

## 4. Settings and Helpers

### What it does

Defines feature-builder constants:

```python
START_ELO = 1500
K_FACTOR = 32
RECENT_N = 3
```

Defines helper functions:

```python
safe_div(a, b)
expected_score(elo_a, elo_b)
```

### Current input/output

Input:

```text
none
```

Output:

```text
Elo and recent-form settings
safe math helpers
```

### Future module destination

```text
pipeline/features/base/elo.py
pipeline/features/base/utils.py
```

### Refactor notes

These settings should become configurable later, but the first migration should preserve current behavior exactly.

---

## 5. Fighter State Initialization

### What it does

Defines `default_state()` for each fighter.

The state tracks:

- Elo
- fights/wins/losses
- knockdowns for/against
- striking landed/attempted/absorbed
- takedowns landed/attempted/allowed
- submission attempts
- control time for/against
- total fight time
- finish wins/losses
- method-specific wins/losses
- opponent Elo history
- streaks
- last fight date
- recent 3-fight queues

### Current input/output

Input:

```text
START_ELO
RECENT_N
```

Output:

```python
fighter_state = defaultdict(default_state)
```

### Future module destination

```text
pipeline/features/base/fighter_state.py
```

### Refactor notes

This is the core point-in-time state engine. It must remain deterministic and chronological.

Do not update a fighter's state until after that fight's pre-fight row has been generated.

---

## 6. Prefight Feature Function

### What it does

Defines:

```python
get_prefight_features(fighter_id, fight_date)
```

This reads a fighter's state before the current fight and emits pre-fight features such as:

- Elo
- fight count
- win/loss record
- win percentage
- knockdown averages
- strikes landed/absorbed per minute
- striking accuracy/defense
- takedown averages/accuracy/defense
- submission averages
- control time per minute
- finish rates
- method rates
- opponent quality metrics
- average fight time
- streaks
- days since last fight
- recent 3-fight form

### Current input/output

Input:

```text
fighter_id
fight_date
fighter_state
```

Output:

```text
single fighter pre-fight feature dictionary
```

### Future module destination

```text
pipeline/features/base/prefight_features.py
```

### Refactor notes

This function is central to leakage prevention.

It must only use information available before the fight being processed.

---

## 7. Fighter State Update Function

### What it does

Updates a fighter's state after a fight is processed.

It updates:

- wins/losses
- streaks
- method-specific wins/losses
- striking totals
- takedown totals
- submission attempts
- control time
- fight time
- opponent Elo
- recent 3-fight queues
- last fight date

### Current input/output

Input:

```text
fighter_id
fight_date
won flag
method
own fight stats
opponent fight stats
fight_time_sec
opponent_elo
```

Output:

```text
mutated fighter_state
```

### Future module destination

```text
pipeline/features/base/fighter_state.py
```

### Refactor notes

This should be kept separate from pre-fight feature extraction.

The order must remain:

```text
1. get pre-fight features
2. create row
3. update Elo
4. update fighter states
```

---

## 8. Build Rolling Dataset

### What it does

Loops through fights chronologically.

For each fight:

1. Gets red fighter pre-fight features.
2. Gets blue fighter pre-fight features.
3. Adds `r_pre_*` columns.
4. Adds `b_pre_*` columns.
5. Adds `*_diff` columns.
6. Appends the row.
7. Updates Elo for both fighters.
8. Updates both fighter states.

### Current input/output

Input:

```text
clean historical fight dataframe
fighter_state
```

Output:

```text
rolling_df: 8190 rows x 237 columns
```

### Future module destination

```text
pipeline/features/base/build_rolling_features.py
```

### Refactor notes

This is the core base rolling feature builder.

The first Python migration should reproduce this output exactly before adding new features.

Efficiency improvement: collect feature dictionaries and create DataFrames in batches rather than repeatedly inserting columns.

---

## 9. Identify EWM Feature Inputs

### What it does

Finds all matching `r_pre_*` / `b_pre_*` stat pairs and builds `stat_names` for EWM processing.

Notebook output indicates:

```text
36 stats to EWM weight
```

### Current input/output

Input:

```text
rolling_df with r_pre_* and b_pre_* columns
```

Output:

```text
stat_names list
```

### Future module destination

```text
pipeline/features/base/ewm_features.py
```

### Refactor notes

The 36-stat list should be validated against the feature inventory and preservation contract.

---

## 10. Create Long Fighter-Level Dataset

### What it does

Converts each fight row into two fighter rows:

```text
one row for red fighter
one row for blue fighter
```

Each fighter row includes:

- fight index
- date
- fighter ID
- corner
- pre-fight stat values

This long format allows EWM calculations by fighter timeline.

### Current input/output

Input:

```text
rolling_df
stat_names
```

Output:

```text
fighter_long_df
```

### Future module destination

```text
pipeline/features/base/ewm_features.py
```

### Refactor notes

This is a transformation step only. It should not mutate the original rolling dataset until EWM values are ready to merge back.

---

## 11. Calculate EWM Recent-Form Features

### What it does

For each fighter and each selected stat, calculates exponentially weighted moving averages using:

```python
EWM_SPAN = 3
```

Creates features like:

```text
ewm_elo
ewm_win_pct
ewm_splm
ewm_sapm
ewm_td_avg
ewm_finish_rate
ewm_recent_win_pct
```

### Current input/output

Input:

```text
fighter_long_df
EWM_SPAN
```

Output:

```text
fighter_long_df with ewm_* columns
```

### Future module destination

```text
pipeline/features/base/ewm_features.py
```

### Refactor notes

This should remain a base feature step, because EWM recent form can support moneyline, props, and market-aware models.

---

## 12. Merge EWM Features Back to Fight Rows

### What it does

Maps fighter-level EWM values back into the fight-level dataset for red and blue corners.

Expected resulting families:

```text
r_ewm_*
b_ewm_*
ewm_*_diff
```

### Current input/output

Input:

```text
rolling_df
fighter_long_df with ewm_* values
```

Output:

```text
rolling_df with EWM corner features and EWM differentials
```

### Future module destination

```text
pipeline/features/base/ewm_features.py
pipeline/features/moneyline/build_moneyline_features.py
```

### Refactor notes

Corner-level EWM features are base features.

EWM differentials are market-ready moneyline features and prop candidates.

---

## 13. Add Recent-Form Differential Features

### What it does

Adds additional recent-form differential columns such as:

```text
recent_form_win_pct_diff
recent_form_splm_diff
recent_form_sapm_diff
recent_form_td_avg_diff
recent_form_finish_rate_diff
recent_form_avg_fight_time_diff
```

### Current input/output

Input:

```text
rolling_df with recent form / EWM data
```

Output:

```text
rolling_df with recent_form_*_diff features
```

### Future module destination

```text
pipeline/features/moneyline/build_moneyline_features.py
pipeline/features/props/*
```

### Refactor notes

These are currently part of the protected 124-feature moneyline contract.

They are also likely valuable future prop features.

---

## 14. Save Final Rolling Feature Artifact

### What it does

Saves the final rolling feature file.

Current protected artifact:

```text
data/features/UFC_enhanced_rolling_features_EWM.parquet
```

### Current input/output

Input:

```text
rolling_df final
```

Output:

```text
483-column rolling feature artifact
```

### Future module destination

```text
pipeline/features/base/build_rolling_features.py
```

### Refactor notes

The first migrated builder should continue producing the current artifact path until parity is proven.

Future split artifacts may include:

```text
data/features/ufc_base_features.parquet
data/features/ufc_moneyline_features.parquet
data/features/ufc_prop_features.parquet
data/features/ufc_market_features.parquet
```

But no split should occur until the 483-column preservation contract passes.

---

## Future Module Map

Recommended module structure:

```text
pipeline/features/base/
├── load_master_fights.py
├── elo.py
├── fighter_state.py
├── prefight_features.py
├── ewm_features.py
└── build_rolling_features.py

pipeline/features/moneyline/
└── build_moneyline_features.py

pipeline/features/props/
├── build_prop_labels.py
├── build_ko_tko_features.py
├── build_submission_features.py
├── build_decision_features.py
├── build_goes_distance_features.py
└── build_round_features.py

pipeline/features/market/
├── build_historical_odds_features.py
├── build_line_movement_features.py
└── build_clv_features.py
```

---

## Market-Aware Feature Note

Historical odds should be incorporated later as a separate market feature layer, not mixed into the base fight-performance layer.

Future market feature artifact:

```text
data/features/ufc_market_features.parquet
```

Candidate market-aware features:

```text
opening_implied_prob
closing_implied_prob
current_implied_prob
line_movement_pct
market_disagreement
sportsbook_consensus_prob
steam_move_flag
model_market_edge
historical_clv_signal
```

The current V5-equivalent moneyline model should remain a pure fighter-performance model until a market-aware model is intentionally trained and validated.

---

## Migration Strategy

## Phase 1: Preserve Current Output

Build Python modules that reproduce the current rolling feature artifact exactly.

Inputs:

```text
data/master/ufc_master.parquet
```

Output:

```text
data/features/UFC_enhanced_rolling_features_EWM.parquet
```

Validation:

```text
483 columns
current moneyline feature count = 124
all protected features present
```

---

## Phase 2: Create Base and Moneyline Views

After parity is proven, create derived views:

```text
data/features/ufc_base_features.parquet
data/features/ufc_moneyline_features.parquet
```

The moneyline view must satisfy:

```text
configs/features/current_moneyline_v5_features.yaml
```

---

## Phase 3: Training Migration

Convert the training notebook to consume:

```text
data/features/ufc_moneyline_features.parquet
```

through the model feature contract.

---

## Phase 4: Backtesting Migration

Convert the backtest notebook to consume:

```text
data/features/ufc_moneyline_features.parquet
historical odds features
```

---

## Phase 5: Prop and Market Feature Expansion

Add:

```text
prop labels
prop features
historical odds features
market-aware overlays
```

only after the moneyline migration is stable.

---

## Validation Requirements

Before replacing notebook behavior, verify:

```text
input uses MASTER_PATH
output path uses ROLLING_FEATURES_PATH
observed rolling column count is 483 or intentionally updated
current moneyline V5-equivalent feature count is 124
all features in full_rolling_feature_inventory.yaml are preserved or mapped
all model-input features are point-in-time safe
no result fields enter live prediction inputs
```

---

## Key Risks

1. The old notebook source is outdated and uses Drive CSV instead of `MASTER_PATH`.
2. The current master parquet may contain newer rows, invalid dates, or unfinished future fights that need filtering.
3. The feature builder must preserve chronological order to prevent leakage.
4. EWM features must be computed by fighter timeline, not by raw row order after accidental resorting.
5. Raw red/blue corner columns must not enter symmetry-safe moneyline training directly.
6. Historical odds should be a market layer, not part of base fighter state.

---

## Recommended Next Implementation Step

No implementation should start until approved.

When approved, create the first production feature module:

```text
pipeline/features/base/build_rolling_features.py
```

Initial goal:

```text
Read MASTER_PATH
Build current rolling feature artifact
Write ROLLING_FEATURES_PATH
Preserve current 483-column output contract as much as possible
Validate 124 moneyline features remain available
```
