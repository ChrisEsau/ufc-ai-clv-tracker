**# Round Fighter State P0.1 Trajectory Spec

## Purpose

This document defines the approved formula specification for the first Round Fighter State feature family:

```text
P0.1 — Round Trajectory and Decay
```

This is a design document only.

It does not implement Python code.

The purpose of this family is to measure whether a fighter sustains, improves, or deteriorates as rounds progress.

This feature family is intended to capture:

* pace sustainability
* cardio decay
* offensive persistence
* accuracy preservation
* wrestling persistence
* defensive deterioration
* late-round function
* recovery after expensive rounds

This family belongs to the broader Round Fighter State architecture.

Input:

```text
data/fight_details/ufc_round_stats.parquet
```

Historical output:

```text
data/features/round_fighter_state_history.parquet
```

Latest output:

```text
data/features/round_latest_fighter_state.parquet
```

---

# Core Modeling Question

This feature family asks:

```text
Does this fighter keep functioning as the fight progresses?
```

It should not simply measure total output.

It should measure whether output, efficiency, wrestling, and defense hold up across rounds.

Examples:

* A fighter who throws 70 strikes in Round 1 and 20 in Round 3 may have poor sustainability.
* A fighter whose accuracy collapses late may be fading even if volume remains high.
* A fighter who allows increasing opponent accuracy by round may be defensively degrading.
* A fighter who continues takedown attempts late may have wrestling/cardio persistence.
* A fighter whose opponent output rises late may be losing control of the fight state.

---

# Scope

## Included in P0.1

P0.1 may include:

* round-to-round output slopes
* late-output ratios
* accuracy slopes
* late-accuracy ratios
* takedown persistence slopes
* control-time trajectory
* opponent accuracy allowed slopes
* opponent output allowed ratios
* late defensive degradation proxies

## Excluded from P0.1

P0.1 must not include:

* opponent suppression baselines
* phase-imposition deltas
* wrestling control conversion
* control-to-damage conversion
* tactical adaptation after losing rounds
* adversity response after knockdowns
* finishing snowball features
* spatial ringcraft assumptions
* in-round momentum assumptions
* timestamped sequence features

Those belong to later P0/P1/P2 families.

---

# Input Grain

Input file:

```text
data/fight_details/ufc_round_stats.parquet
```

Expected grain:

```text
one row per fighter per fight per round
```

Expected primary key:

```text
fight_id
fighter_id
round
corner
```

Each fight round should have:

* one red-corner row
* one blue-corner row

---

# Required Input Columns

Minimum required identity columns:

```text
event_id
fight_id
fighter_id
opponent_id
fighter_name
opponent_name
event_name
date
round
corner
```

Minimum required numeric columns:

```text
sig_str_landed
sig_str_attempted
total_str_landed
total_str_attempted
td_landed
td_attempted
control_seconds
kd
```

Recommended phase columns if available:

```text
distance_str_landed
distance_str_attempted
clinch_str_landed
clinch_str_attempted
ground_str_landed
ground_str_attempted
head_str_landed
head_str_attempted
body_str_landed
body_str_attempted
leg_str_landed
leg_str_attempted
```

P0.1 can be built without phase columns, but phase columns may be used later for extended trajectory variants.

---

# Derived Per-Round Metrics

Before calculating trajectory features, the builder should create safe per-round metrics.

## Safe Division Rule

For any ratio:

```text
ratio = numerator / denominator
```

Use:

```text
safe_div(numerator, denominator)
```

Where:

```text
safe_div(numerator, denominator) = null if denominator is 0 or null
safe_div(numerator, denominator) = numerator / denominator otherwise
```

Do not silently convert zero-denominator ratios to zero at the observation layer.

Null should mean:

```text
the rate was not observable
```

not:

```text
the fighter performed badly
```

---

## Fighter Offensive Metrics

For each fighter-round row:

```text
sig_attempts = sig_str_attempted
sig_landed = sig_str_landed
sig_accuracy = sig_str_landed / sig_str_attempted

total_attempts = total_str_attempted
total_landed = total_str_landed
total_accuracy = total_str_landed / total_str_attempted

td_attempts = td_attempted
td_landed = td_landed
td_accuracy = td_landed / td_attempted

control_seconds = control_seconds
```

---

## Opponent Metrics

Because the input has one row per fighter per round, opponent metrics should be joined from the opposing fighter’s row in the same fight and round.

Join key:

```text
fight_id
round
fighter_id == opponent_id
```

After joining, each fighter-round row should have opponent values:

```text
opp_sig_attempts
opp_sig_landed
opp_sig_accuracy

opp_total_attempts
opp_total_landed
opp_total_accuracy

opp_td_attempts
opp_td_landed
opp_td_accuracy

opp_control_seconds
opp_kd
```

These are used only for defensive trajectory and output-allowed trajectory in P0.1.

---

# Observation Layer vs State Layer

P0.1 has two conceptual layers.

## 1. Fight Observation Layer

The fight observation layer summarizes what happened in one completed fight.

Expected grain:

```text
one row per fighter per completed fight
```

Example feature prefix:

```text
rfs_traj_fight_*
```

These features describe the fighter’s trajectory inside that specific completed fight.

Example:

```text
rfs_traj_fight_sig_attempt_slope
```

Meaning:

```text
In this completed fight, how did the fighter's significant strike attempts change by round?
```

---

## 2. Fighter State Layer

The fighter state layer summarizes the fighter’s historical tendency over prior completed fights.

Expected grain:

```text
one row per fighter per completed fight
```

Example feature prefixes:

```text
rfs_traj_ewm_*
rfs_traj_last3_*
rfs_traj_exp_*
```

These features describe the fighter’s known Round Fighter State after each completed fight.

Example:

```text
rfs_traj_ewm_sig_attempt_slope
```

Meaning:

```text
After this completed fight, what is the fighter's exponentially weighted historical tendency for significant strike attempt slope?
```

---

# Point-in-Time Rule

For model training:

```text
Features for Fighter A entering Fight N must use only Fighter A's fights before Fight N.
```

Correct:

```text
Fights 1 through N-1
        ↓
features entering Fight N
```

Incorrect:

```text
Fight N round stats
        ↓
features for Fight N
```

The feature builder may store post-fight state after Fight N, but the training feature view must shift the fighter state before joining to the target fight.

---

# Minimum-Round Rules

Round trajectory features require enough observed rounds to be meaningful.

## One-Round Fights

If a fighter has only one observed round in a fight:

```text
slope features = null
late-ratio features = null
round3-specific features = null
```

Do not force a slope of zero.

A one-round fight does not prove the fighter sustained or decayed.

---

## Two-Round Fights

If a fighter has two observed rounds:

```text
slope features = allowed
late-ratio features = allowed
round3-specific features = null
```

The slope is calculated from Round 1 to Round 2.

The late ratio is calculated using Round 2 as the late round.

---

## Three-Round Fights

If a fighter has three observed rounds:

```text
slope features = allowed
late-ratio features = allowed
round3-specific features = allowed
```

The late ratio may use Round 3 or average of Rounds 2 and 3 depending on the metric.

---

## Five-Round Fights

If a fighter has four or five observed rounds:

```text
slope features = allowed
late-ratio features = allowed
championship-round features = allowed later, but not in P0.1 V1
```

For P0.1 V1, avoid separate five-round/championship features.

The first implementation should use generic observed-round formulas to avoid overfitting.

---

# Core Formula Definitions

## OLS Slope

For a metric `x` across observed rounds:

```text
round = [1, 2, 3, ...]
x = metric value in each round
```

Calculate:

```text
slope(x) = OLS slope of x ~ round
```

Equivalent simple formula:

```text
slope = covariance(round, x) / variance(round)
```

Rules:

```text
minimum non-null rounds = 2
ignore null metric values
return null if fewer than 2 non-null rounds
```

Interpretation:

```text
positive slope = metric increases as fight progresses
negative slope = metric decreases as fight progresses
zero slope = metric is flat
```

---

## Late Ratio

For a metric `x`:

```text
late_ratio = late_value / round1_value
```

Where:

```text
round1_value = value in Round 1
late_value = value in final observed round
```

Rules:

```text
minimum observed rounds = 2
round1_value must be greater than 0
late_value must be non-null
return null otherwise
```

Interpretation:

```text
late_ratio > 1.00 = late output higher than Round 1
late_ratio = 1.00 = late output equal to Round 1
late_ratio < 1.00 = late output lower than Round 1
```

---

## Late Difference

For a metric `x`:

```text
late_diff = late_value - round1_value
```

Rules:

```text
minimum observed rounds = 2
round1_value must be non-null
late_value must be non-null
return null otherwise
```

Late difference is useful when the denominator can be unstable.

---

## Round 3 vs Round 1 Ratio

For a metric `x`:

```text
round3_vs_round1_ratio = round3_value / round1_value
```

Rules:

```text
Round 1 and Round 3 must both exist
round1_value must be greater than 0
return null otherwise
```

This is optional in P0.1 V1 but useful because third-round state was identified as potentially important.

---

# Approved Fight Observation Features

The following features may be created at the fight observation layer.

These are not directly model features unless shifted into historical state.

---

## Offensive Output Trajectory

### Significant Strike Attempt Slope

Feature:

```text
rfs_traj_fight_sig_attempt_slope
```

Formula:

```text
slope(sig_str_attempted by round)
```

Meaning:

```text
Did significant strike attempt volume rise or fall as the fight progressed?
```

---

### Total Strike Attempt Slope

Feature:

```text
rfs_traj_fight_total_attempt_slope
```

Formula:

```text
slope(total_str_attempted by round)
```

Meaning:

```text
Did total activity rise or fall as the fight progressed?
```

---

### Significant Strike Landed Slope

Feature:

```text
rfs_traj_fight_sig_landed_slope
```

Formula:

```text
slope(sig_str_landed by round)
```

Meaning:

```text
Did effective significant-strike production rise or fall by round?
```

---

### Total Strike Landed Slope

Feature:

```text
rfs_traj_fight_total_landed_slope
```

Formula:

```text
slope(total_str_landed by round)
```

Meaning:

```text
Did total landed production rise or fall by round?
```

---

## Offensive Late Ratios

### Significant Attempt Late Ratio

Feature:

```text
rfs_traj_fight_sig_attempt_late_ratio
```

Formula:

```text
final_round_sig_str_attempted / round1_sig_str_attempted
```

Meaning:

```text
How much significant-strike volume remained late?
```

---

### Total Attempt Late Ratio

Feature:

```text
rfs_traj_fight_total_attempt_late_ratio
```

Formula:

```text
final_round_total_str_attempted / round1_total_str_attempted
```

Meaning:

```text
How much total output remained late?
```

---

### Significant Landed Late Ratio

Feature:

```text
rfs_traj_fight_sig_landed_late_ratio
```

Formula:

```text
final_round_sig_str_landed / round1_sig_str_landed
```

Meaning:

```text
How much significant-strike production remained late?
```

---

## Accuracy Trajectory

### Significant Accuracy Slope

Feature:

```text
rfs_traj_fight_sig_accuracy_slope
```

Formula:

```text
slope(sig_accuracy by round)
```

Where:

```text
sig_accuracy = sig_str_landed / sig_str_attempted
```

Meaning:

```text
Did significant-strike efficiency improve or decay as the fight progressed?
```

---

### Total Accuracy Slope

Feature:

```text
rfs_traj_fight_total_accuracy_slope
```

Formula:

```text
slope(total_accuracy by round)
```

Where:

```text
total_accuracy = total_str_landed / total_str_attempted
```

Meaning:

```text
Did total striking efficiency improve or decay as the fight progressed?
```

---

### Significant Accuracy Late Difference

Feature:

```text
rfs_traj_fight_sig_accuracy_late_diff
```

Formula:

```text
final_round_sig_accuracy - round1_sig_accuracy
```

Meaning:

```text
How much did significant-strike accuracy change from early to late?
```

---

## Wrestling Persistence Trajectory

### Takedown Attempt Slope

Feature:

```text
rfs_traj_fight_td_attempt_slope
```

Formula:

```text
slope(td_attempted by round)
```

Meaning:

```text
Did the fighter continue attempting takedowns as the fight progressed?
```

---

### Takedown Accuracy Slope

Feature:

```text
rfs_traj_fight_td_accuracy_slope
```

Formula:

```text
slope(td_accuracy by round)
```

Where:

```text
td_accuracy = td_landed / td_attempted
```

Meaning:

```text
Did takedown efficiency improve or decay across rounds?
```

---

### Takedown Attempt Late Ratio

Feature:

```text
rfs_traj_fight_td_attempt_late_ratio
```

Formula:

```text
final_round_td_attempted / round1_td_attempted
```

Meaning:

```text
Did wrestling pressure remain available late?
```

---

## Control-Time Trajectory

### Control Seconds Slope

Feature:

```text
rfs_traj_fight_control_seconds_slope
```

Formula:

```text
slope(control_seconds by round)
```

Meaning:

```text
Did the fighter gain or lose control-time influence as the fight progressed?
```

---

### Control Late Ratio

Feature:

```text
rfs_traj_fight_control_late_ratio
```

Formula:

```text
final_round_control_seconds / round1_control_seconds
```

Meaning:

```text
Did control-time presence increase or decrease late?
```

Caution:

Control time alone should not be treated as automatically positive. This feature only belongs to the trajectory family. Control quality belongs to P0.3 Wrestling Control Conversion.

---

## Defensive Degradation Proxies

These use opponent metrics from the same rounds.

### Opponent Significant Accuracy Allowed Slope

Feature:

```text
rfs_traj_fight_opp_sig_accuracy_allowed_slope
```

Formula:

```text
slope(opp_sig_accuracy by round)
```

Meaning:

```text
Did the opponent become more accurate against this fighter as the fight progressed?
```

Interpretation:

```text
positive = possible defensive degradation
negative = possible defensive stabilization
```

---

### Opponent Total Accuracy Allowed Slope

Feature:

```text
rfs_traj_fight_opp_total_accuracy_allowed_slope
```

Formula:

```text
slope(opp_total_accuracy by round)
```

Meaning:

```text
Did the opponent become more efficient overall as the fight progressed?
```

---

### Opponent Significant Attempts Allowed Slope

Feature:

```text
rfs_traj_fight_opp_sig_attempt_allowed_slope
```

Formula:

```text
slope(opp_sig_str_attempted by round)
```

Meaning:

```text
Did the opponent’s significant-strike activity increase as the fight progressed?
```

---

### Opponent Control Allowed Slope

Feature:

```text
rfs_traj_fight_opp_control_allowed_slope
```

Formula:

```text
slope(opp_control_seconds by round)
```

Meaning:

```text
Did the fighter allow increasing control as rounds progressed?
```

---

# Approved State Aggregation Features

Fight observation features should be rolled into fighter-state features.

The initial implementation should support three aggregation families:

```text
expanding mean
last-3 mean
EWM mean
```

Do not start with too many windows.

---

## Expanding Mean

Feature naming pattern:

```text
rfs_traj_exp_<metric>
```

Formula:

```text
mean of all prior non-null fight observation values for that fighter
```

Example:

```text
rfs_traj_exp_sig_attempt_slope
```

Source:

```text
rfs_traj_fight_sig_attempt_slope
```

Meaning:

```text
The fighter's long-run historical significant-strike attempt trajectory.
```

---

## Last-3 Mean

Feature naming pattern:

```text
rfs_traj_last3_<metric>
```

Formula:

```text
mean of the fighter's last 3 prior non-null fight observation values
```

Example:

```text
rfs_traj_last3_sig_attempt_slope
```

Meaning:

```text
The fighter's recent significant-strike attempt trajectory.
```

Purpose:

Capture recent fighter-state changes without relying entirely on one fight.

---

## EWM Mean

Feature naming pattern:

```text
rfs_traj_ewm_<metric>
```

Formula:

```text
exponentially weighted mean of prior fight observation values
```

Recommended initial setting:

```text
alpha = 0.35
adjust = false
ignore_na = true
```

Example:

```text
rfs_traj_ewm_sig_attempt_slope
```

Meaning:

```text
The fighter's recency-weighted significant-strike attempt trajectory.
```

---

# Minimum Fight History Rules

## Zero Prior Fights

If a fighter has no prior UFC fights with round stats:

```text
state features = null
state_missing_flag = 1
```

Do not impute inside the state store.

---

## One Prior Fight

If a fighter has one prior valid fight observation:

```text
expanding mean = allowed
last3 mean = same as one-fight mean
EWM = allowed
```

But include:

```text
rfs_traj_prior_fight_count = 1
```

---

## Two or More Prior Fights

All state aggregations are allowed if the underlying observation values are non-null.

---

# Required Metadata Columns

State history should include metadata that makes feature interpretation and validation possible.

Required metadata:

```text
fighter_id
fighter_name
fight_id
event_id
event_name
date
opponent_id
opponent_name
corner
rfs_traj_prior_fight_count
rfs_traj_prior_valid_trajectory_count
rfs_traj_has_state
```

Where:

```text
rfs_traj_prior_fight_count = number of prior completed fights for this fighter
rfs_traj_prior_valid_trajectory_count = number of prior fights with at least one valid trajectory observation
rfs_traj_has_state = 1 if prior_valid_trajectory_count > 0 else 0
```

---

# Initial Approved State Features

The initial P0.1 V1 state feature set should stay compact.

Recommended first pass:

```text
rfs_traj_prior_fight_count
rfs_traj_prior_valid_trajectory_count
rfs_traj_has_state

rfs_traj_exp_sig_attempt_slope
rfs_traj_last3_sig_attempt_slope
rfs_traj_ewm_sig_attempt_slope

rfs_traj_exp_total_attempt_slope
rfs_traj_last3_total_attempt_slope
rfs_traj_ewm_total_attempt_slope

rfs_traj_exp_sig_landed_slope
rfs_traj_last3_sig_landed_slope
rfs_traj_ewm_sig_landed_slope

rfs_traj_exp_sig_attempt_late_ratio
rfs_traj_last3_sig_attempt_late_ratio
rfs_traj_ewm_sig_attempt_late_ratio

rfs_traj_exp_sig_accuracy_slope
rfs_traj_last3_sig_accuracy_slope
rfs_traj_ewm_sig_accuracy_slope

rfs_traj_exp_td_attempt_slope
rfs_traj_last3_td_attempt_slope
rfs_traj_ewm_td_attempt_slope

rfs_traj_exp_control_seconds_slope
rfs_traj_last3_control_seconds_slope
rfs_traj_ewm_control_seconds_slope

rfs_traj_exp_opp_sig_accuracy_allowed_slope
rfs_traj_last3_opp_sig_accuracy_allowed_slope
rfs_traj_ewm_opp_sig_accuracy_allowed_slope

rfs_traj_exp_opp_sig_attempt_allowed_slope
rfs_traj_last3_opp_sig_attempt_allowed_slope
rfs_traj_ewm_opp_sig_attempt_allowed_slope

rfs_traj_exp_opp_control_allowed_slope
rfs_traj_last3_opp_control_allowed_slope
rfs_traj_ewm_opp_control_allowed_slope
```

This creates a manageable first family.

Do not add every possible derivative in V1.

---

# Model View Transformation

The fighter-state store remains fighter-centric.

The model feature view should convert these features into red/blue side-aware features.

For each fighter-level state feature:

```text
rfs_traj_ewm_sig_attempt_slope
```

Create:

```text
r_rfs_traj_ewm_sig_attempt_slope
b_rfs_traj_ewm_sig_attempt_slope
rfs_traj_ewm_sig_attempt_slope_diff
```

Diff formula:

```text
diff = red_value - blue_value
```

This is consistent with the existing model-view pattern.

---

# Missing Value Handling

## In State Store

Do not force-fill missing state values in:

```text
round_fighter_state_history.parquet
round_latest_fighter_state.parquet
```

Use nulls to represent unavailable state.

## In Model Feature View

Model-view imputation should happen later in the model feature builder.

Recommended model-view behavior:

```text
missing numeric RFS features -> 0 after side-aware diff creation
missingness tracked with feature completeness audit
rfs_traj_has_state retained as signal
```

Reason:

A missing state is different from bad trajectory.

A debuting fighter should not be treated as having poor cardio decay.

---

# Outlier Handling

P0.1 should not aggressively clip observations in the raw state builder.

However, validation should flag extreme values.

Recommended audit thresholds:

```text
absolute slope > 200 attempts per round
late ratio > 10
accuracy slope outside [-1, 1]
accuracy value outside [0, 1]
control seconds per round > 300
opponent control seconds per round > 300
```

The first implementation should audit outliers before deciding whether to clip.

---

# Validation Requirements

P0.1 must include validation before any model experiment.

## Raw Input Validation

Required checks:

```text
required columns present
no duplicate fight_id/fighter_id/round/corner keys
round is numeric
round >= 1
corner in {red, blue}
numeric columns are numeric
landed <= attempted
control_seconds >= 0
control_seconds <= 300 per round
```

---

## Opponent Join Validation

Required checks:

```text
each fighter-round has exactly one opponent-round row
fighter_id != opponent_id
opponent_id matches paired row
round matches paired row
fight_id matches paired row
```

Failures should be audited, not silently ignored.

---

## Observation Feature Validation

Required checks:

```text
one row per fighter per fight
slope features null for one-round fights
slope features non-null only when at least two valid rounds exist
late ratios null when round1 denominator is zero
accuracy values between 0 and 1
accuracy slopes between -1 and 1
no infinite values
```

---

## State Feature Validation

Required checks:

```text
one row per fighter per completed fight
state features use only prior fights
prior_fight_count correct
prior_valid_trajectory_count correct
no target-fight leakage
no infinite values
feature names follow rfs_traj_* convention
latest state has one row per fighter
```

---

## Model View Validation

Required checks:

```text
red fighter state joins correctly
blue fighter state joins correctly
diff columns equal red minus blue
missingness audit generated
feature completeness tracked
no raw round stats included in model view
no target-fight round stats included in model view
```

---

# Ablation Requirements

P0.1 should be evaluated as its own feature family before building P0.2.

Required experiments:

```text
baseline model without RFS
baseline + P0.1 trajectory features
P0.1-only diagnostic model if useful
```

Required metrics:

```text
log loss
ROC-AUC
accuracy
Brier score if available
calibration by bucket
SHAP contribution
feature stability
ROI
CLV
bet count
edge distribution
```

P0.1 should not be promoted if it only improves accuracy while damaging calibration, ROI, or CLV.

---

# Expected First Build Output

The first P0.1 implementation should produce:

```text
data/features/round_fighter_state_history.parquet
data/features/round_latest_fighter_state.parquet
data/audits/round_fighter_state_p0_1_validation.parquet
```

Optional CSV audit exports:

```text
data/audits/round_fighter_state_p0_1_missingness.csv
data/audits/round_fighter_state_p0_1_outliers.csv
```

---

# Recommended Module Path

Future implementation should live under:

```text
pipeline/round_stats/
```

Recommended future files:

```text
pipeline/round_stats/build_round_fighter_state.py
pipeline/round_stats/validate_round_fighter_state.py
pipeline/round_stats/round_state_formulas.py
```

Recommended module execution pattern:

```text
python -m pipeline.round_stats.build_round_fighter_state
python -m pipeline.round_stats.validate_round_fighter_state
```

No implementation should begin until this spec is accepted.

---

# P0.1 V1 Feature Count Target

The first implementation should target approximately:

```text
25 to 35 fighter-state features
```

before red/blue/diff expansion.

After model-view expansion, this may become approximately:

```text
75 to 105 model-view columns
```

This is acceptable for a first experiment but should be monitored carefully.

Do not expand P0.1 beyond this without ablation evidence.

---

# Open Questions Before Implementation

Before coding P0.1, confirm:

1. Exact raw column names in `ufc_round_stats.parquet`
2. Whether `date` exists directly in round stats or must be joined from master
3. Whether all fights have complete red/blue paired rows
4. Whether five-round fights should get special features in V2
5. Whether EWM alpha should match existing fighter-state EWM conventions
6. Whether model-view imputation should use existing production imputer rules

---

# Canonical Summary

P0.1 Round Trajectory and Decay measures how a fighter changes as rounds progress.

It should capture:

```text
output slope
late output ratio
accuracy slope
wrestling persistence
control trajectory
defensive degradation proxies
```

It must follow point-in-time rules.

It must produce fighter-state features, not raw round model inputs.

It must be validated independently before P0.2 begins.

The approved next engineering step after this spec is accepted is:

```text
implement P0.1 only
validate P0.1
run P0.1 ablation
decide keep / revise / reject
```
**
